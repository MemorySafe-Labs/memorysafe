#!/usr/bin/env python3
"""CIFAR-100 hyperparameter sweep — MemorySafe tail-quota vs reservoir."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from benchmark_cifar100 import CONFIG, build_loaders, run_policy, set_seed, try_ttest
from governed_buffer import GovernedBuffer, GovernedBufferConfig, ReservoirBufferUniversal
from quota_policies import TailClassQuota


@dataclass
class CIFARSweepConfig:
    name: str
    replay_prob: float = 0.85
    replay_scale: float = 1.5
    epochs_per_task: int = 4
    min_frac_per_class: float = 0.06
    old_class_boost: float = 1.5
    task_age_weight: float = 0.12
    w_risk: float = 0.50
    w_value: float = 0.35
    mvi_ema: float = 0.70


def run_config(sc: CIFARSweepConfig, seeds: List[int], device, n_tasks: int = 5) -> Dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)
    cfg.update(
        n_tasks=n_tasks,
        classes_per_task=100 // n_tasks,
        replay_prob=sc.replay_prob,
        replay_scale=sc.replay_scale,
        epochs_per_task=sc.epochs_per_task,
        min_frac_per_class=sc.min_frac_per_class,
        old_class_boost=sc.old_class_boost,
        task_age_weight=sc.task_age_weight,
        w_risk=sc.w_risk,
        quota_mode="tail",
    )

    ms_runs, res_runs = [], []
    for seed in seeds:
        set_seed(seed)
        train_loaders, test_loaders = build_loaders(seed, cfg)

        set_seed(seed)
        res_runs.append(
            run_policy(
                "reservoir",
                ReservoirBufferUniversal(cfg["buffer_capacity"]),
                train_loaders,
                test_loaders,
                device,
                cfg,
            )
        )

        set_seed(seed)
        quota = TailClassQuota(
            min_frac_per_class=sc.min_frac_per_class,
            old_class_boost=sc.old_class_boost,
        )
        buf_cfg = GovernedBufferConfig(
            capacity=cfg["buffer_capacity"],
            task_age_weight=sc.task_age_weight,
            w_risk=sc.w_risk,
            w_value=sc.w_value,
            mvi_ema=sc.mvi_ema,
        )
        ms_runs.append(
            run_policy(
                "memorysafe_governed",
                GovernedBuffer(buf_cfg, quota),
                train_loaders,
                test_loaders,
                device,
                cfg,
            )
        )

    ms_acc = [r["summary"]["final_avg_acc"] for r in ms_runs]
    res_acc = [r["summary"]["final_avg_acc"] for r in res_runs]
    ms_t0 = [r["summary"]["task0_retention_acc"] for r in ms_runs]
    res_t0 = [r["summary"]["task0_retention_acc"] for r in res_runs]
    ttest = try_ttest(ms_acc, res_acc)
    wins = sum(1 for m, r in zip(ms_acc, res_acc) if m > r)
    return {
        "name": sc.name,
        "config": asdict(sc),
        "ms_mean": float(np.mean(ms_acc)),
        "ms_std": float(np.std(ms_acc)),
        "res_mean": float(np.mean(res_acc)),
        "res_std": float(np.std(res_acc)),
        "delta": float(np.mean(ms_acc) - np.mean(res_acc)),
        "ms_t0_mean": float(np.mean(ms_t0)),
        "res_t0_mean": float(np.mean(res_t0)),
        "wins": wins,
        "n_seeds": len(seeds),
        "p": ttest["p"],
        "ms_values": ms_acc,
        "res_values": res_acc,
    }


def default_grid() -> List[CIFARSweepConfig]:
    return [
        CIFARSweepConfig("baseline"),
        CIFARSweepConfig("aggressive_replay", replay_prob=0.90, replay_scale=1.75),
        CIFARSweepConfig("strong_tail", old_class_boost=2.5, task_age_weight=0.20),
        CIFARSweepConfig("wide_floor", min_frac_per_class=0.08, old_class_boost=2.0),
        CIFARSweepConfig("age_heavy", task_age_weight=0.28, w_risk=0.62, old_class_boost=2.2),
        CIFARSweepConfig("long_train", epochs_per_task=6, replay_prob=0.78, replay_scale=1.4),
        CIFARSweepConfig("pneumonia_style", replay_prob=0.80, replay_scale=1.25, min_frac_per_class=0.06),
        CIFARSweepConfig("balanced_v2", epochs_per_task=5, replay_prob=0.82, replay_scale=1.6,
                         min_frac_per_class=0.07, old_class_boost=2.0, task_age_weight=0.18),
        CIFARSweepConfig("high_mvi", mvi_ema=0.55, w_risk=0.58, old_class_boost=2.0),
        CIFARSweepConfig("max_tail", min_frac_per_class=0.08, old_class_boost=3.0,
                         replay_prob=0.88, replay_scale=1.65, task_age_weight=0.25),
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="runs/cifar_tune_sweep")
    parser.add_argument("--configs", nargs="*", help="Run only named configs from the grid")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [args.start_seed + i for i in range(args.seeds)]

    grid = default_grid()
    if args.configs:
        names = set(args.configs)
        grid = [c for c in grid if c.name in names]

    results: List[Dict[str, Any]] = []
    out_path = os.path.join(args.save_dir, "sweep_results.json")

    for sc in grid:
        print(f"\n{'='*60}\nSWEEP: {sc.name}\n{'='*60}")
        row = run_config(sc, seeds, device, n_tasks=args.n_tasks)
        results.append(row)
        results.sort(key=lambda r: r["delta"], reverse=True)
        out = {"n_tasks": args.n_tasks, "seeds": seeds, "results": results}
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(
            f"{sc.name}: MS {row['ms_mean']:.4f}±{row['ms_std']:.4f} vs "
            f"RES {row['res_mean']:.4f} (delta {row['delta']:+.4f}, "
            f"wins {row['wins']}/{row['n_seeds']}, p={row['p']})"
        )

    print("\n" + "=" * 60)
    print("TOP CONFIGS (by delta final_avg_acc)")
    print("=" * 60)
    for r in results[:5]:
        print(
            f"{r['name']:20s} delta={r['delta']:+.4f}  "
            f"MS={r['ms_mean']:.4f}  RES={r['res_mean']:.4f}  "
            f"wins={r['wins']}/{r['n_seeds']}  p={r['p']}"
        )
    best = results[0]
    print(f"\nBEST: {best['name']} | apply to config_universal.py")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
