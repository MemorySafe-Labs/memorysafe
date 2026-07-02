#!/usr/bin/env python3
"""Hyperparameter sweep for PathMNIST rare-binary (pneumonia-parity protocol)."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from benchmark_pathmnist_rare import CONFIG, build_loaders, run_policy, set_seed, try_ttest
from buffer_v14 import BufferConfig, MemorySafeBufferV14, ReservoirBuffer


@dataclass
class SweepConfig:
    name: str
    replay_prob: float = 0.80
    epochs_per_task: int = 3
    pos_quota_frac: float = 0.40
    replay_pos_frac: float = 0.45
    replay_scale: float = 1.25
    pos_risk_boost: float = 0.22
    w_risk: float = 0.50
    w_criticality: float = 0.10
    task_age_weight: float = 0.12
    min_pos_per_task: int = 30
    imbalance_ratio_neg_per_pos: int = 20


def run_config(sc: SweepConfig, seeds: List[int], device) -> Dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)
    cfg.update(
        replay_prob=sc.replay_prob,
        epochs_per_task=sc.epochs_per_task,
        pos_quota_frac=sc.pos_quota_frac,
        replay_pos_frac=sc.replay_pos_frac,
        replay_scale=sc.replay_scale,
        min_pos_per_task=sc.min_pos_per_task,
        imbalance_ratio_neg_per_pos=sc.imbalance_ratio_neg_per_pos,
    )

    ms_runs, res_runs = [], []
    for seed in seeds:
        set_seed(seed)
        train_loaders, test_loaders = build_loaders(seed, cfg)

        set_seed(seed)
        res_runs.append(
            run_policy("reservoir", ReservoirBuffer(cfg["buffer_capacity"]), train_loaders, test_loaders, device, cfg)
        )

        set_seed(seed)
        buf_cfg = BufferConfig(
            capacity=cfg["buffer_capacity"],
            pos_quota_frac=sc.pos_quota_frac,
            replay_pos_frac=sc.replay_pos_frac,
            pos_risk_boost=sc.pos_risk_boost,
            w_risk=sc.w_risk,
            w_criticality=sc.w_criticality,
            task_age_weight=sc.task_age_weight,
        )
        ms_runs.append(
            run_policy("memorysafe_v14", MemorySafeBufferV14(buf_cfg), train_loaders, test_loaders, device, cfg)
        )

    ms_auprc = [r["summary"]["combined_auprc"] for r in ms_runs]
    res_auprc = [r["summary"]["combined_auprc"] for r in res_runs]
    ttest = try_ttest(ms_auprc, res_auprc)
    wins = sum(1 for m, r in zip(ms_auprc, res_auprc) if m > r)
    return {
        "name": sc.name,
        "config": asdict(sc),
        "ms_mean": float(np.mean(ms_auprc)),
        "ms_std": float(np.std(ms_auprc)),
        "res_mean": float(np.mean(res_auprc)),
        "res_std": float(np.std(res_auprc)),
        "delta": float(np.mean(ms_auprc) - np.mean(res_auprc)),
        "wins": wins,
        "n_seeds": len(seeds),
        "p": ttest["p"],
        "ms_values": ms_auprc,
        "res_values": res_auprc,
    }


def default_grid() -> List[SweepConfig]:
    return [
        SweepConfig("pneumonia_baseline"),
        SweepConfig("high_replay", replay_prob=0.88, replay_scale=1.5),
        SweepConfig("wide_pos_quota", pos_quota_frac=0.45, replay_pos_frac=0.50),
        SweepConfig("aggressive_pos", pos_quota_frac=0.50, replay_pos_frac=0.55, replay_scale=1.4),
        SweepConfig("long_train", epochs_per_task=5, replay_prob=0.82),
        SweepConfig("more_pos", min_pos_per_task=40, imbalance_ratio_neg_per_pos=15),
        SweepConfig("risk_heavy", pos_risk_boost=0.30, w_risk=0.58, task_age_weight=0.18),
        SweepConfig("v142_clone", replay_prob=0.80, pos_quota_frac=0.40, replay_pos_frac=0.45, replay_scale=1.25),
        SweepConfig("max_govern", replay_prob=0.90, replay_scale=1.6, pos_quota_frac=0.48,
                    replay_pos_frac=0.52, pos_risk_boost=0.28, epochs_per_task=4),
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/pathmnist_rare_tune")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    seeds = [args.start_seed + i for i in range(args.seeds)]

    results: List[Dict[str, Any]] = []
    out_path = os.path.join(args.save_dir, "sweep_results.json")

    for sc in default_grid():
        print(f"\n{'='*60}\nSWEEP: {sc.name}\n{'='*60}")
        row = run_config(sc, seeds, device)
        results.append(row)
        results.sort(key=lambda r: (-(r["p"] is not None and r["p"] < 0.05), r["delta"]), reverse=True)
        with open(out_path, "w") as f:
            json.dump({"seeds": seeds, "results": results}, f, indent=2)
        print(
            f"{sc.name}: MS {row['ms_mean']:.4f} vs RES {row['res_mean']:.4f} "
            f"(delta {row['delta']:+.4f}, wins {row['wins']}/{row['n_seeds']}, p={row['p']})"
        )

    results.sort(key=lambda r: r["delta"], reverse=True)
    print("\nTOP BY DELTA:")
    for r in results[:5]:
        print(f"  {r['name']:18s} delta={r['delta']:+.4f} p={r['p']} wins={r['wins']}/{r['n_seeds']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
