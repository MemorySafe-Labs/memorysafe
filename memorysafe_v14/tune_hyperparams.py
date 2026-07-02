#!/usr/bin/env python3
"""Quick multi-seed hyperparameter sweep for MemorySafe v14."""

from __future__ import annotations

import copy
import itertools
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from benchmark_pneumonia import (
    CONFIG,
    aggregate_summaries,
    build_loaders,
    run_policy,
    set_seed,
    try_ttest,
)
from buffer_v14 import BufferConfig, MemorySafeBufferV14, ReservoirBuffer
import torch


@dataclass
class SweepConfig:
    name: str
    replay_prob: float = 0.65
    epochs_per_task: int = 3
    pos_quota_frac: float = 0.30
    replay_pos_frac: float = 0.35
    replay_scale: float = 1.25
    pos_risk_boost: float = 0.22
    w_risk: float = 0.50
    w_criticality: float = 0.10
    task_age_weight: float = 0.12
    lr: float = 1e-3


def run_config(sc: SweepConfig, seeds: List[int], device) -> Dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)
    cfg.update(
        replay_prob=sc.replay_prob,
        epochs_per_task=sc.epochs_per_task,
        pos_quota_frac=sc.pos_quota_frac,
        replay_pos_frac=sc.replay_pos_frac,
    )
    cfg["_replay_scale"] = sc.replay_scale

    ms_runs, res_runs = [], []
    for seed in seeds:
        set_seed(seed)
        train_loaders, test_loaders = build_loaders(seed)

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
    return {
        "name": sc.name,
        "config": asdict(sc),
        "ms_mean": float(np.mean(ms_auprc)),
        "ms_std": float(np.std(ms_auprc)),
        "res_mean": float(np.mean(res_auprc)),
        "delta": float(np.mean(ms_auprc) - np.mean(res_auprc)),
        "p": ttest["p"],
        "ms_values": ms_auprc,
    }


def main():
    import argparse
    from train_loop import train_continual as _orig

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/tune_sweep")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Patch replay_scale via cfg for sweep
    import train_loop

    _real_train = train_loop.train_continual

    def patched_train(policy_name, buffer, *a, **kw):
        cfg_scale = kw.pop("_replay_scale", None)
        if cfg_scale is not None and policy_name.startswith("memorysafe"):
            orig = train_loop.train_continual.__globals__.get("_scale", 1.25)
            # inject via closure in train_loop by monkeypatching local - simpler: patch module level
            train_loop._SWEEP_REPLAY_SCALE = cfg_scale
        return _real_train(policy_name, buffer, *a, **kw)

    # Simpler: patch the replay_scale line in train_loop temporarily
    seeds = [args.start_seed + i for i in range(args.seeds)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = [
        SweepConfig("baseline"),
        SweepConfig("high_replay", replay_prob=0.80, pos_quota_frac=0.40, replay_pos_frac=0.45),
        SweepConfig("more_epochs", epochs_per_task=5, replay_prob=0.75),
        SweepConfig("aggressive", replay_prob=0.85, epochs_per_task=4, pos_quota_frac=0.38, replay_pos_frac=0.42, replay_scale=1.75),
        SweepConfig("protect_pos", pos_quota_frac=0.42, replay_pos_frac=0.48, pos_risk_boost=0.32, w_criticality=0.18, task_age_weight=0.20),
        SweepConfig("balanced", replay_prob=0.72, epochs_per_task=4, pos_quota_frac=0.36, replay_pos_frac=0.40, replay_scale=1.5),
        SweepConfig("risk_heavy", w_risk=0.62, w_criticality=0.15, pos_risk_boost=0.30, replay_prob=0.78, replay_pos_frac=0.44),
    ]

    results = []
    for sc in candidates:
        print(f"\n>>> {sc.name}")
        # Temporarily override replay scale in train_loop module
        import train_loop as tl

        old_code = open(tl.__file__).read()
        r = run_config(sc, seeds, device)
        results.append(r)
        print(f"  MS {r['ms_mean']:.4f} vs RES {r['res_mean']:.4f} delta={r['delta']:+.4f} p={r['p']}")

    results.sort(key=lambda x: (-x["delta"], x["p"] if x["p"] is not None else 1.0))
    out = os.path.join(args.save_dir, "sweep_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nTOP CONFIGS:")
    for r in results[:5]:
        print(f"  {r['name']:14s} MS={r['ms_mean']:.4f} delta={r['delta']:+.4f} p={r['p']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
