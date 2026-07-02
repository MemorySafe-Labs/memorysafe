#!/usr/bin/env python3
"""Sweep replay_prob for MemorySafe Lite on CPU (Pareto: AUPRC vs replay %)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from benchmark_pneumonia_lite import run_lite_policy
from benchmark_pneumonia import build_loaders, resolve_device, set_seed
from buffer_v14 import BufferConfig, MemorySafeBufferV14
from config_v14 import LITE, lite_to_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--replay-probs", nargs="+", type=float, default=[0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--save-dir", default="runs/pneumonia_lite_tune")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)
    base = lite_to_dict(LITE)
    rows: List[Dict[str, Any]] = []

    for rp in args.replay_probs:
        cfg = dict(base)
        cfg["replay_prob"] = rp
        auprcs: List[float] = []
        replays: List[float] = []
        times: List[float] = []
        for s in range(args.seeds):
            seed = args.start_seed + s
            set_seed(seed)
            train_loaders, test_loaders = build_loaders(seed)
            buf = BufferConfig(
                capacity=cfg["buffer_capacity"],
                pos_quota_frac=cfg["pos_quota_frac"],
                replay_pos_frac=cfg["replay_pos_frac"],
            )
            hist = run_lite_policy(
                f"memorysafe_lite_rp{rp:.2f}",
                MemorySafeBufferV14(buf),
                train_loaders,
                test_loaders,
                device,
                cfg,
            )
            auprcs.append(hist["summary"]["combined_auprc"])
            replays.append(hist["summary"]["replay_step_frac"])
            times.append(hist["summary"]["wall_time_sec"])

        row = {
            "replay_prob": rp,
            "combined_auprc_mean": float(np.mean(auprcs)),
            "combined_auprc_std": float(np.std(auprcs)),
            "replay_step_frac_mean": float(np.mean(replays)),
            "wall_time_sec_mean": float(np.mean(times)),
            "values": auprcs,
        }
        rows.append(row)
        print(
            f"replay_prob={rp:.2f} | AUPRC {row['combined_auprc_mean']:.4f}±{row['combined_auprc_std']:.4f} | "
            f"replay {row['replay_step_frac_mean']:.1%} | time {row['wall_time_sec_mean']:.1f}s"
        )

    best = max(rows, key=lambda r: r["combined_auprc_mean"])
    out = {"device": str(device), "n_seeds": args.seeds, "sweep": rows, "best_replay_prob": best["replay_prob"]}
    path = os.path.join(args.save_dir, "sweep_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBest replay_prob (3-seed mean AUPRC): {best['replay_prob']:.2f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
