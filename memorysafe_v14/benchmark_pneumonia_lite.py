#!/usr/bin/env python3
"""
MemorySafe Lite — cost-reduction benchmark (PneumoniaMNIST).

Compares three tiers on the same 5-task harness:
  - reservoir          — 500-cap baseline (standard replay)
  - memorysafe_v14     — full SOTA knobs (500-cap, high replay)
  - memorysafe_lite    — 80-cap, replay_prob 0.55, scale 1.0, replay_bs 64, health-gated

Logs wall_time_sec and replay_step_frac per run for compute comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from benchmark_pneumonia import (
    aggregate_summaries,
    build_loaders,
    resolve_device,
    run_policy,
    set_seed,
    try_ttest,
)
from buffer_v14 import BufferConfig, MemorySafeBufferV14, ReservoirBuffer
from config_v14 import LITE, PROTOCOL_LITE_VERSION, to_dict, lite_to_dict
from export_results import format_report
from memory_health import lite_controller_config

SOTA_CFG = to_dict()
LITE_CFG = lite_to_dict()


def aggregate_compute(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["wall_time_sec", "replay_step_frac", "buffer_memory_mb"]
    out = {}
    for k in keys:
        vals = [r["summary"][k] for r in runs]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    return out


def run_lite_policy(
    name: str,
    buffer,
    train_loaders,
    test_loaders,
    device,
    cfg: dict,
) -> Dict[str, Any]:
    from train_loop import train_continual

    return train_continual(
        name,
        buffer,
        train_loaders,
        test_loaders,
        device,
        replay_prob=cfg["replay_prob"],
        replay_bs=cfg["replay_batch_size"],
        epochs_per_task=cfg["epochs_per_task"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        mix_loss=cfg["value_mix_loss"],
        mix_unc=cfg["value_mix_unc"],
        pos_quota_frac=cfg["pos_quota_frac"],
        replay_scale=cfg["replay_scale"],
        replay_pos_frac=cfg.get("replay_pos_frac"),
        health_feedback=cfg.get("health_feedback", False),
        health_controller_config=lite_controller_config("binary"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MemorySafe Lite cost benchmark")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/pneumonia_lite_3seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device} | Lite protocol: {PROTOCOL_LITE_VERSION}")

    policies = ["reservoir", "memorysafe_v14", "memorysafe_lite"]
    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in policies}

    for s in range(args.seeds):
        seed = args.start_seed + s
        set_seed(seed)
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        train_loaders, test_loaders = build_loaders(seed)

        set_seed(seed)
        hist = run_policy(
            "reservoir",
            ReservoirBuffer(SOTA_CFG["buffer_capacity"]),
            train_loaders,
            test_loaders,
            device,
            SOTA_CFG,
        )
        all_results["reservoir"].append(hist)

        set_seed(seed)
        buf_cfg = BufferConfig(
            capacity=SOTA_CFG["buffer_capacity"],
            pos_quota_frac=SOTA_CFG["pos_quota_frac"],
            replay_pos_frac=SOTA_CFG["replay_pos_frac"],
        )
        hist = run_policy(
            "memorysafe_v14",
            MemorySafeBufferV14(buf_cfg),
            train_loaders,
            test_loaders,
            device,
            SOTA_CFG,
        )
        all_results["memorysafe_v14"].append(hist)

        set_seed(seed)
        lite_buf = BufferConfig(
            capacity=LITE_CFG["buffer_capacity"],
            pos_quota_frac=LITE_CFG["pos_quota_frac"],
            replay_pos_frac=LITE_CFG["replay_pos_frac"],
        )
        hist = run_lite_policy(
            "memorysafe_lite",
            MemorySafeBufferV14(lite_buf),
            train_loaders,
            test_loaders,
            device,
            LITE_CFG,
        )
        all_results["memorysafe_lite"].append(hist)

    report: Dict[str, Any] = {
        "config": {"sota": SOTA_CFG, "lite": LITE_CFG},
        "n_seeds": args.seeds,
        "device": str(device),
        "aggregates": {p: aggregate_summaries(runs) for p, runs in all_results.items()},
        "compute": {p: aggregate_compute(runs) for p, runs in all_results.items()},
    }

    lite_auprc = [r["summary"]["combined_auprc"] for r in all_results["memorysafe_lite"]]
    sota_auprc = [r["summary"]["combined_auprc"] for r in all_results["memorysafe_v14"]]
    res_auprc = [r["summary"]["combined_auprc"] for r in all_results["reservoir"]]
    report["comparisons"] = {
        "lite_vs_reservoir_auprc": try_ttest(lite_auprc, res_auprc),
        "lite_vs_sota_auprc": try_ttest(lite_auprc, sota_auprc),
        "lite_vs_reservoir_recall": try_ttest(
            [r["summary"]["combined_recall_pos"] for r in all_results["memorysafe_lite"]],
            [r["summary"]["combined_recall_pos"] for r in all_results["reservoir"]],
        ),
    }

    sota_time = report["compute"]["memorysafe_v14"]["wall_time_sec"]["mean"]
    lite_time = report["compute"]["memorysafe_lite"]["wall_time_sec"]["mean"]
    res_time = report["compute"]["reservoir"]["wall_time_sec"]["mean"]
    if sota_time > 0:
        report["compute_savings_vs_sota"] = {
            "lite_wall_time_ratio": lite_time / sota_time,
            "lite_wall_time_pct_saved": 1.0 - lite_time / sota_time,
            "reservoir_wall_time_ratio": res_time / sota_time,
        }

    out_path = os.path.join(args.save_dir, "benchmark_report.json")
    with open(out_path, "w") as f:
        json.dump({"report": report, "raw": all_results}, f, indent=2)

    print("\n" + "=" * 60)
    print("MEMORYSAFE LITE — SUMMARY")
    print("=" * 60)
    for policy in policies:
        agg = report["aggregates"][policy]
        comp = report["compute"][policy]
        a = agg["combined_auprc"]
        r = agg["combined_recall_pos"]
        wt = comp["wall_time_sec"]
        rs = comp["replay_step_frac"]
        bm = comp["buffer_memory_mb"]
        print(
            f"{policy:18s} | AUPRC {a['mean']:.4f}±{a['std']:.4f} | "
            f"Recall {r['mean']:.4f} | "
            f"time {wt['mean']:.1f}s | replay {rs['mean']:.1%} | buf {bm['mean']:.3f}MB"
        )

    if "compute_savings_vs_sota" in report:
        saved = report["compute_savings_vs_sota"]["lite_wall_time_pct_saved"] * 100
        print(f"\nLite vs SOTA wall time: {saved:.1f}% faster")
    p = report["comparisons"]["lite_vs_reservoir_auprc"]["p"]
    print(f"Lite vs Reservoir AUPRC p={p}")

    results_md = os.path.join(args.save_dir, "RESULTS.md")
    with open(results_md, "w") as f:
        f.write(format_report({"report": report, "raw": all_results}))
    print(f"\nSaved: {out_path}")
    print(f"Saved: {results_md}")


if __name__ == "__main__":
    main()
