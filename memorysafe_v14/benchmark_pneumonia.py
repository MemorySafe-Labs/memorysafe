#!/usr/bin/env python3
"""
MemorySafe v14 — PneumoniaMNIST replication benchmark (multi-seed).

Compares:
  - reservoir (uniform replay)
  - loss_priority (GSS-style high-loss buffer)
  - memorysafe_v14 (governed ProtectScore + pos quota + stratified replay)

Metrics aligned with MemorySafe Labs published results:
  - Mean AUPRC across tasks (final)
  - Minority (positive) recall
  - Task-0 retention recall
  - Recall @ 1% FPR
  - Buffer memory footprint (MB)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PneumoniaMNIST

from buffer_v14 import BufferConfig, LossPriorityBuffer, MemorySafeBufferV14, ReservoirBuffer
from config_v14 import CANONICAL, PROTOCOL_VERSION, to_dict
from export_results import format_report
from train_loop import train_continual


CONFIG = to_dict(CANONICAL)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_task_splits(dataset_len: int, n_tasks: int, seed: int) -> List[List[int]]:
    idxs = list(range(dataset_len))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    return [c.tolist() for c in np.array_split(np.array(idxs), n_tasks)]


def get_label(dataset, idx: int) -> int:
    _, y = dataset[idx]
    if isinstance(y, (np.ndarray, list, tuple)):
        return int(y[0])
    return int(y)


def enforce_imbalance(dataset, indices: List[int], neg_per_pos: int, min_pos: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    pos, neg = [], []
    for i in indices:
        (pos if get_label(dataset, i) == 1 else neg).append(i)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_keep = pos[:min_pos] if len(pos) >= min_pos else pos[:]
    n_pos = len(pos_keep)
    neg_keep = neg[: min(len(neg), 500)] if n_pos == 0 else neg[: min(len(neg), n_pos * neg_per_pos)]
    final = pos_keep + neg_keep
    rng.shuffle(final)
    return final


def build_loaders(seed: int):
    tx = transforms.Compose([transforms.ToTensor()])
    train_data = PneumoniaMNIST(split="train", download=True, transform=tx)
    test_data = PneumoniaMNIST(split="test", download=True, transform=tx)

    train_splits = make_task_splits(len(train_data), CONFIG["n_tasks"], seed)
    test_splits = make_task_splits(len(test_data), CONFIG["n_tasks"], seed + 1)
    ratio = CONFIG["imbalance_ratio_neg_per_pos"]
    min_pos = CONFIG["min_pos_per_task"]

    train_idx = [
        enforce_imbalance(train_data, s, ratio, min_pos, seed + k)
        for k, s in enumerate(train_splits)
    ]
    test_idx = [
        enforce_imbalance(test_data, s, ratio, max(10, min_pos // 3), seed + 100 + k)
        for k, s in enumerate(test_splits)
    ]

    bs = CONFIG["batch_size"]
    train_loaders = [DataLoader(Subset(train_data, i), batch_size=bs, shuffle=True, num_workers=0) for i in train_idx]
    test_loaders = [DataLoader(Subset(test_data, i), batch_size=bs, shuffle=False, num_workers=0) for i in test_idx]
    return train_loaders, test_loaders


def run_policy(name: str, buffer, train_loaders, test_loaders, device, cfg: dict) -> Dict[str, Any]:
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
        recall_feedback=cfg.get("recall_feedback", False) and name.startswith("memorysafe"),
        recall_target=cfg.get("recall_target", 0.72),
        recall_feedback_gain=cfg.get("recall_feedback_gain", 0.25),
        replay_scale=cfg.get("replay_scale", 1.25),
        replay_pos_frac=cfg.get("replay_pos_frac"),
        health_feedback=cfg.get("health_feedback", False) and name.startswith("memorysafe"),
    )


def aggregate_summaries(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "final_avg_recall", "final_avg_auprc", "combined_auprc", "combined_recall_pos",
        "task0_retention_recall", "recall_at_1pct_fpr", "combined_recall_at_1pct_fpr", "buffer_memory_mb",
        "mean_mvi", "mean_protect", "fri", "coverage_index", "mean_replay_exposure",
    ]
    out = {}
    for k in keys:
        vals = [r["summary"][k] for r in runs]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    return out


def try_ttest(a: List[float], b: List[float]) -> Dict[str, float]:
    try:
        from scipy import stats
        t, p = stats.ttest_rel(a, b)
        return {"t": float(t), "p": float(p)}
    except Exception:
        return {"t": None, "p": None}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but not available on this machine")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine")
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/pneumonia_benchmark")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["reservoir", "loss_priority", "memorysafe_v14"],
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--recall-feedback",
        action="store_true",
        help="Enable light AR: bump replay prob when eval recall is below target (MemorySafe only)",
    )
    parser.add_argument(
        "--health-feedback",
        action="store_true",
        help="Enable MemoryHealthController (dual-axis self-tune) on MemorySafe policies",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)
    cfg = dict(CONFIG)
    if args.recall_feedback:
        cfg["recall_feedback"] = True
    if args.health_feedback:
        cfg["health_feedback"] = True
    print(f"Device: {device} | Protocol: {PROTOCOL_VERSION}")

    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in args.policies}

    for s in range(args.seeds):
        seed = args.start_seed + s
        set_seed(seed)
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        train_loaders, test_loaders = build_loaders(seed)

        if "reservoir" in args.policies:
            set_seed(seed)
            hist = run_policy("reservoir", ReservoirBuffer(cfg["buffer_capacity"]), train_loaders, test_loaders, device, cfg)
            all_results["reservoir"].append(hist)

        if "loss_priority" in args.policies:
            set_seed(seed)
            hist = run_policy("loss_priority", LossPriorityBuffer(cfg["buffer_capacity"]), train_loaders, test_loaders, device, cfg)
            all_results["loss_priority"].append(hist)

        if "memorysafe_v14" in args.policies:
            set_seed(seed)
            buf_cfg = BufferConfig(
                capacity=cfg["buffer_capacity"],
                pos_quota_frac=cfg["pos_quota_frac"],
                replay_pos_frac=cfg["replay_pos_frac"],
            )
            hist = run_policy("memorysafe_v14", MemorySafeBufferV14(buf_cfg), train_loaders, test_loaders, device, cfg)
            all_results["memorysafe_v14"].append(hist)

        if "memorysafe_compact" in args.policies:
            set_seed(seed)
            compact_capacity = cfg.get("compact_capacity", max(80, int(cfg["buffer_capacity"] * 0.16)))
            buf_cfg = BufferConfig(
                capacity=compact_capacity,
                pos_quota_frac=0.40,
                replay_pos_frac=0.40,
            )
            hist = run_policy(
                f"memorysafe_compact_{compact_capacity}",
                MemorySafeBufferV14(buf_cfg),
                train_loaders,
                test_loaders,
                device,
                cfg,
            )
            all_results["memorysafe_compact"].append(hist)

    report = {
        "config": cfg,
        "n_seeds": args.seeds,
        "aggregates": {p: aggregate_summaries(runs) for p, runs in all_results.items()},
    }

    if "memorysafe_v14" in all_results:
        ms = all_results["memorysafe_v14"]
        ms_auprc = [r["summary"]["combined_auprc"] for r in ms]
        if "reservoir" in all_results:
            res = all_results["reservoir"]
            report["memorysafe_vs_reservoir"] = {
                "auprc": try_ttest(ms_auprc, [r["summary"]["combined_auprc"] for r in res]),
                "recall": try_ttest(
                    [r["summary"]["combined_recall_pos"] for r in ms],
                    [r["summary"]["combined_recall_pos"] for r in res],
                ),
            }
        if "loss_priority" in all_results:
            lp = all_results["loss_priority"]
            report["memorysafe_vs_loss_priority"] = {
                "auprc": try_ttest(ms_auprc, [r["summary"]["combined_auprc"] for r in lp]),
            }

    out_path = os.path.join(args.save_dir, "benchmark_report.json")
    with open(out_path, "w") as f:
        json.dump({"report": report, "raw": all_results}, f, indent=2)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for policy, agg in report["aggregates"].items():
        a = agg["combined_auprc"]
        r = agg["combined_recall_pos"]
        t0 = agg["task0_retention_recall"]
        fpr = agg["combined_recall_at_1pct_fpr"]
        print(
            f"{policy:18s} | AUPRC {a['mean']:.4f}±{a['std']:.4f} | "
            f"Recall {r['mean']:.4f}±{r['std']:.4f} | "
            f"Task0 {t0['mean']:.4f}±{t0['std']:.4f} | "
            f"R@1%FPR {fpr['mean']:.4f}±{fpr['std']:.4f}"
        )

    if "memorysafe_vs_reservoir" in report:
        p = report["memorysafe_vs_reservoir"]["auprc"]["p"]
        print(f"\nMemorySafe vs Reservoir AUPRC paired t-test p={p}")

    print(f"\nSaved: {out_path}")

    results_md = os.path.join(args.save_dir, "RESULTS.md")
    with open(results_md, "w") as f:
        f.write(format_report({"report": report, "raw": all_results}))
    print(f"Saved: {results_md}")
    return report


if __name__ == "__main__":
    main()
