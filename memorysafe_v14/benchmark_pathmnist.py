#!/usr/bin/env python3
"""
MemorySafe v14 — PathMNIST class-incremental benchmark (multi-seed).

Compares:
  - reservoir (uniform replay)
  - loss_priority (GSS-style high-loss buffer)
  - memorysafe_governed (GovernedBuffer + TailClassQuota)

Protocol: v14.2-pathmnist-5task-classil — mirrors pneumonia 5-task / 10-seed structure
on 9-class pathology tiles with subsampled tasks for practical runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from medmnist import PathMNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from buffer_v14 import LossPriorityBuffer, ReservoirBuffer
from config_pathmnist import CANONICAL, PROTOCOL_VERSION, task_splits, to_dict

from governed_buffer import GovernedBuffer, GovernedBufferConfig
from quota_policies import FragilityCLQuota, TailClassQuota
from train_loop_pathmnist import train_continual_pathmnist


CONFIG = to_dict(CANONICAL)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_label(dataset, idx: int) -> int:
    _, y = dataset[idx]
    if isinstance(y, (np.ndarray, list, tuple)):
        return int(y[0])
    return int(y)


def subsample_indices(indices: List[int], cap: int, seed: int) -> List[int]:
    if len(indices) <= cap:
        return indices
    rng = random.Random(seed)
    out = indices[:]
    rng.shuffle(out)
    return out[:cap]


def build_loaders(seed: int):
    tx = transforms.Compose([transforms.ToTensor()])
    train_data = PathMNIST(split="train", download=True, transform=tx)
    test_data = PathMNIST(split="test", download=True, transform=tx)

    splits = task_splits()
    train_loaders, test_loaders = [], []
    bs = CONFIG["batch_size"]
    max_train = CONFIG["max_train_per_task"]
    max_test = CONFIG["max_test_per_task"]

    for k, task_classes in enumerate(splits):
        train_idx = [i for i in range(len(train_data)) if get_label(train_data, i) in task_classes]
        test_idx = [i for i in range(len(test_data)) if get_label(test_data, i) in task_classes]
        train_idx = subsample_indices(train_idx, max_train, seed + k)
        test_idx = subsample_indices(test_idx, max_test, seed + 100 + k)
        rng = random.Random(seed + k)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        train_loaders.append(
            DataLoader(Subset(train_data, train_idx), batch_size=bs, shuffle=True, num_workers=0)
        )
        test_loaders.append(
            DataLoader(Subset(test_data, test_idx), batch_size=bs, shuffle=False, num_workers=0)
        )
    return train_loaders, test_loaders


def run_policy(name: str, buffer, train_loaders, test_loaders, device, cfg: dict) -> Dict[str, Any]:
    return train_continual_pathmnist(
        name,
        buffer,
        train_loaders,
        test_loaders,
        device,
        n_classes=cfg["n_classes"],
        in_channels=cfg.get("in_channels", 3),
        replay_prob=cfg["replay_prob"],
        replay_bs=cfg["replay_batch_size"],
        epochs_per_task=cfg["epochs_per_task"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        mix_loss=cfg["value_mix_loss"],
        mix_unc=cfg["value_mix_unc"],
        replay_scale=cfg.get("replay_scale", 1.25),
    )


def aggregate_summaries(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "final_avg_acc", "combined_acc", "mean_class_acc", "tail_class_acc",
        "task0_retention_acc", "task0_mean_class_acc", "buffer_memory_mb",
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


def format_pathmnist_report(data: dict) -> str:
    report = data.get("report", data)
    cfg = report.get("config", {})
    agg = report.get("aggregates", {})
    lines = [
        "# MemorySafe v14 — PathMNIST Benchmark Results",
        "",
        f"**Protocol:** {cfg.get('protocol_version', 'unknown')}",
        f"**Seeds:** {report.get('n_seeds', '?')}",
        "",
        "## Summary (combined accuracy = primary metric)",
        "",
        "| Policy | Combined acc | Mean class acc | Tail class acc | Task-0 acc | Buffer MB |",
        "|--------|--------------|----------------|----------------|------------|-----------|",
    ]
    for policy, stats in agg.items():
        c = stats["combined_acc"]
        m = stats["mean_class_acc"]
        t = stats["tail_class_acc"]
        t0 = stats["task0_retention_acc"]
        mem = stats.get("buffer_memory_mb", {}).get("mean", 0)
        lines.append(
            f"| {policy} | {c['mean']:.4f} ± {c['std']:.4f} | "
            f"{m['mean']:.4f} ± {m['std']:.4f} | "
            f"{t['mean']:.4f} ± {t['std']:.4f} | "
            f"{t0['mean']:.4f} ± {t0['std']:.4f} | {mem:.2f} |"
        )
    vs = report.get("memorysafe_vs_reservoir")
    if vs:
        p = vs.get("combined_acc", {}).get("p")
        lines.extend(["", f"**MemorySafe vs Reservoir paired t-test p={p}**"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/pathmnist_10seed")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["reservoir", "loss_priority", "memorysafe_governed", "memorysafe_fragility"],
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)
    cfg = dict(CONFIG)
    print(f"Device: {device} | Protocol: {PROTOCOL_VERSION}")

    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in args.policies}

    for s in range(args.seeds):
        seed = args.start_seed + s
        set_seed(seed)
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        train_loaders, test_loaders = build_loaders(seed)

        if "reservoir" in args.policies:
            set_seed(seed)
            hist = run_policy(
                "reservoir",
                ReservoirBuffer(cfg["buffer_capacity"]),
                train_loaders,
                test_loaders,
                device,
                cfg,
            )
            all_results["reservoir"].append(hist)

        if "loss_priority" in args.policies:
            set_seed(seed)
            hist = run_policy(
                "loss_priority",
                LossPriorityBuffer(cfg["buffer_capacity"]),
                train_loaders,
                test_loaders,
                device,
                cfg,
            )
            all_results["loss_priority"].append(hist)

        if "memorysafe_governed" in args.policies:
            set_seed(seed)
            buf_cfg = GovernedBufferConfig(capacity=cfg["buffer_capacity"])
            quota = TailClassQuota(min_frac_per_class=cfg["min_frac_per_class"])
            hist = run_policy(
                "memorysafe_governed",
                GovernedBuffer(buf_cfg, quota),
                train_loaders,
                test_loaders,
                device,
                cfg,
            )
            all_results["memorysafe_governed"].append(hist)

        if "memorysafe_fragility" in args.policies:
            set_seed(seed)
            buf_cfg = GovernedBufferConfig(
                capacity=cfg["buffer_capacity"],
                task_age_weight=0.20,
                w_rarity=0.08,
            )
            quota = FragilityCLQuota(
                min_frac_per_class=cfg["min_frac_per_class"],
                old_task_boost=2.2,
                rare_class_boost=2.0,
            )
            hist = run_policy(
                "memorysafe_fragility",
                GovernedBuffer(buf_cfg, quota),
                train_loaders,
                test_loaders,
                device,
                cfg,
            )
            all_results["memorysafe_fragility"].append(hist)

    report = {
        "config": cfg,
        "n_seeds": args.seeds,
        "aggregates": {p: aggregate_summaries(runs) for p, runs in all_results.items()},
    }

    if "memorysafe_governed" in all_results:
        ms = all_results["memorysafe_governed"]
        ms_acc = [r["summary"]["combined_acc"] for r in ms]
        if "reservoir" in all_results:
            res = all_results["reservoir"]
            report["memorysafe_vs_reservoir"] = {
                "combined_acc": try_ttest(ms_acc, [r["summary"]["combined_acc"] for r in res]),
                "task0_retention_acc": try_ttest(
                    [r["summary"]["task0_retention_acc"] for r in ms],
                    [r["summary"]["task0_retention_acc"] for r in res],
                ),
                "task0_mean_class_acc": try_ttest(
                    [r["summary"]["task0_mean_class_acc"] for r in ms],
                    [r["summary"]["task0_mean_class_acc"] for r in res],
                ),
                "mean_class_acc": try_ttest(
                    [r["summary"]["mean_class_acc"] for r in ms],
                    [r["summary"]["mean_class_acc"] for r in res],
                ),
                "tail_class_acc": try_ttest(
                    [r["summary"]["tail_class_acc"] for r in ms],
                    [r["summary"]["tail_class_acc"] for r in res],
                ),
            }
        if "loss_priority" in all_results:
            lp = all_results["loss_priority"]
            report["memorysafe_vs_loss_priority"] = {
                "combined_acc": try_ttest(ms_acc, [r["summary"]["combined_acc"] for r in lp]),
            }

    out_path = os.path.join(args.save_dir, "benchmark_report.json")
    with open(out_path, "w") as f:
        json.dump({"report": report, "raw": all_results}, f, indent=2)

    print("\n" + "=" * 60)
    print("PATHMNIST BENCHMARK SUMMARY")
    print("=" * 60)
    for policy, agg in report["aggregates"].items():
        c = agg["combined_acc"]
        m = agg["mean_class_acc"]
        t = agg["tail_class_acc"]
        t0 = agg["task0_retention_acc"]
        print(
            f"{policy:22s} | acc {c['mean']:.4f}±{c['std']:.4f} | "
            f"class {m['mean']:.4f}±{m['std']:.4f} | "
            f"tail {t['mean']:.4f}±{t['std']:.4f} | "
            f"task0 {t0['mean']:.4f}±{t0['std']:.4f}"
        )

    if "memorysafe_vs_reservoir" in report:
        p = report["memorysafe_vs_reservoir"]["combined_acc"]["p"]
        print(f"\nMemorySafe vs Reservoir combined_acc paired t-test p={p}")

    print(f"\nSaved: {out_path}")

    results_md = os.path.join(args.save_dir, "RESULTS.md")
    with open(results_md, "w") as f:
        f.write(format_pathmnist_report({"report": report, "raw": all_results}))
    print(f"Saved: {results_md}")
    return report


if __name__ == "__main__":
    main()