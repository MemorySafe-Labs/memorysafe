#!/usr/bin/env python3
"""Split CIFAR-100 class-incremental benchmark (MemorySafe universal governor)."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config_universal import CIFAR100_CANONICAL, cifar100_to_dict
from governed_buffer import (
    GovernedBuffer,
    GovernedBufferConfig,
    HybridCLBuffer,
    HybridCLBufferConfig,
    ReservoirBufferUniversal,
)
from quota_policies import FragilityCLQuota, TailClassQuota, UniformClassQuota
from train_loop_cifar import train_continual_cifar


CONFIG = cifar100_to_dict(CIFAR100_CANONICAL)


def make_hybrid_buffer(cfg: dict) -> HybridCLBuffer:
    hybrid_cfg = HybridCLBufferConfig(
        capacity=cfg["buffer_capacity"],
        core_frac=cfg.get("hybrid_core_frac", 0.90),
        core_replay_frac=cfg.get("hybrid_core_replay_frac", 0.85),
        promote_frac=cfg.get("hybrid_promote_frac", 0.08),
    )
    shell_cfg = GovernedBufferConfig(
        task_age_weight=0.35,
        w_rarity=0.08,
    )
    shell_quota = FragilityCLQuota(
        min_frac_per_class=0.0,
        old_task_boost=cfg.get("hybrid_old_task_boost", 2.0),
        rare_class_boost=1.4,
    )
    return HybridCLBuffer(hybrid_cfg, shell_cfg, shell_quota)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_splits(n_classes: int, n_tasks: int) -> List[List[int]]:
    per = n_classes // n_tasks
    return [list(range(i * per, (i + 1) * per)) for i in range(n_tasks)]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_loaders(seed: int, cfg: dict | None = None):
    cfg = cfg or CONFIG
    tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_full = datasets.CIFAR100(root="./data", train=True, download=True, transform=tx)
    test_full = datasets.CIFAR100(root="./data", train=False, download=True, transform=tx)

    splits = class_splits(100, cfg["n_tasks"])
    train_loaders, test_loaders = [], []
    bs = cfg["batch_size"]

    for task_classes in splits:
        train_idx = [i for i, (_, y) in enumerate(train_full) if y in task_classes]
        test_idx = [i for i, (_, y) in enumerate(test_full) if y in task_classes]
        rng = random.Random(seed)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        train_loaders.append(DataLoader(Subset(train_full, train_idx), batch_size=bs, shuffle=True, num_workers=0))
        test_loaders.append(DataLoader(Subset(test_full, test_idx), batch_size=bs, shuffle=False, num_workers=0))
    return train_loaders, test_loaders


def run_policy(name: str, buffer, train_loaders, test_loaders, device, cfg: dict) -> Dict[str, Any]:
    return train_continual_cifar(
        name,
        buffer,
        train_loaders,
        test_loaders,
        device,
        n_classes=100,
        replay_prob=cfg["replay_prob"],
        replay_bs=cfg["replay_batch_size"],
        epochs_per_task=cfg["epochs_per_task"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        mix_loss=cfg["value_mix_loss"],
        mix_unc=cfg["value_mix_unc"],
        replay_scale=cfg.get("replay_scale", 1.2),
        health_feedback=cfg.get("health_feedback", False) and name.startswith("memorysafe"),
        min_frac_per_class=cfg.get("min_frac_per_class", 0.06),
        old_task_boost=getattr(
            getattr(getattr(buffer, "shell", buffer), "quota", getattr(buffer, "quota", None)),
            "old_task_boost",
            2.2,
        ),
    )


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "final_avg_acc", "combined_acc", "mean_class_acc", "task0_retention_acc", "buffer_memory_mb",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/cifar100_benchmark")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["reservoir", "memorysafe_hybrid", "memorysafe_fragility"],
    )
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--epochs-per-task", type=int, default=5)
    parser.add_argument("--replay-scale", type=float, default=1.8)
    parser.add_argument("--replay-prob", type=float, default=0.78)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--health-feedback",
        action="store_true",
        help="Enable MemoryHealthController on MemorySafe policies",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)
    cfg = dict(CONFIG)
    cfg["n_tasks"] = args.n_tasks
    cfg["classes_per_task"] = 100 // args.n_tasks
    cfg["epochs_per_task"] = args.epochs_per_task
    cfg["replay_scale"] = args.replay_scale
    cfg["replay_prob"] = args.replay_prob
    cfg["protocol_version"] = f"v14.3-cifar100-{args.n_tasks}task-fragility-cl"
    if args.health_feedback:
        cfg["health_feedback"] = True
    print(f"Device: {device} | Protocol: {cfg['protocol_version']}")

    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in args.policies}

    for s in range(args.seeds):
        seed = args.start_seed + s
        set_seed(seed)
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        train_loaders, test_loaders = build_loaders(seed, cfg)

        if "reservoir" in args.policies:
            set_seed(seed)
            hist = run_policy("reservoir", ReservoirBufferUniversal(cfg["buffer_capacity"]), train_loaders, test_loaders, device, cfg)
            all_results["reservoir"].append(hist)

        if "memorysafe" in args.policies:
            set_seed(seed)
            buf_cfg = GovernedBufferConfig(capacity=cfg["buffer_capacity"])
            quota = UniformClassQuota(min_frac_per_class=cfg["min_frac_per_class"])
            hist = run_policy("memorysafe_governed", GovernedBuffer(buf_cfg, quota), train_loaders, test_loaders, device, cfg)
            all_results["memorysafe"].append(hist)

        if "memorysafe_tail" in args.policies:
            set_seed(seed)
            buf_cfg = GovernedBufferConfig(capacity=cfg["buffer_capacity"], task_age_weight=0.22)
            quota = TailClassQuota(min_frac_per_class=cfg["min_frac_per_class"], old_class_boost=2.2)
            hist = run_policy("memorysafe_tail", GovernedBuffer(buf_cfg, quota), train_loaders, test_loaders, device, cfg)
            all_results["memorysafe_tail"].append(hist)

        if "memorysafe_fragility" in args.policies:
            set_seed(seed)
            buf_cfg = GovernedBufferConfig(
                capacity=cfg["buffer_capacity"],
                task_age_weight=0.22,
                w_rarity=0.08,
            )
            quota = FragilityCLQuota(min_frac_per_class=cfg["min_frac_per_class"], old_task_boost=2.2, rare_class_boost=1.8)
            hist = run_policy("memorysafe_fragility", GovernedBuffer(buf_cfg, quota), train_loaders, test_loaders, device, cfg)
            all_results["memorysafe_fragility"].append(hist)

        if "memorysafe_hybrid" in args.policies:
            set_seed(seed)
            hybrid_cfg = dict(cfg)
            hybrid_cfg["replay_scale"] = cfg.get("hybrid_replay_scale", 1.05)
            hist = run_policy(
                "memorysafe_hybrid",
                make_hybrid_buffer(hybrid_cfg),
                train_loaders,
                test_loaders,
                device,
                hybrid_cfg,
            )
            all_results["memorysafe_hybrid"].append(hist)

    report = {
        "config": cfg,
        "n_seeds": args.seeds,
        "aggregates": {p: aggregate(runs) for p, runs in all_results.items()},
    }
    for ms_key in ("memorysafe", "memorysafe_tail", "memorysafe_fragility", "memorysafe_hybrid"):
        if ms_key in all_results and "reservoir" in all_results:
            ms = [r["summary"]["combined_acc"] for r in all_results[ms_key]]
            res = [r["summary"]["combined_acc"] for r in all_results["reservoir"]]
            report[f"{ms_key}_vs_reservoir"] = {"combined_acc": try_ttest(ms, res)}

    out_path = os.path.join(args.save_dir, "benchmark_report.json")
    payload = {"report": report, "raw": all_results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 60)
    print("CIFAR-100 SUMMARY")
    print("=" * 60)
    for policy, agg in report["aggregates"].items():
        a = agg["combined_acc"]
        t0 = agg["task0_retention_acc"]
        print(f"{policy:20s} | combined acc {a['mean']:.4f}±{a['std']:.4f} | task0 {t0['mean']:.4f}±{t0['std']:.4f}")
    for ms_key in ("memorysafe", "memorysafe_tail", "memorysafe_fragility", "memorysafe_hybrid"):
        vs = report.get(f"{ms_key}_vs_reservoir")
        if vs:
            print(f"\n{ms_key} vs Reservoir combined_acc p={vs['combined_acc']['p']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()