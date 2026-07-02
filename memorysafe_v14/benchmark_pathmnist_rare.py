#!/usr/bin/env python3
"""
PathMNIST rare-tissue binary benchmark — pneumonia-parity protocol.

Positive = rare pathology classes {2, 7}; 5-task stream with induced imbalance;
MemorySafeBufferV14 vs reservoir vs loss_priority. Primary metric: combined AUPRC.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Set

import numpy as np
import torch
from medmnist import PathMNIST
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from buffer_v14 import BufferConfig, LossPriorityBuffer, MemorySafeBufferV14, ReservoirBuffer
from config_pathmnist_rare import CANONICAL, PROTOCOL_VERSION, to_dict
from train_loop_pathmnist_rare import train_continual_pathmnist_rare


CONFIG = to_dict(CANONICAL)


class BinaryRareWrapper(Dataset):
    def __init__(self, base: Dataset, rare_classes: Set[int]):
        self.base = base
        self.rare = rare_classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if isinstance(y, (np.ndarray, list, tuple)):
            orig = int(y[0])
        else:
            orig = int(y)
        label = 1 if orig in self.rare else 0
        return x, np.array([label], dtype=np.int64)


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


def build_loaders(seed: int, cfg: dict | None = None):
    cfg = cfg or CONFIG
    rare = set(cfg["rare_classes"])
    tx = transforms.Compose([transforms.ToTensor()])
    train_base = PathMNIST(split="train", download=True, transform=tx)
    test_base = PathMNIST(split="test", download=True, transform=tx)
    train_data = BinaryRareWrapper(train_base, rare)
    test_data = BinaryRareWrapper(test_base, rare)

    train_splits = make_task_splits(len(train_data), cfg["n_tasks"], seed)
    test_splits = make_task_splits(len(test_data), cfg["n_tasks"], seed + 1)
    ratio = cfg["imbalance_ratio_neg_per_pos"]
    min_pos = cfg["min_pos_per_task"]

    train_idx = [enforce_imbalance(train_data, s, ratio, min_pos, seed + k) for k, s in enumerate(train_splits)]
    test_idx = [
        enforce_imbalance(test_data, s, ratio, max(10, min_pos // 3), seed + 100 + k)
        for k, s in enumerate(test_splits)
    ]

    bs = cfg["batch_size"]
    train_loaders = [DataLoader(Subset(train_data, i), batch_size=bs, shuffle=True, num_workers=0) for i in train_idx]
    test_loaders = [DataLoader(Subset(test_data, i), batch_size=bs, shuffle=False, num_workers=0) for i in test_idx]
    return train_loaders, test_loaders


def run_policy(name: str, buffer, train_loaders, test_loaders, device, cfg: dict) -> Dict[str, Any]:
    return train_continual_pathmnist_rare(
        name,
        buffer,
        train_loaders,
        test_loaders,
        device,
        in_channels=cfg.get("in_channels", 3),
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
    )


def aggregate_summaries(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "final_avg_auprc",
        "combined_auprc",
        "combined_recall_pos",
        "task0_retention_recall",
        "task0_auprc",
        "combined_recall_at_1pct_fpr",
        "buffer_memory_mb",
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
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/pathmnist_rare_10seed")
    parser.add_argument("--policies", nargs="+", default=["reservoir", "loss_priority", "memorysafe_v14"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)
    cfg = dict(CONFIG)
    print(f"Device: {device} | Protocol: {PROTOCOL_VERSION} | rare_classes={cfg['rare_classes']}")

    all_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in args.policies}

    for s in range(args.seeds):
        seed = args.start_seed + s
        set_seed(seed)
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        train_loaders, test_loaders = build_loaders(seed, cfg)

        if "reservoir" in args.policies:
            set_seed(seed)
            all_results["reservoir"].append(
                run_policy("reservoir", ReservoirBuffer(cfg["buffer_capacity"]), train_loaders, test_loaders, device, cfg)
            )
        if "loss_priority" in args.policies:
            set_seed(seed)
            all_results["loss_priority"].append(
                run_policy("loss_priority", LossPriorityBuffer(cfg["buffer_capacity"]), train_loaders, test_loaders, device, cfg)
            )
        if "memorysafe_v14" in args.policies:
            set_seed(seed)
            buf_cfg = BufferConfig(
                capacity=cfg["buffer_capacity"],
                pos_quota_frac=cfg["pos_quota_frac"],
                replay_pos_frac=cfg["replay_pos_frac"],
                pos_risk_boost=cfg.get("pos_risk_boost", 0.28),
            )
            all_results["memorysafe_v14"].append(
                run_policy("memorysafe_v14", MemorySafeBufferV14(buf_cfg), train_loaders, test_loaders, device, cfg)
            )

    report = {
        "config": cfg,
        "n_seeds": args.seeds,
        "aggregates": {p: aggregate_summaries(runs) for p, runs in all_results.items()},
    }

    if "memorysafe_v14" in all_results and "reservoir" in all_results:
        ms = all_results["memorysafe_v14"]
        res = all_results["reservoir"]
        report["memorysafe_vs_reservoir"] = {
            "auprc": try_ttest([r["summary"]["combined_auprc"] for r in ms], [r["summary"]["combined_auprc"] for r in res]),
            "recall": try_ttest([r["summary"]["combined_recall_pos"] for r in ms], [r["summary"]["combined_recall_pos"] for r in res]),
        }

    out_path = os.path.join(args.save_dir, "benchmark_report.json")
    with open(out_path, "w") as f:
        json.dump({"report": report, "raw": all_results}, f, indent=2)

    print("\n" + "=" * 60)
    print("PATHMNIST RARE BINARY SUMMARY")
    print("=" * 60)
    for policy, agg in report["aggregates"].items():
        a = agg["combined_auprc"]
        r = agg["combined_recall_pos"]
        t0 = agg["task0_retention_recall"]
        print(f"{policy:18s} | AUPRC {a['mean']:.4f}±{a['std']:.4f} | Recall {r['mean']:.4f}±{r['std']:.4f} | Task0 {t0['mean']:.4f}±{t0['std']:.4f}")

    if "memorysafe_vs_reservoir" in report:
        p = report["memorysafe_vs_reservoir"]["auprc"]["p"]
        print(f"\nMemorySafe vs Reservoir AUPRC paired t-test p={p}")

    print(f"\nSaved: {out_path}")

    results_md = os.path.join(args.save_dir, "RESULTS.md")
    with open(results_md, "w") as f:
        f.write(f"# PathMNIST Rare Binary — {cfg['protocol_version']}\n\n")
        f.write(f"**Seeds:** {args.seeds} | **Rare classes:** {cfg['rare_classes']}\n\n")
        f.write("| Policy | Combined AUPRC | Recall_pos | Task-0 recall |\n")
        f.write("|--------|----------------|------------|---------------|\n")
        for policy, agg in report["aggregates"].items():
            a, r, t0 = agg["combined_auprc"], agg["combined_recall_pos"], agg["task0_retention_recall"]
            f.write(
                f"| {policy} | {a['mean']:.4f} ± {a['std']:.4f} | "
                f"{r['mean']:.4f} ± {r['std']:.4f} | {t0['mean']:.4f} ± {t0['std']:.4f} |\n"
            )
        if "memorysafe_vs_reservoir" in report:
            p = report["memorysafe_vs_reservoir"]["auprc"]["p"]
            f.write(f"\n**MemorySafe vs Reservoir AUPRC paired t-test p={p}**\n")
    print(f"Saved: {results_md}")
    return report


if __name__ == "__main__":
    main()