#!/usr/bin/env python3
"""
A/B: static MemorySafe vs health-feedback controller (Layer 3).

Pneumonia: memorysafe_v14 vs memorysafe_v14_health
CIFAR:     memorysafe_hybrid vs memorysafe_hybrid_health (+ reservoir)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch

from benchmark_cifar100 import (
    aggregate as aggregate_cifar,
    build_loaders as build_cifar_loaders,
    make_hybrid_buffer,
    try_ttest,
)
from benchmark_pneumonia import (
    aggregate_summaries as aggregate_pneumonia,
    build_loaders as build_pneumonia_loaders,
    resolve_device,
    set_seed,
)
from buffer_v14 import BufferConfig, MemorySafeBufferV14
from config_v14 import PROTOCOL_VERSION, to_dict
from governed_buffer import GovernedBuffer, GovernedBufferConfig, ReservoirBufferUniversal
from quota_policies import FragilityCLQuota
from train_loop import train_continual
from train_loop_cifar import train_continual_cifar

PNEUMONIA_CFG = to_dict()


def run_pneumonia_policy(
    name: str,
    buffer: MemorySafeBufferV14,
    train_loaders,
    test_loaders,
    device: torch.device,
    health_feedback: bool,
) -> Dict[str, Any]:
    cfg = PNEUMONIA_CFG
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
        replay_pos_frac=cfg["replay_pos_frac"],
        health_feedback=health_feedback,
    )


def run_cifar_policy(
    name: str,
    buffer,
    train_loaders,
    test_loaders,
    device: torch.device,
    cfg: dict,
    health_feedback: bool,
) -> Dict[str, Any]:
    quota = getattr(getattr(buffer, "shell", buffer), "quota", getattr(buffer, "quota", None))
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
        replay_scale=cfg["replay_scale"],
        health_feedback=health_feedback,
        min_frac_per_class=getattr(quota, "min_frac_per_class", cfg["min_frac_per_class"]),
        old_task_boost=getattr(quota, "old_task_boost", 2.2),
    )


def make_pneumonia_buffer() -> MemorySafeBufferV14:
    cfg = PNEUMONIA_CFG
    buf_cfg = BufferConfig(
        capacity=cfg["buffer_capacity"],
        pos_quota_frac=cfg["pos_quota_frac"],
        replay_pos_frac=cfg["replay_pos_frac"],
    )
    return MemorySafeBufferV14(buf_cfg)


def make_cifar_buffer(cfg: dict):
    return make_hybrid_buffer(cfg)


def run_pneumonia_ab(seeds: int, start_seed: int, device: torch.device) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {
        "memorysafe_v14": [],
        "memorysafe_v14_health": [],
    }
    for s in range(seeds):
        seed = start_seed + s
        print(f"\n{'='*60}\nPNEUMONIA SEED {seed}\n{'='*60}")
        train_loaders, test_loaders = build_pneumonia_loaders(seed)

        set_seed(seed)
        hist = run_pneumonia_policy(
            "memorysafe_v14",
            make_pneumonia_buffer(),
            train_loaders,
            test_loaders,
            device,
            health_feedback=False,
        )
        results["memorysafe_v14"].append(hist)

        set_seed(seed)
        hist = run_pneumonia_policy(
            "memorysafe_v14_health",
            make_pneumonia_buffer(),
            train_loaders,
            test_loaders,
            device,
            health_feedback=True,
        )
        results["memorysafe_v14_health"].append(hist)
    return results


def run_cifar_ab(seeds: int, start_seed: int, device: torch.device, cfg: dict) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {
        "reservoir": [],
        "memorysafe_hybrid": [],
        "memorysafe_hybrid_health": [],
    }
    for s in range(seeds):
        seed = start_seed + s
        print(f"\n{'='*60}\nCIFAR SEED {seed}\n{'='*60}")
        train_loaders, test_loaders = build_cifar_loaders(seed, cfg)

        set_seed(seed)
        hist = run_cifar_policy(
            "reservoir",
            ReservoirBufferUniversal(cfg["buffer_capacity"]),
            train_loaders,
            test_loaders,
            device,
            {**cfg, "replay_scale": 1.0},
            health_feedback=False,
        )
        results["reservoir"].append(hist)

        set_seed(seed)
        hist = run_cifar_policy(
            "memorysafe_hybrid",
            make_cifar_buffer(cfg),
            train_loaders,
            test_loaders,
            device,
            cfg,
            health_feedback=False,
        )
        results["memorysafe_hybrid"].append(hist)

        set_seed(seed)
        hist = run_cifar_policy(
            "memorysafe_hybrid_health",
            make_cifar_buffer(cfg),
            train_loaders,
            test_loaders,
            device,
            cfg,
            health_feedback=True,
        )
        results["memorysafe_hybrid_health"].append(hist)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory health feedback A/B")
    parser.add_argument("--save-dir", default="runs/health_feedback_ab")
    parser.add_argument("--pneumonia-seeds", type=int, default=10)
    parser.add_argument("--cifar-seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip-pneumonia", action="store_true")
    parser.add_argument("--skip-cifar", action="store_true")
    parser.add_argument("--cifar-epochs-per-task", type=int, default=5)
    parser.add_argument("--cifar-replay-scale", type=float, default=1.05)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    report: Dict[str, Any] = {
        "protocol": "v14-health-feedback-ab",
        "pneumonia_protocol": PROTOCOL_VERSION,
        "device": str(device),
        "raw": {},
        "aggregates": {},
        "comparisons": {},
    }

    if not args.skip_pneumonia:
        pneu = run_pneumonia_ab(args.pneumonia_seeds, args.start_seed, device)
        report["raw"]["pneumonia"] = pneu
        report["aggregates"]["pneumonia"] = {
            k: aggregate_pneumonia(v) for k, v in pneu.items()
        }
        static = [r["summary"]["combined_auprc"] for r in pneu["memorysafe_v14"]]
        health = [r["summary"]["combined_auprc"] for r in pneu["memorysafe_v14_health"]]
        report["comparisons"]["pneumonia_auprc"] = try_ttest(health, static)
        report["comparisons"]["pneumonia_fri"] = try_ttest(
            [r["summary"]["fri"] for r in pneu["memorysafe_v14_health"]],
            [r["summary"]["fri"] for r in pneu["memorysafe_v14"]],
        )

    if not args.skip_cifar:
        from config_universal import CIFAR100_CANONICAL, cifar100_to_dict

        cifar_cfg = cifar100_to_dict(CIFAR100_CANONICAL)
        cifar_cfg["n_tasks"] = 5
        cifar_cfg["classes_per_task"] = 20
        cifar_cfg["epochs_per_task"] = args.cifar_epochs_per_task
        cifar_cfg["replay_scale"] = args.cifar_replay_scale
        cifar_cfg["hybrid_replay_scale"] = args.cifar_replay_scale
        cifar_cfg["protocol_version"] = "v14.4-cifar100-5task-hybrid-cl"
        cifar = run_cifar_ab(args.cifar_seeds, args.start_seed, device, cifar_cfg)
        report["raw"]["cifar"] = cifar
        report["aggregates"]["cifar"] = {k: aggregate_cifar(v) for k, v in cifar.items()}
        static = [r["summary"]["combined_acc"] for r in cifar["memorysafe_hybrid"]]
        health = [r["summary"]["combined_acc"] for r in cifar["memorysafe_hybrid_health"]]
        hybrid = [r["summary"]["combined_acc"] for r in cifar["memorysafe_hybrid"]]
        res = [r["summary"]["combined_acc"] for r in cifar["reservoir"]]
        report["comparisons"]["cifar_hybrid_vs_health"] = try_ttest(health, static)
        report["comparisons"]["cifar_hybrid_vs_reservoir"] = try_ttest(hybrid, res)
        report["comparisons"]["cifar_fri"] = try_ttest(
            [r["summary"]["fri"] for r in cifar["memorysafe_hybrid_health"]],
            [r["summary"]["fri"] for r in cifar["memorysafe_hybrid"]],
        )

    out = os.path.join(args.save_dir, "benchmark_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("HEALTH FEEDBACK A/B SUMMARY")
    print("=" * 60)
    if "pneumonia" in report.get("aggregates", {}):
        for policy, agg in report["aggregates"]["pneumonia"].items():
            a = agg["combined_auprc"]
            f = agg["fri"]
            print(
                f"PNEU {policy:28s} | AUPRC {a['mean']:.4f}±{a['std']:.4f} | "
                f"FRI {f['mean']:.4f}±{f['std']:.4f}"
            )
        p = report["comparisons"]["pneumonia_auprc"]["p"]
        print(f"Pneumonia health vs static AUPRC p={p}")
    if "cifar" in report.get("aggregates", {}):
        for policy, agg in report["aggregates"]["cifar"].items():
            a = agg["combined_acc"]
            f = agg["fri"]
            print(
                f"CIFAR {policy:27s} | acc {a['mean']:.4f}±{a['std']:.4f} | "
                f"FRI {f['mean']:.4f}±{f['std']:.4f}"
            )
        p_h = report["comparisons"]["cifar_hybrid_vs_health"]["p"]
        p_r = report["comparisons"]["cifar_hybrid_vs_reservoir"]["p"]
        print(f"CIFAR hybrid health vs static acc p={p_h}")
        print(f"CIFAR hybrid vs reservoir acc p={p_r}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()