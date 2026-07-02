#!/usr/bin/env python3
"""
Real integration: NeMo Guardrails policy + PathMNIST rare-tissue continual learning.

Uses the canonical v14.2-pathmnist-rare protocol (real RGB tiles, real CNN, real buffer).
Compares MemorySafe v14 vs reservoir on one seed — same harness as benchmark_pathmnist_rare.py.

Run:
  python demo_nemo_pathmnist_real.py
  python demo_nemo_pathmnist_real.py --device mps
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from benchmark_pathmnist_rare import build_loaders, run_policy, set_seed, try_ttest
from buffer_v14 import BufferConfig, MemorySafeBufferV14, ReservoirBuffer
from config_pathmnist_rare import CANONICAL, PROTOCOL_VERSION, to_dict
from integrations.nemo_runtime import GuardrailAction, check_prompt
from integrations.production_pipeline import AgentRequest


PATHOLOGY_PROMPT = (
    "Ingest PathMNIST rare-tissue stream (classes 2+7) for pathology continual learning — "
    "protect vulnerable classes under replay pressure"
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real NeMo + PathMNIST integration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/nemo_integration_demo/pathmnist_real_report.json"),
    )
    args = parser.parse_args()

    guard = check_prompt(PATHOLOGY_PROMPT)
    if guard.action == GuardrailAction.BLOCK:
        print("Guardrails blocked request — aborting.")
        return

    cfg = dict(to_dict(CANONICAL))
    device = resolve_device(args.device)
    print(f"Guardrails: {guard.action.value} ({guard.engine})")
    print(f"Protocol: {PROTOCOL_VERSION} | seed={args.seed} | device={device}")
    print("Loading PathMNIST (real tiles)...")

    set_seed(args.seed)
    train_loaders, test_loaders = build_loaders(args.seed, cfg)

    buf_cfg = BufferConfig(
        capacity=cfg["buffer_capacity"],
        pos_quota_frac=cfg["pos_quota_frac"],
        replay_pos_frac=cfg["replay_pos_frac"],
        pos_risk_boost=cfg.get("pos_risk_boost", 0.28),
    )

    set_seed(args.seed)
    res_hist = run_policy(
        "reservoir",
        ReservoirBuffer(cfg["buffer_capacity"]),
        train_loaders,
        test_loaders,
        device,
        cfg,
    )

    set_seed(args.seed)
    ms_hist = run_policy(
        "memorysafe_v14",
        MemorySafeBufferV14(buf_cfg),
        train_loaders,
        test_loaders,
        device,
        cfg,
    )

    ms_s = ms_hist["summary"]
    res_s = res_hist["summary"]
    auprc_p = try_ttest([ms_s["combined_auprc"]], [res_s["combined_auprc"]])["p"]

    payload: Dict[str, Any] = {
        "integration": "nemo_guardrails + pathmnist_rare_real",
        "protocol": PROTOCOL_VERSION,
        "rare_classes": cfg["rare_classes"],
        "seed": args.seed,
        "device": str(device),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_prompt": PATHOLOGY_PROMPT,
        "guardrails": {
            "action": guard.action.value,
            "message": guard.message,
            "engine": guard.engine,
        },
        "reservoir": res_s,
        "memorysafe_v14": ms_s,
        "delta": {
            "combined_auprc": ms_s["combined_auprc"] - res_s["combined_auprc"],
            "combined_recall_pos": ms_s["combined_recall_pos"] - res_s["combined_recall_pos"],
            "task0_recall_pos": ms_s["task0_retention_recall"] - res_s["task0_retention_recall"],
            "mean_mvi": ms_s.get("mean_mvi", 0) - res_s.get("mean_mvi", 0),
            "fri": ms_s.get("fri", 0) - res_s.get("fri", 0),
        },
        "memorysafe_vs_reservoir_1seed": {"auprc_p": auprc_p},
        "task_metrics_memorysafe": ms_hist["task_metrics"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("REAL PathMNIST rare-tissue — MemorySafe vs Reservoir (1 seed)")
    print("=" * 60)
    print(f"Reservoir   AUPRC {res_s['combined_auprc']:.4f} | recall_pos {res_s['combined_recall_pos']:.4f} | task0 {res_s['task0_retention_recall']:.4f}")
    print(f"MemorySafe  AUPRC {ms_s['combined_auprc']:.4f} | recall_pos {ms_s['combined_recall_pos']:.4f} | task0 {ms_s['task0_retention_recall']:.4f}")
    print(f"Delta AUPRC {payload['delta']['combined_auprc']:+.4f}")
    print(f"MS pos buffer (final): {ms_hist['task_metrics'][-1].get('buffer_size', '?')}")
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()