"""
MemorySafe CL Game — what 'solving continual learning' means for this project.

Two-lane hypothesis (product + research):
  Lane A — Frequency: uniform / task-balanced replay keeps common classes alive.
  Lane B — Fragility: MVI + ProtectScore + quota keeps rare and old tasks alive.

Full CL win = beat reservoir on ALL four lanes at α=0.05 (10 seeds each).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


CL_GAME_VERSION = "cl-game-v1"
CL_ALPHA = 0.05
CL_SEEDS_REQUIRED = 10


@dataclass(frozen=True)
class CLLane:
    id: str
    name: str
    question: str
    evidence_path: str
    metric_key: str
    ms_policy: str
    baseline_policy: str
    direction: str  # "higher" | "lower"


CL_LANES: Tuple[CLLane, ...] = (
    CLLane(
        "rare_medical",
        "Rare medical detection",
        "Do we beat reservoir on rare-class AUPRC in medical streams?",
        "runs/pneumonia_10seed_sota/benchmark_report.json",
        "combined_auprc",
        "memorysafe_v14",
        "reservoir",
        "higher",
    ),
    CLLane(
        "rare_pathology",
        "Rare pathology tissue",
        "Does the win replicate on PathMNIST rare-tissue binary?",
        "runs/pathmnist_rare_10seed_sota/benchmark_report.json",
        "combined_auprc",
        "memorysafe_v14",
        "reservoir",
        "higher",
    ),
    CLLane(
        "anti_forgetting",
        "Task-0 anti-forgetting",
        "After N tasks, do we retain the first task better than reservoir?",
        "runs/pathmnist_10seed/benchmark_report.json",
        "task0_retention_acc",
        "memorysafe_governed",
        "reservoir",
        "higher",
    ),
    CLLane(
        "general_il",
        "General class-incremental",
        "Do we beat reservoir on combined accuracy (CIFAR-100 5-task)?",
        "runs/cifar100_fragility_10seed/benchmark_report.json",
        "combined_acc",
        "memorysafe_fragility",
        "reservoir",
        "higher",
    ),
)


def lane_status(passed: bool, p: float | None, delta: float) -> str:
    if passed:
        return "WON"
    if p is not None and p < CL_ALPHA and delta < 0:
        return "BEHIND"
    if p is not None and p < 0.10 and delta > 0:
        return "CLOSE"
    return "OPEN"
