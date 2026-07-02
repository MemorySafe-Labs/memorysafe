"""Frozen protocol: PathMNIST rare-tissue binary detection (Pneumonia-parity harness)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


PROTOCOL_VERSION = "v14.2-pathmnist-rare-5task-sota-tuned"
PROTOCOL_DATE = "2026-06-18"

# Test-set rarest pathology classes → positive label
RARE_CLASSES: Tuple[int, ...] = (2, 7)


@dataclass(frozen=True)
class PathMNISTRareProtocol:
    dataset: str = "PathMNIST-rare-binary"
    in_channels: int = 3
    rare_classes: Tuple[int, ...] = RARE_CLASSES
    n_tasks: int = 5
    min_pos_per_task: int = 40
    imbalance_ratio_neg_per_pos: int = 15
    buffer_capacity: int = 500
    batch_size: int = 128
    epochs_per_task: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_prob: float = 0.85
    replay_batch_size: int = 128
    pos_quota_frac: float = 0.50
    replay_pos_frac: float = 0.55
    replay_scale: float = 1.50
    pos_risk_boost: float = 0.28
    value_mix_loss: float = 0.5
    value_mix_unc: float = 0.5
    recall_feedback: bool = False
    default_seeds: int = 10
    start_seed: int = 42


CANONICAL = PathMNISTRareProtocol()


def to_dict(cfg: PathMNISTRareProtocol = CANONICAL) -> Dict[str, Any]:
    d = asdict(cfg)
    d["protocol_version"] = PROTOCOL_VERSION
    d["protocol_date"] = PROTOCOL_DATE
    d["rare_classes"] = list(cfg.rare_classes)
    return d
