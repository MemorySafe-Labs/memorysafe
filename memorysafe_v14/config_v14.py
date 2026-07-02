"""
Frozen protocol for reproducible MemorySafe v14 benchmarks.

Bump PROTOCOL_VERSION when any of these change in a way that breaks comparability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


PROTOCOL_VERSION = "v14.2-pneumonia-5task-sota"
PROTOCOL_DATE = "2026-06-13"


@dataclass(frozen=True)
class PneumoniaProtocol:
    dataset: str = "PneumoniaMNIST"
    n_tasks: int = 5
    imbalance_ratio_neg_per_pos: int = 20
    min_pos_per_task: int = 30
    buffer_capacity: int = 500
    compact_capacity: int = 80
    batch_size: int = 128
    epochs_per_task: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_prob: float = 0.80
    replay_batch_size: int = 128
    pos_quota_frac: float = 0.40
    replay_pos_frac: float = 0.45
    replay_scale: float = 1.25
    value_mix_loss: float = 0.5
    value_mix_unc: float = 0.5
    # Light AR: optional recall feedback after each task (v13.5 lesson)
    recall_feedback: bool = False
    recall_target: float = 0.72
    recall_feedback_gain: float = 0.25
    num_workers: int = 0
    default_seeds: int = 10
    start_seed: int = 42


CANONICAL = PneumoniaProtocol()

PROTOCOL_LITE_VERSION = "v14.3-pneumonia-5task-lite"
PROTOCOL_LITE_DATE = "2026-06-19"


@dataclass(frozen=True)
class PneumoniaProtocolLite:
    """Compact buffer + lower replay + health-gated boost (cost-reduction SKU)."""

    dataset: str = "PneumoniaMNIST"
    n_tasks: int = 5
    imbalance_ratio_neg_per_pos: int = 20
    min_pos_per_task: int = 30
    buffer_capacity: int = 80
    compact_capacity: int = 80
    batch_size: int = 128
    epochs_per_task: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_prob: float = 0.55
    replay_batch_size: int = 64
    pos_quota_frac: float = 0.40
    replay_pos_frac: float = 0.40
    replay_scale: float = 1.0
    value_mix_loss: float = 0.5
    value_mix_unc: float = 0.5
    health_feedback: bool = True
    recall_feedback: bool = False
    recall_target: float = 0.72
    recall_feedback_gain: float = 0.25
    num_workers: int = 0
    default_seeds: int = 10
    start_seed: int = 42


LITE = PneumoniaProtocolLite()


def to_dict(cfg: PneumoniaProtocol = CANONICAL) -> Dict[str, Any]:
    d = asdict(cfg)
    d["protocol_version"] = PROTOCOL_VERSION
    d["protocol_date"] = PROTOCOL_DATE
    return d


def lite_to_dict(cfg: PneumoniaProtocolLite = LITE) -> Dict[str, Any]:
    d = asdict(cfg)
    d["protocol_version"] = PROTOCOL_LITE_VERSION
    d["protocol_date"] = PROTOCOL_LITE_DATE
    return d
