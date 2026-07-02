"""Frozen protocol for PathMNIST continual-learning benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


PROTOCOL_VERSION = "v14.2-pathmnist-5task-task0-sota"
PROTOCOL_DATE = "2026-06-18"


@dataclass(frozen=True)
class PathMNISTProtocol:
    dataset: str = "PathMNIST"
    in_channels: int = 3
    n_classes: int = 9
    n_tasks: int = 5
    # Class-incremental splits (9 classes → 2+2+2+2+1)
    task_class_splits: tuple = ((0, 1), (2, 3), (4, 5), (6, 7), (8,))
    buffer_capacity: int = 500
    batch_size: int = 128
    epochs_per_task: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_prob: float = 0.80
    replay_batch_size: int = 128
    replay_scale: float = 1.25
    min_frac_per_class: float = 0.06
    value_mix_loss: float = 0.5
    value_mix_unc: float = 0.5
    # Subsample large MedMNIST splits to keep runtime near Pneumonia protocol
    max_train_per_task: int = 600
    max_test_per_task: int = 200
    num_workers: int = 0
    default_seeds: int = 10
    start_seed: int = 42


CANONICAL = PathMNISTProtocol()


def task_splits() -> List[List[int]]:
    return [list(t) for t in CANONICAL.task_class_splits]


def to_dict(cfg: PathMNISTProtocol = CANONICAL) -> Dict[str, Any]:
    d = asdict(cfg)
    d["protocol_version"] = PROTOCOL_VERSION
    d["protocol_date"] = PROTOCOL_DATE
    d["task_class_splits"] = task_splits()
    return d
