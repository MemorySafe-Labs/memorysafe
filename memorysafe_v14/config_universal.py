"""Protocol configs for universal MemorySafe benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


CIFAR100_PROTOCOL_VERSION = "v14.2-cifar100-10task-classil"
CIFAR100_PROTOCOL_DATE = "2026-06-12"


@dataclass(frozen=True)
class CIFAR100Protocol:
    dataset: str = "CIFAR100"
    n_tasks: int = 10
    classes_per_task: int = 10
    buffer_capacity: int = 2000
    batch_size: int = 128
    epochs_per_task: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_prob: float = 0.70
    replay_batch_size: int = 128
    replay_scale: float = 1.2
    min_frac_per_class: float = 0.04
    value_mix_loss: float = 0.5
    value_mix_unc: float = 0.5
    num_workers: int = 0
    default_seeds: int = 10
    start_seed: int = 42


CIFAR100_CANONICAL = CIFAR100Protocol()


def cifar100_to_dict(cfg: CIFAR100Protocol = CIFAR100_CANONICAL) -> Dict[str, Any]:
    d = asdict(cfg)
    d["protocol_version"] = CIFAR100_PROTOCOL_VERSION
    d["protocol_date"] = CIFAR100_PROTOCOL_DATE
    return d
