"""
Pluggable quota policies for MemorySafe governed buffers.

Each policy defines how samples are grouped for eviction floors and replay stratification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Hashable, List, Sequence


class QuotaPolicy(ABC):
    """Maps labels → groups with buffer/replay quota targets."""

    @abstractmethod
    def group_key(self, label: int) -> Hashable:
        ...

    @abstractmethod
    def rarity_weight(self, label: int) -> float:
        ...

    @abstractmethod
    def criticality(self, label: int) -> float:
        ...

    @abstractmethod
    def risk_boost(self, label: int) -> float:
        ...

    @abstractmethod
    def target_buffer_frac(self, group: Hashable, n_seen_groups: int, capacity: int) -> float:
        """Target fraction of buffer capacity for this group (sum ≤ 1 across active groups)."""
        ...

    @abstractmethod
    def replay_frac(self, group: Hashable, n_seen_groups: int) -> float:
        """Target fraction of a replay batch for this group."""
        ...

    def groups_from_labels(self, labels: Sequence[int]) -> List[Hashable]:
        return sorted({self.group_key(int(y)) for y in labels})


class RareClassQuota(QuotaPolicy):
    """Binary rare-positive quota (PneumoniaMNIST / fraud / medical detection)."""

    def __init__(self, positive_label: int = 1, buffer_frac: float = 0.40, replay_frac_pos: float = 0.45):
        self.positive_label = positive_label
        self.buffer_frac = buffer_frac
        self.replay_frac_pos = replay_frac_pos

    def group_key(self, label: int) -> Hashable:
        return "pos" if label == self.positive_label else "neg"

    def rarity_weight(self, label: int) -> float:
        return 1.0 if label == self.positive_label else 0.2

    def criticality(self, label: int) -> float:
        return 1.0 if label == self.positive_label else 0.0

    def risk_boost(self, label: int) -> float:
        return 0.22 if label == self.positive_label else 0.0

    def target_buffer_frac(self, group: Hashable, n_seen_groups: int, capacity: int) -> float:
        if group == "pos":
            return self.buffer_frac
        return max(0.0, 1.0 - self.buffer_frac)

    def replay_frac(self, group: Hashable, n_seen_groups: int) -> float:
        if group == "pos":
            return self.replay_frac_pos
        return max(0.0, 1.0 - self.replay_frac_pos)


class UniformClassQuota(QuotaPolicy):
    """Multi-class class-incremental: uniform floor per class label."""

    def __init__(self, min_frac_per_class: float = 0.04, replay_oversample_old: float = 1.15):
        self.min_frac_per_class = min_frac_per_class
        self.replay_oversample_old = replay_oversample_old
        self._seen_classes: set[int] = set()

    def register_seen(self, label: int) -> None:
        self._seen_classes.add(int(label))

    def n_seen(self) -> int:
        return max(1, len(self._seen_classes))

    def group_key(self, label: int) -> Hashable:
        return int(label)

    def rarity_weight(self, label: int) -> float:
        return 1.0

    def criticality(self, label: int) -> float:
        return 0.5

    def risk_boost(self, label: int) -> float:
        return 0.08

    def target_buffer_frac(self, group: Hashable, n_seen_groups: int, capacity: int) -> float:
        n = max(1, n_seen_groups)
        return max(self.min_frac_per_class, 0.85 / n)

    def replay_frac(self, group: Hashable, n_seen_groups: int) -> float:
        n = max(1, n_seen_groups)
        return 1.0 / n


class TailClassQuota(UniformClassQuota):
    """Boosts replay weight for earlier (older) class ids — anti-forgetting bias."""

    def __init__(self, min_frac_per_class: float = 0.04, old_class_boost: float = 1.5):
        super().__init__(min_frac_per_class=min_frac_per_class)
        self.old_class_boost = old_class_boost
        self._current_max_class = 0

    def register_seen(self, label: int) -> None:
        super().register_seen(label)
        self._current_max_class = max(self._current_max_class, int(label))

    def replay_frac(self, group: Hashable, n_seen_groups: int) -> float:
        base = super().replay_frac(group, n_seen_groups)
        if self._current_max_class > 0 and int(group) < self._current_max_class // 2:
            return base * self.old_class_boost
        return base


class FragilityCLQuota(QuotaPolicy):
    """
    CL Game move #1 — unify frequency floors + dynamic rarity + old-task boost.

    Lane A: min_frac_per_class keeps every class represented (frequency).
    Lane B: rarity_weight + old_task_boost protect fragile / stale knowledge.
    """

    def __init__(
        self,
        min_frac_per_class: float = 0.06,
        old_task_boost: float = 2.0,
        rare_class_boost: float = 1.8,
        old_class_id_threshold: bool = True,
    ):
        self.min_frac_per_class = min_frac_per_class
        self.old_task_boost = old_task_boost
        self.rare_class_boost = rare_class_boost
        self.old_class_id_threshold = old_class_id_threshold
        self._seen_classes: Dict[int, int] = {}
        self._current_max_class = 0

    def register_seen(self, label: int) -> None:
        y = int(label)
        self._seen_classes[y] = self._seen_classes.get(y, 0) + 1
        self._current_max_class = max(self._current_max_class, y)

    def n_seen(self) -> int:
        return max(1, len(self._seen_classes))

    def _rarity(self, label: int) -> float:
        if not self._seen_classes:
            return 0.5
        freq = self._seen_classes.get(int(label), 1)
        max_freq = max(self._seen_classes.values())
        return float(1.0 - 0.75 * (freq / max(max_freq, 1)))

    def group_key(self, label: int) -> Hashable:
        return int(label)

    def rarity_weight(self, label: int) -> float:
        return self._rarity(label)

    def criticality(self, label: int) -> float:
        return 0.35 + 0.65 * self._rarity(label)

    def risk_boost(self, label: int) -> float:
        return 0.08 + 0.14 * self._rarity(label)

    def target_buffer_frac(self, group: Hashable, n_seen_groups: int, capacity: int) -> float:
        n = max(1, n_seen_groups)
        base = max(self.min_frac_per_class, 0.80 / n)
        if self._rarity(int(group)) > 0.55:
            return min(0.35, base * self.rare_class_boost)
        return base

    def replay_frac(self, group: Hashable, n_seen_groups: int) -> float:
        n = max(1, n_seen_groups)
        frac = max(self.min_frac_per_class, 0.85 / n)
        g = int(group)
        if self.old_class_id_threshold and self._current_max_class > 0 and g < self._current_max_class // 2:
            frac *= self.old_task_boost
        if self._rarity(g) > 0.55:
            frac *= self.rare_class_boost
        return frac
