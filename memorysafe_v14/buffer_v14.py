"""
MemorySafe v14 production buffer — efficient continual-learning memory governance.

Combines proven PneumoniaMNIST mechanics (pos quota, stratified replay, MVI EMA)
with v14 ProtectScore signals (rarity, criticality, uncertainty, task-age).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class MemoryItem:
    x: torch.Tensor
    y: int
    task_id: int
    value: float
    risk: float
    protect: float
    seen: int = 0


@dataclass
class BufferConfig:
    capacity: int = 500
    pos_quota_frac: float = 0.30
    replay_pos_frac: float = 0.35
    w_risk: float = 0.50
    w_value: float = 0.35
    w_criticality: float = 0.10
    w_rarity: float = 0.05
    mvi_ema: float = 0.70
    pos_risk_boost: float = 0.22
    task_age_weight: float = 0.12
    task_balanced_replay: bool = True


class ReservoirBuffer:
    """Uniform reservoir sampling baseline."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List[MemoryItem] = []
        self.n_seen_total = 0

    def __len__(self) -> int:
        return len(self.items)

    def add_batch(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        values: np.ndarray,
        task_id: int = 0,
    ) -> None:
        xs = xs.detach().cpu()
        ys = ys.detach().cpu().view(-1).int()
        for i in range(xs.size(0)):
            self.n_seen_total += 1
            it = MemoryItem(
                x=xs[i],
                y=int(ys[i].item()),
                task_id=task_id,
                value=float(values[i]),
                risk=float(values[i]),
                protect=float(values[i]),
            )
            if len(self.items) < self.capacity:
                self.items.append(it)
            else:
                j = np.random.randint(0, self.n_seen_total)
                if j < self.capacity:
                    self.items[j] = it

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        k = min(batch_size, len(self.items))
        idxs = np.random.choice(len(self.items), size=k, replace=False)
        xs = torch.stack([self.items[i].x for i in idxs], dim=0)
        ys = torch.tensor([self.items[i].y for i in idxs], dtype=torch.long)
        return xs, ys, idxs.tolist()

    def count_pos(self) -> int:
        return sum(1 for it in self.items if it.y == 1)

    def memory_bytes(self) -> int:
        return sum(it.x.numel() * it.x.element_size() for it in self.items)


class LossPriorityBuffer:
    """Greedy high-loss buffer (GSS-style baseline)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def __len__(self) -> int:
        return len(self.items)

    def add_batch(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        values: np.ndarray,
        task_id: int = 0,
    ) -> None:
        xs = xs.detach().cpu()
        ys = ys.detach().cpu().view(-1).int()
        for i in range(xs.size(0)):
            it = MemoryItem(
                x=xs[i],
                y=int(ys[i].item()),
                task_id=task_id,
                value=float(values[i]),
                risk=float(values[i]),
                protect=float(values[i]),
            )
            if len(self.items) < self.capacity:
                self.items.append(it)
                continue
            worst = int(np.argmin([z.protect for z in self.items]))
            if it.protect > self.items[worst].protect:
                self.items[worst] = it

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        k = min(batch_size, len(self.items))
        scores = np.array([max(it.protect, 1e-6) for it in self.items], dtype=np.float64)
        probs = scores / scores.sum()
        idxs = np.random.choice(len(self.items), size=k, replace=False, p=probs)
        xs = torch.stack([self.items[i].x for i in idxs], dim=0)
        ys = torch.tensor([self.items[i].y for i in idxs], dtype=torch.long)
        return xs, ys, idxs.tolist()

    def count_pos(self) -> int:
        return sum(1 for it in self.items if it.y == 1)

    def memory_bytes(self) -> int:
        return sum(it.x.numel() * it.x.element_size() for it in self.items)


class MemorySafeBufferV14:
    """
    Governed replay buffer for rare-class continual learning.

    Design choices (ML engineering):
    - Single bounded buffer (no unbounded side queues)
    - Positive quota eviction protects minority class under fixed memory
    - ProtectScore = risk EMA + value + criticality + rarity + task-age pressure
    - Stratified replay oversamples positives proportional to replay_pos_frac
    - Weighted sampling by ProtectScore (not uniform like reservoir)
    """

    def __init__(self, cfg: BufferConfig, current_task: int = 0):
        self.cfg = cfg
        self.current_task = current_task
        self.items: List[MemoryItem] = []

    def __len__(self) -> int:
        return len(self.items)

    def set_task(self, task_id: int) -> None:
        self.current_task = task_id

    def _recompute_protect(self, it: MemoryItem) -> None:
        cfg = self.cfg
        task_age = max(0, self.current_task - it.task_id)
        age_factor = min(task_age / 4.0, 1.0)
        criticality = 1.0 if it.y == 1 else 0.0
        rarity = 1.0 if it.y == 1 else 0.2
        it.protect = float(
            cfg.w_risk * it.risk
            + cfg.w_value * it.value
            + cfg.w_criticality * criticality
            + cfg.w_rarity * rarity
            + cfg.task_age_weight * age_factor
        )

    def count_pos(self) -> int:
        return sum(1 for it in self.items if it.y == 1)

    def _indices_by_class(self, y: int) -> List[int]:
        return [i for i, it in enumerate(self.items) if it.y == y]

    def add_batch(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        values: np.ndarray,
        task_id: Optional[int] = None,
    ) -> None:
        if task_id is None:
            task_id = self.current_task
        xs = xs.detach().cpu()
        ys = ys.detach().cpu().view(-1).int()
        cfg = self.cfg

        for i in range(xs.size(0)):
            y = int(ys[i].item())
            val = float(values[i])
            risk0 = val + (cfg.pos_risk_boost if y == 1 else 0.0)
            it = MemoryItem(
                x=xs[i],
                y=y,
                task_id=task_id,
                value=val,
                risk=risk0,
                protect=0.0,
            )
            self._recompute_protect(it)

            if len(self.items) < cfg.capacity:
                self.items.append(it)
                continue

            pos_count = self.count_pos()
            target_pos = int(cfg.capacity * cfg.pos_quota_frac)

            if pos_count < target_pos and y == 1:
                neg_idxs = self._indices_by_class(0)
                worst_idx = (
                    min(neg_idxs, key=lambda k: self.items[k].protect)
                    if neg_idxs
                    else int(np.argmin([z.protect for z in self.items]))
                )
            elif pos_count < target_pos:
                neg_idxs = self._indices_by_class(0)
                if neg_idxs:
                    worst_idx = min(neg_idxs, key=lambda k: self.items[k].protect)
                else:
                    worst_idx = int(np.argmin([z.protect for z in self.items]))
            else:
                same_idxs = self._indices_by_class(y)
                worst_idx = (
                    min(same_idxs, key=lambda k: self.items[k].protect)
                    if same_idxs
                    else int(np.argmin([z.protect for z in self.items]))
                )

            if it.protect > self.items[worst_idx].protect:
                self.items[worst_idx] = it

    def _weighted_sample(self, idxs: List[int], k: int) -> List[int]:
        if k <= 0 or not idxs:
            return []
        k = min(k, len(idxs))
        scores = np.array([max(self.items[i].protect, 1e-6) for i in idxs], dtype=np.float64)
        probs = scores / scores.sum()
        return np.random.choice(idxs, size=k, replace=False, p=probs).tolist()

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if not self.items:
            raise ValueError("Buffer empty")

        pos_idxs = self._indices_by_class(1)
        neg_idxs = self._indices_by_class(0)
        k = min(batch_size, len(self.items))
        k_pos = min(len(pos_idxs), max(1, int(round(k * self.cfg.replay_pos_frac)))) if pos_idxs else 0
        k_neg = k - k_pos

        chosen: List[int] = []
        if k_pos > 0:
            chosen.extend(self._weighted_sample(pos_idxs, k_pos))

        if k_neg > 0 and neg_idxs:
            k_neg = min(k_neg, len(neg_idxs))
            chosen.extend(self._weighted_sample(neg_idxs, k_neg))

        if self.cfg.task_balanced_replay and len(chosen) < k:
            seen_tasks = sorted({self.items[i].task_id for i in range(len(self.items))})
            for tid in seen_tasks:
                if len(chosen) >= k:
                    break
                task_idxs = [i for i in range(len(self.items)) if self.items[i].task_id == tid and i not in chosen]
                if task_idxs:
                    chosen.append(int(np.random.choice(task_idxs)))

        if len(chosen) < k:
            remaining = [i for i in range(len(self.items)) if i not in set(chosen)]
            need = k - len(chosen)
            if need > 0 and remaining:
                chosen.extend(
                    np.random.choice(remaining, size=min(need, len(remaining)), replace=False).tolist()
                )

        xs = torch.stack([self.items[i].x for i in chosen], dim=0)
        ys = torch.tensor([self.items[i].y for i in chosen], dtype=torch.long)
        return xs, ys, chosen

    def update_risk_from_losses(self, idxs: List[int], losses: np.ndarray) -> None:
        cfg = self.cfg
        for j, bi in enumerate(idxs):
            it = self.items[bi]
            it.risk = float(cfg.mvi_ema * it.risk + (1.0 - cfg.mvi_ema) * float(losses[j]))
            self._recompute_protect(it)
            it.seen += 1

    def memory_bytes(self) -> int:
        return sum(it.x.numel() * it.x.element_size() for it in self.items)
