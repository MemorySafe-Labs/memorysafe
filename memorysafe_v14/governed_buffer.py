"""
Environment-agnostic governed replay buffer (MemorySafe kernel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch

from quota_policies import QuotaPolicy


@dataclass
class GovernedItem:
    x: torch.Tensor
    y: int
    task_id: int
    value: float
    risk: float
    protect: float
    seen: int = 0


@dataclass
class GovernedBufferConfig:
    capacity: int = 500
    w_risk: float = 0.50
    w_value: float = 0.35
    w_criticality: float = 0.10
    w_rarity: float = 0.05
    mvi_ema: float = 0.70
    task_age_weight: float = 0.12
    task_balanced_replay: bool = True


class GovernedBuffer:
    """Universal MemorySafe buffer driven by a QuotaPolicy plugin."""

    def __init__(
        self,
        cfg: GovernedBufferConfig,
        quota: QuotaPolicy,
        current_task: int = 0,
    ):
        self.cfg = cfg
        self.quota = quota
        self.current_task = current_task
        self.items: List[GovernedItem] = []

    def __len__(self) -> int:
        return len(self.items)

    def set_task(self, task_id: int) -> None:
        self.current_task = task_id

    def _group_counts(self) -> Dict[Hashable, int]:
        counts: Dict[Hashable, int] = {}
        for it in self.items:
            g = self.quota.group_key(it.y)
            counts[g] = counts.get(g, 0) + 1
        return counts

    def _seen_groups(self) -> int:
        if hasattr(self.quota, "n_seen"):
            return self.quota.n_seen()
        return max(1, len(self._group_counts()))

    def _recompute_protect(self, it: GovernedItem) -> None:
        cfg = self.cfg
        task_age = max(0, self.current_task - it.task_id)
        age_factor = min(task_age / 4.0, 1.0)
        it.protect = float(
            cfg.w_risk * it.risk
            + cfg.w_value * it.value
            + cfg.w_criticality * self.quota.criticality(it.y)
            + cfg.w_rarity * self.quota.rarity_weight(it.y)
            + cfg.task_age_weight * age_factor
        )

    def count_pos(self) -> int:
        """Backward compat: count positive-label items if binary rare policy."""
        if hasattr(self.quota, "positive_label"):
            pl = self.quota.positive_label
            return sum(1 for it in self.items if it.y == pl)
        return len(self.items)

    def _indices_by_group(self, group: Hashable) -> List[int]:
        return [i for i, it in enumerate(self.items) if self.quota.group_key(it.y) == group]

    def _pick_eviction_index(self, incoming_y: int) -> int:
        g_in = self.quota.group_key(incoming_y)
        counts = self._group_counts()
        n_groups = self._seen_groups()
        cap = self.cfg.capacity
        target_in = int(cap * self.quota.target_buffer_frac(g_in, n_groups, cap))
        cur_in = counts.get(g_in, 0)

        if cur_in < target_in:
            over_groups = [
                g for g, c in counts.items()
                if g != g_in and c > int(cap * self.quota.target_buffer_frac(g, n_groups, cap))
            ]
            if over_groups:
                victim_g = over_groups[0]
                idxs = self._indices_by_group(victim_g)
                return min(idxs, key=lambda k: self.items[k].protect)
            neg_idxs = [i for i, it in enumerate(self.items) if self.quota.group_key(it.y) != g_in]
            if neg_idxs:
                return min(neg_idxs, key=lambda k: self.items[k].protect)

        same = self._indices_by_group(g_in)
        if same:
            return min(same, key=lambda k: self.items[k].protect)
        return int(np.argmin([z.protect for z in self.items]))

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
            if hasattr(self.quota, "register_seen"):
                self.quota.register_seen(y)
            val = float(values[i])
            risk0 = val + self.quota.risk_boost(y)
            it = GovernedItem(x=xs[i], y=y, task_id=task_id, value=val, risk=risk0, protect=0.0)
            self._recompute_protect(it)

            if len(self.items) < cfg.capacity:
                self.items.append(it)
                continue

            worst_idx = self._pick_eviction_index(y)
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

        counts = self._group_counts()
        n_groups = max(1, len(counts))
        k = min(batch_size, len(self.items))
        chosen: List[int] = []

        fracs = {g: self.quota.replay_frac(g, n_groups) for g in counts}
        total_f = sum(fracs.values()) or 1.0
        for g, c in counts.items():
            if len(chosen) >= k:
                break
            idxs = self._indices_by_group(g)
            k_g = min(len(idxs), max(1, int(round(k * fracs[g] / total_f))))
            chosen.extend(self._weighted_sample(idxs, k_g))

        if self.cfg.task_balanced_replay and len(chosen) < k:
            seen_tasks = sorted({it.task_id for it in self.items})
            for tid in seen_tasks:
                if len(chosen) >= k:
                    break
                task_idxs = [i for i, it in enumerate(self.items) if it.task_id == tid and i not in chosen]
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

    def group_distribution(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for it in self.items:
            key = str(self.quota.group_key(it.y))
            out[key] = out.get(key, 0) + 1
        return out


class ReservoirBufferUniversal:
    """Uniform reservoir baseline (model-agnostic)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List[GovernedItem] = []
        self.n_seen_total = 0
        self.current_task = 0

    def __len__(self) -> int:
        return len(self.items)

    def set_task(self, task_id: int) -> None:
        self.current_task = task_id

    def count_pos(self) -> int:
        return len(self.items)

    def add_batch(self, xs, ys, values, task_id: int = 0) -> None:
        xs = xs.detach().cpu()
        ys = ys.detach().cpu().view(-1).int()
        for i in range(xs.size(0)):
            self.n_seen_total += 1
            it = GovernedItem(
                x=xs[i], y=int(ys[i].item()), task_id=task_id,
                value=float(values[i]), risk=float(values[i]), protect=float(values[i]),
            )
            if len(self.items) < self.capacity:
                self.items.append(it)
            else:
                j = np.random.randint(0, self.n_seen_total)
                if j < self.capacity:
                    self.items[j] = it

    def sample(self, batch_size: int):
        k = min(batch_size, len(self.items))
        idxs = np.random.choice(len(self.items), size=k, replace=False)
        xs = torch.stack([self.items[i].x for i in idxs], dim=0)
        ys = torch.tensor([self.items[i].y for i in idxs], dtype=torch.long)
        return xs, ys, idxs.tolist()

    def group_distribution(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for it in self.items:
            key = str(it.y)
            out[key] = out.get(key, 0) + 1
        return out

    def memory_bytes(self) -> int:
        return sum(it.x.numel() * it.x.element_size() for it in self.items)


@dataclass
class HybridCLBufferConfig:
    """Reservoir core (frequency) + governed shell (fragility)."""

    capacity: int = 2000
    core_frac: float = 0.90
    core_replay_frac: float = 0.85
    promote_frac: float = 0.08
    shell_value_pct: float = 90.0
    shell_rarity_thresh: float = 0.55


class HybridCLBuffer:
    """
    Split buffer: reservoir core preserves stream frequency; governed shell
    protects fragile / old-task memories without collapsing FRI.
    """

    def __init__(
        self,
        cfg: HybridCLBufferConfig,
        shell_cfg: GovernedBufferConfig,
        shell_quota: QuotaPolicy,
    ):
        self.cfg = cfg
        self.quota = shell_quota
        core_cap = max(1, int(cfg.capacity * cfg.core_frac))
        shell_cap = max(1, cfg.capacity - core_cap)
        shell_cfg = GovernedBufferConfig(**{**shell_cfg.__dict__, "capacity": shell_cap})
        self.core = ReservoirBufferUniversal(core_cap)
        self.shell = GovernedBuffer(shell_cfg, shell_quota)
        self.core_replay_frac = cfg.core_replay_frac
        self.current_task = 0

    def __len__(self) -> int:
        return len(self.core) + len(self.shell)

    def set_task(self, task_id: int) -> None:
        if task_id > self.current_task:
            self._promote_old_tasks_from_core()
        self.current_task = task_id
        self.core.set_task(task_id)
        self.shell.set_task(task_id)

    def count_pos(self) -> int:
        return len(self.core) + len(self.shell)

    def _register_stream(self, ys: torch.Tensor) -> None:
        if hasattr(self.shell.quota, "register_seen"):
            for y in ys.view(-1).tolist():
                self.shell.quota.register_seen(int(y))

    def _promote_old_tasks_from_core(self) -> None:
        """At task boundaries, vault high-value memories from prior tasks into the shell."""
        if not self.core.items:
            return
        cap = self.shell.cfg.capacity
        budget = max(1, int(cap * self.cfg.promote_frac))
        candidates = [
            it for it in self.core.items if it.task_id < self.current_task
        ]
        if not candidates:
            return
        thresh = float(
            np.percentile(
                [it.value for it in candidates],
                self.cfg.shell_value_pct,
            )
        )
        ranked = sorted(
            [it for it in candidates if it.value >= thresh],
            key=lambda z: (z.task_id, -z.value),
        )
        if not ranked:
            ranked = sorted(candidates, key=lambda z: (-z.value, z.task_id))[:budget]
        for it in ranked[:budget]:
            xs = it.x.unsqueeze(0)
            ys = torch.tensor([it.y], dtype=torch.long)
            vals = np.array([it.value], dtype=np.float64)
            self.shell.add_batch(xs, ys, vals, it.task_id)

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
        self._register_stream(ys)
        self.core.add_batch(xs, ys, values, task_id)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if not self.items_available():
            raise ValueError("Buffer empty")

        k = min(batch_size, len(self))
        k_core = 0
        k_shell = 0
        if len(self.core) > 0 and len(self.shell) > 0:
            k_core = min(len(self.core), max(0, int(round(k * self.core_replay_frac))))
            k_shell = min(len(self.shell), k - k_core)
            if k_core + k_shell < k:
                if len(self.core) > k_core:
                    k_core = min(len(self.core), k - k_shell)
                elif len(self.shell) > k_shell:
                    k_shell = min(len(self.shell), k - k_core)
        elif len(self.core) > 0:
            k_core = min(len(self.core), k)
        else:
            k_shell = min(len(self.shell), k)

        xs_list: List[torch.Tensor] = []
        ys_list: List[torch.Tensor] = []
        chosen: List[int] = []
        core_len = len(self.core)

        if k_core > 0:
            bx, by, idxs = self.core.sample(k_core)
            xs_list.append(bx)
            ys_list.append(by)
            chosen.extend(idxs)

        if k_shell > 0:
            bx, by, idxs = self.shell.sample(k_shell)
            xs_list.append(bx)
            ys_list.append(by)
            chosen.extend([core_len + i for i in idxs])

        xs_out = torch.cat(xs_list, dim=0) if xs_list else torch.empty(0)
        ys_out = torch.cat(ys_list, dim=0) if ys_list else torch.empty(0)
        return xs_out, ys_out, chosen

    def items_available(self) -> bool:
        return len(self.core) > 0 or len(self.shell) > 0

    def update_risk_from_losses(self, idxs: List[int], losses: np.ndarray) -> None:
        core_len = len(self.core)
        shell_pairs: List[Tuple[int, float]] = []
        for j, bi in enumerate(idxs):
            if bi >= core_len:
                shell_pairs.append((bi - core_len, float(losses[j])))
        if not shell_pairs:
            return
        shell_idxs = [p[0] for p in shell_pairs]
        shell_losses = np.array([p[1] for p in shell_pairs], dtype=np.float64)
        self.shell.update_risk_from_losses(shell_idxs, shell_losses)

    def memory_bytes(self) -> int:
        return self.core.memory_bytes() + self.shell.memory_bytes()

    def group_distribution(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for dist in (self.core.group_distribution(), self.shell.group_distribution()):
            for k, v in dist.items():
                out[k] = out.get(k, 0) + int(v)
        return out
