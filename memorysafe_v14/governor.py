"""
MemorySafeGovernor — product API for any continual-learning training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from governed_buffer import GovernedBuffer, GovernedBufferConfig, ReservoirBufferUniversal
from quota_policies import QuotaPolicy, RareClassQuota, UniformClassQuota


@dataclass
class GovernorConfig:
    capacity: int = 500
    replay_prob: float = 0.80
    replay_batch_size: int = 128
    replay_scale: float = 1.25
    mix_loss: float = 0.5
    mix_unc: float = 0.5
    buffer: GovernedBufferConfig = field(default_factory=GovernedBufferConfig)


@dataclass
class AuditEvent:
    step: int
    action: str
    detail: Dict[str, Any]


class MemorySafeGovernor:
    """
    Drop-in memory governance for any PyTorch trainer.

    Usage:
        gov = MemorySafeGovernor.for_rare_binary(capacity=500)
        bx, by, idx = gov.maybe_sample()
        # ... train ...
        gov.observe(x, y, per_sample_losses, task_id=0, replay_idxs=idx, replay_losses=...)
    """

    def __init__(self, cfg: GovernorConfig, quota: QuotaPolicy, *, use_reservoir: bool = False):
        self.cfg = cfg
        self.quota = quota
        self.step = 0
        self.audit: List[AuditEvent] = []
        if use_reservoir:
            self._buffer = ReservoirBufferUniversal(cfg.capacity)
        else:
            buf_cfg = cfg.buffer
            buf_cfg.capacity = cfg.capacity
            self._buffer = GovernedBuffer(buf_cfg, quota)

    @classmethod
    def for_rare_binary(
        cls,
        capacity: int = 500,
        buffer_frac: float = 0.40,
        replay_frac_pos: float = 0.45,
        replay_prob: float = 0.80,
        positive_label: int = 1,
    ) -> "MemorySafeGovernor":
        cfg = GovernorConfig(capacity=capacity, replay_prob=replay_prob)
        quota = RareClassQuota(positive_label=positive_label, buffer_frac=buffer_frac, replay_frac_pos=replay_frac_pos)
        return cls(cfg, quota)

    @classmethod
    def for_class_incremental(
        cls,
        capacity: int = 2000,
        replay_prob: float = 0.70,
        min_frac_per_class: float = 0.04,
    ) -> "MemorySafeGovernor":
        cfg = GovernorConfig(capacity=capacity, replay_prob=replay_prob, replay_scale=1.2)
        quota = UniformClassQuota(min_frac_per_class=min_frac_per_class)
        return cls(cfg, quota)

    @classmethod
    def reservoir(cls, capacity: int = 500) -> "MemorySafeGovernor":
        cfg = GovernorConfig(capacity=capacity)
        return cls(cfg, RareClassQuota(), use_reservoir=True)

    @property
    def buffer(self):
        return self._buffer

    def set_task(self, task_id: int) -> None:
        if hasattr(self._buffer, "set_task"):
            self._buffer.set_task(task_id)

    def __len__(self) -> int:
        return len(self._buffer)

    def maybe_sample(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[int]]]:
        import random
        if len(self._buffer) == 0 or random.random() >= self.cfg.replay_prob:
            return None, None, None
        bx, by, idx = self._buffer.sample(self.cfg.replay_batch_size)
        self._log("replay_sample", {"n": len(idx), "groups": getattr(self._buffer, "group_distribution", lambda: {})()})
        return bx, by, idx

    def observe(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        per_sample_values: np.ndarray,
        *,
        task_id: int = 0,
        replay_idxs: Optional[List[int]] = None,
        replay_losses: Optional[np.ndarray] = None,
    ) -> None:
        self._buffer.add_batch(xs, ys, per_sample_values, task_id=task_id)
        if replay_idxs and replay_losses is not None and hasattr(self._buffer, "update_risk_from_losses"):
            self._buffer.update_risk_from_losses(replay_idxs, replay_losses)
        self.step += 1

    def compute_value_scores(self, losses: torch.Tensor, probs: Optional[torch.Tensor] = None) -> np.ndarray:
        if probs is None:
            return losses.detach().cpu().numpy()
        unc = (1.0 - torch.abs(2.0 * probs - 1.0)).clamp(0.0, 1.0)
        val = self.cfg.mix_loss * losses.detach() + self.cfg.mix_unc * unc.detach()
        return val.cpu().numpy()

    def _log(self, action: str, detail: Dict[str, Any]) -> None:
        self.audit.append(AuditEvent(step=self.step, action=action, detail=detail))

    def audit_summary(self) -> Dict[str, Any]:
        return {
            "steps": self.step,
            "buffer_size": len(self._buffer),
            "group_distribution": getattr(self._buffer, "group_distribution", lambda: {})(),
            "events": len(self.audit),
        }