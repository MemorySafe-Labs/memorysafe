"""
Production integration: NeMo Guardrails (runtime) + MemorySafeGovernor (training memory).

Agent prompt → runtime policy → governed buffer ingest → audit JSON for pilots / NVIDIA.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from governor import MemorySafeGovernor
from integrations.nemo_runtime import GuardrailAction, GuardrailDecision, check_prompt
from memory_health import StreamCounter, snapshot_memory_health
from simulated_cl_mvi_test import SimulatedCLConfig, make_task_batch, set_seed


@dataclass
class AgentRequest:
    prompt: str
    task_id: int = 0
    ingest_batch: bool = True
    rare_class_emphasis: bool = True


@dataclass
class MemoryDecision:
    action: str  # PROTECT | REINFORCE | DEFER
    mean_mvi: float
    mean_protect: float
    pos_in_buffer: int
    buffer_size: int
    group_distribution: Dict[str, Any]


@dataclass
class PipelineResult:
    prompt: str
    guardrails: GuardrailDecision
    memory: Optional[MemoryDecision]
    blocked: bool
    governor_audit: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["guardrails"] = asdict(self.guardrails)
        if self.memory:
            d["memory"] = asdict(self.memory)
        return d


def _action_from_buffer(gov: MemorySafeGovernor, rare_emphasis: bool) -> MemoryDecision:
    buf = gov.buffer
    health = snapshot_memory_health(buf, StreamCounter())
    mean_mvi = float(health.get("mean_mvi", 0.0))
    mean_protect = float(health.get("mean_protect", 0.0))
    pos = int(getattr(buf, "count_pos", lambda: 0)())
    size = len(buf)
    groups = getattr(buf, "group_distribution", lambda: {})()
    pos_frac = pos / max(size, 1)
    target_pos_frac = float(getattr(gov.quota, "buffer_frac", 0.40))

    if rare_emphasis and pos > 0 and pos_frac >= target_pos_frac * 0.70:
        action = "PROTECT"
    elif rare_emphasis and pos > 0 and pos_frac >= 0.08:
        action = "REINFORCE"
    elif mean_protect >= 0.45:
        action = "REINFORCE"
    else:
        action = "DEFER"

    return MemoryDecision(
        action=action,
        mean_mvi=mean_mvi,
        mean_protect=mean_protect,
        pos_in_buffer=pos,
        buffer_size=size,
        group_distribution={str(k): v for k, v in groups.items()},
    )


class MemorySafeProductionPipeline:
    """
    Two-layer stack for pathology continual-learning agents:
      1. NeMo Guardrails Colang policy (runtime)
      2. MemorySafeGovernor + GovernedBuffer (training memory)
    """

    def __init__(
        self,
        *,
        capacity: int = 80,
        seed: int = 42,
        prefer_nemo: bool = False,
    ) -> None:
        set_seed(seed)
        self.seed = seed
        self.prefer_nemo = prefer_nemo
        self.device = torch.device("cpu")
        self.cfg = SimulatedCLConfig(
            buffer_capacity=capacity,
            batch_size=32,
            pos_frac=0.10,
            pos_quota_frac=0.40,
            replay_pos_frac=0.45,
        )
        self.governor = MemorySafeGovernor.for_rare_binary(
            capacity=capacity,
            buffer_frac=0.40,
            replay_frac_pos=0.45,
            replay_prob=0.75,
        )
        self.stream = StreamCounter()

    def _ingest_batch(self, task_id: int, rare_emphasis: bool, *, n_waves: int = 1) -> MemoryDecision:
        self.governor.set_task(task_id)
        for _ in range(max(1, n_waves)):
            x, y = make_task_batch(task_id, self.cfg.batch_size, self.cfg, self.device)
            losses = torch.rand(x.size(0)) * 0.4 + 0.1
            if rare_emphasis:
                losses = losses + (y.float() * 0.35)
            values = self.governor.compute_value_scores(losses)
            self.governor.observe(x, y, values, task_id=task_id)
            self.stream.observe_labels(y, self.governor.buffer)
        return _action_from_buffer(self.governor, rare_emphasis)

    def process(self, request: AgentRequest) -> PipelineResult:
        decision = check_prompt(request.prompt, prefer_nemo=self.prefer_nemo)
        blocked = decision.action == GuardrailAction.BLOCK

        memory: Optional[MemoryDecision] = None
        if not blocked and request.ingest_batch:
            waves = 8 if request.rare_class_emphasis else 1
            memory = self._ingest_batch(
                request.task_id, request.rare_class_emphasis, n_waves=waves
            )

        return PipelineResult(
            prompt=request.prompt,
            guardrails=decision,
            memory=memory,
            blocked=blocked,
            governor_audit=self.governor.audit_summary(),
        )

    def save_report(self, results: List[PipelineResult], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "integration": "nemo_guardrails + memorysafe_v14_governor",
            "buffer_capacity": self.cfg.buffer_capacity,
            "seed": self.seed,
            "runs": [r.to_dict() for r in results],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
