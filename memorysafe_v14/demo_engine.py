"""
Live demo engine — real MemorySafeBufferV14 snapshots for Streamlit / sandbox UIs.

Replaces demo_live.py synthetic drift with governed replay + MVI EMA from
simulated_cl_mvi_test.PolicyRunner (same kernel as benchmarks).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from memory_health import StreamCounter, snapshot_memory_health
from simulated_cl_mvi_test import (
    PolicyRunner,
    SimulatedCLConfig,
    buffer_live_feed,
    buffer_mvi_stats,
    set_seed,
)


def demo_config() -> SimulatedCLConfig:
    """Fast interactive settings — Compact Governor story (80-cap buffer)."""
    return SimulatedCLConfig(
        n_tasks=5,
        steps_per_task=8,
        batch_size=48,
        pos_frac=0.08,
        buffer_capacity=80,
        replay_prob=0.75,
        replay_bs=32,
        replay_scale=1.25,
        pos_quota_frac=0.40,
        replay_pos_frac=0.45,
        lr=3e-3,
    )


class GovernedDemoSession:
    """
    Step-by-step simulated CL session backed by MemorySafeBufferV14.

    Call tick() once per UI refresh; returns buffer feed + memory health metrics.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.cfg = demo_config()
        self.device = torch.device("cpu")
        set_seed(seed)
        self.runner = PolicyRunner("mvi", self.cfg, self.device)
        self.stream = StreamCounter()
        self.cycles = 0
        self.last_event: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        set_seed(self.seed)
        self.runner = PolicyRunner("mvi", self.cfg, self.device)
        self.stream = StreamCounter()
        self.cycles += 1
        self.last_event = {}

    def _observe_last_batch(self) -> None:
        labels = self.runner.last_batch_labels
        if labels:
            self.stream.observe_labels(labels, self.runner.buffer)

    def tick(self) -> Dict[str, Any]:
        if self.runner.done:
            self.reset(self.seed + self.cycles)

        event = self.runner.step_once()
        self.last_event = event
        self._observe_last_batch()

        buffer = self.runner.buffer
        feed = buffer_live_feed(buffer) if hasattr(buffer, "items") else []
        health = snapshot_memory_health(buffer, self.stream) if hasattr(buffer, "items") else {}
        mvi_stats = buffer_mvi_stats(buffer) if hasattr(buffer, "items") else {}

        wave = self.runner.task_id + 1
        combined_auprc = None
        task0_auprc = None
        if self.runner.history:
            combined_auprc = self.runner.history[-1].get("combined_auprc")
            task0_auprc = self.runner.history[-1].get("task0_auprc")

        protected = sum(1 for s in feed if s["action"] == "protect")
        replay_n = sum(1 for s in feed if s["action"] == "replay")
        high_n = sum(1 for s in feed if s["status"] == "high")

        cap = self.cfg.buffer_capacity
        pos = buffer.count_pos() if hasattr(buffer, "count_pos") else 0
        fill_pct = round(100 * len(buffer) / max(cap, 1))

        return {
            "samples": feed,
            "event": event.get("event", "step"),
            "wave": wave,
            "total_waves": self.cfg.n_tasks,
            "step": self.runner.step_in_task,
            "steps_per_wave": self.cfg.steps_per_task,
            "buffer_fill": len(buffer),
            "buffer_cap": cap,
            "buffer_fill_pct": fill_pct,
            "pos_in_buffer": pos,
            "pos_quota_target": int(cap * self.cfg.pos_quota_frac),
            "protected": protected,
            "replay_n": replay_n,
            "high_n": high_n,
            "mean_mvi": health.get("mean_mvi", mvi_stats.get("mean_risk", 0.0)),
            "fri": health.get("fri", 1.0),
            "coverage_index": health.get("coverage_index", 1.0),
            "mean_replay_exposure": health.get("mean_replay_exposure", 0.0),
            "combined_auprc": combined_auprc,
            "task0_auprc": task0_auprc,
            "last_replayed": self.runner.last_step_replayed,
            "done": self.runner.done,
            "source": "MemorySafeBufferV14",
        }


def get_or_create_session(state: Any, *, seed: int = 42) -> GovernedDemoSession:
    """Streamlit session_state helper."""
    key = "governed_demo_session"
    if key not in state or state[key] is None:
        state[key] = GovernedDemoSession(seed=seed)
    return state[key]


def _self_test() -> None:
    session = GovernedDemoSession(seed=7)
    snapshots: List[Dict[str, Any]] = []
    for _ in range(12):
        snapshots.append(session.tick())
    assert snapshots[-1]["source"] == "MemorySafeBufferV14"
    assert snapshots[-1]["buffer_fill"] > 0
    assert len(snapshots[-1]["samples"]) > 0
    assert "mvi" in snapshots[-1]["samples"][0]
    print("demo_engine self-test OK", snapshots[-1]["buffer_fill"], "slots")


if __name__ == "__main__":
    _self_test()
