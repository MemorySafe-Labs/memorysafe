"""
Memory health metrics for continual-learning benchmarks.

Fragility axis — mean MVI (buffer risk EMA) and mean ProtectScore.
Frequency axis — FRI (buffer vs stream alignment) and coverage index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np


def group_key_for_label(buffer: Any, label: int) -> Hashable:
    if hasattr(buffer, "quota") and hasattr(buffer.quota, "group_key"):
        return buffer.quota.group_key(label)
    return int(label)


class StreamCounter:
    """Counts how often each group appears in the training stream."""

    def __init__(self) -> None:
        self.counts: Dict[Hashable, int] = {}
        self.total: int = 0

    def observe_labels(self, ys: Any, buffer: Optional[Any] = None) -> None:
        if hasattr(ys, "detach"):
            labels = ys.detach().cpu().view(-1).tolist()
        else:
            labels = list(ys)
        for y in labels:
            key = group_key_for_label(buffer, int(y)) if buffer is not None else int(y)
            self.counts[key] = self.counts.get(key, 0) + 1
            self.total += 1

    def freq_scores(self) -> Dict[str, float]:
        if self.total <= 0:
            return {}
        return {str(k): v / self.total for k, v in self.counts.items()}


def buffer_group_counts(buffer: Any) -> Dict[str, int]:
    if hasattr(buffer, "group_distribution"):
        return {str(k): int(v) for k, v in buffer.group_distribution().items()}
    if not hasattr(buffer, "items"):
        return {}
    out: Dict[str, int] = {}
    for it in buffer.items:
        key = str(group_key_for_label(buffer, int(it.y)))
        out[key] = out.get(key, 0) + 1
    return out


def _count_for_key(dist: Mapping[Hashable, float], key: str) -> float:
    for g, v in dist.items():
        if str(g) == key:
            return float(v)
    return 0.0


def _aligned_probs(
    a: Mapping[Hashable, float],
    b: Mapping[Hashable, float],
) -> Tuple[np.ndarray, np.ndarray]:
    keys = sorted({str(k) for k in a} | {str(k) for k in b})
    if not keys:
        return np.array([]), np.array([])
    ta = float(sum(a.values()))
    tb = float(sum(b.values()))
    if ta <= 0 or tb <= 0:
        return np.array([]), np.array([])
    pa = np.array([_count_for_key(a, k) / ta for k in keys], dtype=np.float64)
    pb = np.array([_count_for_key(b, k) / tb for k in keys], dtype=np.float64)
    return pa, pb


def js_divergence(a: Mapping[Hashable, float], b: Mapping[Hashable, float]) -> float:
    """Jensen–Shannon divergence (natural log)."""
    pa, pb = _aligned_probs(a, b)
    if pa.size == 0:
        return 0.0
    m = 0.5 * (pa + pb)
    eps = 1e-12

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        return float(np.sum(p * np.log(p / q)))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def frequency_representation_index(
    buffer_dist: Mapping[Hashable, float],
    stream_dist: Mapping[Hashable, float],
) -> float:
    """
    FRI: 1 − normalized JS divergence. 1 = buffer matches stream; 0 = maximally misaligned.
    """
    if not stream_dist:
        return 1.0
    js = js_divergence(buffer_dist, stream_dist)
    return float(max(0.0, 1.0 - js / np.log(2.0)))


def coverage_index(
    buffer_dist: Mapping[Hashable, float],
    stream_dist: Mapping[Hashable, float],
    min_frac: float = 0.04,
) -> float:
    """Fraction of seen stream groups with ≥ min_frac representation in buffer."""
    if not stream_dist:
        return 1.0
    total_buf = float(sum(buffer_dist.values()))
    if total_buf <= 0:
        return 0.0
    covered = 0
    for g in stream_dist:
        buf_n = next((buffer_dist[k] for k in buffer_dist if str(k) == str(g)), 0)
        if buf_n / total_buf >= min_frac:
            covered += 1
    return float(covered / len(stream_dist))


def _mean_item_field(buffer: Any, field: str) -> float:
    if not hasattr(buffer, "items") or not buffer.items:
        return 0.0
    vals = [float(getattr(it, field, 0.0)) for it in buffer.items]
    return float(np.mean(vals))


def snapshot_memory_health(buffer: Any, stream: StreamCounter) -> Dict[str, Any]:
    buf_dist = buffer_group_counts(buffer)
    stream_dist = dict(stream.counts)
    return {
        "mean_mvi": _mean_item_field(buffer, "risk"),
        "mean_protect": _mean_item_field(buffer, "protect"),
        "mean_replay_exposure": _mean_item_field(buffer, "seen"),
        "fri": frequency_representation_index(buf_dist, stream_dist),
        "coverage_index": coverage_index(buf_dist, stream_dist),
        "stream_groups": {str(k): int(v) for k, v in stream.counts.items()},
        "buffer_groups": buf_dist,
        "freq_scores": stream.freq_scores(),
    }


SUMMARY_HEALTH_KEYS = (
    "mean_mvi",
    "mean_protect",
    "mean_replay_exposure",
    "fri",
    "coverage_index",
)


def _is_hybrid_buffer(buffer: Any) -> bool:
    return hasattr(buffer, "core") and hasattr(buffer, "shell")


def apply_memory_health(agg: MutableMapping[str, Any], buffer: Any, stream: StreamCounter) -> None:
    if _is_hybrid_buffer(buffer):
        health_core = snapshot_memory_health(buffer.core, stream)
        health_shell = snapshot_memory_health(buffer.shell, stream)
        health_combined = snapshot_memory_health(buffer, stream)
        n_core = max(1, len(buffer.core))
        n_shell = max(0, len(buffer.shell))
        n_total = n_core + n_shell
        agg["mean_mvi"] = float(
            (health_core["mean_mvi"] * n_core + health_shell["mean_mvi"] * n_shell) / n_total
        )
        agg["mean_protect"] = float(
            (health_core["mean_protect"] * n_core + health_shell["mean_protect"] * n_shell) / n_total
        )
        agg["mean_replay_exposure"] = float(
            (health_core["mean_replay_exposure"] * n_core + health_shell["mean_replay_exposure"] * n_shell)
            / n_total
        )
        agg["fri"] = health_core["fri"]
        agg["fri_core"] = health_core["fri"]
        agg["fri_combined"] = health_combined["fri"]
        agg["coverage_index"] = health_combined["coverage_index"]
        agg["memory_health"] = {
            **health_combined,
            "fri": health_core["fri"],
            "fri_core": health_core["fri"],
            "fri_combined": health_combined["fri"],
            "mean_mvi_core": health_core["mean_mvi"],
            "mean_mvi_shell": health_shell["mean_mvi"],
            "core_size": len(buffer.core),
            "shell_size": len(buffer.shell),
        }
        return

    health = snapshot_memory_health(buffer, stream)
    for k in SUMMARY_HEALTH_KEYS:
        agg[k] = health[k]
    agg["memory_health"] = health


def extend_summary_with_health(summary: MutableMapping[str, Any], final_task: Mapping[str, Any]) -> None:
    for k in SUMMARY_HEALTH_KEYS:
        summary[k] = final_task[k]


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


@dataclass
class HealthControllerConfig:
    """Dual-axis closed-loop targets (Layer 3)."""

    mode: str = "binary"  # binary | multiclass
    fri_floor: float = 0.88
    fri_target: float = 0.93
    mvi_high: float = 1.0
    rare_target: float = 0.72
    task0_target: float = 0.70
    outcome_drop_tol: float = 0.01
    replay_prob_step: float = 0.04
    replay_scale_step: float = 0.10
    pos_quota_step: float = 0.02
    min_frac_step: float = 0.01
    old_task_boost_step: float = 0.15
    replay_prob_bounds: Tuple[float, float] = (0.55, 0.95)
    replay_scale_bounds: Tuple[float, float] = (1.0, 2.2)
    pos_quota_bounds: Tuple[float, float] = (0.30, 0.45)
    min_frac_bounds: Tuple[float, float] = (0.04, 0.10)
    old_task_boost_bounds: Tuple[float, float] = (1.0, 2.5)


class MemoryHealthController:
    """
    MemorySafe-AR v2 — adapt replay/quota knobs from MVI + FRI after each task.

    Can increase OR decrease governance spend (unlike rejected recall_feedback).
    """

    def __init__(self, base_knobs: Mapping[str, float], cfg: Optional[HealthControllerConfig] = None):
        self.cfg = cfg or HealthControllerConfig()
        self.knobs: Dict[str, float] = {k: float(v) for k, v in base_knobs.items()}
        self.history: List[Dict[str, Any]] = []
        self._prev_outcome: Optional[float] = None
        self._prev_fri: Optional[float] = None

    def _outcome(self, task_metrics: Mapping[str, Any]) -> float:
        if self.cfg.mode == "binary":
            return float(task_metrics.get("combined_auprc", 0.0))
        return float(task_metrics.get("combined_acc", 0.0))

    def _task0(self, task_metrics: Mapping[str, Any]) -> float:
        if self.cfg.mode == "binary":
            return float(task_metrics.get("task0_recall", 0.0))
        return float(task_metrics.get("task0_acc", 0.0))

    def _rare(self, task_metrics: Mapping[str, Any]) -> float:
        if self.cfg.mode == "binary":
            return float(task_metrics.get("combined_recall_pos", 0.0))
        return float(task_metrics.get("mean_class_acc", 0.0))

    def _apply_deltas(self, deltas: Mapping[str, float]) -> None:
        c = self.cfg
        if "replay_prob" in deltas:
            self.knobs["replay_prob"] = _clamp(
                self.knobs["replay_prob"] + deltas["replay_prob"],
                *c.replay_prob_bounds,
            )
        if "replay_scale" in deltas:
            self.knobs["replay_scale"] = _clamp(
                self.knobs["replay_scale"] + deltas["replay_scale"],
                *c.replay_scale_bounds,
            )
        if "pos_quota_frac" in deltas:
            self.knobs["pos_quota_frac"] = _clamp(
                self.knobs["pos_quota_frac"] + deltas["pos_quota_frac"],
                *c.pos_quota_bounds,
            )
            self.knobs["replay_pos_frac"] = _clamp(
                self.knobs["pos_quota_frac"] * 1.125,
                0.30,
                0.50,
            )
        if "min_frac_per_class" in deltas:
            self.knobs["min_frac_per_class"] = _clamp(
                self.knobs.get("min_frac_per_class", 0.06) + deltas["min_frac_per_class"],
                *c.min_frac_bounds,
            )
        if "old_task_boost" in deltas:
            self.knobs["old_task_boost"] = _clamp(
                self.knobs.get("old_task_boost", 2.0) + deltas["old_task_boost"],
                *c.old_task_boost_bounds,
            )
        if "core_replay_frac" in deltas:
            self.knobs["core_replay_frac"] = _clamp(
                self.knobs.get("core_replay_frac", 0.60) + deltas["core_replay_frac"],
                0.40,
                0.80,
            )

    def step(self, task_metrics: Mapping[str, Any]) -> Dict[str, Any]:
        c = self.cfg
        fri = float(task_metrics.get("fri", 1.0))
        mvi = float(task_metrics.get("mean_mvi", 0.0))
        outcome = self._outcome(task_metrics)
        task0 = self._task0(task_metrics)
        rare = self._rare(task_metrics)
        outcome_drop = (
            self._prev_outcome is not None and outcome < self._prev_outcome - c.outcome_drop_tol
        )
        fri_drop = self._prev_fri is not None and fri < self._prev_fri - 0.04

        deltas: Dict[str, float] = {}
        action = "hold"

        if (fri < c.fri_floor and outcome_drop) or (fri_drop and outcome_drop and fri < c.fri_target):
            action = "restore_frequency"
            deltas["replay_scale"] = -c.replay_scale_step
            if c.mode == "binary":
                deltas["pos_quota_frac"] = -c.pos_quota_step
            else:
                deltas["min_frac_per_class"] = c.min_frac_step
                deltas["old_task_boost"] = -c.old_task_boost_step
                if "core_replay_frac" in self.knobs:
                    deltas["core_replay_frac"] = 0.05
        elif mvi > c.mvi_high and (task0 < c.task0_target or rare < c.rare_target):
            action = "spend_fragility"
            deltas["replay_scale"] = c.replay_scale_step * 0.5
            deltas["replay_prob"] = c.replay_prob_step * 0.5
            if c.mode == "binary":
                deltas["pos_quota_frac"] = c.pos_quota_step
            else:
                deltas["old_task_boost"] = c.old_task_boost_step * 0.5
                if "core_replay_frac" in self.knobs:
                    deltas["core_replay_frac"] = -0.05
        elif fri >= c.fri_target and rare < c.rare_target:
            action = "boost_practice"
            deltas["replay_prob"] = c.replay_prob_step * 0.5
            deltas["replay_scale"] = c.replay_scale_step * 0.5

        self._apply_deltas(deltas)
        self._prev_outcome = outcome
        self._prev_fri = fri

        state = {
            "action": action,
            "deltas": deltas,
            "knobs": dict(self.knobs),
            "inputs": {
                "fri": fri,
                "mvi": mvi,
                "outcome": outcome,
                "task0": task0,
                "rare": rare,
                "outcome_drop": outcome_drop,
                "fri_drop": fri_drop,
            },
        }
        self.history.append(state)
        return state


def lite_controller_config(mode: str = "binary") -> HealthControllerConfig:
    """Tighter replay bounds for MemorySafe Lite — spend GPU only when health drifts."""
    cfg = HealthControllerConfig(mode=mode)
    cfg.replay_prob_bounds = (0.40, 0.70)
    cfg.replay_scale_bounds = (1.0, 1.5)
    cfg.replay_prob_step = 0.05
    cfg.replay_scale_step = 0.08
    return cfg


def make_controller(
    *,
    mode: str,
    replay_prob: float,
    replay_scale: float,
    pos_quota_frac: float = 0.40,
    replay_pos_frac: Optional[float] = None,
    min_frac_per_class: float = 0.06,
    old_task_boost: float = 2.0,
    core_replay_frac: Optional[float] = None,
    fri_floor: Optional[float] = None,
    controller_cfg: Optional[HealthControllerConfig] = None,
) -> MemoryHealthController:
    cfg = controller_cfg or HealthControllerConfig(mode=mode)
    if mode == "binary":
        cfg.fri_floor = 0.90 if fri_floor is None else fri_floor
        cfg.fri_target = 0.93
    else:
        cfg.fri_floor = 0.90 if fri_floor is None else fri_floor
        cfg.fri_target = 0.94
        cfg.mvi_high = 1.8
        cfg.task0_target = 0.15
        cfg.rare_target = 0.14

    knobs = {
        "replay_prob": replay_prob,
        "replay_scale": replay_scale,
        "pos_quota_frac": pos_quota_frac,
        "replay_pos_frac": replay_pos_frac if replay_pos_frac is not None else pos_quota_frac * 1.125,
        "min_frac_per_class": min_frac_per_class,
        "old_task_boost": old_task_boost,
    }
    if core_replay_frac is not None:
        knobs["core_replay_frac"] = core_replay_frac
    return MemoryHealthController(knobs, cfg)


def apply_controller_knobs(buffer: Any, knobs: Mapping[str, float]) -> None:
    if hasattr(buffer, "cfg"):
        if "pos_quota_frac" in knobs:
            buffer.cfg.pos_quota_frac = knobs["pos_quota_frac"]
        if "replay_pos_frac" in knobs:
            buffer.cfg.replay_pos_frac = knobs["replay_pos_frac"]
    quota_target = buffer
    if _is_hybrid_buffer(buffer):
        if "core_replay_frac" in knobs:
            buffer.core_replay_frac = knobs["core_replay_frac"]
        quota_target = buffer.shell
    if hasattr(quota_target, "quota"):
        if "min_frac_per_class" in knobs and hasattr(quota_target.quota, "min_frac_per_class"):
            quota_target.quota.min_frac_per_class = knobs["min_frac_per_class"]
        if "old_task_boost" in knobs and hasattr(quota_target.quota, "old_task_boost"):
            quota_target.quota.old_task_boost = knobs["old_task_boost"]


def _self_test() -> None:
    stream = StreamCounter()
    for y in [0, 0, 0, 1]:
        stream.observe_labels([y])
    buf = {"0": 75, "1": 25}
    fri = frequency_representation_index(buf, stream.counts)
    assert 0.0 <= fri <= 1.0
    assert coverage_index(buf, stream.counts, min_frac=0.20) == 1.0
    assert abs(frequency_representation_index({"0": 50, "1": 50}, stream.counts) - fri) >= 0.0
    ctrl = make_controller(mode="binary", replay_prob=0.8, replay_scale=1.25)
    s1 = ctrl.step({"fri": 0.95, "mean_mvi": 1.1, "combined_auprc": 0.5, "task0_recall": 0.5, "combined_recall_pos": 0.5})
    assert s1["action"] in ("hold", "spend_fragility", "boost_practice")
    s2 = ctrl.step({"fri": 0.80, "mean_mvi": 2.0, "combined_auprc": 0.4, "task0_recall": 0.0, "combined_recall_pos": 0.3})
    assert s2["action"] == "restore_frequency"
    print("memory_health self-test OK")


if __name__ == "__main__":
    _self_test()
