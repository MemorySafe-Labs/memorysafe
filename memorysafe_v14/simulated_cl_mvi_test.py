#!/usr/bin/env python3
"""
Simulated continual-learning environment — test MVI (risk EMA) in isolation.

Builds a dynamic simulated CL stream (shifting Gaussian tasks, rare positives),
then compares:
  - memorysafe_mvi   — full buffer with replay-loss risk EMA (production path)
  - memorysafe_no_mvi — same buffer, risk frozen at insert (MVI ablation)
  - reservoir        — uniform replay baseline

Run:
  python simulated_cl_mvi_test.py
  python simulated_cl_mvi_test.py --seeds 5 --tasks 6 --save-dir runs/simulated_cl_mvi
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from buffer_v14 import BufferConfig, MemorySafeBufferV14, ReservoirBuffer
from train_loop import (
    SmallCNN,
    adaptive_replay_prob,
    per_sample_bce_pos_weight,
    uncertainty_from_probs,
)


@dataclass
class SimulatedCLConfig:
    n_tasks: int = 5
    steps_per_task: int = 60
    batch_size: int = 64
    pos_frac: float = 0.08
    feature_dim: int = 28 * 28
    task_shift: float = 6.0
    noise: float = 0.25
    buffer_capacity: int = 120
    replay_prob: float = 0.75
    replay_bs: int = 48
    replay_scale: float = 1.25
    pos_quota_frac: float = 0.40
    replay_pos_frac: float = 0.45
    lr: float = 3e-3
    seed: int = 42


class NoMVIUpdateBuffer(MemorySafeBufferV14):
    """MemorySafe buffer with MVI disabled — risk never updated after replay."""

    def update_risk_from_losses(self, idxs: List[int], losses: np.ndarray) -> None:
        return


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_task_batch(
    task_id: int,
    batch_size: int,
    cfg: SimulatedCLConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic stream: each task shifts the positive/negative Gaussian centers."""
    rng = np.random.default_rng(cfg.seed + task_id * 9973 + batch_size)
    dim = cfg.feature_dim
    n_pos = max(1, int(round(batch_size * cfg.pos_frac)))
    n_neg = batch_size - n_pos

    direction = rng.standard_normal(dim).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    center_pos = direction * cfg.task_shift * (task_id + 1)
    center_neg = -direction * cfg.task_shift * (task_id + 1) * 0.35

    x_pos = center_pos + rng.standard_normal((n_pos, dim)).astype(np.float32) * cfg.noise
    x_neg = center_neg + rng.standard_normal((n_neg, dim)).astype(np.float32) * cfg.noise
    x = np.vstack([x_pos, x_neg])
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    perm = rng.permutation(batch_size)
    x_t = torch.tensor(x[perm], device=device).view(batch_size, 1, 28, 28)
    y_t = torch.tensor(y[perm], device=device)
    return x_t, y_t


def make_task_eval_set(task_id: int, cfg: SimulatedCLConfig, device: torch.device, n: int = 512) -> DataLoader:
    xs, ys = [], []
    for _ in range(max(1, n // cfg.batch_size)):
        x, y = make_task_batch(task_id, cfg.batch_size, cfg, device)
        xs.append(x.cpu())
        ys.append(y.cpu())
    return DataLoader(
        TensorDataset(torch.cat(xs), torch.cat(ys)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )


@torch.no_grad()
def eval_auprc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    from sklearn.metrics import average_precision_score

    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(y.numpy().reshape(-1))
    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def risk_by_task_id(buffer: MemorySafeBufferV14) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {}
    for it in buffer.items:
        buckets.setdefault(it.task_id, []).append(it.risk)
    return {tid: float(np.mean(v)) for tid, v in sorted(buckets.items())}


def buffer_mvi_stats(buffer: MemorySafeBufferV14) -> Dict[str, float]:
    if not buffer.items:
        return {"mean_risk": 0.0, "mean_protect": 0.0, "pos_risk": 0.0, "neg_risk": 0.0, "task_age_mean": 0.0}
    risks = np.array([it.risk for it in buffer.items])
    protects = np.array([it.protect for it in buffer.items])
    pos_r = [it.risk for it in buffer.items if it.y == 1]
    neg_r = [it.risk for it in buffer.items if it.y == 0]
    ages = [max(0, buffer.current_task - it.task_id) for it in buffer.items]
    return {
        "mean_risk": float(risks.mean()),
        "mean_protect": float(protects.mean()),
        "pos_risk": float(np.mean(pos_r)) if pos_r else 0.0,
        "neg_risk": float(np.mean(neg_r)) if neg_r else 0.0,
        "task_age_mean": float(np.mean(ages)),
        "n_items": len(buffer.items),
        "n_pos": buffer.count_pos(),
    }


def _make_buffer(variant: str, cfg: SimulatedCLConfig) -> Tuple[Any, bool]:
    if variant == "reservoir":
        return ReservoirBuffer(cfg.buffer_capacity), False
    if variant == "no_mvi":
        return (
            NoMVIUpdateBuffer(
                BufferConfig(
                    capacity=cfg.buffer_capacity,
                    pos_quota_frac=cfg.pos_quota_frac,
                    replay_pos_frac=cfg.replay_pos_frac,
                )
            ),
            False,
        )
    return (
        MemorySafeBufferV14(
            BufferConfig(
                capacity=cfg.buffer_capacity,
                pos_quota_frac=cfg.pos_quota_frac,
                replay_pos_frac=cfg.replay_pos_frac,
            )
        ),
        True,
    )


class PolicyRunner:
    """Stateful runner — one task (wave) at a time for live UIs."""

    def __init__(self, variant: str, cfg: SimulatedCLConfig, device: torch.device) -> None:
        self.variant = variant
        self.cfg = cfg
        self.device = device
        self.buffer, self.enable_mvi = _make_buffer(variant, cfg)
        self.model = SmallCNN().to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        self.history: List[Dict[str, Any]] = []
        self.replay_loss_traces: List[float] = []
        self.task_id = 0
        self.step_in_task = 0
        self.done = False
        self.last_step_replayed = False
        self.last_batch_labels: List[int] = []
        if hasattr(self.buffer, "set_task"):
            self.buffer.set_task(0)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            set_seed(seed)
        self.__init__(self.variant, self.cfg, self.device)

    def _train_one_batch(self, task_id: int) -> Dict[str, Any]:
        buffer = self.buffer
        cfg = self.cfg
        device = self.device
        variant = self.variant
        enable_mvi = self.enable_mvi

        x, y = make_task_batch(task_id, cfg.batch_size, cfg, device)
        pos = (y == 1).sum().item()
        neg = (y == 0).sum().item()
        pos_weight = torch.tensor([neg / (pos + 1e-12)], device=device)

        bx = by = rep_idxs = None
        rep_losses = None
        cap = cfg.buffer_capacity
        eff_prob = adaptive_replay_prob(
            cfg.replay_prob,
            buffer.count_pos() if len(buffer) > 0 else 0,
            len(buffer) if len(buffer) > 0 else cap,
            cfg.pos_quota_frac,
        )

        if len(buffer) > 0 and random.random() < eff_prob:
            bx, by, rep_idxs = buffer.sample(cfg.replay_bs)
            bx, by = bx.to(device), by.to(device).long()

        if bx is not None:
            x_all = torch.cat([x, bx], dim=0)
            logits_all = self.model(x_all)
            n_cur = x.size(0)
            logits = logits_all[:n_cur]
            blogits = logits_all[n_cur:]
            cur_losses = per_sample_bce_pos_weight(logits, y, pos_weight)
            rep_pos = (by == 1).sum().item()
            rep_neg = (by == 0).sum().item()
            rep_pw = torch.tensor([rep_neg / (rep_pos + 1e-12)], device=device)
            rep_losses = per_sample_bce_pos_weight(blogits, by, rep_pw)
            scale = cfg.replay_scale if variant != "reservoir" else 1.0
            loss = cur_losses.mean() + scale * rep_losses.mean()
            self.replay_loss_traces.append(float(rep_losses.mean().item()))
            self.last_step_replayed = True
        else:
            logits = self.model(x)
            cur_losses = per_sample_bce_pos_weight(logits, y, pos_weight)
            loss = cur_losses.mean()
            self.last_step_replayed = False

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            p = torch.sigmoid(logits).detach()
            unc = uncertainty_from_probs(p)
        value = (0.5 * cur_losses.detach() + 0.5 * unc.detach()).cpu().numpy()
        buffer.add_batch(x, y, value, task_id=task_id)

        if (
            enable_mvi
            and rep_idxs is not None
            and rep_losses is not None
            and hasattr(buffer, "update_risk_from_losses")
        ):
            buffer.update_risk_from_losses(rep_idxs, rep_losses.detach().cpu().numpy())

        self.last_batch_labels = y.detach().cpu().view(-1).int().tolist()
        return {
            "batch_pos": int(pos),
            "batch_neg": int(neg),
            "replayed": self.last_step_replayed,
            "buffer_size": len(buffer),
            "buffer_pos": buffer.count_pos() if hasattr(buffer, "count_pos") else 0,
        }

    def _eval_task(self, task_id: int) -> Dict[str, Any]:
        buffer = self.buffer
        cfg = self.cfg
        device = self.device

        per_task_auprc = []
        for j in range(task_id + 1):
            loader = make_task_eval_set(j, cfg, device)
            per_task_auprc.append(eval_auprc(self.model, loader, device))

        combined_x, combined_y = [], []
        for j in range(task_id + 1):
            loader = make_task_eval_set(j, cfg, device, n=256)
            for xb, yb in loader:
                combined_x.append(xb)
                combined_y.append(yb.view(-1))
        combined_loader = DataLoader(
            TensorDataset(torch.cat(combined_x), torch.cat(combined_y)),
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        combined_auprc = eval_auprc(self.model, combined_loader, device)

        mvi = buffer_mvi_stats(buffer) if isinstance(buffer, MemorySafeBufferV14) else {}
        risk_by_task = risk_by_task_id(buffer) if isinstance(buffer, MemorySafeBufferV14) else {}
        row = {
            "task": task_id + 1,
            "per_task_auprc": per_task_auprc,
            "task0_auprc": per_task_auprc[0],
            "combined_auprc": combined_auprc,
            "mvi": mvi,
            "risk_by_task_id": risk_by_task,
        }
        self.history.append(row)
        return row

    def step_once(self) -> Dict[str, Any]:
        """Advance one training batch; finalize wave when steps_per_task reached."""
        if self.done:
            return {"event": "done", "task": self.task_id, "history": self.history}

        if hasattr(self.buffer, "set_task"):
            self.buffer.set_task(self.task_id)

        batch_info = self._train_one_batch(self.task_id)
        self.step_in_task += 1
        wave_complete = self.step_in_task >= self.cfg.steps_per_task

        if wave_complete:
            row = self._eval_task(self.task_id)
            self.step_in_task = 0
            self.task_id += 1
            if self.task_id >= self.cfg.n_tasks:
                self.done = True
            elif hasattr(self.buffer, "set_task"):
                self.buffer.set_task(self.task_id)
            return {"event": "wave_complete", "wave": row, "batch": batch_info}

        return {"event": "step", "task": self.task_id + 1, "step": self.step_in_task, "batch": batch_info}

    def run_task(self, task_id: int) -> Dict[str, Any]:
        if hasattr(self.buffer, "set_task"):
            self.buffer.set_task(task_id)
        for _ in range(self.cfg.steps_per_task):
            self._train_one_batch(task_id)
        return self._eval_task(task_id)

    def finalize(self) -> Dict[str, Any]:
        buffer = self.buffer
        enable_mvi = self.enable_mvi
        variant = self.variant
        if not self.history:
            final = {"combined_auprc": 0.0, "task0_auprc": 0.0}
        else:
            final = self.history[-1]

        old_task_risk = 0.0
        new_task_risk = 0.0
        risk_gradient_ok = False
        if isinstance(buffer, MemorySafeBufferV14) and buffer.items and enable_mvi:
            by_task = risk_by_task_id(buffer)
            if len(by_task) >= 2:
                tasks = sorted(by_task.keys())
                old_task_risk = by_task[tasks[0]]
                new_task_risk = by_task[tasks[-1]]
                risk_gradient_ok = old_task_risk >= new_task_risk
        elif isinstance(buffer, MemorySafeBufferV14) and buffer.items and not enable_mvi:
            by_task = risk_by_task_id(buffer)
            if len(by_task) >= 2:
                tasks = sorted(by_task.keys())
                old_task_risk = by_task[tasks[0]]
                new_task_risk = by_task[tasks[-1]]

        label = {"reservoir": "reservoir", "no_mvi": "memorysafe_no_mvi", "mvi": "memorysafe_mvi"}[variant]
        return {
            "policy": label,
            "variant": variant,
            "enable_mvi": enable_mvi,
            "task_history": self.history,
            "summary": {
                "final_combined_auprc": final["combined_auprc"],
                "final_task0_auprc": final["task0_auprc"],
                "mean_replay_loss": float(np.mean(self.replay_loss_traces)) if self.replay_loss_traces else 0.0,
                "old_task_mean_risk": old_task_risk,
                "new_task_mean_risk": new_task_risk,
                "mvi_separates_age": risk_gradient_ok,
                "risk_by_task_id": risk_by_task_id(buffer) if isinstance(buffer, MemorySafeBufferV14) else {},
            },
        }


def run_policy(
    variant: str,
    cfg: SimulatedCLConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """variant: reservoir | no_mvi | mvi"""
    runner = PolicyRunner(variant, cfg, device)
    for task_id in range(cfg.n_tasks):
        runner.run_task(task_id)
    return runner.finalize()


def _classify_buffer_slot(it: Any, *, buffer: Any) -> Tuple[str, str]:
    """Map real buffer item → UI status + governance action."""
    risk = float(it.risk)
    protect = float(it.protect)
    is_rare = int(it.y) == 1
    task_age = max(0, getattr(buffer, "current_task", 0) - int(it.task_id))

    if is_rare and (risk >= 0.55 or protect >= 0.45):
        return "high", "protect"
    if task_age >= 1 and protect >= 0.40:
        return "high", "protect"
    if risk >= 0.40 or protect >= 0.32:
        return "medium", "replay"
    return "stable", "ignore"


def buffer_live_feed(buffer: MemorySafeBufferV14, limit: int = 8) -> List[Dict[str, Any]]:
    """Snapshot of MemorySafeBufferV14 slots for live dashboards."""
    if not buffer.items:
        return []

    ranked = sorted(enumerate(buffer.items), key=lambda pair: pair[1].protect, reverse=True)
    feed: List[Dict[str, Any]] = []
    for slot_idx, it in ranked[:limit]:
        risk = float(it.risk)
        protect = float(it.protect)
        status, action = _classify_buffer_slot(it, buffer=buffer)
        mvi_display = round(min(0.99, risk), 2)
        feed.append(
            {
                "sample_id": f"MS-T{it.task_id + 1}-{slot_idx:03d}",
                "wave": it.task_id + 1,
                "rarity": "rare" if it.y == 1 else "common",
                "risk": round(risk, 3),
                "mvi": mvi_display,
                "protect": round(protect, 3),
                "seen": int(it.seen),
                "status": status,
                "action": action,
            }
        )
    return feed


def iter_live_demo(cfg: SimulatedCLConfig, device: torch.device, seed: int):
    """
    Yield after each wave so Streamlit (or any UI) can paint live updates.
    Runs all three policies in lockstep — one wave at a time.
    """
    cfg = SimulatedCLConfig(**{**asdict(cfg), "seed": seed})
    set_seed(seed)
    runners = {
        "memorysafe_mvi": PolicyRunner("mvi", cfg, device),
        "memorysafe_no_mvi": PolicyRunner("no_mvi", cfg, device),
        "reservoir": PolicyRunner("reservoir", cfg, device),
    }
    for wave in range(cfg.n_tasks):
        wave_rows: Dict[str, Dict[str, Any]] = {}
        for key, runner in runners.items():
            wave_rows[key] = runner.run_task(wave)
        ms_buffer = runners["memorysafe_mvi"].buffer
        feed = buffer_live_feed(ms_buffer) if isinstance(ms_buffer, MemorySafeBufferV14) else []
        mvi = buffer_mvi_stats(ms_buffer) if isinstance(ms_buffer, MemorySafeBufferV14) else {}
        yield {
            "wave": wave + 1,
            "total_waves": cfg.n_tasks,
            "policies": wave_rows,
            "feed": feed,
            "protected": sum(1 for x in feed if x["action"] == "protect"),
            "buffer_fill": len(ms_buffer),
            "buffer_cap": cfg.buffer_capacity,
            "mean_risk": mvi.get("mean_risk", 0.0),
        }
    yield {
        "done": True,
        "seed": seed,
        "policies": {k: runners[k].finalize() for k in runners},
    }


def test_mvi_ema_mechanism() -> Dict[str, Any]:
    """Direct unit check: replay loss spikes → risk EMA rises on those slots."""
    cfg = BufferConfig(capacity=20, mvi_ema=0.7)
    buf = MemorySafeBufferV14(cfg)
    buf.set_task(1)
    x = torch.randn(4, 1, 28, 28)
    y = torch.tensor([1, 0, 1, 0])
    buf.add_batch(x, y, np.array([0.3, 0.2, 0.35, 0.15]), task_id=0)
    idxs = [0, 2]
    risk_before = [buf.items[i].risk for i in idxs]
    buf.update_risk_from_losses(idxs, np.array([2.5, 3.0]))
    risk_after = [buf.items[i].risk for i in idxs]
    increased = all(a > b for a, b in zip(risk_after, risk_before))
    return {
        "risk_before": risk_before,
        "risk_after": risk_after,
        "replay_loss_spike_increases_risk": increased,
    }


def run_seed(seed: int, cfg: SimulatedCLConfig, device: torch.device) -> Dict[str, Any]:
    cfg = SimulatedCLConfig(**{**asdict(cfg), "seed": seed})
    set_seed(seed)
    key_map = {"reservoir": "reservoir", "no_mvi": "memorysafe_no_mvi", "mvi": "memorysafe_mvi"}
    results = {}
    for variant, key in key_map.items():
        set_seed(seed)
        results[key] = run_policy(variant, cfg, device)
    return {"seed": seed, "policies": results}


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    policies = ("reservoir", "memorysafe_no_mvi", "memorysafe_mvi")
    out = {}
    for p in policies:
        vals = [r["policies"][p]["summary"]["final_combined_auprc"] for r in runs]
        t0 = [r["policies"][p]["summary"]["final_task0_auprc"] for r in runs]
        sep = [r["policies"][p]["summary"]["mvi_separates_age"] for r in runs]
        out[p] = {
            "combined_auprc": {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals},
            "task0_auprc": {"mean": float(np.mean(t0)), "std": float(np.std(t0))},
            "mvi_separates_old_risk": {"fraction": float(np.mean(sep))},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated CL environment — MVI validation harness")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--steps-per-task", type=int, default=80)
    parser.add_argument("--save-dir", type=str, default="runs/simulated_cl_mvi")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = SimulatedCLConfig(n_tasks=args.tasks, steps_per_task=args.steps_per_task)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    unit = test_mvi_ema_mechanism()
    print("=== MVI MECHANISM UNIT TEST ===")
    print(f"  risk before replay EMA: {unit['risk_before']}")
    print(f"  risk after replay EMA:  {unit['risk_after']}")
    print(f"  spike increases risk:   {unit['replay_loss_spike_increases_risk']}\n")

    print(f"Device: {device}")
    print(f"Simulated CL: {base.n_tasks} tasks × {base.steps_per_task} steps | rare pos {base.pos_frac:.0%}")
    print("Policies: reservoir | memorysafe_no_mvi (ablation) | memorysafe_mvi (production)\n")

    runs = []
    for i in range(args.seeds):
        seed = args.start_seed + i
        print(f"--- SEED {seed} ---")
        run = run_seed(seed, base, device)
        runs.append(run)
        for p in ("reservoir", "memorysafe_no_mvi", "memorysafe_mvi"):
            s = run["policies"][p]["summary"]
            rbt = s.get("risk_by_task_id", {})
            rbt_str = ", ".join(f"t{k}={v:.3f}" for k, v in sorted(rbt.items())[:4])
            if len(rbt) > 4:
                rbt_str += ", ..."
            print(
                f"  {p:22s} combined={s['final_combined_auprc']:.3f} "
                f"task0={s['final_task0_auprc']:.3f} "
                f"risk[{rbt_str}] mvi_grad={s['mvi_separates_age']}"
            )

    agg = aggregate(runs)
    report = {
        "protocol": "simulated-cl-mvi-test",
        "mvi_unit_test": unit,
        "config": asdict(base),
        "n_seeds": args.seeds,
        "aggregates": agg,
        "interpretation": {
            "mvi_working": "early-task buffer risk >= latest-task risk after distribution shift",
            "mvi_helps": "memorysafe_mvi combined_auprc > memorysafe_no_mvi and > reservoir",
        },
    }

    out = save_dir / "simulated_cl_mvi_report.json"
    out.write_text(json.dumps({"report": report, "raw": runs}, indent=2))

    print("\n=== AGGREGATE (final combined AUPRC) ===")
    for p in ("reservoir", "memorysafe_no_mvi", "memorysafe_mvi"):
        a = agg[p]["combined_auprc"]
        sep = agg[p]["mvi_separates_old_risk"]["fraction"]
        print(f"  {p:22s} {a['mean']:.4f} ± {a['std']:.4f}  | MVI ages old>new: {sep:.0%} seeds")

    mvi_mean = agg["memorysafe_mvi"]["combined_auprc"]["mean"]
    no_mvi_mean = agg["memorysafe_no_mvi"]["combined_auprc"]["mean"]
    res_mean = agg["reservoir"]["combined_auprc"]["mean"]
    print(f"\nMVI lift vs no-MVI: {mvi_mean - no_mvi_mean:+.4f}")
    print(f"MVI lift vs reservoir: {mvi_mean - res_mean:+.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
