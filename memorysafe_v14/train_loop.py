"""
Efficient continual-learning training loop for MemorySafe v14 benchmarks.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from buffer_v14 import BufferConfig, MemorySafeBufferV14
from memory_health import (
    HealthControllerConfig,
    MemoryHealthController,
    StreamCounter,
    apply_controller_knobs,
    apply_memory_health,
    extend_summary_with_health,
    make_controller,
)


class ReplayBuffer(Protocol):
    def __len__(self) -> int: ...
    def add_batch(self, xs, ys, values, task_id: int = 0) -> None: ...
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]: ...
    def count_pos(self) -> int: ...


def per_sample_bce_pos_weight(
    logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits.view(-1), y.float().view(-1), reduction="none", pos_weight=pos_weight
    )


def uncertainty_from_probs(p: torch.Tensor) -> torch.Tensor:
    return (1.0 - torch.abs(2.0 * p - 1.0)).clamp(0.0, 1.0)


def adaptive_replay_prob(base_prob: float, buffer_pos: int, buffer_size: int, target_pos_frac: float) -> float:
    """Increase replay when buffer is starved of positives."""
    if buffer_size <= 0:
        return base_prob
    target = int(buffer_size * target_pos_frac)
    if buffer_pos < target:
        deficit = (target - buffer_pos) / max(target, 1)
        return min(0.95, base_prob + 0.35 * deficit)
    return base_prob


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


@torch.no_grad()
def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, confusion_matrix

    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_prob.append(probs)
        y_true.append(y.numpy().reshape(-1))

    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)
    auprc = float(average_precision_score(y_true, y_prob))

    best_f1, best_thr, best_rec, best_prec = -1.0, 0.5, 0.0, 0.0
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1, best_thr, best_rec, best_prec = float(f1), float(thr), float(rec), float(prec)

    # Recall @ 1% FPR
    neg_mask = y_true == 0
    pos_mask = y_true == 1
    recall_at_1fpr = 0.0
    if neg_mask.any() and pos_mask.any():
        neg_probs = y_prob[neg_mask]
        thr_1fpr = np.quantile(neg_probs, 1.0 - 0.01)
        recall_at_1fpr = float((y_prob[pos_mask] >= thr_1fpr).mean())

    acc = float(((y_prob >= best_thr).astype(int) == y_true).mean())
    return {
        "acc": acc,
        "recall_pos": best_rec,
        "precision_pos": best_prec,
        "f1_pos": best_f1,
        "auprc": auprc,
        "recall_at_1pct_fpr": recall_at_1fpr,
        "best_thr": best_thr,
    }


def train_continual(
    policy_name: str,
    buffer: ReplayBuffer,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    device: torch.device,
    *,
    replay_prob: float = 0.5,
    replay_bs: int = 128,
    epochs_per_task: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mix_loss: float = 0.5,
    mix_unc: float = 0.5,
    pos_quota_frac: float = 0.30,
    recall_feedback: bool = False,
    recall_target: float = 0.72,
    recall_feedback_gain: float = 0.25,
    replay_scale: float = 1.25,
    health_feedback: bool = False,
    replay_pos_frac: Optional[float] = None,
    health_controller_config: Optional[HealthControllerConfig] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict[str, Any] = {"policy": policy_name, "task_metrics": []}
    stream_counter = StreamCounter()
    total_steps = 0
    replay_steps = 0
    replay_prob_current = replay_prob
    replay_scale_current = replay_scale
    pos_quota_current = pos_quota_frac
    if replay_pos_frac is None:
        replay_pos_frac = pos_quota_frac * 1.125
    controller: Optional[MemoryHealthController] = None
    if health_feedback and policy_name.startswith("memorysafe"):
        controller = make_controller(
            mode="binary",
            replay_prob=replay_prob,
            replay_scale=replay_scale,
            pos_quota_frac=pos_quota_frac,
            replay_pos_frac=replay_pos_frac,
            controller_cfg=health_controller_config,
        )
        if hasattr(buffer, "cfg"):
            apply_controller_knobs(buffer, controller.knobs)

    if hasattr(buffer, "set_task"):
        buffer.set_task(0)

    for t, train_loader in enumerate(train_loaders, start=1):
        if hasattr(buffer, "set_task"):
            buffer.set_task(t - 1)

        for _ in range(epochs_per_task):
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device).long().view(-1)

                pos = (y == 1).sum().item()
                neg = (y == 0).sum().item()
                pos_weight = torch.tensor([neg / (pos + 1e-12)], device=device)

                rep_idxs: Optional[List[int]] = None
                rep_losses: Optional[torch.Tensor] = None
                bx = by = None

                eff_replay_prob = adaptive_replay_prob(
                    replay_prob_current,
                    buffer.count_pos() if len(buffer) > 0 else 0,
                    len(buffer) if len(buffer) > 0 else getattr(buffer, "cfg", BufferConfig()).capacity,
                    pos_quota_current,
                )

                if len(buffer) > 0 and random.random() < eff_replay_prob:
                    bx, by, rep_idxs = buffer.sample(replay_bs)
                    bx = bx.to(device)
                    by = by.to(device).long().view(-1)

                # Single forward pass when replaying (efficiency)
                if bx is not None:
                    x_all = torch.cat([x, bx], dim=0)
                    logits_all = model(x_all)
                    n_cur = x.size(0)
                    logits = logits_all[:n_cur]
                    blogits = logits_all[n_cur:]
                    cur_losses = per_sample_bce_pos_weight(logits, y, pos_weight)
                    rep_pos = (by == 1).sum().item()
                    rep_neg = (by == 0).sum().item()
                    rep_pos_weight = torch.tensor([rep_neg / (rep_pos + 1e-12)], device=device)
                    rep_losses = per_sample_bce_pos_weight(blogits, by, rep_pos_weight)
                    eff_scale = replay_scale_current if policy_name.startswith("memorysafe") else 1.0
                    loss = cur_losses.mean() + eff_scale * rep_losses.mean()
                else:
                    logits = model(x)
                    cur_losses = per_sample_bce_pos_weight(logits, y, pos_weight)
                    loss = cur_losses.mean()

                total_steps += 1
                if bx is not None:
                    replay_steps += 1

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    p = torch.sigmoid(logits).detach()
                    unc = uncertainty_from_probs(p)
                value = (mix_loss * cur_losses.detach() + mix_unc * unc.detach()).cpu().numpy()
                stream_counter.observe_labels(y, buffer)
                buffer.add_batch(x, y, value, task_id=t - 1)

                if rep_idxs is not None and rep_losses is not None and hasattr(buffer, "update_risk_from_losses"):
                    buffer.update_risk_from_losses(rep_idxs, rep_losses.detach().cpu().numpy())

        per_task = [evaluate_binary(model, test_loaders[j], device) for j in range(t)]
        # Combined eval: all seen tasks in one pass (primary ranking metric)
        combined_x, combined_y = [], []
        for j in range(t):
            for xb, yb in test_loaders[j]:
                combined_x.append(xb)
                combined_y.append(yb.view(-1))
        if combined_x:
            combined_loader = DataLoader(
                TensorDataset(torch.cat(combined_x), torch.cat(combined_y)),
                batch_size=128,
                shuffle=False,
            )
        else:
            combined_loader = None
        combined_metrics = (
            evaluate_binary(model, combined_loader, device) if combined_loader else per_task[-1]
        )
        task0 = per_task[0]
        agg = {
            "task_completed": t,
            "avg_acc_up_to_t": float(np.mean([m["acc"] for m in per_task])),
            "avg_recall_pos_up_to_t": float(np.mean([m["recall_pos"] for m in per_task])),
            "avg_auprc_up_to_t": float(np.mean([m["auprc"] for m in per_task])),
            "task0_recall": task0["recall_pos"],
            "task0_auprc": task0["auprc"],
            "recall_at_1pct_fpr": float(np.mean([m["recall_at_1pct_fpr"] for m in per_task])),
            "combined_auprc": combined_metrics["auprc"],
            "combined_recall_pos": combined_metrics["recall_pos"],
            "combined_recall_at_1pct_fpr": combined_metrics["recall_at_1pct_fpr"],
            "pos_in_buffer": buffer.count_pos(),
            "buffer_size": len(buffer),
            "per_task_eval": per_task,
        }
        if hasattr(buffer, "memory_bytes"):
            agg["buffer_memory_mb"] = buffer.memory_bytes() / (1024 * 1024)
        agg["replay_prob_effective"] = replay_prob_current
        agg["replay_scale_effective"] = replay_scale_current
        agg["pos_quota_effective"] = pos_quota_current
        apply_memory_health(agg, buffer, stream_counter)
        if controller is not None:
            state = controller.step(agg)
            agg["controller_state"] = state
            replay_prob_current = state["knobs"]["replay_prob"]
            replay_scale_current = state["knobs"]["replay_scale"]
            pos_quota_current = state["knobs"]["pos_quota_frac"]
            apply_controller_knobs(buffer, state["knobs"])
        history["task_metrics"].append(agg)

        if recall_feedback and policy_name.startswith("memorysafe") and controller is None:
            gap = max(0.0, recall_target - combined_metrics["recall_pos"])
            replay_prob_current = min(0.95, replay_prob + recall_feedback_gain * gap)

        print(
            f"[{policy_name}] task {t}: "
            f"recall={agg['avg_recall_pos_up_to_t']:.3f} "
            f"auprc={agg['avg_auprc_up_to_t']:.3f} "
            f"task0_recall={agg['task0_recall']:.3f} "
            f"buf_pos={agg['pos_in_buffer']}/{agg['buffer_size']} "
            f"mvi={agg['mean_mvi']:.3f} fri={agg['fri']:.3f}"
            + (
                f" ctrl={agg['controller_state']['action']}"
                if "controller_state" in agg
                else ""
            )
        )

    final = history["task_metrics"][-1]
    if controller is not None:
        history["controller_history"] = controller.history
        history["health_feedback"] = True
    wall_time_sec = time.perf_counter() - t0
    replay_step_frac = float(replay_steps / total_steps) if total_steps > 0 else 0.0
    history["summary"] = {
        "final_avg_recall": final["avg_recall_pos_up_to_t"],
        "final_avg_auprc": final["avg_auprc_up_to_t"],
        "combined_auprc": final["combined_auprc"],
        "combined_recall_pos": final["combined_recall_pos"],
        "task0_retention_recall": final["task0_recall"],
        "recall_at_1pct_fpr": final["recall_at_1pct_fpr"],
        "combined_recall_at_1pct_fpr": final["combined_recall_at_1pct_fpr"],
        "buffer_pos": final["pos_in_buffer"],
        "buffer_memory_mb": final.get("buffer_memory_mb", 0.0),
        "wall_time_sec": wall_time_sec,
        "replay_step_frac": replay_step_frac,
        "total_train_steps": total_steps,
    }
    extend_summary_with_health(history["summary"], final)
    return history
