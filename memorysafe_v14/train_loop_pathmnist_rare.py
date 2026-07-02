"""Binary rare-class continual learning on PathMNIST RGB tiles."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from buffer_v14 import BufferConfig, MemorySafeBufferV14
from memory_health import StreamCounter, apply_memory_health, extend_summary_with_health
from train_loop import (
    adaptive_replay_prob,
    evaluate_binary,
    per_sample_bce_pos_weight,
    uncertainty_from_probs,
)


class ReplayBuffer(Protocol):
    def __len__(self) -> int: ...
    def add_batch(self, xs, ys, values, task_id: int = 0) -> None: ...
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]: ...
    def count_pos(self) -> int: ...


class PathMNISTRareCNN(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
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


def train_continual_pathmnist_rare(
    policy_name: str,
    buffer: ReplayBuffer,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    device: torch.device,
    *,
    in_channels: int = 3,
    replay_prob: float = 0.8,
    replay_bs: int = 128,
    epochs_per_task: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mix_loss: float = 0.5,
    mix_unc: float = 0.5,
    pos_quota_frac: float = 0.40,
    recall_feedback: bool = False,
    recall_target: float = 0.72,
    recall_feedback_gain: float = 0.25,
    replay_scale: float = 1.25,
) -> Dict[str, Any]:
    model = PathMNISTRareCNN(in_channels=in_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict[str, Any] = {"policy": policy_name, "task_metrics": []}
    stream_counter = StreamCounter()
    replay_prob_current = replay_prob

    for t, train_loader in enumerate(train_loaders, start=1):
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
                    len(buffer) if len(buffer) > 0 else BufferConfig().capacity,
                    pos_quota_frac,
                )

                if len(buffer) > 0 and random.random() < eff_replay_prob:
                    bx, by, rep_idxs = buffer.sample(replay_bs)
                    bx, by = bx.to(device), by.to(device).long().view(-1)

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
                    scale = replay_scale if policy_name.startswith("memorysafe") else 1.0
                    loss = cur_losses.mean() + scale * rep_losses.mean()
                else:
                    logits = model(x)
                    cur_losses = per_sample_bce_pos_weight(logits, y, pos_weight)
                    loss = cur_losses.mean()

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
        combined_x, combined_y = [], []
        for j in range(t):
            for xb, yb in test_loaders[j]:
                combined_x.append(xb)
                combined_y.append(yb.view(-1))
        combined_loader = (
            DataLoader(TensorDataset(torch.cat(combined_x), torch.cat(combined_y)), batch_size=128, shuffle=False)
            if combined_x
            else test_loaders[t - 1]
        )
        combined = evaluate_binary(model, combined_loader, device) if combined_x else per_task[-1]
        task0 = per_task[0]
        agg = {
            "task_completed": t,
            "avg_auprc_up_to_t": float(np.mean([m["auprc"] for m in per_task])),
            "combined_auprc": combined["auprc"],
            "combined_recall_pos": combined["recall_pos"],
            "task0_auprc": task0["auprc"],
            "task0_recall_pos": task0["recall_pos"],
            "combined_recall_at_1pct_fpr": combined["recall_at_1pct_fpr"],
            "buffer_size": len(buffer),
            "per_task_eval": per_task,
        }
        if hasattr(buffer, "memory_bytes"):
            agg["buffer_memory_mb"] = buffer.memory_bytes() / (1024 * 1024)
        apply_memory_health(agg, buffer, stream_counter)
        history["task_metrics"].append(agg)

        print(
            f"[{policy_name}] task {t}: "
            f"combined_auprc={agg['combined_auprc']:.3f} "
            f"recall_pos={agg['combined_recall_pos']:.3f} "
            f"task0={agg['task0_auprc']:.3f} "
            f"buf={agg['buffer_size']} "
            f"mvi={agg['mean_mvi']:.3f} fri={agg['fri']:.3f}"
        )

        if recall_feedback and policy_name.startswith("memorysafe"):
            if combined["recall_pos"] < recall_target:
                replay_prob_current = min(0.95, replay_prob_current + recall_feedback_gain * (recall_target - combined["recall_pos"]))

    final = history["task_metrics"][-1]
    history["summary"] = {
        "final_avg_auprc": final["avg_auprc_up_to_t"],
        "combined_auprc": final["combined_auprc"],
        "combined_recall_pos": final["combined_recall_pos"],
        "task0_retention_recall": final["task0_recall_pos"],
        "task0_auprc": final["task0_auprc"],
        "combined_recall_at_1pct_fpr": final["combined_recall_at_1pct_fpr"],
        "buffer_memory_mb": final.get("buffer_memory_mb", 0.0),
    }
    extend_summary_with_health(history["summary"], final)
    return history
