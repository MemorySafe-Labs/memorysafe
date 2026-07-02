"""Class-incremental training loop for PathMNIST (grayscale MedMNIST)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from memory_health import StreamCounter, apply_memory_health, extend_summary_with_health


class ReplayBuffer(Protocol):
    def __len__(self) -> int: ...
    def add_batch(self, xs, ys, values, task_id: int = 0) -> None: ...
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]: ...
    def set_task(self, task_id: int) -> None: ...


class PathMNISTCNN(nn.Module):
    def __init__(self, n_classes: int = 9, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)


def per_sample_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y, reduction="none")


def uncertainty_from_probs(probs: torch.Tensor) -> torch.Tensor:
    ent = -(probs * (probs + 1e-12).log()).sum(dim=1)
    return (ent / np.log(probs.size(1))).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_multiclass(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    per_class_correct: Dict[int, int] = {}
    per_class_total: Dict[int, int] = {}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for c in y.unique().tolist():
            mask = y == c
            per_class_correct[c] = per_class_correct.get(c, 0) + (pred[mask] == y[mask]).sum().item()
            per_class_total[c] = per_class_total.get(c, 0) + mask.sum().item()
    acc = correct / max(total, 1)
    class_accs = [per_class_correct[c] / per_class_total[c] for c in per_class_total if per_class_total[c] > 0]
    tail_classes = sorted(per_class_total.keys())[-2:]
    tail_accs = [per_class_correct[c] / per_class_total[c] for c in tail_classes if per_class_total[c] > 0]
    return {
        "acc": float(acc),
        "mean_class_acc": float(np.mean(class_accs)) if class_accs else 0.0,
        "tail_class_acc": float(np.mean(tail_accs)) if tail_accs else 0.0,
        "n_classes": len(class_accs),
    }


def _combined_loader(test_loaders: List[DataLoader], t: int) -> DataLoader:
    all_x, all_y = [], []
    for j in range(t):
        for xb, yb in test_loaders[j]:
            all_x.append(xb)
            all_y.append(yb.view(-1))
    return DataLoader(
        TensorDataset(torch.cat(all_x), torch.cat(all_y)),
        batch_size=128,
        shuffle=False,
    )


def train_continual_pathmnist(
    policy_name: str,
    buffer: ReplayBuffer,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    device: torch.device,
    *,
    n_classes: int = 9,
    in_channels: int = 3,
    replay_prob: float = 0.8,
    replay_bs: int = 128,
    epochs_per_task: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mix_loss: float = 0.5,
    mix_unc: float = 0.5,
    replay_scale: float = 1.25,
) -> Dict[str, Any]:
    model = PathMNISTCNN(n_classes=n_classes, in_channels=in_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict[str, Any] = {"policy": policy_name, "task_metrics": []}
    stream_counter = StreamCounter()

    if hasattr(buffer, "set_task"):
        buffer.set_task(0)

    for t, train_loader in enumerate(train_loaders, start=1):
        if hasattr(buffer, "set_task"):
            buffer.set_task(t - 1)

        for _ in range(epochs_per_task):
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device).long().view(-1)

                rep_idxs: Optional[List[int]] = None
                rep_losses: Optional[torch.Tensor] = None
                bx = by = None

                if len(buffer) > 0 and random.random() < replay_prob:
                    bx, by, rep_idxs = buffer.sample(replay_bs)
                    bx, by = bx.to(device), by.to(device).long().view(-1)

                if bx is not None:
                    x_all = torch.cat([x, bx], dim=0)
                    logits_all = model(x_all)
                    n_cur = x.size(0)
                    logits = logits_all[:n_cur]
                    blogits = logits_all[n_cur:]
                    cur_losses = per_sample_ce(logits, y)
                    rep_losses = per_sample_ce(blogits, by)
                    scale = replay_scale if policy_name.startswith("memorysafe") else 1.0
                    loss = cur_losses.mean() + scale * rep_losses.mean()
                else:
                    logits = model(x)
                    cur_losses = per_sample_ce(logits, y)
                    loss = cur_losses.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    probs = F.softmax(logits, dim=1).detach()
                    unc = uncertainty_from_probs(probs)
                value = (mix_loss * cur_losses.detach() + mix_unc * unc.detach()).cpu().numpy()
                stream_counter.observe_labels(y, buffer)
                buffer.add_batch(x, y, value, task_id=t - 1)

                if rep_idxs is not None and rep_losses is not None and hasattr(buffer, "update_risk_from_losses"):
                    buffer.update_risk_from_losses(rep_idxs, rep_losses.detach().cpu().numpy())

        per_task = [evaluate_multiclass(model, test_loaders[j], device) for j in range(t)]
        combined = evaluate_multiclass(model, _combined_loader(test_loaders, t), device)
        task0 = per_task[0]
        agg = {
            "task_completed": t,
            "avg_acc_up_to_t": float(np.mean([m["acc"] for m in per_task])),
            "combined_acc": combined["acc"],
            "mean_class_acc": combined["mean_class_acc"],
            "tail_class_acc": combined["tail_class_acc"],
            "task0_acc": task0["acc"],
            "task0_mean_class_acc": task0["mean_class_acc"],
            "buffer_size": len(buffer),
            "per_task_eval": per_task,
        }
        if hasattr(buffer, "memory_bytes"):
            agg["buffer_memory_mb"] = buffer.memory_bytes() / (1024 * 1024)
        if hasattr(buffer, "group_distribution"):
            agg["buffer_groups"] = buffer.group_distribution()
        apply_memory_health(agg, buffer, stream_counter)
        history["task_metrics"].append(agg)

        print(
            f"[{policy_name}] task {t}: "
            f"avg_acc={agg['avg_acc_up_to_t']:.3f} "
            f"combined={agg['combined_acc']:.3f} "
            f"tail={agg['tail_class_acc']:.3f} "
            f"task0={agg['task0_acc']:.3f} "
            f"buf={agg['buffer_size']} "
            f"mvi={agg['mean_mvi']:.3f} fri={agg['fri']:.3f}"
        )

    final = history["task_metrics"][-1]
    history["summary"] = {
        "final_avg_acc": final["avg_acc_up_to_t"],
        "combined_acc": final["combined_acc"],
        "mean_class_acc": final["mean_class_acc"],
        "tail_class_acc": final["tail_class_acc"],
        "task0_retention_acc": final["task0_acc"],
        "task0_mean_class_acc": final["task0_mean_class_acc"],
        "buffer_memory_mb": final.get("buffer_memory_mb", 0.0),
    }
    extend_summary_with_health(history["summary"], final)
    return history
