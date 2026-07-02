#!/usr/bin/env python3
"""
MemorySafe v14 — continual learning training loop (wired, not mocked).

Runs class-incremental CIFAR-10 with MemorySafeBrain governing protect /
reinforce / defer and real replay from stored tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from memorysafe_brain import MemorySafeBrain, memorysafe_training_step

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return self.head(h)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x)
            return torch.flatten(h, 1)


def get_incremental_loaders(batch_size: int = 64) -> Tuple[List[DataLoader], List[DataLoader]]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loaders, test_loaders = [], []
    for pair in CLASS_PAIRS:
        train_idx = [i for i, (_, y) in enumerate(train_ds) if y in pair]
        test_idx = [i for i, (_, y) in enumerate(test_ds) if y in pair]
        train_loaders.append(
            DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
        )
        test_loaders.append(
            DataLoader(Subset(test_ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)
        )
    return train_loaders, test_loaders


@torch.no_grad()
def evaluate(model: nn.Module, loaders: List[DataLoader], up_to_task: int) -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    for loader in loaders[:up_to_task]:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / max(total, 1), "samples": total}


def train_continual(
    epochs_per_task: int = 2,
    lr: float = 1e-3,
    replay_prob: float = 0.5,
    replay_batch_size: int = 32,
    buffer_size: int = 200,
    save_dir: str = "runs/v14_cifar10",
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    train_loaders, test_loaders = get_incremental_loaders()
    model = SmallCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    brain = MemorySafeBrain(buffer_size=buffer_size, reactivation_pool_size=80)

    history: List[dict] = []
    global_step = 0
    class_counts: Dict[int, int] = Counter()
    class_counts["total"] = 0

    def embedding_fn(m, batch):
        return m.embed(batch)

    def loss_fn(m, batch, labels):
        return F.cross_entropy(m(batch), labels, reduction="none")

    for task_id, train_loader in enumerate(train_loaders):
        print(f"\n=== Task {task_id + 1}/{len(train_loaders)} classes {CLASS_PAIRS[task_id]} ===")
        for epoch in range(epochs_per_task):
            for x, y in train_loader:
                global_step += 1
                x, y = x.to(DEVICE), y.to(DEVICE)

                for label in y.tolist():
                    class_counts[label] = class_counts.get(label, 0) + 1
                    class_counts["total"] += 1

                step_info = memorysafe_training_step(
                    model,
                    optimizer,
                    x,
                    y,
                    brain,
                    task_id=task_id,
                    class_counts=class_counts,
                    replay_prob=replay_prob,
                    replay_batch_size=replay_batch_size,
                    critical_classes=set(CLASS_PAIRS[0]),
                    embedding_fn=embedding_fn,
                    loss_fn=loss_fn,
                    global_step=global_step,
                )

                if global_step % 100 == 0:
                    status = step_info["brain_status"]
                    print(
                        f"  step {global_step:5d} | loss {step_info['loss']:.4f} "
                        f"| replay {step_info['replay_loss']:.4f} "
                        f"| buf {status['protected_buffer']} "
                        f"| queue {status['replay_queue']} "
                        f"| pool {status['reactivation_pool']}"
                    )

        metrics = evaluate(model, test_loaders, task_id + 1)
        status = brain.get_status()
        record = {
            "task": task_id + 1,
            "classes": CLASS_PAIRS[task_id],
            "accuracy_up_to_task": metrics["accuracy"],
            "brain_status": status,
        }
        history.append(record)
        print(
            f"  >> after task {task_id + 1}: acc={metrics['accuracy']:.4f} "
            f"| protected={status['protected_buffer']} "
            f"| replay_q={status['replay_queue']}"
        )

    brain.print_status()
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"history": history, "final_status": brain.get_status()}, f, indent=2)
    print(f"\nSaved model and results to {save_dir}")
    return {"history": history, "final_status": brain.get_status()}


def main():
    parser = argparse.ArgumentParser(description="MemorySafe v14 continual learning on CIFAR-10")
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--replay-prob", type=float, default=0.5)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/v14_cifar10")
    args = parser.parse_args()

    set_seed(args.seed)
    train_continual(
        epochs_per_task=args.epochs_per_task,
        lr=args.lr,
        replay_prob=args.replay_prob,
        replay_batch_size=args.replay_batch_size,
        buffer_size=args.buffer_size,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
