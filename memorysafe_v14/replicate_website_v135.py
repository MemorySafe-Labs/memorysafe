#!/usr/bin/env python3
"""
Replicate website-era PneumoniaMNIST results (v13.5 Colab protocol).

Source: MemorySafe-Chat-Archive/conversations/2026-03-26_memorysafe-v13-gentler-quota-tuning.md
        (10-seed v13.5 block, ~line 9176)

Website claim: AUPRC 0.941 ± 0.007, minority recall ~64.1%, 2-task split, buffer 1500.
This is NOT the v14.2 5-task harness (0.706 combined AUPRC).

Usage:
  python replicate_website_v135.py --seeds 10 --policy memorysafe
  python replicate_website_v135.py --seeds 10 --policy reservoir
  python replicate_website_v135.py --seeds 3 --policy both
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from medmnist import PneumoniaMNIST
from sklearn.metrics import average_precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(__file__).parent / "runs" / "website_v135_replication"


@dataclass(frozen=True)
class V135Config:
    buffer_capacity: int = 1500
    target_minority_ratio: float = 0.40
    minority_class: int = 1
    batch_size: int = 64
    replay_batch_size: int = 32
    replay_weight: float = 0.60
    epochs_per_task: int = 8
    lr: float = 1e-3
    mvi_conf_weight: float = 0.70
    mvi_rarity_weight: float = 0.30
    ps_minority_multiplier: float = 1.75
    coreshield_boost: float = 0.40
    ar_step: float = 0.06
    target_minority_recall: float = 0.70
    quota_strength_start: float = 0.75
    quota_strength_end: float = 0.25


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class SimpleTensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels.squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class LightMVI:
    def __init__(self, cfg: V135Config):
        self.cfg = cfg
        self.class_counts: dict[int, int] = {}

    def compute(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        confidence = probs[torch.arange(len(labels), device=labels.device), labels]
        uncertainty = 1.0 - confidence
        rarity = torch.zeros(len(labels), device=labels.device)
        for i, lab in enumerate(labels):
            lab_i = int(lab)
            self.class_counts[lab_i] = self.class_counts.get(lab_i, 0) + 1
            total = sum(self.class_counts.values())
            avg = total / max(1, len(self.class_counts))
            if self.class_counts[lab_i] < avg * 0.8:
                rarity[i] = self.cfg.mvi_rarity_weight
        mvi = self.cfg.mvi_conf_weight * uncertainty + (rarity / max(1, len(labels)))
        return mvi.clamp(0.0, 1.0).detach()


class SmartBuffer:
    def __init__(self, cfg: V135Config):
        self.cfg = cfg
        self.capacity = cfg.buffer_capacity
        self.target_ratio = cfg.target_minority_ratio
        self.items: list[tuple] = []
        self.total_steps = 0

    def add(self, x, y, mvi_score):
        is_minority = int(y) == self.cfg.minority_class
        score = float(mvi_score)
        if is_minority:
            score *= self.cfg.ps_minority_multiplier
        self.items.append((x.cpu(), int(y), score, float(mvi_score)))
        self.total_steps += 1
        if len(self.items) > self.capacity:
            self._evict()

    def _evict(self):
        if not self.items:
            return
        minority_count = sum(1 for _, y, _, _ in self.items if y == self.cfg.minority_class)
        current_ratio = minority_count / len(self.items)
        progress = min(1.0, self.total_steps / 12000.0)
        quota_strength = (
            self.cfg.quota_strength_start * (1 - progress)
            + self.cfg.quota_strength_end * progress
        )
        if current_ratio > self.target_ratio + 0.04:
            excess = current_ratio - self.target_ratio
            bias_factor = quota_strength * min(excess * 2.0, 0.75)

            def eviction_key(t):
                _, y, boosted_score, original_mvi = t
                if y == self.cfg.minority_class:
                    return boosted_score * (1.0 - bias_factor) - (original_mvi * 0.5)
                return boosted_score

            self.items.sort(key=eviction_key)
            self.items.pop(0)
            return
        self.items.sort(key=lambda t: t[2])
        self.items.pop(0)

    def sample(self, batch_size):
        if not self.items:
            return None, None
        n = min(batch_size, len(self.items))
        scores = torch.tensor([score for _, _, score, _ in self.items])
        probs = F.softmax(scores, dim=0)
        idx = torch.multinomial(probs, n, replacement=False)
        xs = torch.stack([self.items[i][0] for i in idx]).to(DEVICE)
        ys = torch.tensor([self.items[i][1] for i in idx], device=DEVICE)
        return xs, ys

    def stats(self):
        if not self.items:
            return {"size": 0, "minority_ratio": 0.0}
        minority = sum(1 for _, y, _, _ in self.items if y == self.cfg.minority_class)
        return {"size": len(self.items), "minority_ratio": minority / len(self.items)}


class ReservoirBuffer:
    def __init__(self, cfg: V135Config):
        self.cfg = cfg
        self.capacity = cfg.buffer_capacity
        self.items: list[tuple] = []

    def add(self, x, y, _score=None):
        if len(self.items) < self.capacity:
            self.items.append((x.cpu(), int(y)))
        else:
            idx = np.random.randint(0, self.capacity)
            self.items[idx] = (x.cpu(), int(y))

    def sample(self, batch_size):
        if not self.items:
            return None, None
        n = min(batch_size, len(self.items))
        idx = np.random.choice(len(self.items), n, replace=False)
        xs = torch.stack([self.items[i][0] for i in idx]).to(DEVICE)
        ys = torch.tensor([self.items[i][1] for i in idx], device=DEVICE)
        return xs, ys

    def stats(self):
        if not self.items:
            return {"size": 0, "minority_ratio": 0.0}
        minority = sum(1 for _, y in self.items if y == self.cfg.minority_class)
        return {"size": len(self.items), "minority_ratio": minority / len(self.items)}


class LightCoreShield:
    def __init__(self, cfg: V135Config):
        self.cfg = cfg

    def boost(self, score, mvi):
        return score * (1.0 + self.cfg.coreshield_boost * float(mvi))


class LightAR:
    def __init__(self, cfg: V135Config):
        self.cfg = cfg
        self.replay_weight = cfg.replay_weight

    def update(self, minority_recall: float):
        if minority_recall < self.cfg.target_minority_recall:
            self.replay_weight = min(1.0, self.replay_weight + self.cfg.ar_step)
        else:
            self.replay_weight = max(0.4, self.replay_weight - self.cfg.ar_step * 0.5)


def load_pneumonia_tensors(split: str = "train"):
    ds = PneumoniaMNIST(split=split, download=True, as_rgb=True)
    x = torch.tensor(
        np.stack(
            [np.transpose(np.array(ds[i][0]) / 255.0, (2, 0, 1)) for i in range(len(ds))]
        ),
        dtype=torch.float32,
    )
    y = torch.tensor([int(ds[i][1]) for i in range(len(ds))], dtype=torch.long)
    return x, y


@torch.no_grad()
def evaluate(model, loader, minority_class: int):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).squeeze()
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, minority_class]
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    acc = float((all_preds == all_labels).mean())
    minority_recall = float(
        recall_score(
            all_labels == minority_class,
            all_preds == minority_class,
            zero_division=0,
        )
    )
    auprc = float(average_precision_score(all_labels == minority_class, all_probs))
    return {"accuracy": acc, "minority_recall": minority_recall, "auprc": auprc}


def run_seed(seed: int, policy: str, cfg: V135Config):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train = load_pneumonia_tensors("train")
    x_test, y_test = load_pneumonia_tensors("test")

    split = len(y_train) // 2
    tasks = [(x_train[:split], y_train[:split]), (x_train[split:], y_train[split:])]

    model = SmallCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    if policy == "memorysafe":
        buffer = SmartBuffer(cfg)
        mvi = LightMVI(cfg)
        coreshield = LightCoreShield(cfg)
        ar = LightAR(cfg)
        replay_w = None
    else:
        buffer = ReservoirBuffer(cfg)
        mvi = coreshield = ar = None
        replay_w = cfg.replay_weight

    for task_id, (tx, ty) in enumerate(tasks):
        loader = DataLoader(
            SimpleTensorDataset(tx, ty),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        for epoch in range(cfg.epochs_per_task):
            for batch_x, batch_y in tqdm(
                loader,
                desc=f"seed={seed} {policy} task={task_id} ep={epoch+1}",
                leave=False,
            ):
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE).squeeze()

                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)

                if len(getattr(buffer, "items", [])) > 0:
                    rx, ry = buffer.sample(cfg.replay_batch_size)
                    if rx is not None:
                        rloss = F.cross_entropy(model(rx), ry)
                        w = ar.replay_weight if ar else replay_w
                        loss = loss + w * rloss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    if policy == "memorysafe":
                        mvi_scores = mvi.compute(logits, batch_y)
                        for i in range(len(batch_y)):
                            mvi_val = float(mvi_scores[i])
                            boosted = coreshield.boost(mvi_val, mvi_val)
                            buffer.add(batch_x[i], batch_y[i].item(), boosted)
                    else:
                        for i in range(len(batch_y)):
                            buffer.add(batch_x[i], batch_y[i].item())

            if ar is not None:
                metrics = evaluate(model, loader, cfg.minority_class)
                ar.update(metrics["minority_recall"])

    test_loader = DataLoader(
        SimpleTensorDataset(x_test, y_test),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    final = evaluate(model, test_loader, cfg.minority_class)
    final["buffer"] = buffer.stats()
    final["seed"] = seed
    final["policy"] = policy
    return final


def summarize(results: list[dict]) -> dict:
    keys = ["auprc", "minority_recall", "accuracy"]
    out = {}
    for k in keys:
        vals = [r[k] for r in results]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--policy", choices=["memorysafe", "reservoir", "both"], default="both")
    args = parser.parse_args()

    cfg = V135Config()
    policies = ["memorysafe", "reservoir"] if args.policy == "both" else [args.policy]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Protocol: website v13.5 — 2-task, buffer={cfg.buffer_capacity}, epochs/task={cfg.epochs_per_task}")
    print(f"Seeds: {args.seeds} (base={args.base_seed})\n")

    report = {"protocol": "website-v13.5-2task", "config": cfg.__dict__, "policies": {}}

    for policy in policies:
        results = []
        for i in range(args.seeds):
            seed = args.base_seed + i
            results.append(run_seed(seed, policy, cfg))
        summary = summarize(results)
        report["policies"][policy] = {"summary": summary, "runs": results}

        print(f"=== {policy} ===")
        print(f"  AUPRC (test):          {summary['auprc']['mean']:.4f} ± {summary['auprc']['std']:.4f}")
        print(f"  Minority recall (test): {summary['minority_recall']['mean']:.4f} ± {summary['minority_recall']['std']:.4f}")
        print(f"  Accuracy (test):        {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
        print()

    out_path = OUT_DIR / f"website_v135_{args.seeds}seed.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
