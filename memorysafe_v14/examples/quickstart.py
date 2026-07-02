#!/usr/bin/env python3
"""
MemorySafe v14 — minimal integration example (~20 lines of core logic).

Run from repo root:
  python examples/quickstart.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from buffer_v14 import BufferConfig, MemorySafeBufferV14  # noqa: E402


def main():
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cpu")

    # --- Your model (any architecture) ---
    model = torch.nn.Linear(28 * 28, 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # --- MemorySafe buffer (the product) ---
    buffer = MemorySafeBufferV14(BufferConfig(capacity=500, pos_quota_frac=0.40, replay_pos_frac=0.45))
    replay_prob, replay_bs = 0.80, 64

    # --- Simulated stream: rare positives (y=1) are 5% of batches ---
    for step in range(200):
        x = torch.randn(32, 1, 28, 28, device=device)
        y = (torch.rand(32, device=device) < 0.05).long()

        pos, neg = (y == 1).sum().item(), (y == 0).sum().item()
        pos_weight = torch.tensor([neg / (pos + 1e-12)], device=device)

        bx = by = None
        if len(buffer) > 0 and random.random() < replay_prob:
            bx, by, _ = buffer.sample(replay_bs)
            bx, by = bx.to(device), by.to(device).long()

        if bx is not None:
            x_all = torch.cat([x, bx])
            y_all = torch.cat([y, by])
            logits = model(x_all.flatten(1)).squeeze(-1)
            n = x.size(0)
            cur_loss = F.binary_cross_entropy_with_logits(logits[:n], y.float(), pos_weight=pos_weight)
            rep_pos = (by == 1).sum().item()
            rep_neg = (by == 0).sum().item()
            rep_pw = torch.tensor([rep_neg / (rep_pos + 1e-12)], device=device)
            rep_loss = F.binary_cross_entropy_with_logits(logits[n:], by.float(), pos_weight=rep_pw)
            loss = cur_loss + 1.25 * rep_loss
        else:
            logits = model(x.flatten(1)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            p = torch.sigmoid(logits[: x.size(0)]).detach()
            unc = (1.0 - torch.abs(2.0 * p - 1.0)).clamp(0, 1)
            value = (0.5 * loss.detach().expand(x.size(0)) + 0.5 * unc).cpu().numpy()
        buffer.add_batch(x, y, value)

        if step % 50 == 49:
            print(f"step {step+1}: buffer {len(buffer)} | positives {buffer.count_pos()}")

    print("Done — buffer holds governed rare-class memory under fixed budget.")


if __name__ == "__main__":
    main()
