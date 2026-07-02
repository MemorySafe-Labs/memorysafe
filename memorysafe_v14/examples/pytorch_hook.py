#!/usr/bin/env python3
"""
MemorySafeGovernor — universal PyTorch training hook (any CL loop).

  python examples/pytorch_hook.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from governor import MemorySafeGovernor  # noqa: E402


def main():
    device = torch.device("cpu")
    torch.manual_seed(0)
    random.seed(0)

    # --- Your model ---
    model = torch.nn.Linear(16, 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # --- MemorySafe governor (class-incremental quota) ---
    gov = MemorySafeGovernor.for_class_incremental(capacity=200, replay_prob=0.7)
    gov.set_task(0)

    for step in range(150):
        x = torch.randn(32, 16, device=device)
        y = torch.randint(0, 4, (32,), device=device)

        bx, by, rep_idxs = gov.maybe_sample()
        if bx is not None:
            bx, by = bx.to(device), by.to(device)
            x_all = torch.cat([x, bx])
            y_all = torch.cat([y, by])
            logits = model(x_all)
            n = x.size(0)
            cur_losses = F.cross_entropy(logits[:n], y, reduction="none")
            rep_losses = F.cross_entropy(logits[n:], by, reduction="none")
            loss = cur_losses.mean() + gov.cfg.replay_scale * rep_losses.mean()
            rep_loss_np = rep_losses.detach().cpu().numpy()
        else:
            logits = model(x)
            cur_losses = F.cross_entropy(logits, y, reduction="none")
            loss = cur_losses.mean()
            rep_loss_np = None

        opt.zero_grad()
        loss.backward()
        opt.step()

        probs = F.softmax(logits[: x.size(0)], dim=1)
        values = gov.compute_value_scores(cur_losses, probs)
        gov.observe(
            x, y, values,
            task_id=step // 50,
            replay_idxs=rep_idxs,
            replay_losses=rep_loss_np,
        )

        if step % 50 == 49:
            print(f"step {step+1}: {gov.audit_summary()}")

    print("Governor hook demo complete.")


if __name__ == "__main__":
    main()