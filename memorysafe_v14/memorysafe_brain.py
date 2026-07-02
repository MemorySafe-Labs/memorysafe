"""
MemorySafe v14 — governance engine for continual learning.

Decides what to PROTECT, REINFORCE (replay), or DEFER (reactivation pool)
using MVI + ProtectScore under bounded memory constraints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MemorySafeBrain:
    """
    Predict which samples are vulnerable to forgetting, govern memory buffers,
    and support replay under fixed capacity.
    """

    def __init__(
        self,
        buffer_size: int = 150,
        reactivation_pool_size: int = 80,
        protect_threshold: float = 0.70,
        reinforce_threshold: float = 0.45,
        reactivation_threshold: float = 0.72,
        alpha: float = 0.45,
        beta: float = 0.25,
        gamma: float = 0.20,
        delta: float = 0.10,
        mvi_ema: float = 0.7,
    ):
        self.buffer_size = buffer_size
        self.reactivation_pool_size = reactivation_pool_size
        self.protect_threshold = protect_threshold
        self.reinforce_threshold = reinforce_threshold
        self.reactivation_threshold = reactivation_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mvi_ema = mvi_ema

        self.protected_buffer: Dict[int, dict] = {}
        self.replay_queue: Dict[int, dict] = {}
        self.reactivation_pool: Dict[int, dict] = {}
        self.explanations: List[str] = []
        self._mvi_state: Dict[int, float] = {}

    def compute_protect_score(
        self,
        mvi: float,
        rarity: float,
        criticality: float,
        uncertainty: float,
    ) -> float:
        score = (
            self.alpha * mvi
            + self.beta * rarity
            + self.gamma * criticality
            + self.delta * uncertainty
        )
        return float(min(max(score, 0.0), 1.0))

    @staticmethod
    def compute_signals(
        loss: float,
        uncertainty: float,
        rarity: float,
        criticality: float,
        task_age: int = 0,
        prev_mvi: Optional[float] = None,
        mvi_ema: float = 0.7,
    ) -> Tuple[float, float]:
        """
        Derive MVI and ProtectScore inputs from model outputs.

        MVI blends per-sample loss (forgetting proxy) with task-age pressure.
        """
        raw_mvi = 0.55 * min(loss, 2.0) / 2.0 + 0.25 * uncertainty + 0.20 * min(task_age / 5.0, 1.0)
        if prev_mvi is None:
            mvi = raw_mvi
        else:
            mvi = mvi_ema * prev_mvi + (1.0 - mvi_ema) * raw_mvi
        mvi = float(min(max(mvi, 0.0), 1.0))
        protect_score = (
            0.45 * mvi + 0.25 * rarity + 0.20 * criticality + 0.10 * uncertainty
        )
        return mvi, float(min(max(protect_score, 0.0), 1.0))

    def decide(
        self,
        sample_id: int,
        label: int,
        embedding: torch.Tensor,
        mvi: float,
        rarity: float = 0.0,
        criticality: float = 0.0,
        uncertainty: float = 0.0,
        x: Optional[torch.Tensor] = None,
        new_data_embedding: Optional[torch.Tensor] = None,
    ) -> dict:
        reactivated: List[int] = []
        if new_data_embedding is not None:
            reactivated = self.check_reactivation(new_data_embedding)

        protect_score = self.compute_protect_score(mvi, rarity, criticality, uncertainty)
        self._mvi_state[sample_id] = mvi

        if protect_score >= self.protect_threshold:
            action = "PROTECT"
            self.protect(sample_id, label, embedding, mvi, protect_score, x=x)
        elif protect_score >= self.reinforce_threshold:
            action = "REINFORCE"
            self.reinforce(sample_id, label, embedding, mvi, protect_score, x=x)
        else:
            action = "DEFER"
            self.defer(sample_id, label, embedding, mvi, protect_score, x=x)

        explanation = {
            "sample_id": sample_id,
            "label": label,
            "mvi": round(mvi, 4),
            "protect_score": round(protect_score, 4),
            "action": action,
            "reactivated_samples": reactivated,
        }
        self.explanations.append(str(explanation))
        return explanation

    def _entry(
        self,
        label: int,
        embedding: torch.Tensor,
        mvi: float,
        protect_score: float,
        x: Optional[torch.Tensor],
    ) -> dict:
        entry = {
            "label": label,
            "embedding": embedding.detach().clone(),
            "mvi": mvi,
            "protect_score": protect_score,
        }
        if x is not None:
            entry["x"] = x.detach().cpu().clone()
        return entry

    def protect(
        self,
        sample_id: int,
        label: int,
        embedding: torch.Tensor,
        mvi: float,
        protect_score: float,
        x: Optional[torch.Tensor] = None,
    ) -> None:
        if len(self.protected_buffer) >= self.buffer_size:
            self.evict_lowest_priority()
        self.protected_buffer[sample_id] = self._entry(label, embedding, mvi, protect_score, x)
        self.replay_queue.pop(sample_id, None)

    def reinforce(
        self,
        sample_id: int,
        label: int,
        embedding: torch.Tensor,
        mvi: float,
        protect_score: float,
        x: Optional[torch.Tensor] = None,
    ) -> None:
        self.replay_queue[sample_id] = self._entry(label, embedding, mvi, protect_score, x)

    def defer(
        self,
        sample_id: int,
        label: int,
        embedding: torch.Tensor,
        mvi: float,
        protect_score: float,
        x: Optional[torch.Tensor] = None,
    ) -> None:
        if len(self.reactivation_pool) >= self.reactivation_pool_size:
            oldest = min(self.reactivation_pool.keys())
            del self.reactivation_pool[oldest]
        self.reactivation_pool[sample_id] = self._entry(label, embedding, mvi, protect_score, x)

    def check_reactivation(self, new_embedding: torch.Tensor) -> List[int]:
        reactivated_ids: List[int] = []
        for old_id, data in list(self.reactivation_pool.items()):
            sim = F.cosine_similarity(
                new_embedding.unsqueeze(0),
                data["embedding"].unsqueeze(0),
            ).item()
            if sim >= self.reactivation_threshold:
                old = self.reactivation_pool.pop(old_id)
                self.protect(
                    sample_id=old_id,
                    label=old["label"],
                    embedding=old["embedding"],
                    mvi=old["mvi"],
                    protect_score=max(old["protect_score"], sim),
                    x=old.get("x"),
                )
                reactivated_ids.append(old_id)
        return reactivated_ids

    def evict_lowest_priority(self) -> None:
        if not self.protected_buffer:
            return
        lowest_id = min(
            self.protected_buffer,
            key=lambda sid: self.protected_buffer[sid]["protect_score"],
        )
        del self.protected_buffer[lowest_id]

    def replay_batch(self, max_items: int = 16) -> List[dict]:
        ranked = sorted(
            self.replay_queue.items(),
            key=lambda item: item[1]["protect_score"],
            reverse=True,
        )
        return [
            {
                "sample_id": sid,
                "label": data["label"],
                "embedding": data["embedding"],
                "mvi": data["mvi"],
                "protect_score": data["protect_score"],
                "x": data.get("x"),
            }
            for sid, data in ranked[:max_items]
        ]

    def sample_tensors(
        self,
        batch_size: int,
        device: torch.device,
        include_protected: bool = True,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Sample stored inputs for replay training.
        Pulls from replay queue first, then protected buffer by ProtectScore.
        """
        candidates: List[Tuple[int, dict, str]] = []
        for sid, data in self.replay_queue.items():
            if data.get("x") is not None:
                candidates.append((sid, data, "replay"))
        if include_protected:
            for sid, data in self.protected_buffer.items():
                if data.get("x") is not None:
                    candidates.append((sid, data, "protected"))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[1]["protect_score"], reverse=True)
        k = min(batch_size, len(candidates))
        scores = torch.tensor([c[1]["protect_score"] for c in candidates[:k]], dtype=torch.float32)
        probs = scores / scores.sum().clamp(min=1e-8)
        chosen_idx = torch.multinomial(probs, num_samples=k, replacement=False).tolist()

        xs, ys, ids = [], [], []
        for i in chosen_idx:
            sid, data, _ = candidates[i]
            xs.append(data["x"])
            ys.append(data["label"])
            ids.append(sid)
            data["protect_score"] = min(1.0, data["protect_score"] + 0.02)

        x_batch = torch.stack(xs).to(device)
        y_batch = torch.tensor(ys, dtype=torch.long, device=device)
        return x_batch, y_batch, ids

    def update_mvi_from_losses(self, sample_ids: List[int], losses: List[float]) -> None:
        for sid, loss in zip(sample_ids, losses):
            prev = self._mvi_state.get(sid, float(loss))
            self._mvi_state[sid] = self.mvi_ema * prev + (1.0 - self.mvi_ema) * float(loss)
            for store in (self.protected_buffer, self.replay_queue):
                if sid in store:
                    store[sid]["mvi"] = self._mvi_state[sid]
                    store[sid]["protect_score"] = self.compute_protect_score(
                        self._mvi_state[sid],
                        rarity=0.5,
                        criticality=0.5,
                        uncertainty=min(float(loss), 1.0),
                    )

    def get_status(self) -> dict:
        return {
            "protected_buffer": len(self.protected_buffer),
            "replay_queue": len(self.replay_queue),
            "reactivation_pool": len(self.reactivation_pool),
            "last_decisions": self.explanations[-10:],
        }

    def print_status(self) -> None:
        status = self.get_status()
        print("\nMemorySafe Brain Status")
        print(f"  Protected buffer   : {status['protected_buffer']}")
        print(f"  Replay queue       : {status['replay_queue']}")
        print(f"  Reactivation pool  : {status['reactivation_pool']}")
        print("  Last decisions:")
        for line in status["last_decisions"]:
            print(f"    {line}")


def memorysafe_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    brain: MemorySafeBrain,
    *,
    task_id: int = 0,
    class_counts: Optional[Dict[int, int]] = None,
    replay_prob: float = 0.5,
    replay_batch_size: int = 32,
    critical_classes: Optional[set] = None,
    embedding_fn=None,
    loss_fn=None,
    global_step: int = 0,
) -> dict:
    """
    One continual-learning step: forward, govern memory, replay, backward.

    embedding_fn(model, x) -> [B, D] feature vectors
    loss_fn(model, x, y) -> per-sample losses [B]
    """
    device = x.device
    model.train()

    if embedding_fn is None:
        def embedding_fn(m, batch):
            with torch.no_grad():
                logits = m(batch)
                return logits.detach()

    if loss_fn is None:
        def loss_fn(m, batch, labels):
            logits = m(batch)
            return F.cross_entropy(logits, labels, reduction="none")

    logits = model(x)
    per_loss = loss_fn(model, x, y)
    probs = F.softmax(logits.detach(), dim=1)
    uncertainty = 1.0 - probs.max(dim=1).values
    embeddings = embedding_fn(model, x)

    total = class_counts.get("total", 1) if class_counts else max(len(y), 1)
    crit = critical_classes or set()

    replay_loss = torch.tensor(0.0, device=device)
    replay_ids: List[int] = []

    if brain.sample_tensors(1, device) and torch.rand(1).item() < replay_prob:
        sampled = brain.sample_tensors(replay_batch_size, device)
        if sampled is not None:
            bx, by, replay_ids = sampled
            replay_loss = loss_fn(model, bx, by).mean()

    loss = per_loss.mean() + replay_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    decisions = []
    for i in range(x.size(0)):
        label = int(y[i].item())
        count = class_counts.get(label, 1) if class_counts else 1
        rarity = float(1.0 - count / total)
        criticality = 1.0 if label in crit else 0.3
        prev_mvi = brain._mvi_state.get(global_step * 10000 + i)
        mvi, _ = brain.compute_signals(
            loss=float(per_loss[i].item()),
            uncertainty=float(uncertainty[i].item()),
            rarity=rarity,
            criticality=criticality,
            task_age=task_id,
            prev_mvi=prev_mvi,
            mvi_ema=brain.mvi_ema,
        )
        sid = global_step * 10000 + i
        decisions.append(
            brain.decide(
                sample_id=sid,
                label=label,
                embedding=embeddings[i],
                mvi=mvi,
                rarity=rarity,
                criticality=criticality,
                uncertainty=float(uncertainty[i].item()),
                x=x[i],
                new_data_embedding=embeddings[i],
            )
        )

    if replay_ids:
        brain.update_mvi_from_losses(
            replay_ids,
            [float(replay_loss.item())] * len(replay_ids),
        )

    return {
        "loss": float(loss.item()),
        "replay_loss": float(replay_loss.item()),
        "decisions": decisions,
        "brain_status": brain.get_status(),
    }
