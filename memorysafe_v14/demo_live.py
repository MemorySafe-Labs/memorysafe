"""
Legacy synthetic feed for sandbox UI prototyping.

Production demo uses demo_engine.py → MemorySafeBufferV14 (demo_streamlit.py).
"""

from __future__ import annotations

import math
import random
from typing import Any


def classify_risk(mvi: float) -> str:
    if mvi >= 0.75:
        return "high"
    if mvi >= 0.4:
        return "medium"
    return "stable"


def decide_action(mvi: float, rarity: str, *, auto_protect: bool, mode: str) -> str:
    if not auto_protect:
        return "ignore"
    if mode == "before":
        return "replay" if mvi >= 0.6 else "ignore"
    if rarity == "rare" and mvi >= 0.7:
        return "protect"
    if mvi >= 0.45:
        return "replay"
    return "ignore"


def evaluate_sample(sample: dict, *, auto_protect: bool = True, mode: str = "after") -> dict[str, Any]:
    score = float(sample.get("score", 0.5))
    mvi = max(0.0, min(1.0, score))
    rarity = sample.get("rarity", "common")
    status = classify_risk(mvi)
    action = decide_action(mvi, rarity, auto_protect=auto_protect, mode=mode)
    return {
        "sample_id": sample.get("id", "unknown"),
        "mvi": round(mvi, 2),
        "rarity": rarity,
        "status": status,
        "action": action,
    }


_TEMPLATES = [
    {"id": "MS-1042", "base": 0.91, "rarity": "rare", "phase": 0.2},
    {"id": "MS-1043", "base": 0.62, "rarity": "common", "phase": 1.1},
    {"id": "MS-1044", "base": 0.28, "rarity": "rare", "phase": 2.4},
    {"id": "MS-1045", "base": 0.81, "rarity": "common", "phase": 0.8},
    {"id": "MS-1046", "base": 0.55, "rarity": "common", "phase": 1.7},
    {"id": "MS-1047", "base": 0.88, "rarity": "rare", "phase": 2.9},
]


def generate_streaming_samples(
    tick: int,
    *,
    auto_protect: bool = True,
    mode: str = "after",
) -> list[dict[str, Any]]:
    """Drifting scores so the feed feels live (like emitter.py + data.json polling)."""
    rng = random.Random(tick // 2)
    out = []
    for t in _TEMPLATES:
        drift = 0.09 * math.sin(tick * 0.35 + t["phase"])
        noise = rng.uniform(-0.04, 0.04)
        score = max(0.05, min(0.98, t["base"] + drift + noise))
        out.append(
            evaluate_sample(
                {"id": t["id"], "score": score, "rarity": t["rarity"]},
                auto_protect=auto_protect,
                mode=mode,
            )
        )
    return out
