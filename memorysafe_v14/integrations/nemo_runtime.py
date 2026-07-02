"""
Runtime policy layer — NeMo Guardrails Colang config with deterministic fallback.

Uses the same pathology_guardrails.co rules offline (no API). When NVIDIA_API_KEY is
set and nemoguardrails is installed, can run the full NeMo stack (optional).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

COLANG_PATH = Path(__file__).resolve().parent / "pathology_guardrails.co"

_MEDICAL = re.compile(
    r"pneumonia|x-?ray|pathology|pathmnist|rare\s+(class|tissue)|continual\s+learning|"
    r"memory\s*safe|replay\s+buffer|medical\s+imaging",
    re.I,
)
_UNSAFE = re.compile(
    r"ignore\s+all\s+rules|steal|hack|jailbreak|bypass|harmful|illegal",
    re.I,
)


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    NEUTRAL = "neutral"


@dataclass
class GuardrailDecision:
    action: GuardrailAction
    message: str
    engine: str  # "colang_rules" | "nemo_guardrails"


def _rules_engine(prompt: str) -> GuardrailDecision:
    text = prompt.strip()
    if _UNSAFE.search(text):
        return GuardrailDecision(
            action=GuardrailAction.BLOCK,
            message="Blocked: request outside approved pathology agent scope.",
            engine="colang_rules",
        )
    if _MEDICAL.search(text):
        return GuardrailDecision(
            action=GuardrailAction.ALLOW,
            message="Approved: pathology / continual-learning scope. Forwarding to MemorySafe governor.",
            engine="colang_rules",
        )
    return GuardrailDecision(
        action=GuardrailAction.NEUTRAL,
        message="Neutral: no pathology keywords — MemorySafe may still process generic batches.",
        engine="colang_rules",
    )


def _nemo_engine(prompt: str) -> Optional[GuardrailDecision]:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import nest_asyncio

        nest_asyncio.apply()
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from nemoguardrails import LLMRails, RailsConfig
    except ImportError:
        return None

    colang = COLANG_PATH.read_text(encoding="utf-8")
    config = RailsConfig.from_content(colang_content=colang)
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0.2)
    rails = LLMRails(config, llm=llm)
    out = rails.generate(prompt)
    text = str(out).lower()
    if "blocked" in text or "can't assist" in text or "cannot provide" in text:
        return GuardrailDecision(
            action=GuardrailAction.BLOCK,
            message=str(out),
            engine="nemo_guardrails",
        )
    if "approved" in text or "forwarding" in text or "safe medical" in text:
        return GuardrailDecision(
            action=GuardrailAction.ALLOW,
            message=str(out),
            engine="nemo_guardrails",
        )
    return GuardrailDecision(
        action=GuardrailAction.NEUTRAL,
        message=str(out),
        engine="nemo_guardrails",
    )


def check_prompt(prompt: str, *, prefer_nemo: bool = False) -> GuardrailDecision:
    """Evaluate agent prompt against pathology runtime policy."""
    if prefer_nemo:
        nemo = _nemo_engine(prompt)
        if nemo is not None:
            return nemo
    return _rules_engine(prompt)
