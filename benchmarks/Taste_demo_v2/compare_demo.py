#!/usr/bin/env python3
"""Side-by-side Reservoir vs MemorySafe governance demo plot for Colab."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BRAND = {
    "bg": "#020408",
    "cyan": "#00F0FF",
    "blue": "#2F7BFF",
    "danger": "#FF2A6D",
    "gray": "#94A3B8",
    "dim": "#475569",
}

DEMO_DIR = Path(__file__).resolve().parent
SIM = DEMO_DIR / "simulate_stream.py"
RUNS = DEMO_DIR / "_runs"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_policy(policy: str, steps: int, capacity: int, seed: int, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SIM),
        "--policy",
        policy,
        "--steps",
        str(steps),
        "--capacity",
        str(capacity),
        "--seed",
        str(seed),
        "--log_every",
        "50",
        "--outdir",
        str(outdir),
    ]
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=DEMO_DIR)
    latest = outdir / "latest.jsonl"
    if not latest.exists():
        raise FileNotFoundError(f"Missing log: {latest}")
    return latest


def series(rows: List[Dict[str, Any]], *keys: str) -> np.ndarray:
    out = rows
    for key in keys:
        if isinstance(out, list) and out and isinstance(out[0], dict):
            out = [r[key] for r in out]
        else:
            raise KeyError(keys)
    return np.asarray(out, dtype=float)


def style_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_facecolor(BRAND["bg"])
    ax.set_title(title, color="white", fontsize=12, pad=10, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel, color=BRAND["gray"])
    ax.set_xlabel("Training steps", color=BRAND["dim"])
    ax.tick_params(colors=BRAND["dim"])
    for spine in ax.spines.values():
        spine.set_color("#1e293b")


def final_recall(rows: List[Dict[str, Any]]) -> float:
    return float(rows[-1]["metrics"]["recall_pos"]) if rows else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--capacity", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

    plt.rcParams.update(
        {
            "figure.facecolor": BRAND["bg"],
            "axes.edgecolor": "#1e293b",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )

    logs: Dict[str, Path] = {}
    for policy in ("reservoir", "memorysafe"):
        run_dir = RUNS / f"{policy}_steps{args.steps}_cap{args.capacity}_seed{args.seed}"
        logs[policy] = run_policy(policy, args.steps, args.capacity, args.seed, run_dir)

    res_rows = load_jsonl(logs["reservoir"])
    ms_rows = load_jsonl(logs["memorysafe"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    fig.suptitle(
        "MemorySafe Governance Demo — Rare-class retention under buffer pressure",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0, 0]
    style_axes(ax, "Rare positive recall", "Recall")
    ax.plot(series(res_rows, "step"), series(res_rows, "metrics", "recall_pos"), color=BRAND["gray"], lw=2, label="Reservoir")
    ax.plot(series(ms_rows, "step"), series(ms_rows, "metrics", "recall_pos"), color=BRAND["cyan"], lw=2.5, label="MemorySafe")
    ax.legend(facecolor=BRAND["bg"], edgecolor="#1e293b", labelcolor="white")
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    style_axes(ax, "Rare cases kept in buffer", "Positive fraction")
    ax.plot(series(res_rows, "step"), series(res_rows, "buffer", "pos_frac"), color=BRAND["gray"], lw=2, label="Reservoir")
    ax.plot(series(ms_rows, "step"), series(ms_rows, "buffer", "pos_frac"), color=BRAND["blue"], lw=2.5, label="MemorySafe")
    ax.legend(facecolor=BRAND["bg"], edgecolor="#1e293b", labelcolor="white")

    ax = axes[1, 0]
    style_axes(ax, "Governance signals (MemorySafe)", "Score / count")
    ax.plot(series(ms_rows, "step"), series(ms_rows, "buffer", "mvi_mean"), color=BRAND["cyan"], lw=2, label="MVI mean")
    ax.plot(series(ms_rows, "step"), series(ms_rows, "buffer", "protected"), color=BRAND["danger"], lw=2, label="Protected slots")
    ax.legend(facecolor=BRAND["bg"], edgecolor="#1e293b", labelcolor="white")

    ax = axes[1, 1]
    ax.axis("off")
    res_final = final_recall(res_rows)
    ms_final = final_recall(ms_rows)
    delta = ms_final - res_final
    summary = (
        f"Toy continual-learning stream · {args.steps} steps · buffer {args.capacity}\n\n"
        f"Final rare-class recall\n"
        f"  Reservoir:   {res_final:.3f}\n"
        f"  MemorySafe:  {ms_final:.3f}  ({delta:+.3f})\n\n"
        f"MemorySafe governs protect / replay / forget using MVI.\n"
        f"This is a behavioral demo — not the v14.2 medical benchmark.\n\n"
        f"Canonical results: memorysafe.ca/#validation"
    )
    ax.text(
        0.02,
        0.95,
        summary,
        va="top",
        ha="left",
        color=BRAND["gray"],
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#0A0E17", edgecolor="#1e293b"),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.save) if args.save else DEMO_DIR / "governance_comparison.png"
    fig.savefig(out, facecolor=BRAND["bg"], bbox_inches="tight")
    print(f"\nSaved figure → {out}")
    if os.environ.get("DISPLAY") or os.environ.get("COLAB_RELEASE_TAG"):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
