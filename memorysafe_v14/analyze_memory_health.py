#!/usr/bin/env python3
"""Link memory health metrics (MVI, FRI) to model outcomes from benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


OUTCOME_KEYS = {
    "PneumoniaMNIST": ("combined_auprc", "task0_retention_recall", "combined_recall_pos"),
    "PathMNIST": ("combined_acc", "task0_retention_acc", "tail_class_acc"),
    "CIFAR100": ("combined_acc", "task0_retention_acc", "mean_class_acc"),
}


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _final_task_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    return run["task_metrics"][-1]


def _task_trajectory(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for tm in run["task_metrics"]:
        rows.append(
            {
                "task": tm["task_completed"],
                "outcome": tm.get("combined_auprc", tm.get("combined_acc", 0.0)),
                "task0": tm.get("task0_recall", tm.get("task0_acc", 0.0)),
                "mvi": tm.get("mean_mvi", 0.0),
                "fri": tm.get("fri", 0.0),
                "replay_exp": tm.get("mean_replay_exposure", 0.0),
                "buf_pos": tm.get("pos_in_buffer"),
                "buf_size": tm.get("buffer_size", 0),
            }
        )
    return rows


def paired_seed_table(
    raw: Dict[str, List[Dict[str, Any]]],
    policy_a: str,
    policy_b: str,
    outcome_key: str,
) -> List[Dict[str, float]]:
    rows = []
    for ra, rb in zip(raw[policy_a], raw[policy_b]):
        sa = ra["summary"]
        sb = rb["summary"]
        rows.append(
            {
                "outcome_a": sa[outcome_key],
                "outcome_b": sb[outcome_key],
                "delta_outcome": sb[outcome_key] - sa[outcome_key],
                "mvi_a": sa.get("mean_mvi", 0.0),
                "mvi_b": sb.get("mean_mvi", 0.0),
                "delta_mvi": sb.get("mean_mvi", 0.0) - sa.get("mean_mvi", 0.0),
                "fri_a": sa.get("fri", 0.0),
                "fri_b": sb.get("fri", 0.0),
                "delta_fri": sb.get("fri", 0.0) - sa.get("fri", 0.0),
                "replay_b": sb.get("mean_replay_exposure", 0.0),
            }
        )
    return rows


def correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def format_report(data: Dict[str, Any], baseline: str, governed: str) -> str:
    report = data.get("report", data)
    raw = data["raw"]
    dataset = report.get("config", {}).get("dataset", "PneumoniaMNIST")
    keys = OUTCOME_KEYS.get(dataset, ("combined_auprc", "task0_retention_recall", "combined_recall_pos"))
    primary, task0_key, secondary = keys

    agg = report["aggregates"]
    lines = [
        "# Memory Health → Model Outcome Report",
        "",
        f"**Dataset:** {dataset}  ",
        f"**Compare:** `{baseline}` → `{governed}`  ",
        f"**Seeds:** {report.get('n_seeds', '?')}",
        "",
        "## How the metrics help (mechanism)",
        "",
        "- **MVI** does not train the model directly — it **governs which memories stay and get replayed**.",
        "- **FRI** measures whether that governance **distorted** the buffer away from the stream.",
        "- **Model lift** comes from **practicing the right memories** (replay exposure ↑ on fragile/rare slots).",
        "",
        "## Final summary — health vs outcomes",
        "",
        "| Policy | Primary outcome | Task-0 | Mean MVI | FRI | Replay exposure |",
        "|--------|-----------------|--------|----------|-----|-----------------|",
    ]

    for policy in (baseline, governed):
        if policy not in agg:
            continue
        a = agg[policy]
        lines.append(
            f"| {policy} | "
            f"{a[primary]['mean']:.4f} ± {a[primary]['std']:.4f} | "
            f"{a.get(task0_key, {'mean': 0})['mean']:.4f} | "
            f"{a.get('mean_mvi', {'mean': 0})['mean']:.4f} | "
            f"{a.get('fri', {'mean': 0})['mean']:.4f} | "
            f"{a.get('mean_replay_exposure', {'mean': 0})['mean']:.2f} |"
        )

    if baseline in raw and governed in raw:
        paired = paired_seed_table(raw, baseline, governed, primary)
        deltas = [r["delta_outcome"] for r in paired]
        wins = sum(1 for d in deltas if d > 0)
        lines.extend(
            [
                "",
                f"**Governed wins on {primary}:** {wins}/{len(paired)} seeds",
                "",
                "## Per-seed tradeoff (governed − reservoir)",
                "",
                "| Seed | Δ outcome | Δ MVI | Δ FRI | Replay exp | Interpretation |",
                "|------|-----------|-------|-------|------------|----------------|",
            ]
        )
        for i, row in enumerate(paired):
            if row["delta_outcome"] > 0 and row["delta_mvi"] > 0:
                note = "Fragility spend → outcome win"
            elif row["delta_outcome"] < 0 and row["delta_fri"] < -0.05:
                note = "FRI drop hurt common stream"
            elif row["delta_outcome"] > 0:
                note = "Outcome win"
            else:
                note = "No lift this seed"
            lines.append(
                f"| {i} | {row['delta_outcome']:+.4f} | {row['delta_mvi']:+.3f} | "
                f"{row['delta_fri']:+.4f} | {row['replay_b']:.1f} | {note} |"
            )

        corr_mvi = correlation([r["delta_mvi"] for r in paired], [r["delta_outcome"] for r in paired])
        corr_fri = correlation([r["delta_fri"] for r in paired], [r["delta_outcome"] for r in paired])
        corr_replay = correlation([r["replay_b"] for r in paired], [r["delta_outcome"] for r in paired])
        lines.extend(
            [
                "",
                "## Correlations (per seed)",
                "",
                f"- Δ MVI vs Δ {primary}: **r = {corr_mvi:.3f}**",
                f"- Δ FRI vs Δ {primary}: **r = {corr_fri:.3f}**",
                f"- Replay exposure vs Δ {primary}: **r = {corr_replay:.3f}**",
                "",
                "> Positive r on Δ MVI + replay ≈ governance is buying model lift by protecting fragile memories.  ",
                "> Negative r on Δ FRI ≈ intentional frequency sacrifice; bad only when outcome also drops.",
            ]
        )

        # Best example seed
        best_i = int(np.argmax(deltas))
        lines.extend(["", f"## Task-by-task trajectory (best seed #{best_i})", ""])
        lines.append(f"### {governed}")
        lines.append("| Task | Outcome | Task-0 | MVI | FRI | Replay exp | pos/buf |")
        lines.append("|------|---------|--------|-----|-----|------------|---------|")
        for row in _task_trajectory(raw[governed][best_i]):
            pos = row["buf_pos"]
            pos_s = f"{pos}/{row['buf_size']}" if pos is not None else "—"
            lines.append(
                f"| {row['task']} | {row['outcome']:.3f} | {row['task0']:.3f} | "
                f"{row['mvi']:.3f} | {row['fri']:.3f} | {row['replay_exp']:.1f} | {pos_s} |"
            )
        lines.append("")
        lines.append(f"### {baseline}")
        lines.append("| Task | Outcome | Task-0 | MVI | FRI | pos/buf |")
        lines.append("|------|---------|--------|-----|-----|---------|")
        for row in _task_trajectory(raw[baseline][best_i]):
            pos = row["buf_pos"]
            pos_s = f"{pos}/{row['buf_size']}" if pos is not None else "—"
            lines.append(
                f"| {row['task']} | {row['outcome']:.3f} | {row['task0']:.3f} | "
                f"{row['mvi']:.3f} | {row['fri']:.3f} | {pos_s} |"
            )

    lines.extend(
        [
            "",
            "## Read this for product",
            "",
            "1. **When MVI ↑ and FRI ↓ slightly but rare/task-0 metrics ↑** → governance is working (Pneumonia pattern).",
            "2. **When MVI ↑ and FRI ↓ hard and combined acc ↓** → over-governing fragility (CIFAR BEHIND pattern).",
            "3. **FRI tells you what to fix next** — hybrid buffer: reservoir core (frequency) + MemorySafe shell (fragility).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--baseline", default="reservoir")
    parser.add_argument("--governed", default="memorysafe_v14")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = _load(args.json)
    text = format_report(data, args.baseline, args.governed)
    out = args.out or args.json.parent / "HEALTH_ANALYSIS.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()