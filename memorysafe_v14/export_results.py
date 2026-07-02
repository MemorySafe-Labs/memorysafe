#!/usr/bin/env python3
"""Write RESULTS.md from benchmark_report.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def format_report(data: dict) -> str:
    report = data.get("report", data)
    cfg = report.get("config", {})
    agg = report.get("aggregates", {})
    lines = [
        "# MemorySafe v14 — Canonical Benchmark Results",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Protocol:** {cfg.get('protocol_version', 'unknown')}",
        f"**Seeds:** {report.get('n_seeds', '?')}",
        "",
        "## Protocol",
        "",
        f"- Dataset: {cfg.get('dataset', 'PneumoniaMNIST')}",
        f"- Tasks: {cfg.get('n_tasks', 5)}",
        f"- Buffer: {cfg.get('buffer_capacity', 500)}",
        f"- Replay prob: {cfg.get('replay_prob', 0.65)}",
        f"- Pos quota: {cfg.get('pos_quota_frac', 0.30)}",
        f"- Recall feedback (light AR): {cfg.get('recall_feedback', False)}",
        "",
        "## Summary (combined AUPRC = primary metric)",
        "",
        "| Policy | Combined AUPRC | Combined recall_pos | Task-0 recall | R@1%FPR | Buffer MB |",
        "|--------|----------------|---------------------|---------------|---------|-----------|",
    ]
    has_health = any("mean_mvi" in stats for stats in agg.values())
    for policy, stats in agg.items():
        a = stats["combined_auprc"]
        r = stats["combined_recall_pos"]
        t0 = stats["task0_retention_recall"]
        fpr = stats["combined_recall_at_1pct_fpr"]
        mem = stats.get("buffer_memory_mb", {}).get("mean", 0)
        lines.append(
            f"| {policy} | {a['mean']:.4f} ± {a['std']:.4f} | "
            f"{r['mean']:.4f} ± {r['std']:.4f} | "
            f"{t0['mean']:.4f} ± {t0['std']:.4f} | "
            f"{fpr['mean']:.4f} ± {fpr['std']:.4f} | {mem:.2f} |"
        )

    if has_health:
        lines.extend([
            "",
            "## Memory health (fragility + frequency)",
            "",
            "| Policy | Mean MVI | FRI | Coverage | Replay exposure |",
            "|--------|----------|-----|----------|-----------------|",
        ])
        for policy, stats in agg.items():
            if "mean_mvi" not in stats:
                continue
            mvi = stats["mean_mvi"]
            fri = stats["fri"]
            cov = stats["coverage_index"]
            exp = stats.get("mean_replay_exposure", {"mean": 0.0})
            lines.append(
                f"| {policy} | {mvi['mean']:.4f} ± {mvi['std']:.4f} | "
                f"{fri['mean']:.4f} ± {fri['std']:.4f} | "
                f"{cov['mean']:.4f} ± {cov['std']:.4f} | "
                f"{exp['mean']:.2f} ± {exp['std']:.2f} |"
            )

    vs = report.get("memorysafe_vs_reservoir")
    if vs:
        p = vs.get("auprc", {}).get("p")
        lines.extend([
            "",
            "## MemorySafe v14 vs Reservoir",
            "",
            f"- Paired t-test (combined AUPRC): p = {p}",
            "",
        ])

    lines.extend([
        "## Reproduce",
        "",
        "```bash",
        "cd ~/Desktop/memorysafe_v14",
        "pip install -r requirements.txt",
        "python benchmark_pneumonia.py --seeds 10 --save-dir runs/pneumonia_10seed",
        "python export_results.py --json runs/pneumonia_10seed/benchmark_report.json",
        "```",
        "",
        "## Honest claim",
        "",
        "Governed bounded replay under rare-class pressure. Not a claim of universal CL SOTA.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("RESULTS.md"))
    args = parser.parse_args()
    data = json.loads(args.json.read_text())
    args.out.write_text(format_report(data), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
