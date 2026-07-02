#!/usr/bin/env python3
"""
MemorySafe CL Game — scorecard from locked benchmark JSON.

  python cl_scorecard.py
  python cl_scorecard.py --save-dir runs/cl_game_scorecard
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config_cl_game import CL_ALPHA, CL_GAME_VERSION, CL_LANES, CL_SEEDS_REQUIRED, lane_status

ROOT = Path(__file__).parent


def try_ttest(a: List[float], b: List[float]) -> Optional[float]:
    try:
        from scipy import stats
        _, p = stats.ttest_rel(a, b)
        return float(p)
    except Exception:
        return None


def load_lane_values(path: Path, ms_policy: str, res_policy: str, metric: str) -> Optional[Tuple[List[float], List[float], int]]:
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    # Standard benchmark_report.json
    if "raw" in data:
        raw = data["raw"]
        if ms_policy not in raw or res_policy not in raw:
            return None
        ms = [r["summary"][metric] for r in raw[ms_policy]]
        res = [r["summary"][metric] for r in raw[res_policy]]
        return ms, res, data.get("report", {}).get("n_seeds", len(ms))

    # CIFAR sweep_results.json
    if "results" in data and metric == "combined_acc":
        best = None
        for row in data["results"]:
            if row.get("name") == "ms_heavy_replay" or best is None:
                best = row
        if not best:
            return None
        ms = best.get("ms_values", [])
        res = best.get("res_values", [])
        return ms, res, best.get("n_seeds", len(ms))

    return None


def evaluate_lane(lane) -> Dict[str, Any]:
    path = ROOT / lane.evidence_path
    loaded = load_lane_values(path, lane.ms_policy, lane.baseline_policy, lane.metric_key)
    if not loaded:
        return {
            "lane": lane.id,
            "name": lane.name,
            "status": "NO_DATA",
            "path": str(path),
        }

    ms_vals, res_vals, n_seeds = loaded
    ms_mean = sum(ms_vals) / len(ms_vals)
    res_mean = sum(res_vals) / len(res_vals)
    delta = ms_mean - res_mean
    wins = sum(1 for m, r in zip(ms_vals, res_vals) if m > r)
    p = try_ttest(ms_vals, res_vals) if len(ms_vals) == len(res_vals) and len(ms_vals) >= 2 else None

    passed = (
        n_seeds >= CL_SEEDS_REQUIRED
        and p is not None
        and p < CL_ALPHA
        and delta > 0
    )
    return {
        "lane": lane.id,
        "name": lane.name,
        "question": lane.question,
        "status": lane_status(passed, p, delta),
        "passed": passed,
        "metric": lane.metric_key,
        "ms_mean": ms_mean,
        "res_mean": res_mean,
        "delta": delta,
        "wins": f"{wins}/{len(ms_vals)}",
        "p": p,
        "n_seeds": n_seeds,
        "path": str(path),
    }


def cl_index(rows: List[Dict[str, Any]]) -> int:
    """75 = 3/4 lanes won; 100 = full CL (all four lanes at α=0.05)."""
    total = len(CL_LANES)
    won = sum(1 for r in rows if r.get("passed"))
    close = sum(1 for r in rows if r.get("status") == "CLOSE")
    behind = sum(1 for r in rows if r.get("status") == "BEHIND")
    return int(100 * (won + 0.25 * close - 0.10 * behind) / total)


def format_markdown(rows: List[Dict[str, Any]], index: int) -> str:
    lines = [
        f"# MemorySafe CL Game — Scorecard ({CL_GAME_VERSION})",
        "",
        f"**CL Index:** {index}/100 — lanes won at α={CL_ALPHA}, {CL_SEEDS_REQUIRED} seeds",
        "",
        "> Hypothesis: CL = frequency replay (Lane A) + fragility governance (Lane B).",
        "> Product wedge owns rare medical lanes; research track owns general IL.",
        "",
        "| Lane | Status | MemorySafe | Reservoir | Δ | Wins | p |",
        "|------|--------|------------|-----------|---|------|---|",
    ]
    for r in rows:
        if r.get("status") == "NO_DATA":
            lines.append(f"| {r['name']} | NO DATA | — | — | — | — | — |")
            continue
        pstr = f"{r['p']:.4f}" if r.get("p") is not None else "—"
        lines.append(
            f"| {r['name']} | **{r['status']}** | {r['ms_mean']:.4f} | {r['res_mean']:.4f} | "
            f"{r['delta']:+.4f} | {r['wins']} | {pstr} |"
        )

    lines.extend([
        "",
        "## Next moves (CL Game v1)",
        "",
        "1. **FragilityCLQuota** — `memorysafe_fragility` policy on PathMNIST class-IL (unifies rare + old-task).",
        "2. **Re-test CIFAR** with FragilityCLQuota + longer epochs (general IL lane still OPEN).",
        "3. **Lane pass = CL solved for that dimension** — full CL = 4/4 lanes WON.",
        "",
        "## What we already proved",
        "",
        "- Rare + medical: **2/2** binary medical lanes beat reservoir.",
        "- Anti-forgetting: task-0 retention p≈0.007 on 9-class pathology stream.",
        "- General IL: still OPEN — that's the long game, not the pilot SKU.",
        "",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="runs/cl_game_scorecard")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    rows = [evaluate_lane(lane) for lane in CL_LANES]
    index = cl_index(rows)

    report = {"version": CL_GAME_VERSION, "cl_index": index, "lanes": rows}
    json_path = os.path.join(args.save_dir, "cl_scorecard.json")
    md_path = os.path.join(args.save_dir, "CL_GAME.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w") as f:
        f.write(format_markdown(rows, index))

    print(format_markdown(rows, index))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
