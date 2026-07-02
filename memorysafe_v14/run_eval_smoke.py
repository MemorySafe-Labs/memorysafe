#!/usr/bin/env python3
"""Partner smoke eval — 1-seed PneumoniaMNIST repro + eval_report.json."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "runs" / "partner_eval_smoke"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "benchmark_pneumonia.py"),
        "--seeds",
        "1",
        "--policies",
        "reservoir",
        "memorysafe_v14",
        "--save-dir",
        str(OUT_DIR),
    ]
    print("Running partner smoke eval (1 seed)...")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    src = OUT_DIR / "benchmark_report.json"
    if not src.exists():
        print(f"ERROR: missing {src}", file=sys.stderr)
        return 1

    with open(src) as f:
        payload = json.load(f)

    report = payload.get("report", {})
    ms = report.get("aggregates", {}).get("memorysafe_v14", {})
    res = report.get("aggregates", {}).get("reservoir", {})
    vs = report.get("memorysafe_vs_reservoir", {})

    eval_report = {
        "eval_type": "partner_smoke",
        "protocol": report.get("config", {}).get("protocol_version", "v14.2-pneumonia-5task-sota"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_seeds": 1,
        "memorysafe_v14": {
            "combined_auprc_mean": ms.get("combined_auprc", {}).get("mean"),
            "combined_recall_pos_mean": ms.get("combined_recall_pos", {}).get("mean"),
        },
        "reservoir": {
            "combined_auprc_mean": res.get("combined_auprc", {}).get("mean"),
            "combined_recall_pos_mean": res.get("combined_recall_pos", {}).get("mean"),
        },
        "memorysafe_vs_reservoir_p": vs.get("auprc", {}).get("p"),
        "note": "1-seed smoke only — not a statistical claim. Run --seeds 10 for canonical repro.",
        "full_report_path": str(src),
    }

    out_json = OUT_DIR / "eval_report.json"
    with open(out_json, "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nWrote {out_json}")
    print(json.dumps(eval_report, indent=2))

    results_md = OUT_DIR / "RESULTS.md"
    if results_md.exists():
        print(f"Wrote {results_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
