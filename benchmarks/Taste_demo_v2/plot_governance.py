# plot_governance.py
from __future__ import annotations

import argparse
import json
from typing import Dict, Any, List

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=str, help="Path to runs/*.jsonl (e.g., runs/latest.jsonl)")
    args = ap.parse_args()

    rows = load_jsonl(args.logfile)
    if not rows:
        raise SystemExit("No rows found in log.")

    steps = [r["step"] for r in rows]
    recall = [r["metrics"]["recall_pos"] for r in rows]
    f1 = [r["metrics"]["f1_pos"] for r in rows]

    mvi_mean = [r["buffer"]["mvi_mean"] for r in rows]
    mvi_p90 = [r["buffer"]["mvi_p90"] for r in rows]
    prot = [r["buffer"]["protected"] for r in rows]
    buf_size = [r["buffer"]["buffer_size"] for r in rows]

    forgot = [r["actions"].get("forgotten", 0) for r in rows]
    repl = [r["actions"].get("replaced", 0) for r in rows]
    pos_frac = [r["buffer"].get("pos_frac", 0.0) for r in rows]

    policy = rows[-1].get("policy", "unknown")

    # 1) Performance
    plt.figure()
    plt.plot(steps, recall)
    plt.title(f"Rare-class Recall over time ({policy})")
    plt.xlabel("Step")
    plt.ylabel("Recall (pos)")
    plt.show()

    plt.figure()
    plt.plot(steps, f1)
    plt.title(f"F1 (positive class) over time ({policy})")
    plt.xlabel("Step")
    plt.ylabel("F1 (pos)")
    plt.show()

    # 2) MVI curves
    plt.figure()
    plt.plot(steps, mvi_mean)
    plt.title(f"MVI mean over time ({policy})")
    plt.xlabel("Step")
    plt.ylabel("MVI mean")
    plt.show()

    plt.figure()
    plt.plot(steps, mvi_p90)
    plt.title(f"MVI p90 over time ({policy})")
    plt.xlabel("Step")
    plt.ylabel("MVI p90")
    plt.show()

    # 3) Governance actions
    plt.figure()
    plt.plot(steps, forgot)
    plt.title(f"Forgetting events per log step ({policy})")
    plt.xlabel("Step")
    plt.ylabel("Forgotten count (delta)")
    plt.show()

    plt.figure()
    plt.plot(steps, repl)
    plt.title(f"Replacement events per log step ({policy})")
    plt.xlabel("Step")
    plt.ylabel("Replaced count (delta)")
    plt.show()

    # 4) Buffer composition and protection
    plt.figure()
    plt.plot(steps, prot)
    plt.title(f"Protected items over time ({policy})")
    plt.xlabel("Step")
    plt.ylabel("Protected count")
    plt.show()

    plt.figure()
    plt.plot(steps, pos_frac)
    plt.title(f"Positive fraction in buffer over time ({policy})")
    plt.xlabel("Step")
    plt.ylabel("pos_frac")
    plt.show()

    plt.figure()
    plt.plot(steps, buf_size)
    plt.title(f"Buffer size over time ({policy})")
    plt.xlabel("Step")
    plt.ylabel("Buffer size")
    plt.show()


if __name__ == "__main__":
    main()
