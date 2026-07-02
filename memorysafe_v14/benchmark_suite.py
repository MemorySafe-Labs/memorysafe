#!/usr/bin/env python3
"""
MemorySafe universal benchmark suite.

  python benchmark_suite.py --suite all --seeds 3
  python benchmark_suite.py --suite pneumonia --seeds 10
  python benchmark_suite.py --suite pathmnist --seeds 10
  python benchmark_suite.py --suite cifar100 --seeds 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["pneumonia", "pathmnist", "cifar100", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=42)
    args = parser.parse_args()

    suites = []
    if args.suite in ("pneumonia", "all"):
        suites.append([
            sys.executable, "benchmark_pneumonia.py",
            "--seeds", str(args.seeds),
            "--start-seed", str(args.start_seed),
            "--policies", "reservoir", "loss_priority", "memorysafe_v14",
            "--save-dir", f"runs/suite_pneumonia_{args.seeds}seed",
        ])
    if args.suite in ("pathmnist", "all"):
        suites.append([
            sys.executable, "benchmark_pathmnist.py",
            "--seeds", str(args.seeds),
            "--start-seed", str(args.start_seed),
            "--save-dir", f"runs/suite_pathmnist_{args.seeds}seed",
        ])
    if args.suite in ("cifar100", "all"):
        suites.append([
            sys.executable, "benchmark_cifar100.py",
            "--seeds", str(args.seeds),
            "--start-seed", str(args.start_seed),
            "--save-dir", f"runs/suite_cifar100_{args.seeds}seed",
        ])

    rc = 0
    for cmd in suites:
        rc = run(cmd) or rc
    sys.exit(rc)


if __name__ == "__main__":
    main()
