#!/usr/bin/env python3
"""Assemble post-NDA partner zip packages for Maher (tech) and Grégoire (GTM)."""

from __future__ import annotations

import shutil
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MS_ROOT = ROOT.parents[1]
V14 = MS_ROOT / "memorysafe_v14"
SALES = ROOT.parent / "sales"
OUT = ROOT / "output"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  skip (missing): {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  + {dst.relative_to(ROOT)}")
    return True


def build_maher(staging: Path) -> None:
    kit = staging / "MemorySafe_PartnerKit_Maher"
    kit.mkdir(parents=True, exist_ok=True)

    for f in sorted((ROOT / "maher").glob("*")):
        if f.is_file():
            shutil.copy2(f, kit / f.name)

    evidence = kit / "evidence"
    evidence.mkdir(exist_ok=True)
    pairs = [
        (V14 / "v14_SPEC.md", evidence / "v14_SPEC.md"),
        (V14 / "runs/pneumonia_10seed_sota/RESULTS.md", evidence / "pneumonia_10seed_RESULTS.md"),
        (V14 / "runs/pneumonia_10seed_sota/benchmark_report.json", evidence / "pneumonia_10seed_report.json"),
        (V14 / "runs/cl_game_scorecard/CL_GAME.md", evidence / "CL_GAME.md"),
        (V14 / "runs/pathmnist_rare_10seed_sota/RESULTS.md", evidence / "pathmnist_rare_RESULTS.md"),
    ]
    for src, dst in pairs:
        copy_if_exists(src, dst)

    code_ref = kit / "code_reference"
    code_ref.mkdir(exist_ok=True)
    for rel in [
        "examples/pytorch_hook.py",
        "governor.py",
        "run_eval_smoke.py",
        "README.md",
        "requirements.txt",
    ]:
        copy_if_exists(V14 / rel, code_ref / Path(rel).name)


def build_gregoire(staging: Path) -> None:
    kit = staging / "MemorySafe_PartnerKit_Gregoire"
    kit.mkdir(parents=True, exist_ok=True)

    for f in sorted((ROOT / "gregoire").glob("*")):
        if f.is_file():
            shutil.copy2(f, kit / f.name)

    assets = kit / "assets"
    assets.mkdir(exist_ok=True)
    copy_if_exists(SALES / "MemorySafe_Sales_OnePager.docx", assets / "MemorySafe_Sales_OnePager.docx")
    copy_if_exists(SALES / "PILOT_OUTREACH_READY.txt", assets / "PILOT_OUTREACH_READY.txt")
    copy_if_exists(
        MS_ROOT / "MemorySafe-Docs/pitch-decks/MemorySafe Labs - Investor Pitch Deck NVIDIA V2.pdf",
        assets / "MemorySafe_Pitch_Deck_NVIDIA_V2.pdf",
    )


def zip_dir(folder: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(folder.parent))
    print(f"Wrote {zip_path} ({zip_path.stat().st_size // 1024} KB)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    staging = OUT / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    print("Building Maher technical kit...")
    build_maher(staging)
    print("Building Grégoire commercial kit...")
    build_gregoire(staging)

    today = date.today().isoformat()
    zip_dir(
        staging / "MemorySafe_PartnerKit_Maher",
        OUT / f"MemorySafe_PartnerKit_Maher_{today}.zip",
    )
    zip_dir(
        staging / "MemorySafe_PartnerKit_Gregoire",
        OUT / f"MemorySafe_PartnerKit_Gregoire_{today}.zip",
    )

    shutil.rmtree(staging)
    print(f"\nDone. Packages in {OUT}")


if __name__ == "__main__":
    main()