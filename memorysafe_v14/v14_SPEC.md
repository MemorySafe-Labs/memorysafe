# MemorySafe v14 — Canonical Specification

**Protocol version:** `v14.2-pneumonia-5task-sota`  
**Status:** Frozen for reproduction (June 2026)

## What this version is

A **bounded replay buffer policy** for class-incremental learning under rare-positive pressure.

**Includes:** MVI (risk EMA) + ProtectScore + hard positive quota + stratified replay + task-balanced sampling.

**Excludes:** CoreShield, full v13.5 MemorySafe-AR eval loop, MemorySafeBrain (separate research path).

## Canonical code path

```
config_v14.py          → frozen hyperparameters
buffer_v14.py          → MemorySafeBufferV14
train_loop.py          → training + evaluation
benchmark_pneumonia.py → multi-seed reproduction
export_results.py      → RESULTS.md from JSON
```

**Not canonical for benchmarks:** `memorysafe_brain.py`, `continual_train.py` (CIFAR demo / governance narrative).

## Tuned protocol (v14.2 — Jun 2026 sweep)

Hyperparameters selected by 3-seed sweep, validated on 10 seeds:

| Parameter | v14.1 | v14.2 (SOTA) |
|-----------|-------|--------------|
| replay_prob | 0.65 | **0.80** |
| pos_quota_frac | 0.30 | **0.40** |
| replay_pos_frac | 0.35 | **0.45** |
| epochs_per_task | 3 | 3 |
| replay_scale | 1.25 | 1.25 |

## 10-seed results (Jun 2026)

| Policy | Combined AUPRC |
|--------|----------------|
| Reservoir | 0.6629 ± 0.0656 |
| Loss-priority (GSS) | 0.6778 ± 0.0621 |
| **MemorySafe v14** | **0.7058 ± 0.0513** |

- vs Reservoir: **p = 0.0165** (10/10 seed wins)
- vs Loss-priority: **p = 0.0015** (9/10 seed wins)

Raw: `runs/pneumonia_10seed_sota/benchmark_report.json`

## MVI (in v14)

- On buffer insert: initial `risk` from per-sample loss + positive boost.
- After replay: `risk ← EMA(old_risk, replay_loss)` with `mvi_ema=0.7`.

## ProtectScore (in v14)

```
protect = w_risk*risk + w_value*value + w_criticality*criticality
        + w_rarity*rarity + task_age_weight*age_factor
```

Used for eviction (lowest score out) and replay sampling (weighted).

## Light feedback (optional — off by default)

`--recall-feedback`: after each task, if combined recall_pos < target (0.72), increase replay probability for the next task.

**A/B result (10 seeds, v14.1):** recall feedback did not help — rejected. See `COMPARISON.md`.

## Reproduce

```bash
cd ~/Desktop/memorysafe_v14
pip install -r requirements.txt
python benchmark_pneumonia.py --seeds 10 --policies reservoir loss_priority memorysafe_v14 --save-dir runs/pneumonia_10seed_sota
python export_results.py --json runs/pneumonia_10seed_sota/benchmark_report.json
```

## Claim (this benchmark)

SOTA on PneumoniaMNIST 5-task rare-class continual learning (combined AUPRC) vs reservoir and loss-priority replay at matched 500-sample buffer (10 seeds, p < 0.05). Not universal CL SOTA (e.g. DER++ on CIFAR-100 accuracy).