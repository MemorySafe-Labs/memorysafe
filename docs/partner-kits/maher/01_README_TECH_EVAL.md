# MemorySafe — Technical Eval Package (Maher)

**Version:** v14.2 · July 2026  
**Your role:** Prospective Technical Partner (personal capacity)  
**Goal:** Test whether MemorySafe's replay governor is real, reproducible, and integrable.

---

## What MemorySafe is (and is not)

**Is:** A PyTorch replay-governance hook — `MemorySafeGovernor` / `GovernedBuffer` — that decides what stays in a bounded replay buffer under rare-class pressure. In-VPC only. No SaaS.

**Is not:** Universal continual-learning SOTA. Not a hosted platform. Not a claim that every benchmark wins (CIFAR lane is behind reservoir).

**Canonical win (10 seeds, CPU):** PneumoniaMNIST 5-task combined AUPRC **0.706 ± 0.051** vs reservoir **0.663 ± 0.066** (p=0.017).

---

## Eval path (recommended)

| Step | Time | What | Pass criteria |
|------|------|------|---------------|
| **1** | 2 min | Colab behavioral demo | Side-by-side reservoir vs MemorySafe on synthetic stream |
| **2** | 15–30 min | 1-seed pneumonia smoke | `run_eval_smoke.py` completes; JSON matches ballpark |
| **3** | 30 min | Read `pytorch_hook.py` | Hook pattern is clear for your CL loops |
| **4** | 2–4 h | 10-seed full repro | Matches published `RESULTS.md` within tolerance |
| **5** | call | Architecture review | You deliver written notes (see `04_TECH_REVIEW_BRIEF.md`) |

---

## Step 1 — Colab (behavioral)

**Link:** https://colab.research.google.com/github/MemorySafe-Labs/memorysafe/blob/main/benchmarks/Taste_demo_v2/demo.ipynb

**Important:** This demo shows *governance behavior* on a synthetic stream. It is **not** the v14.2 medical benchmark. Use it to understand MVI / buffer dynamics quickly.

---

## Step 2 — 1-seed smoke (local)

```bash
cd ~/Desktop/MemorySafe/memorysafe_v14
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_eval_smoke.py
```

Outputs: `runs/partner_eval_smoke/eval_report.json` + `RESULTS.md`

Expected ballpark (1 seed, not a statistical claim):
- MemorySafe combined AUPRC > reservoir on same seed
- Runtime: ~10–20 min CPU

---

## Step 3 — Integration sketch

Read: `memorysafe_v14/examples/pytorch_hook.py`

Core pattern:
```python
gov = MemorySafeGovernor.for_class_incremental(capacity=200, replay_prob=0.7)
gov.set_task(task_id)
bx, by, rep_idxs = gov.maybe_sample()
# ... train on concat(current_batch, replay_batch) ...
gov.observe(x, y, values, task_id=..., replay_idxs=..., replay_losses=...)
```

Full integration checklist: `03_INTEGRATION.md`

---

## Step 4 — Full 10-seed repro

```bash
python benchmark_pneumonia.py \
  --seeds 10 \
  --policies reservoir loss_priority memorysafe_v14 \
  --save-dir runs/pneumonia_10seed_sota
```

Compare output to `evidence/pneumonia_10seed_RESULTS.md` in this package.

---

## Files in this package

| File | Purpose |
|------|---------|
| `evidence/v14_SPEC.md` | Frozen protocol + hyperparams |
| `evidence/pneumonia_10seed_RESULTS.md` | Canonical 10-seed table |
| `evidence/pneumonia_10seed_report.json` | Raw JSON for audit |
| `evidence/CL_GAME.md` | 4-lane scorecard (3 won, 1 behind) |
| `03_INTEGRATION.md` | VPC / PyTorch integration checklist |
| `04_TECH_REVIEW_BRIEF.md` | What to assess + deliverable format |

---

## CAE reminder

Personal capacity only. Do not use CAE resources, accounts, or confidential information. CAE is not a Party to this eval.

Questions: carla@memorysafe.ca