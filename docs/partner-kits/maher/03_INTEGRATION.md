# MemorySafe Integration Checklist

Use this when evaluating fit on a real PyTorch continual-learning pipeline.

---

## Prerequisites

- [ ] PyTorch ≥ 2.0
- [ ] Python 3.10+
- [ ] Training loop exposes: current batch `(x, y)`, per-sample loss, optional softmax probs
- [ ] Agreed metric: combined AUPRC, rare-class recall, or task-0 retention
- [ ] Buffer cap agreed upfront (e.g. 500 samples ≈ 1.5 MB for small MNIST-scale tensors)

---

## Hook integration (minimum viable)

1. **Instantiate governor** at task/stream start:
   - Rare binary / medical: `MemorySafeGovernor.for_rare_binary(capacity=..., replay_prob=0.80)`
   - Class-incremental: `MemorySafeGovernor.for_class_incremental(capacity=..., replay_prob=0.80)`

2. **Each training step:**
   - `bx, by, rep_idxs = gov.maybe_sample()`
   - If replay batch exists: concat with current batch, compute joint loss
   - `gov.observe(x, y, value_scores, task_id=..., replay_idxs=..., replay_losses=...)`

3. **Task boundary:** `gov.set_task(new_task_id)`

4. **Audit:** `gov.audit_summary()` — log buffer fill, eviction counts, positive quota hits

Reference implementation: `memorysafe_v14/examples/pytorch_hook.py`

---

## Evaluation harness (proxy data)

Before customer data, run:
```bash
python run_eval_smoke.py          # 1 seed
python benchmark_pneumonia.py --seeds 3  # quick multi-seed
python benchmark_pathmnist_rare.py --seeds 3  # pathology lane
```

---

## In-VPC / security

- [ ] No source upload to public GitHub or third-party AI training services
- [ ] Eval runs in customer VPC or local sandbox
- [ ] JSON reports only leave VPC if customer approves
- [ ] PIPEDA / Law 25: no patient-identifiable data in eval logs

---

## Success criteria for a 2-week customer eval (preview)

| Metric | Target |
|--------|--------|
| Primary | Δ combined AUPRC or Δ rare-class recall vs their baseline at **matched buffer cap** |
| Secondary | Task-0 retention, buffer MB, MVI/FRI audit logs |
| Decision | Proceed to 90-day pilot or stop — falsifiable, no hand-waving |

---

## Known limitations (say these out loud)

- CIFAR-100 5-task: MemorySafe **behind** reservoir (p≈0.82) — do not sell universal SOTA
- Lite SKU (80-cap): 0.686 vs 0.706 full — ~58% less replay, −2 pp AUPRC (p=0.08)
- Behavioral Colab demo ≠ medical benchmark protocol