# MemorySafe — Safe Claims Cheat Sheet

**Rule:** Only cite v14.2 metrics below. Retire legacy 0.941 AUPRC (different protocol, v13.5).

---

## Lead with (medical imaging)

### PneumoniaMNIST — 5-task class-IL (10 seeds, CPU)
- MemorySafe: **0.706 ± 0.051** combined AUPRC
- Reservoir: 0.663 ± 0.066
- **p = 0.017** (10/10 seed wins)
- Task-0 recall: **0.800 ± 0.118** vs reservoir 0.750

### PathMNIST rare-tissue (10 seeds) — pathology wedge
- MemorySafe: **0.389 ± 0.121** vs reservoir **0.303 ± 0.128**
- **p = 0.028** (7/10 wins)
- Use for **digital pathology** conversations

### PathMNIST task-0 retention (10 seeds)
- MemorySafe vs reservoir: **p = 0.007**
- Anti-forgetting story for early-task classes

---

## Supporting (not headline)

### Lite SKU (~80 buffer cap)
- **0.686 ± 0.056** AUPRC (~58% less replay vs 500-cap)
- Gap vs full: −2.0 pp (p=0.08 — not α-claimable)
- Use for "tight memory budget" buyers

---

## Do NOT claim

| Claim | Why |
|-------|-----|
| "0.941 AUPRC" | v13.5 2-task GPU Colab — different protocol |
| "Universal CL SOTA" | CIFAR-100 5-task: behind reservoir (p≈0.82) |
| "Production-validated" | Pre-revenue; public benchmarks only |
| "SaaS / hosted" | In-VPC PyTorch hook only |
| "Beats DER++ on CIFAR" | Outdated site copy — do not use in sales |

---

## Elevator proof stack

1. **Site:** https://memorysafe.ca/#validation
2. **Colab demo** (behavioral, 2 min): link in README
3. **JSON report:** available on request after NDA with prospect
4. **2-week eval:** falsifiable Δ metric on their stream

---

## One sentence per vertical

- **Pathology:** "Rare tissue classes stay in replay at equal buffer cap — PathMNIST rare-tissue p=0.028."
- **Radiology / pneumonia:** "Rare-positive retention under 5-task class-IL — PneumoniaMNIST p=0.017."
- **Enterprise ML:** "Governed replay buffer policy with audit logs — in-VPC, PyTorch hook."