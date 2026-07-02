# MemorySafe Labs

**Replay governance for continual-learning PyTorch pipelines** — in-VPC hook, not SaaS.

MemorySafe governs what stays in a bounded replay buffer under rare-class pressure. It does not replace your training algorithm; it disciplines retention when memory is full.

**Site:** https://memorysafe.ca  
**Contact:** carla@memorysafe.ca

---

## Try in 2 minutes (behavioral demo)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MemorySafe-Labs/memorysafe/blob/main/benchmarks/Taste_demo_v2/demo.ipynb)

Synthetic stream — shows governance dynamics. **Not** the v14.2 medical benchmark (see `memorysafe_v14/`).

---

## v14.2 validation (honest claims)

Protocol: `v14.2-pneumonia-5task-sota` · 10 seeds · CPU · 500-sample buffer

| Benchmark | MemorySafe | Baseline | p-value |
|-----------|------------|----------|---------|
| PneumoniaMNIST 5-task (combined AUPRC) | **0.706 ± 0.051** | Reservoir 0.663 ± 0.066 | **0.017** |
| PathMNIST rare-tissue (combined AUPRC) | **0.389 ± 0.121** | Reservoir 0.303 ± 0.128 | **0.028** |
| PathMNIST task-0 retention | wins vs reservoir | — | **0.007** |

**Lite SKU (80-cap):** 0.686 ± 0.056 (~58% less replay; −2 pp vs full, p=0.08 — not α-claimable).

**Do not claim:** universal CL SOTA (CIFAR-100 5-task is behind reservoir, p≈0.82).

Full spec: [`memorysafe_v14/v14_SPEC.md`](memorysafe_v14/v14_SPEC.md)

---

## Quick start — reproduce canonical benchmark

```bash
cd memorysafe_v14
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Partner smoke (1 seed, ~15 min CPU)
python run_eval_smoke.py

# Full 10-seed reproduction
python benchmark_pneumonia.py --seeds 10 --policies reservoir loss_priority memorysafe_v14 \
  --save-dir runs/pneumonia_10seed_sota
```

Pre-computed results: [`memorysafe_v14/runs/pneumonia_10seed_sota/RESULTS.md`](memorysafe_v14/runs/pneumonia_10seed_sota/RESULTS.md)

---

## PyTorch integration (universal hook)

```python
from governor import MemorySafeGovernor

gov = MemorySafeGovernor.for_class_incremental(capacity=200, replay_prob=0.7)
gov.set_task(task_id)
bx, by, rep_idxs = gov.maybe_sample()
# concat replay batch with current batch, train, then:
gov.observe(x, y, value_scores, task_id=task_id, replay_idxs=rep_idxs, replay_losses=...)
```

Full example: [`memorysafe_v14/examples/pytorch_hook.py`](memorysafe_v14/examples/pytorch_hook.py)

---

## Repository layout

| Path | Description |
|------|-------------|
| `memorysafe_v14/` | v14.2 engine — governor, buffers, medical benchmarks |
| `benchmarks/Taste_demo_v2/` | Colab behavioral demo |
| `docs/partner-kits/` | Post-NDA partner guides (technical eval + commercial GTM) |

---

## What MemorySafe is (and is not)

**Is:**
- PyTorch replay-governance hook (`MemorySafeGovernor` / `GovernedBuffer`)
- MVI + ProtectScore + quota policies
- In-VPC evaluation kit

**Is not:**
- Hosted SaaS platform
- Universal continual-learning SOTA on every dataset
- A replacement for your CL algorithm

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
