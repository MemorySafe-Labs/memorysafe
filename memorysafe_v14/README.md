## Universal continual learning (any environment)

Drop-in governor for **any** PyTorch training loop — not only PneumoniaMNIST:

```bash
# First time on this machine (creates .venv + installs deps):
bash run_demo.sh

# Or manually:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run demo_streamlit.py

python examples/pytorch_hook.py
python benchmark_cifar100.py --seeds 3 --n-tasks 5
python benchmark_suite.py --suite all --seeds 3
```

| File | Role |
|------|------|
| `quota_policies.py` | `RareClassQuota` · `TaskBalancedQuota` · `UniformClassQuota` |
| `governed_buffer.py` | `GovernedBuffer` kernel + reservoir baseline |
| `governor.py` | `MemorySafeGovernor.for_rare_binary()` / `.for_class_incremental()` |
| `config_universal.py` | Split CIFAR-100 protocol |
| `train_loop_cifar.py` | Class-incremental CIFAR training |
| `benchmark_cifar100.py` | CIFAR benchmark (`--quota task\|uniform\|tail`) |
| `benchmark_suite.py` | Pneumonia + CIFAR suite runner |
| `tune_cifar100.py` | CIFAR hyperparameter sweep |
| `examples/pytorch_hook.py` | Universal training-hook demo |

**CIFAR-100 status:** plumbing validated; task-level quota (`TaskBalancedQuota`) replaces per-class floors that over-dilute the buffer. Run `tune_cifar100.py` before claiming wins on CIFAR.

## Layout (Pneumonia fast path)

| File | Role |
|------|------|
| `config_v14.py` | Frozen protocol `v14.2-pneumonia-5task-sota` |
| `buffer_v14.py` | `MemorySafeBufferV14` + baselines |
| `train_loop.py` | Training loop |
| `benchmark_pneumonia.py` | Multi-seed reproduction |
| `generate_sales_onepager.py` | Sales one-pager Word doc |
| `demo_streamlit.py` | Buyer-facing demo (wired to `MemorySafeBufferV14` via `demo_engine.py`) |
| `demo_engine.py` | Step-by-step governed CL session for live UI |
| `memorysafe_brain.py` | Governance research demo (not the integration SKU) |