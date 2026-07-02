# MemorySafe v14 — Canonical Benchmark Results

**Generated:** 2026-06-17 15:28 UTC
**Protocol:** v14.2-pneumonia-5task-sota
**Seeds:** 10

## Protocol

- Dataset: PneumoniaMNIST
- Tasks: 5
- Buffer: 500
- Replay prob: 0.8
- Pos quota: 0.4
- Recall feedback (light AR): False

## Summary (combined AUPRC = primary metric)

| Policy | Combined AUPRC | Combined recall_pos | Task-0 recall | R@1%FPR | Buffer MB |
|--------|----------------|---------------------|---------------|---------|-----------|
| reservoir | 0.6629 ± 0.0656 | 0.7220 ± 0.0629 | 0.7500 ± 0.1118 | 0.1860 ± 0.0959 | 1.50 |
| loss_priority | 0.6778 ± 0.0621 | 0.7160 ± 0.0697 | 0.7400 ± 0.1200 | 0.1740 ± 0.0938 | 1.50 |
| memorysafe_v14 | 0.7058 ± 0.0513 | 0.7500 ± 0.0602 | 0.8000 ± 0.1183 | 0.1820 ± 0.0787 | 1.50 |

## MemorySafe v14 vs Reservoir

- Paired t-test (combined AUPRC): p = 0.016527398822756976

## Reproduce

```bash
cd ~/Desktop/memorysafe_v14
pip install -r requirements.txt
python benchmark_pneumonia.py --seeds 10 --save-dir runs/pneumonia_10seed
python export_results.py --json runs/pneumonia_10seed/benchmark_report.json
```

## Honest claim

Governed bounded replay under rare-class pressure. Not a claim of universal CL SOTA.
