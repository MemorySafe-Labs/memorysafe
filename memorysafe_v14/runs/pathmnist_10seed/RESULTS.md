# MemorySafe v14 — PathMNIST Benchmark Results

**Protocol:** v14.2-pathmnist-5task-classil
**Seeds:** 10

## Summary (combined accuracy = primary metric)

| Policy | Combined acc | Mean class acc | Tail class acc | Task-0 acc | Buffer MB |
|--------|--------------|----------------|----------------|------------|-----------|
| reservoir | 0.1411 ± 0.0668 | 0.1053 ± 0.0353 | 0.2952 ± 0.2374 | 0.0040 ± 0.0120 | 4.49 |
| loss_priority | 0.1120 ± 0.0298 | 0.1108 ± 0.0009 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 4.49 |
| memorysafe_governed | 0.1079 ± 0.0428 | 0.1143 ± 0.0095 | 0.0500 ± 0.1500 | 0.2925 ± 0.2517 | 4.49 |

**MemorySafe vs Reservoir paired t-test p=0.16654199959784866**
