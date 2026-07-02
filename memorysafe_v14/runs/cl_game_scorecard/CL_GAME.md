# MemorySafe CL Game — Scorecard (cl-game-v1)

**CL Index:** 72/100 — lanes won at α=0.05, 10 seeds

> Hypothesis: CL = frequency replay (Lane A) + fragility governance (Lane B).
> Product wedge owns rare medical lanes; research track owns general IL.

| Lane | Status | MemorySafe | Reservoir | Δ | Wins | p |
|------|--------|------------|-----------|---|------|---|
| Rare medical detection | **WON** | 0.7058 | 0.6629 | +0.0429 | 10/10 | 0.0165 |
| Rare pathology tissue | **WON** | 0.3892 | 0.3034 | +0.0858 | 7/10 | 0.0275 |
| Task-0 anti-forgetting | **WON** | 0.2925 | 0.0040 | +0.2885 | 6/10 | 0.0073 |
| General class-incremental | **BEHIND** | 0.1299 | 0.1588 | -0.0289 | 2/10 | 0.0446 |

## Next moves (CL Game v1)

1. **FragilityCLQuota** — `memorysafe_fragility` policy on PathMNIST class-IL (unifies rare + old-task).
2. **Re-test CIFAR** with FragilityCLQuota + longer epochs (general IL lane still OPEN).
3. **Lane pass = CL solved for that dimension** — full CL = 4/4 lanes WON.

## What we already proved

- Rare + medical: **2/2** binary medical lanes beat reservoir.
- Anti-forgetting: task-0 retention p≈0.007 on 9-class pathology stream.
- General IL: still OPEN — that's the long game, not the pilot SKU.
