# MemorySafe Decision Layer: MVI + ProtectScore  
**Predictive Memory Governance for Continual Learning**

**MemorySafe Labs**  
Carla Centeno, Founder  
February 2026  
https://memorysafe.ca | NVIDIA Inception Program Member

## License & IP Notice
This document provides a high-level conceptual overview of the MemorySafe Decision Layer, comprising the **Memory Vulnerability Index (MVI)** and the **ProtectScore Policy Engine**.  
All rights reserved. This is proprietary technology developed by MemorySafe Labs. No part of this method may be reproduced, adapted, implemented, or used commercially without explicit written permission. For collaboration, licensing, or inquiries: carla@memorysafe.ca.

## 1. Overview – From Passive Replay to Predictive Governance
In continual learning, replay buffers are essential to combat catastrophic forgetting — but standard approaches (reservoir sampling, FIFO, uncertainty-based selection) are largely passive or heuristic. They do not anticipate which experiences are most at risk of being forgotten, nor do they make intelligent trade-offs when memory is constrained.

MemorySafe introduces a **predictive decision layer** with two tightly integrated components:

- **Memory Vulnerability Index (MVI)** — quantifies **how vulnerable** each buffered experience is to future forgetting (risk forecasting).  
- **ProtectScore Policy Engine** — the **decision engine** that translates MVI risk scores into concrete actions: protect, prioritize replay, evict, or apply special handling.

Together, they transform the replay buffer into an **active, risk-aware memory system** that intentionally preserves what matters most — especially rare or safety-critical experiences — while maintaining overall performance and efficiency.

## 2. Memory Vulnerability Index (MVI) – The Risk Forecaster
MVI assigns each buffered sample a scalar score ∈ [0, 1] representing **predicted forgetting risk** if the sample is not replayed or protected soon.

### Core Idea
MVI predicts degradation **before** it visibly impacts accuracy, enabling proactive decisions.

### High-Level Signals
MVI combines complementary, literature-grounded indicators of forgetting:

- Forgetting Velocity — rate of recent performance drop on the sample  
- Feature / Representation Drift — shift in embedding space since storage  
- Rarity & Imbalance Signal — inverse frequency of class/pseudo-class  
- Temporal / Task Age Decay — exponential penalty for older experiences  

These signals are normalized and fused into a single MVI value using a proprietary combination method.

### Outcome
High MVI flags samples that are fragile (e.g., rare classes, drifted representations) → candidates for protection.

## 3. ProtectScore Policy Engine – The Decision Layer
ProtectScore takes MVI as primary input and produces a final **protection priority score** that guides buffer actions.

### Core Idea
While MVI answers “how at-risk is this?”, ProtectScore answers “what should we do about it?” in a balanced, context-aware way.

### High-Level Mechanism
At each buffer management step:

1. Compute fresh MVI for buffer samples (and optionally new arrivals).  
2. Feed MVI + contextual state (buffer composition, task progress, class distribution, memory pressure) into the ProtectScore function.  
3. Obtain ProtectScore per sample — higher score = higher priority to keep/replay.  
4. Execute policy-driven actions:
   - Evict lowest ProtectScore samples when buffer is full  
   - Bias replay sampling toward high ProtectScore items  
   - Force inclusion of very high ProtectScore samples in upcoming batches  
   - Optional: apply special treatment (duplication, stronger regularization, etc.)

The ProtectScore function is a **proprietary policy** that prevents extremes:
- Over-protection (starving diversity / majority performance collapse)  
- Under-protection (losing rare/safety-critical items)

Exact mapping, contextual rules, thresholds, and balancing logic are proprietary to MemorySafe Labs.

## 4. How MVI + ProtectScore Work Together (Decision Layer Flow)
Incoming samples ──► Add to buffer (initial MVI)
│
▼
Periodic / post-task trigger ──► Recompute MVI for buffer
│
▼
Compute ProtectScore
│
▼
Apply decisions:
• Evict low ProtectScore
• Prioritize high ProtectScore in replay
• Enforce protection for critical items
• Maintain buffer health & diversity
textThis closed loop creates **intelligent rarity enrichment** and **proactive rare-event safeguarding** without naive over-retention.

## 5. Empirical Validation Highlights
Multi-seed results across benchmarks:

- **PneumoniaMNIST** (imbalanced medical binary)  
  → Perfect rare-class recall (1.000)  
  → Buffer positive fraction ~0.11–0.12 (vs ~2% in stream)  
  → Strong specificity (0.004–0.012 best configs) — no majority collapse

- **CIFAR-100 Class-Incremental** (5 tasks, ResNet-18)  
  → ~53.17% final acc (10-seed avg)  
  → +8.2% relative vs standard reservoir replay  
  → ~18% forgetting reduction vs plain replay  
  → Low variance

- Memory savings (downsampling + feature replay synergy): 81–99% realistic reduction

## 6. Target Applications
Especially valuable where forgetting rare events is unacceptable:
- Medical imaging (rare pathologies on edge ultrasound/X-ray devices)  
- Robotics (anomalies, failures, novel obstacles in dynamic settings)  
- Industrial monitoring (rare fault detection)  
- Any memory-constrained continual system

## 7. Limitations & Roadmap
- Validation so far concentrated on image-based benchmarks  
- Robotics / sequential data may require extensions (trajectory-level signals, action-conditioned risk)  
- Future: hybrid with distillation, full on-device execution, real-world pilots

For licensing, collaboration, or detailed discussions:  
**Carla Centeno** — carla@memorysafe.ca  
MemorySafe Labs — https://memorysafe.ca

© 2026 MemorySafe Labs. All rights reserved.
