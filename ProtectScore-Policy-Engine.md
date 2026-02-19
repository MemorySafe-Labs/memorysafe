# ProtectScore Policy Engine  
**Intelligent Decision Layer for Replay Buffer Governance in Continual Learning**

**MemorySafe Labs**  
Carla Centeno, Founder  
February 2026  
https://memorysafe.ca | NVIDIA Inception Program Member

## License & IP Notice
This document describes the conceptual foundation of the ProtectScore Policy Engine, a core component of the MemorySafe system used in conjunction with the Memory Vulnerability Index (MVI).  
All rights reserved. This is proprietary technology developed by MemorySafe Labs. No part of this method may be reproduced, adapted, or used commercially without explicit written permission. For collaboration, licensing, or inquiries: carla@memorysafe.ca.

## 1. The Role of ProtectScore
The ProtectScore Policy Engine is the **decision-making layer** that translates predictive risk signals (primarily the Memory Vulnerability Index — MVI) into concrete buffer management actions during continual learning.

While MVI quantifies **how vulnerable** each buffered experience is to forgetting, ProtectScore determines **what to do about it**: protect, prioritize for replay, allow eviction, or apply special handling.

This turns raw risk scores into **actionable, balanced buffer governance** — preventing over-protection (which can collapse majority-class performance) while ensuring safety-critical and easily-forgotten experiences are preserved.

## 2. Core Design Principles
ProtectScore operates under three guiding principles:

- **Risk-Aware Prioritization**  
  Higher MVI → higher protection priority (harder to evict, more likely to be replayed).

- **Balanced Preservation**  
  Avoids naive thresholding that would over-protect common experiences and starve diversity in the buffer.

- **Efficiency Under Constraint**  
  Works within fixed memory budgets (especially important for edge devices) and minimal computational overhead.

## 3. High-Level Mechanism
At each buffer update point (after task completion, minibatch, or periodic interval):

1. Compute MVI for every sample in the current buffer (and optionally for new incoming samples).
2. Apply the ProtectScore function to produce a final **protection priority score** per sample.
3. Use ProtectScore to guide one or more of the following actions:
   - **Eviction decisions** — when buffer is full, remove lowest ProtectScore samples first.
   - **Replay sampling** — bias sampling distribution toward higher ProtectScore samples.
   - **Forced inclusion** — guarantee replay of very high ProtectScore samples in the next training step(s).
   - **Special handling** (optional) — e.g., duplicate high-risk samples or apply stronger regularization.

The ProtectScore function is a **proprietary, tunable policy** that combines:
- The raw MVI value
- Contextual factors (current buffer composition, task progress, class distribution drift)
- Hard/soft protection rules to maintain overall buffer health

Exact functional form, thresholds, weighting, and contextual adjustments are proprietary to MemorySafe Labs.

## 4. Key Outcomes Enabled by ProtectScore
- **Rarity Enrichment Without Collapse**  
  In imbalanced streams (e.g., PneumoniaMNIST), buffer positive-class fraction increased to ~0.11–0.12 (vs. ~2% in raw stream) while maintaining strong specificity (0.004–0.012 in best configurations).

- **Rare-Event Protection**  
  Achieved perfect rare-class recall (1.000) across multiple seeds on PneumoniaMNIST without sacrificing majority-class performance.

- **Forgetting Reduction**  
  ~18% relative reduction in catastrophic forgetting vs. plain reservoir replay on CIFAR-100 class-incremental (5 tasks, ResNet-18, 10-seed average).

- **Memory Efficiency Synergy**  
  When combined with downsampled images or feature-level replay, enables 81–99% realistic memory savings while preserving rare-event protection.

## 5. Why ProtectScore Matters
In safety-critical continual learning domains (medical imaging, robotics, industrial anomaly detection), standard replay methods can silently forget rare but catastrophic events. ProtectScore provides a principled way to **intentionally remember what matters most** — turning the replay buffer from a passive storage queue into an active, risk-managed memory system.

## 6. Limitations & Future Directions
- Current results are primarily on image-based continual benchmarks.
- Adaptation to sequential, embodied robotics data may require domain-specific policy extensions (e.g., trajectory-level protection, action-conditioned risk).
- Ongoing exploration: integration with distillation losses, on-device policy execution, real-world pilot tuning.

For more information, licensing discussions, or collaboration:  
**Carla Centeno** — carla@memorysafe.ca  
MemorySafe Labs — https://memorysafe.ca

© 2026 MemorySafe Labs. All rights reserved.
