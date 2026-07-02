# Pitch & Proof — What You Can Say Today

**Source of truth:** https://memorysafe.ca/#validation · GitHub `memorysafe_v14/`

Only cite **v14.2**. No other numbers.

---

## Headline (always)

**Up to 84% smaller replay footprint** with MemorySafe Lite (80-cap vs 500-cap Performance profile).

That's the door-opener. Everything else supports it.

---

## The proof stack (current)

### 1. Memory efficiency (money story)
- **Lite SKU:** 80-sample buffer
- **Performance SKU:** 500-sample buffer  
- **Footprint reduction:** ~**84%** smaller replay buffer (80 ÷ 500)
- **Retention at Lite:** **0.686 ± 0.056** combined AUPRC (PneumoniaMNIST 5-task, 10 seeds, CPU)
- **Retention at Performance:** **0.706 ± 0.051** on the same protocol

**How to say it:**  
*"We're within ~2 points of full-buffer performance while using one-sixth of the replay memory."*

### 2. Rare-class pressure (clinical story)
- **Performance vs reservoir:** **0.706 vs 0.663** combined AUPRC, **p = 0.017** (10/10 seed wins)
- **Positive recall:** **75.0%** · **Task-0 recall:** **80.0%**
- Use when the buyer cares about **what gets forgotten**, not just MB

### 3. Pathology lane (digital pathology prospects)
- PathMNIST rare-tissue: MemorySafe **0.389** vs reservoir **0.303**, **p = 0.028**
- Pull this when the logo says pathology

---

## Three creative angles (pick one per prospect)

**① The cloud bill**  
*"You're paying for GPU partly to store old cases. We shrink that storage ~84% and keep the dangerous tail."*

**② The edge bet**  
*"Lite mode is for the box that doesn't have 80GB — ambulance, clinic, on-prem PACS adjacency."*

**③ The audit trail**  
*"Governed replay = you can explain why a rare case stayed in memory. That's regulatory air cover."*

---

## One-liners (rotate these)

- "Same GPU budget. Smaller buffer. Smarter retention."
- "84% less replay RAM — rare cases don't get voted off the island."
- "Not a new platform — a hook in your existing PyTorch loop."
- "2-week eval: we prove it on your stream or we stop."

---

## Hard stops (never say)

| Don't | Why |
|-------|-----|
| Universal CL SOTA | We're a memory governor, not a foundation model |
| Hosted SaaS | In-VPC hook only |
| Production guarantees | Eval first, always |
| Numbers not on memorysafe.ca | Stay on v14.2 public benchmarks |

---

## Leave-behinds

1. https://memorysafe.ca/#profiles (Lite vs Performance)
2. https://memorysafe.ca/#validation (numbers)
3. Colab demo (2 min, behavioral): GitHub `benchmarks/Taste_demo_v2/demo.ipynb`
4. `assets/MemorySafe_Sales_OnePager.docx` if Carla approves for that prospect
