# ROI Talk Track — Creative, Honest

Use on discovery calls. **Listen first.** Pick one track.

---

## Open (Grégoire or Carla)

*"I'll be blunt — we sell memory efficiency on GPUs you already pay for. Can I ask how big your replay buffer is today and what happens when you try to shrink it?"*

---

## Track A — The CFO whisper

**Setup:** They mention cloud spend or capex.

**You:**  
*"Most teams oversize replay because shrinking it kills rare classes. MemorySafe Lite targets **~84% smaller replay footprint** — 80 vs 500 samples on our public benchmark — with **0.686 vs 0.706** AUPRC. That's not a research flex; it's **more model per dollar** on the same box."*

**Ask:**  
*"If we freed even 1–2 GB in training, what would you run on that headroom?"*

---

## Track B — The edge / deploy play

**Setup:** On-prem, hospital IT, latency-sensitive.

**You:**  
*"Performance profile is for max retention. **Lite is the edge story** — governed replay when the device doesn't have datacenter RAM. Same hook, smaller buffer, auditable decisions."*

**Ask:**  
*"Any deploy target where a 500-sample replay buffer is a non-starter?"*

---

## Track C — The clinical risk play

**Setup:** Pathology, radiology, regulated AI.

**You:**  
*"When memory is full, the buffer keeps frequent cases and drops rare ones — that's a safety story, not just a math story. We govern eviction. Public pathology benchmark: **p = 0.028** vs reservoir on rare-tissue AUPRC. Pair that with **84% memory reduction** in Lite and you get **risk + cost** in one eval."*

**Ask:**  
*"Which rare class or early task hurts most when you ship a new model version?"*

---

## Objection → pivot

| They say | You say |
|----------|---------|
| "Too early / startup" | "Fair — that's why we do a **free 2-week eval** on your stream. You keep the report either way." |
| "We already have replay" | "We don't replace it — we **govern** it. Question is whether you can **shrink** it 84% without bleeding rare cases." |
| "Send a paper" | "Site has v14.2 numbers + GitHub repro. Paper comes after your eval JSON." |
| "What's the price?" | "Eval free. Pilot $25–40k if data supports it. **ROI question first:** what's GPU memory worth to you?" |
| "Is it SOTA?" | "It's a **memory governor**. We win on medical lanes we publish; eval wins on **your** P&L." |

---

## Close

*"Next step I'd suggest: 30 min with your ML lead + whoever owns infra. We scope a 2-week eval — Lite profile, your stream, memory MB + rare metric in the report. If it doesn't beat your baseline, we shake hands and stop."*

---

## Cheat card (keep on screen)

```
HEADLINE: 84% smaller replay footprint (Lite)
PROOF:    0.686 vs 0.706 AUPRC · v14.2 · memorysafe.ca
PRODUCT:  PyTorch hook · in-VPC · no SaaS
ASK:      2-week eval on their data
```
