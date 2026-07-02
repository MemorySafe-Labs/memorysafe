# Outreach Templates — Money First

**Tone:** Direct, curious, no jargon dump. Lead with **84% memory** or **GPU bill**.

---

## LinkedIn — infra / VP Eng angle

```
Hi [Name] — quick one from Montreal.

We help continual-learning teams shrink replay buffers ~84% (Lite profile) without dumping rare cases — PyTorch hook, in-VPC.

Public v14.2 benchmark: near-full retention at 1/6th the replay memory on medical imaging data.

Worth 15 min if GPU cost or buffer size is on your roadmap?

memorysafe.ca
```

---

## LinkedIn — pathology / product angle

```
Hi [Name] — Carla, MemorySafe Labs (Montreal).

Digital pathology CL teams often face two pains: rare tissue classes disappearing from replay, and GPU memory limits on retrain.

We govern what stays in the buffer — Lite mode uses ~84% less replay footprint than our Performance profile, with close retention on public benchmarks.

Open to a short call to see if a 2-week in-VPC eval is worth it?

memorysafe.ca
```

---

## Email — subject lines that work

- `Cut replay memory ~84%? (PyTorch hook, in-VPC)`
- `GPU headroom for your CL pipeline`
- `Rare cases + smaller replay buffer — 15 min?`

---

## Email — full (copy-paste)

**Subject:** Cut replay memory ~84% on your CL pipeline?

```
Hi [Name],

I'm Carla Centeno — MemorySafe Labs, Montreal. We built a PyTorch hook that governs replay-buffer retention for continual-learning vision models.

The practical pitch: **MemorySafe Lite runs with ~84% smaller replay footprint** (80 vs 500 sample cap on our public medical benchmark) while staying close to full-buffer retention (0.686 vs 0.706 combined AUPRC, v14.2, 10 seeds).

Why teams care:
- GPU / memory headroom on the same hardware
- Rare, high-impact cases protected when buffer is tight
- In-VPC only — no SaaS, no rip-and-replace

Happy to do 15 minutes, or point me to whoever owns retrain infra. If it's not a fit, a quick "not now" is perfect.

carla@memorysafe.ca
https://memorysafe.ca
```

---

## After they bite — eval offer

```
Great speaking today. Proposed next step — free 2-week in-VPC eval:

• Integrate MemorySafe Lite on one update stream
• Compare vs your current replay at reduced buffer budget
• Deliverable: MB saved + rare-class metric + go/no-go

No pilot commitment unless the numbers work on your data.
```

---

## Follow-up (Day 7)

```
Hi — gentle bump. Still happy to show the Lite vs Performance memory story (84% smaller replay footprint on our public benchmark) or run a short eval on your stream. No worries if timing's off.
```

---

## Targets (one at a time)

1. **Aiforia** — pathology · Gary Chisholm · contact@aiforia.com  
2. **contextflow** — radiology · contextflow.com/contact  

Log send date. Follow up Day 7.
