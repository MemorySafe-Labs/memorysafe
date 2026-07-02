# Outreach Templates — Copy-Paste

**Before sending:** Read `02_SAFE_CLAIMS.md`. Pre-revenue, honest tone. No spray-and-pray.

---

## LinkedIn — pathology (Aiforia-class)

```
Hi [Name] — I'm Carla, founder of MemorySafe Labs (Montreal, NVIDIA Inception). Pre-revenue, still validating fit.

Small PyTorch hook for continual learning — protect rare pathology classes when replay memory is tight. PathMNIST (10-seed): beat reservoir on rare-tissue AUPRC (p=0.028). May not apply to your stack.

Open to 15 min of honest feedback? No pressure.

memorysafe.ca
```

---

## Email — pathology team

**Subject:** Early-stage startup — rare-class retention in pathology CL?

```
Hi [Name] / team,

I'm Carla Centeno, founder of MemorySafe Labs — early-stage, Montreal (NVIDIA Inception, Google for Startups). Pre-revenue.

We built a lightweight PyTorch hook for continual-learning pipelines: when replay buffers are full, rare tissue classes often get dropped — we try to protect what's vulnerable, not just what's frequent.

On public pathology benchmarks only (10-seed CPU, JSON available):
- PathMNIST rare-tissue: 0.389 vs 0.303 combined AUPRC vs reservoir (p=0.028)
- PneumoniaMNIST (supporting): 0.706 vs 0.663 (p=0.017)

Not a production claim — looking for a first conversation with a pathology team who can tell us if this matters. Integration is in-VPC only, no SaaS.

Open to 15 minutes, or a pointer to the right person? Even "not now" helps.

Thank you,
Carla Centeno
carla@memorysafe.ca
https://memorysafe.ca
```

---

## Email — radiology / general medical CL

**Subject:** Replay governance for rare-class retention in medical CL?

```
Hi [Name],

I'm Carla Centeno (MemorySafe Labs, Montreal). We govern replay-buffer retention for continual-learning vision models — protecting rare, high-impact cases when GPU memory is fixed.

On public medical benchmarks (10-seed CPU): PneumoniaMNIST class-IL 0.706 vs 0.663 combined AUPRC (p=0.017); pathology rare-tissue lane p=0.028 vs reservoir.

Integration is a PyTorch training hook (in-VPC, no SaaS). Worth 15 minutes to see if a 2-week eval on one update stream is useful for your roadmap?

carla@memorysafe.ca · https://memorysafe.ca
```

---

## Follow-up (Day 7)

```
Hi — just bumping this gently in case it got buried. Still early-stage on our side, but happy to share benchmark JSON or jump on a short call if useful. Totally fine if timing isn't right — thank you either way.
```

---

## After discovery — eval offer

```
Thanks for the call. As discussed, we'd propose a free 2-week in-VPC eval:

- One retrain/CL stream
- A/B: your replay vs MemorySafe at matched buffer cap
- Readout: Δ AUPRC or Δ rare-class recall + audit logs
- No obligation to pilot

If useful, I can send a one-pager and JSON from our public benchmarks. When works for a quick scoping call with your ML lead?
```

---

## Priority targets (Phase 1)

1. **Aiforia** — digital pathology, Gary Chisholm (Americas), contact@aiforia.com
2. **contextflow** — radiology workflow, via contextflow.com/contact

One intro at a time. Log send date; follow up Day 7.