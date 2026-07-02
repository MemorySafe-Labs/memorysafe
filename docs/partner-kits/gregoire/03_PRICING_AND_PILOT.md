# Pricing, Pilot & ROI Frame

---

## Price the problem before the product

Ask on discovery:
1. What does a **GPU hour** cost you all-in? (cloud or amortized on-prem)
2. How big is your **replay buffer** today (samples / MB)?
3. What happens when you **halve** buffer size today? (rare class drop? retrain churn?)

If they feel #3, they'll fund an eval.

---

## Commercial ladder

| Stage | Price | Buyer gets |
|-------|-------|------------|
| **Discovery** | Free | 15 min — qualify memory pain |
| **2-week eval** | Free (first) | A/B: their replay vs MemorySafe **Lite** on one stream |
| **90-day pilot** | **$25k–$40k** | Production-adjacent integration + monitoring |
| **Annual SDK** | **$50k–$120k/yr** | Lite or Performance profile + support hours |

Pilot only after eval shows signal on **their** metric.

---

## Eval design (sell this)

**Question we answer:**  
*"Can MemorySafe Lite hit our rare-class target with ≤20% of our current replay memory?"*

**Deliverables:**
1. Hook in their PyTorch loop
2. A/B at reduced buffer cap
3. Report: **MB saved**, Δ rare metric, audit log sample
4. Go / no-go recommendation

**Timeline:** 10 business days after VPC access

---

## ROI talk (illustrative — adapt on the call)

Use their numbers when you have them. Otherwise:

> "If replay is even **500 MB–2 GB** of your training footprint, an **84% cut** frees headroom — smaller instance, second model on the same card, or edge deploy without a new hardware line item."

> "Lite is **0.686 vs 0.706** on our public medical benchmark — you're not trading 84% memory for zero quality; you're trading for *governed* quality."

**CFO line:**  
*"This is a memory-efficiency lever on infrastructure you already pay for."*

**ML line:**  
*"Governed retention so the buffer shrink doesn't eat your rare positives."*

---

## Eval success criteria (put in writing)

| Metric | Example target |
|--------|----------------|
| Memory | ≥ **70%** replay footprint reduction vs baseline |
| Quality | No worse than agreed Δ on rare-class recall / AUPRC |
| Ops | Hook integrated without forking their trainer |

---

## Partner economics (MOU — not binding here)

Referral on closed pilot: **10–15%** range discussed with Carla.  
Hilo corporate channel: separate agreement.
