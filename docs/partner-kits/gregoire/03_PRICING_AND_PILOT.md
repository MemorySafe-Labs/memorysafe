# Pricing & Pilot Motion

---

## Tier 0 — Discovery (free)

- **15 minutes** — qualify ICP, explain hook, share safe claims
- **Output:** Go / no-go for eval

---

## Tier 1 — 2-week technical eval (free first engagement)

**Deliverables:**
1. `MemorySafeGovernor` integrated into one CL/retrain stream
2. A/B vs their current replay (or reservoir baseline) on **agreed metric**
3. Short report: Δ AUPRC or Δ rare-class recall, buffer stats, audit logs
4. Recommendation: pilot or stop

**Timeline:** 10 business days after VPC/sandbox access agreed

**What we need from customer:**
- One PyTorch training pipeline (or proxy benchmark first)
- Agreed rare-class metric
- In-VPC or sandbox access

**Price:** Free for first engagement. If they insist on paid eval: **$5k** (your discretion).

---

## Tier 2 — 90-day paid pilot

**Price:** **$25,000 – $40,000** (scope-dependent)

**Scope:**
- Production-adjacent integration on 1–2 streams
- Monitoring + audit JSON in their pipeline
- Success criteria defined upfront (falsifiable)

**Only discuss after eval shows signal.**

---

## Tier 3 — Annual SDK license

**Price:** **$50,000 – $120,000 / year**

**Includes:**
- Governor + buffer policies (Lite or Performance profile)
- Integration support hours (negotiated)
- Benchmark harness for regression checks

---

## Partner economics (Phase 1 — MOU, not this doc)

Discussed separately with Grégoire after first intro meeting:
- Referral commission on closed pilot: **10–15%** (typical range discussed)
- No Hilo co-brand until corporate agreement

---

## Eval success criteria (put in every proposal)

| Item | Example |
|------|---------|
| Primary metric | Δ combined AUPRC ≥ 0.03 at matched 500-cap buffer |
| Secondary | Δ rare-class recall or task-0 retention |
| Guardrail | No regression > 5% on majority class |
| Decision date | Day 10 readout call |

---

## What NOT to discount

- In-VPC-only deployment (non-negotiable for v1)
- Falsifiable metric agreement before eval starts
- No production claims without their data