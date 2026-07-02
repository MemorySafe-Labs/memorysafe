# MemorySafe — Commercial Partner Kit (Grégoire)

**July 2026 · v14.2 · Personal capacity**

---

## Sell this sentence first

> **"Cut replay memory up to 84% without throwing away the cases that matter."**

Buyers move when something hits **cost, capacity, or risk**. Lead there. The science is backup.

---

## The story (30 seconds)

Hospitals and imaging AI teams retrain on new data all the time. Replay buffers eat GPU RAM. Bigger buffer = bigger bill. Smaller buffer = rare cases vanish.

MemorySafe is a **PyTorch hook** that governs what stays in the buffer — so teams can run **Lite mode (80-cap, ~84% smaller footprint)** and still protect rare, high-impact classes. In-VPC. No SaaS. No rip-and-replace.

**Proof today:** public v14.2 benchmarks on memorysafe.ca — Lite **0.686** combined AUPRC vs Performance **0.706** on the same protocol (PneumoniaMNIST 5-task, 10 seeds).

---

## Two buyers, two doors

| Buyer | Pain | Your opener |
|-------|------|-------------|
| **VP Eng / Infra** | GPU $, memory ceiling, edge deploy | "84% smaller replay footprint — more headroom on the same box." |
| **ML lead / Product** | Rare class drift, audit anxiety | "Governed retention — you choose what survives when memory is tight." |

Same product. Different receipt.

---

## SKUs you quote

| Profile | Buffer | When to pitch |
|---------|--------|---------------|
| **Performance** | 500-cap | "Max retention — benchmark-winning lane" |
| **Lite** | 80-cap | **Default pitch** — edge, cost-sensitive, multi-tenant GPU |

---

## Motion (simple)

```
Hook (84% memory) → 15 min discovery → free 2-week in-VPC eval → pilot → SDK
```

**Eval promise:** A/B on *their* stream — MemorySafe Lite vs their replay at **matched or reduced** buffer budget. Report: memory MB, rare-class metric, recommendation.

**Do not promise:** production accuracy on their data until eval runs.

---

## ICP (who pays)

- Medical imaging AI (pathology, radiology, ophthalmology)
- Continual learning or frequent retraining
- **Fixed GPU budget** or edge deployment pressure
- PyTorch, can do in-VPC eval

**First targets:** digital pathology · radiology workflow AI · any team complaining about **GPU cost + model updates**.

---

## Phase 1 (you)

1. Read kit + alignment call with Carla
2. One warm intro (qualified)
3. Co-run discovery — you open with **money**, Carla closes with **eval**

Details: `05_PHASE1_PARTNER_SCOPE.md`

---

## Kit map

| File | Use |
|------|-----|
| `02_PITCH_AND_PROOF.md` | Lines that are safe to say |
| `03_PRICING_AND_PILOT.md` | Eval → pilot → SDK + ROI framing |
| `04_OUTREACH_TEMPLATES.md` | Emails / LinkedIn |
| `06_ROI_TALK_TRACK.md` | Call scripts |

carla@memorysafe.ca · https://memorysafe.ca
