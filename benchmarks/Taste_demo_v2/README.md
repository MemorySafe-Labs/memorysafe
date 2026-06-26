# MemorySafe — Taste Demo v2

Interactive Colab demo: **Reservoir vs MemorySafe** on a synthetic rare-class stream.

## What this shows

- MVI-governed protect / replay / forget decisions
- Rare positive recall when the buffer is full
- Side-by-side comparison chart (~2 min on CPU)

## What this is NOT

- Not the v14.2 PneumoniaMNIST / PathMNIST benchmark
- Not a production accuracy claim

Canonical 10-seed results live on [memorysafe.ca](https://memorysafe.ca/#validation).

## Run locally

```bash
cd benchmarks/Taste_demo_v2
python compare_demo.py --steps 1200 --capacity 500
```

## Colab

Open `demo.ipynb` in Google Colab — **Runtime → Run all**.
