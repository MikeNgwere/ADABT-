# ADABT: An AI-Driven Adaptive B-Tree for Optimised Deletion on Large-Scale Datasets with Proactive Overflow and Underflow Elimination

**Module:** MCS501 — Advanced Data Structures & Algorithms
**University:** University of Zimbabwe · MSc Computer Science · 2026
**Authors:** Mike Ngwere (R186209Q) & John Mberi (R2425845)

---

## Research Focus
Standard B-Trees handle deletion reactively — underflow, borrow, and merge
operations fire *after* the structural violation occurs, causing expensive
cascading disk writes on large-scale datasets.

ADABT introduces an AI layer that predicts impending overflow and underflow
before they happen, proactively restructuring nodes in a background pass so
deletion never encounters underflow on the critical path.

## Key Results
| Metric | Value |
|--------|-------|
| Single-op delete speedup | ~2× (3.1ms vs 6.0ms) |
| Bulk delete speedup | 1.28× |
| Insert speedup (disk) | 1.16× |
| Overflow elimination | 34.6% |
| Overflow recall (predictor) | 97.96% |
| Underflow recall (predictor) | 97.88% |
| MAE (fill ratio error) | 0.031 |

## Files
| File | Purpose |
|------|---------|
| `node_monitor.py` | Per-node operation tracker & feature extractor |
| `fill_rate_predictor.py` | MLP fill-rate prediction model |
| `adabt_tree.py` | Full ADABT B-Tree implementation |
| `benchmark.py` | ADABT vs Standard B-Tree comparison |
| `dashboard.py` | Streamlit interactive dashboard |

## Quick Start
```bash
pip install streamlit scikit-learn plotly numpy
streamlit run dashboard.py
```

## Model Architecture
```
Input (11 features) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
```
Training: offline warm-up (5000 synthetic samples) + online partial_fit

## Why In-Memory Tests Show ADABT Slower
Python sklearn inference (~38µs) exceeds in-memory B-Tree op cost (~35µs).
In real disk-based systems (1–10ms per node access), ML inference is negligible
and proactive restructuring produces the speedups shown above.
Enable **Disk Simulation** in the dashboard to see the real-world advantage.
