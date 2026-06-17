# 🌳 ADABT — AI-Driven Adaptive B-Tree

## An AI-Driven Adaptive B-Tree for Optimised Deletion on Large-Scale Datasets with Proactive Overflow and Underflow Elimination

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Academic-green?style=flat)]()

---

**Module:** MCS501 — Advanced Data Structures & Algorithms  
**Institution:** University of Zimbabwe, Faculty of Science, Department of Computer Science  
**Degree:** Master of Computer Science (MCS) — 2025  
**Authors:** Mike Ngwere (R186209Q) &nbsp;|&nbsp; John Mberi (R2425845)

---

## 📌 Overview

ADABT is a research prototype that investigates and addresses a long-standing performance gap in B-Tree data structures: **deletion on large-scale datasets**.

Decades of research have produced near-optimal algorithms for B-Tree insertion and search. Deletion — the most structurally complex operation — has remained algorithmically unchanged since Bayer and McCreight's original 1972 formulation. ADABT applies a lightweight machine learning model **exclusively to the deletion path**, leaving insertion and search completely unchanged, to proactively detect and prevent the underflow and overflow events that make deletion expensive in disk-based systems.

---

## 🔬 The Research Gap

| Operation | Research Status | AI Applied |
|-----------|----------------|------------|
| **Search** | Exhaustively studied — O(log n) optimal | ✗ Not needed |
| **Insertion** | Heavily optimised (LSM-Trees, Bw-Tree, bulk loading) | ✗ Not needed |
| **Deletion** | Unchanged since 1972 — cascade merges unaddressed | ✅ **ADABT targets this** |

---

## ⚡ Key Results

| Metric | Value |
|--------|-------|
| Insert Speedup | **1.00×** — no AI, identical to standard B-Tree |
| Search Speedup | **1.00×** — no AI, identical to standard B-Tree |
| Delete Speedup (bulk) | **1.27×–1.61×** under disk simulation |
| Delete Speedup (single op) | **1.92×** (3.128ms vs 6.023ms) |
| Underflow Recall (predictor) | **97.88%** |
| Overflow Recall (predictor) | **97.96%** |
| Overflow Elimination | **34.6%** prototype / **97.96%** theoretical |
| Predictor MAE | **0.031** (±3.1% fill ratio error) |

---

## 🗂️ File Structure

```
ADABT/
├── dashboard.py            ← Streamlit interactive dashboard (main app)
├── adabt_tree.py           ← ADABT B-Tree implementation (AI on delete only)
├── fill_rate_predictor.py  ← MLP Fill-Rate Predictor (11→64→32→1)
├── node_monitor.py         ← Per-node structural feature extractor
├── benchmark.py            ← Side-by-side benchmark script
└── requirements.txt        ← Python dependencies
```

---

## 🚀 Quick Start — Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ADABT.git
cd ADABT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```

The dashboard opens automatically at `http://localhost:8501`

### 4. Run the benchmark script (optional)
```bash
python benchmark.py
```

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Push all files to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click **New app** → select your repo → set main file to `dashboard.py`
5. Click **Deploy** — your public URL is ready in ~2 minutes

---

## 🏗️ Architecture

ADABT adds an AI intelligence layer to a standard B-Tree. The layer has four components that **only activate during deletion**:

```
Query Interface
    ├── INSERT  ──────────────────────────► Standard B-Tree Core  (no AI)
    ├── SEARCH  ──────────────────────────► Standard B-Tree Core  (no AI)
    └── DELETE  ──► AI Intelligence Layer ► Standard B-Tree Core
                        │
                        ├── Node Monitor         (tracks fill ratios)
                        ├── Fill-Rate Predictor  (MLP: 11→64→32→1)
                        ├── Proactive Restructure (flags nodes early)
                        └── Adaptive Degree       (adjusts capacity)
```

---

## 🧠 The Fill-Rate Predictor

A two-layer MLP that predicts a node's fill ratio after the next 50 operations.

**Architecture:** `Input(11) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)`  
**Parameters:** 3,009 (deliberately tiny for low inference cost)  
**Inference cost:** ~0.1ms (vs 1–10ms per disk I/O in production)

### 11 Input Features

| # | Feature | Role |
|---|---------|------|
| 1 | Current fill ratio | Primary overflow/underflow signal |
| 2 | Insert rate | Upward fill pressure |
| 3 | Delete rate | Downward fill pressure |
| 4 | Search rate | Access pattern context |
| 5 | Fill trend | Direction of structural change |
| 6 | Parent fill ratio | Upstream structural pressure |
| 7 | Left sibling fill | Borrow availability (left) |
| 8 | Right sibling fill | Borrow availability (right) |
| 9 | Depth (normalised) | Level-in-tree signal |
| 10 | Time since split | Recency signal |
| 11 | Time since merge | Recency signal |

### Decision Thresholds

| Predicted fill ≥ 0.82 | → Proactive **split** flagged |
|---|---|
| Predicted fill ≤ 0.28 | → Proactive **merge** flagged |
| 0.28 < fill < 0.82 | → No action needed |

---

## 💾 Why Disk Simulation Matters

In a pure Python in-memory test:
- B-Tree operation cost: ~35 µs
- AI inference cost: ~38 µs
- Result: AI overhead > operation cost → ADABT appears slower *(expected and honest)*

In a real disk-based database:
- Each node access: **1–10 ms** (disk seek)
- Each merge operation: **3+ disk writes** = 3–30ms
- AI inference: **0.1ms** (negligible)
- Result: Preventing merges saves far more than inference costs → **ADABT wins**

Toggle **💾 Disk Simulation** in the dashboard to model realistic conditions. The 1.61× deletion speedup holds stable across all tested latency values (0.5ms–10ms), confirming a structural advantage.

---

## 📊 Dashboard Guide

### 🌳 B-Tree Visualiser Tab
- Insert keys individually or in bulk (5–100 random keys)
- Both trees update live and are always structurally identical after insert/search
- Enable Disk Simulation in the sidebar, then delete keys to see the timing difference
- Node colours: 🟦 Healthy &nbsp; 🟥 Near overflow (>82%) &nbsp; 🟨 Near underflow (<28%) &nbsp; 🟩 Search result
- Separate operation logs show exact timing for both trees

### 📊 Benchmark Tab
- Enable 💾 Disk Simulation for realistic results
- Select dataset size (1,000–2,000 recommended for clear results)
- Click **▶ Run Benchmark**
- Insert and Search speedup cards = **1.00×** (no AI — identical)
- Delete speedup card = **1.27×+** (AI active — ADABT faster)

### 🧠 Predictor Analytics Tab
- Shows Fill-Rate Predictor evaluation on 1,000 held-out samples
- Scatter plot: predicted vs actual fill ratios with split/merge threshold zones
- Feature importance chart: which of the 11 features matter most
- Residual histogram: prediction errors are small and centred near zero

---

## 📐 Complexity Analysis

| Operation | Standard B-Tree | ADABT (Worst Case) | ADABT (Expected, Disk) |
|-----------|----------------|-------------------|----------------------|
| Search | O(t log_t n) | O(t log_t n) | O(t log_t n) — identical |
| Insert | O(t log_t n) | O(t log_t n) | O(t log_t n) — identical |
| Delete | O(t log_t n) | O(t log_t n) | O(t log_t n − D_saved) |
| Merges / n deletes | O(n/t) worst | O(n/t) worst | O(n × 0.0212 / t) |

Where α = predictor underflow recall = 0.9788. The 47× reduction in expected reactive merges per n deletions directly translates to eliminated disk write operations.

---

## 📚 References

1. Bayer, R., & McCreight, E. M. (1972). Organization and maintenance of large ordered indexes. *Acta Informatica, 1*(3), 173–189.
2. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to algorithms* (4th ed.). MIT Press.
3. Comer, D. (1979). The ubiquitous B-tree. *ACM Computing Surveys, 11*(2), 121–137.
4. Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018). The case for learned index structures. *Proceedings of ACM SIGMOD*, 489–504.
5. Ding, J., et al. (2020). ALEX: An updatable adaptive learned index. *Proceedings of ACM SIGMOD*, 969–984.
6. Graefe, G. (2011). Modern B-tree techniques. *Foundations and Trends in Databases, 3*(4), 203–402.
7. Levandoski, J. J., Lomet, D. B., & Sengupta, S. (2013). The Bw-Tree: A B-tree for new hardware. *ICDE*, 302–313.
8. Ramakrishnan, R., & Gehrke, J. (2003). *Database management systems* (3rd ed.). McGraw-Hill.

---

## 🛠️ Requirements

```
streamlit
scikit-learn
plotly
numpy
pandas
```

Python 3.10 or higher required.

---

## 📄 Licence

This project is submitted for academic purposes as part of the MCS501 coursework at the University of Zimbabwe. All rights reserved by the authors.

---

*"B-Tree deletion has been the forgotten operation for 50 years. ADABT changes that."*
