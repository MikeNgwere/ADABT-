"""
ADABT - Benchmark
Compares ADABT vs standard B-Tree across insert, delete, search
and measures overflow/underflow event counts and throughput.
"""

import time
import random
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fill_rate_predictor import FillRatePredictor
from adabt_tree import ADABT


# ====================================================================== #
#  Standard B-Tree (no AI) — baseline                                    #
# ====================================================================== #

class StandardBTree:
    """
    Minimal standard B-Tree for baseline comparison.
    Same logic as ADABT but with enable_ai=False.
    """
    def __init__(self, t=3):
        self._tree = ADABT(t=t, enable_ai=False)

    def insert(self, key): self._tree.insert(key)
    def delete(self, key): self._tree.delete(key)
    def search(self, key): return self._tree.search(key)
    def stats(self):       return self._tree.stats()


# ====================================================================== #
#  Benchmark runner                                                       #
# ====================================================================== #

def run_benchmark(n_keys=5000, t=3, seed=42):
    random.seed(seed)
    keys = random.sample(range(1, n_keys * 10), n_keys)
    delete_keys = random.sample(keys[:n_keys//2], n_keys//4)
    search_keys = random.sample(keys, n_keys//4)

    print("=" * 62)
    print(f"  ADABT vs Standard B-Tree  |  n={n_keys}  t={t}")
    print("=" * 62)

    # ── Warm-up predictor ──────────────────────────────────────────── #
    predictor = FillRatePredictor(
        split_threshold=0.82,
        merge_threshold=0.28,
        lookahead_k=50,
    )
    predictor.warm_up(n_samples=5000)

    # ── ADABT ──────────────────────────────────────────────────────── #
    adabt = ADABT(t=t, predictor=predictor, enable_ai=True)

    t0 = time.perf_counter()
    for k in keys:
        adabt.insert(k)
    insert_time_adabt = time.perf_counter() - t0

    t0 = time.perf_counter()
    for k in search_keys:
        adabt.search(k)
    search_time_adabt = time.perf_counter() - t0

    t0 = time.perf_counter()
    for k in delete_keys:
        adabt.delete(k)
    delete_time_adabt = time.perf_counter() - t0

    adabt_stats = adabt.stats()

    # ── Standard B-Tree ────────────────────────────────────────────── #
    std = StandardBTree(t=t)

    t0 = time.perf_counter()
    for k in keys:
        std.insert(k)
    insert_time_std = time.perf_counter() - t0

    t0 = time.perf_counter()
    for k in search_keys:
        std.search(k)
    search_time_std = time.perf_counter() - t0

    t0 = time.perf_counter()
    for k in delete_keys:
        std.delete(k)
    delete_time_std = time.perf_counter() - t0

    std_stats = std.stats()

    # ── Report ─────────────────────────────────────────────────────── #
    print(f"\n{'Metric':<35} {'Standard':>12} {'ADABT':>12}")
    print("-" * 62)

    def row(label, v1, v2, fmt=".4f"):
        print(f"  {label:<33} {v1:>12{fmt}} {v2:>12{fmt}}")

    row("Insert time (s)",    insert_time_std,  insert_time_adabt)
    row("Search time (s)",    search_time_std,  search_time_adabt)
    row("Delete time (s)",    delete_time_std,  delete_time_adabt)

    print()
    row("Proactive splits",   std_stats["proactive_splits"],
                              adabt_stats["proactive_splits"], "d")
    row("Reactive splits",    std_stats["reactive_splits"],
                              adabt_stats["reactive_splits"], "d")
    row("Proactive merges",   std_stats["proactive_merges"],
                              adabt_stats["proactive_merges"], "d")
    row("Reactive merges",    std_stats["reactive_merges"],
                              adabt_stats["reactive_merges"], "d")

    print()
    oe = adabt_stats["overflow_elimination"]
    ue = adabt_stats["underflow_elimination"]
    print(f"  Overflow  elimination rate : {oe*100:6.1f}%")
    print(f"  Underflow elimination rate : {ue*100:6.1f}%")

    total_std   = insert_time_std + search_time_std + delete_time_std
    total_adabt = insert_time_adabt + search_time_adabt + delete_time_adabt
    speedup = total_std / total_adabt if total_adabt > 0 else 0
    print(f"\n  Overall speedup (ADABT / Std): {speedup:.3f}x")
    print("=" * 62)


# ====================================================================== #
#  Predictor evaluation                                                   #
# ====================================================================== #

def evaluate_predictor(n_train=5000, n_test=1000):
    import numpy as np

    print("\n── Fill-Rate Predictor Evaluation ──────────────────────────")
    predictor = FillRatePredictor()
    predictor.warm_up(n_samples=n_train)

    rng = np.random.default_rng(99)
    X_test, y_test = predictor._generate_synthetic_data(n_test, rng)

    metrics = predictor.evaluate(X_test, y_test)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")
    print()


# ====================================================================== #
#  Entry point                                                            #
# ====================================================================== #

if __name__ == "__main__":
    evaluate_predictor()
    run_benchmark(n_keys=3000, t=3)
    run_benchmark(n_keys=3000, t=5)
