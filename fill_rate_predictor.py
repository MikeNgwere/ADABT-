"""
ADABT - Fill-Rate Predictor
Lightweight MLP that predicts a node's fill ratio after k future operations.
Supports offline batch training and online incremental updates via partial_fit.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import pickle
import os


FEATURE_NAMES = [
    "current_fill_ratio",
    "insert_rate",
    "delete_rate",
    "search_rate",
    "fill_trend",
    "parent_fill_ratio",
    "left_sibling_fill_ratio",
    "right_sibling_fill_ratio",
    "depth_norm",
    "time_since_split_norm",
    "time_since_merge_norm",
]

N_FEATURES = len(FEATURE_NAMES)


class FillRatePredictor:
    """
    Predicts the fill ratio of a B-Tree node after k future operations.

    Architecture: StandardScaler -> MLP (11 -> 64 -> 32 -> 1)
    Training:     Offline warm-up on synthetic data + online partial_fit
    Thresholds:   split_threshold (default 0.85) and merge_threshold (default 0.30)
    """

    DEFAULT_SPLIT_THRESHOLD = 0.85   # predict fill > this  → proactive split
    DEFAULT_MERGE_THRESHOLD = 0.30   # predict fill < this  → proactive merge

    def __init__(
        self,
        split_threshold: float = DEFAULT_SPLIT_THRESHOLD,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
        lookahead_k: int = 50,
        random_state: int = 42,
    ):
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.lookahead_k = lookahead_k

        self._scaler = StandardScaler()
        self._model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=1,          # one epoch per partial_fit call
            warm_start=True,     # keeps weights between partial_fit calls
            random_state=random_state,
        )
        self._is_fitted = False
        self._n_online_updates = 0

        # Rolling buffer for online training samples
        self._online_X = []
        self._online_y = []
        self._online_batch_size = 32

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def warm_up(self, n_samples: int = 5000, random_state: int = 42):
        """
        Offline warm-up on synthetic data to initialise weights.
        Generates plausible (feature, target) pairs from a simulation.
        """
        rng = np.random.default_rng(random_state)
        X, y = self._generate_synthetic_data(n_samples, rng)

        # Fit scaler on full batch, then train
        X_scaled = self._scaler.fit_transform(X)
        self._model.set_params(max_iter=200)
        self._model.fit(X_scaled, y)
        self._model.set_params(max_iter=1)
        self._is_fitted = True
        print(f"[FillRatePredictor] Warm-up complete on {n_samples} synthetic samples.")
        return self

    def observe(self, features: list, actual_future_fill: float):
        """
        Record a ground-truth observation for online learning.
        Called when we know what the fill became k ops later.
        Triggers partial_fit when the online buffer is full.
        """
        self._online_X.append(features)
        self._online_y.append(np.clip(actual_future_fill, 0.0, 1.0))

        if len(self._online_X) >= self._online_batch_size:
            self._flush_online_buffer()

    def _flush_online_buffer(self):
        """Train on buffered online samples."""
        if not self._online_X:
            return

        X = np.array(self._online_X)
        y = np.array(self._online_y)

        if not self._is_fitted:
            # First ever fit — must fit scaler too
            X_scaled = self._scaler.fit_transform(X)
            self._model.fit(X_scaled, y)
            self._is_fitted = True
        else:
            X_scaled = self._scaler.transform(X)
            self._model.partial_fit(X_scaled, y)

        self._n_online_updates += 1
        self._online_X.clear()
        self._online_y.clear()

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, features: list) -> float:
        """
        Predict fill ratio after lookahead_k operations.
        Returns 0.5 (neutral) if model not yet fitted.
        """
        if not self._is_fitted:
            return 0.5

        X = np.array(features).reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        pred = self._model.predict(X_scaled)[0]
        return float(np.clip(pred, 0.0, 1.0))

    def should_split(self, features: list) -> bool:
        """Returns True if proactive split is recommended."""
        return self.predict(features) >= self.split_threshold

    def should_merge(self, features: list) -> bool:
        """Returns True if proactive merge is recommended."""
        return self.predict(features) <= self.merge_threshold

    def action(self, features: list) -> str:
        """Returns 'split', 'merge', or 'none'."""
        pred = self.predict(features)
        if pred >= self.split_threshold:
            return "split"
        if pred <= self.merge_threshold:
            return "merge"
        return "none"

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Computes regression metrics on a held-out test set.
        Returns MAE, RMSE, and prediction accuracy within ±0.05 tolerance.
        """
        if not self._is_fitted:
            raise NotFittedError("Predictor has not been trained yet.")

        X_scaled = self._scaler.transform(X_test)
        y_pred   = self._model.predict(X_scaled)
        y_pred   = np.clip(y_pred, 0.0, 1.0)

        mae  = float(np.mean(np.abs(y_pred - y_test)))
        rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        within_tolerance = float(np.mean(np.abs(y_pred - y_test) <= 0.05))

        # Overflow detection accuracy (did we correctly flag nodes that overflowed?)
        true_overflow  = y_test  >= self.split_threshold
        pred_overflow  = y_pred  >= self.split_threshold
        true_underflow = y_test  <= self.merge_threshold
        pred_underflow = y_pred  <= self.merge_threshold

        overflow_recall  = _safe_recall(true_overflow, pred_overflow)
        underflow_recall = _safe_recall(true_underflow, pred_underflow)
        overflow_prec    = _safe_precision(true_overflow, pred_overflow)
        underflow_prec   = _safe_precision(true_underflow, pred_underflow)

        return {
            "mae":                mae,
            "rmse":               rmse,
            "accuracy_within_5pct": within_tolerance,
            "overflow_recall":    overflow_recall,
            "overflow_precision": overflow_prec,
            "underflow_recall":   underflow_recall,
            "underflow_precision":underflow_prec,
            "n_test_samples":     len(y_test),
            "n_online_updates":   self._n_online_updates,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler,
                         "fitted": self._is_fitted}, f)
        print(f"[FillRatePredictor] Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model    = data["model"]
        self._scaler   = data["scaler"]
        self._is_fitted = data["fitted"]
        print(f"[FillRatePredictor] Loaded from {path}")
        return self

    # ------------------------------------------------------------------ #
    # Synthetic data generation (for warm-up)                             #
    # ------------------------------------------------------------------ #

    def _generate_synthetic_data(
        self, n_samples: int, rng: np.random.Generator
    ):
        """
        Simulates plausible (feature, target_fill) pairs.
        Target is computed from a deterministic approximation:
            predicted_fill ≈ current_fill + insert_rate*k/max_keys
                            - delete_rate*k/max_keys + noise
        This gives the model a meaningful starting signal.
        """
        k = self.lookahead_k

        # ---- Feature columns ----
        current_fill  = rng.uniform(0.1, 0.95, n_samples)
        insert_rate   = rng.beta(2, 5,  n_samples)   # skewed low
        delete_rate   = rng.beta(2, 8,  n_samples)
        search_rate   = 1.0 - insert_rate - delete_rate
        search_rate   = np.clip(search_rate, 0.0, 1.0)
        fill_trend    = rng.uniform(-0.02, 0.02, n_samples)
        parent_fill   = rng.uniform(0.3, 0.9, n_samples)
        left_fill     = rng.uniform(0.2, 0.9, n_samples)
        right_fill    = rng.uniform(0.2, 0.9, n_samples)
        depth_norm    = rng.uniform(0.0, 0.5, n_samples)
        split_norm    = rng.uniform(0.0, 1.0, n_samples)
        merge_norm    = rng.uniform(0.0, 1.0, n_samples)

        X = np.column_stack([
            current_fill, insert_rate, delete_rate, search_rate,
            fill_trend, parent_fill, left_fill, right_fill,
            depth_norm, split_norm, merge_norm,
        ])

        # ---- Target: approximate future fill ----
        # Each insert adds ~1/max_keys fill, each delete removes ~1/max_keys
        # We assume max_keys ~ 10 for this simulation
        max_keys_approx = 10.0
        net_change = (insert_rate - delete_rate) * k / max_keys_approx
        noise = rng.normal(0, 0.03, n_samples)
        y = np.clip(current_fill + net_change + noise, 0.0, 1.0)

        return X, y


# ------------------------------------------------------------------ #
# Utility functions                                                    #
# ------------------------------------------------------------------ #

def _safe_recall(true_pos_mask, pred_pos_mask) -> float:
    total = true_pos_mask.sum()
    if total == 0:
        return 1.0   # no positives to recall
    return float((true_pos_mask & pred_pos_mask).sum() / total)


def _safe_precision(true_pos_mask, pred_pos_mask) -> float:
    total = pred_pos_mask.sum()
    if total == 0:
        return 1.0   # no predictions made
    return float((true_pos_mask & pred_pos_mask).sum() / total)
