"""ADABT Fill-Rate Predictor — MLP that predicts node fill ratio after k operations."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class FillRatePredictor:
    DEFAULT_SPLIT_THRESHOLD = 0.85
    DEFAULT_MERGE_THRESHOLD = 0.30

    def __init__(self, split_threshold=0.85, merge_threshold=0.30, lookahead_k=50, random_state=42):
        self.split_threshold = split_threshold
        self.merge_threshold  = merge_threshold
        self.lookahead_k      = lookahead_k
        self._scaler = StandardScaler()
        self._model  = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu',
                                     solver='adam', learning_rate='adaptive',
                                     learning_rate_init=0.001, max_iter=1,
                                     warm_start=True, random_state=random_state)
        self._is_fitted = False
        self._online_X, self._online_y = [], []
        self._online_batch_size = 32
        self._n_online_updates = 0

    def warm_up(self, n_samples=5000, random_state=42):
        rng = np.random.default_rng(random_state)
        X, y = self._generate_synthetic_data(n_samples, rng)
        X_scaled = self._scaler.fit_transform(X)
        self._model.set_params(max_iter=200)
        self._model.fit(X_scaled, y)
        self._model.set_params(max_iter=1)
        self._is_fitted = True

    def observe(self, features, actual_future_fill):
        self._online_X.append(features)
        self._online_y.append(np.clip(actual_future_fill, 0.0, 1.0))
        if len(self._online_X) >= self._online_batch_size:
            self._flush()

    def _flush(self):
        if not self._online_X: return
        X = np.array(self._online_X); y = np.array(self._online_y)
        if not self._is_fitted:
            self._scaler.fit_transform(X); self._model.fit(self._scaler.transform(X), y)
            self._is_fitted = True
        else:
            self._model.partial_fit(self._scaler.transform(X), y)
        self._n_online_updates += 1
        self._online_X.clear(); self._online_y.clear()

    def predict(self, features):
        if not self._is_fitted: return 0.5
        X = np.array(features).reshape(1, -1)
        return float(np.clip(self._model.predict(self._scaler.transform(X))[0], 0.0, 1.0))

    def should_split(self, features): return self.predict(features) >= self.split_threshold
    def should_merge(self, features): return self.predict(features) <= self.merge_threshold

    def action(self, features):
        p = self.predict(features)
        if p >= self.split_threshold: return "split"
        if p <= self.merge_threshold: return "merge"
        return "none"

    def evaluate(self, X_test, y_test):
        if not self._is_fitted: return {}
        X_scaled = self._scaler.transform(X_test)
        y_pred   = np.clip(self._model.predict(X_scaled), 0.0, 1.0)
        mae  = float(np.mean(np.abs(y_pred - y_test)))
        rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
        within = float(np.mean(np.abs(y_pred - y_test) <= 0.05))
        to = y_test >= self.split_threshold; po = y_pred >= self.split_threshold
        tu = y_test <= self.merge_threshold; pu = y_pred <= self.merge_threshold
        def recall(t, p): return float((t & p).sum() / t.sum()) if t.sum() > 0 else 1.0
        def prec(t, p):   return float((t & p).sum() / p.sum()) if p.sum() > 0 else 1.0
        return {"mae": mae, "rmse": rmse, "accuracy_within_5pct": within,
                "overflow_recall": recall(to, po), "overflow_precision": prec(to, po),
                "underflow_recall": recall(tu, pu), "underflow_precision": prec(tu, pu),
                "n_test_samples": len(y_test), "n_online_updates": self._n_online_updates}

    def _generate_synthetic_data(self, n_samples, rng):
        k = self.lookahead_k
        curr = rng.uniform(0.1, 0.95, n_samples)
        ins  = rng.beta(2, 5, n_samples)
        dlt  = rng.beta(2, 8, n_samples)
        srch = np.clip(1.0 - ins - dlt, 0.0, 1.0)
        trend= rng.uniform(-0.02, 0.02, n_samples)
        par  = rng.uniform(0.3, 0.9, n_samples)
        left = rng.uniform(0.2, 0.9, n_samples)
        right= rng.uniform(0.2, 0.9, n_samples)
        dep  = rng.uniform(0.0, 0.5, n_samples)
        sp   = rng.uniform(0.0, 1.0, n_samples)
        mg   = rng.uniform(0.0, 1.0, n_samples)
        X = np.column_stack([curr, ins, dlt, srch, trend, par, left, right, dep, sp, mg])
        net = (ins - dlt) * k / 10.0
        y   = np.clip(curr + net + rng.normal(0, 0.03, n_samples), 0.0, 1.0)
        return X, y
