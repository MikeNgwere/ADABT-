"""ADABT Node Monitor — tracks per-node structural features for the Fill-Rate Predictor."""
from collections import deque

class NodeMonitor:
    WINDOW_SIZE = 100
    HISTORY_SIZE = 20
    INSERT = 0; DELETE = 1; SEARCH = 2

    def __init__(self, max_keys: int, depth: int = 0):
        self.max_keys = max_keys
        self.depth = depth
        self._ops = deque(maxlen=self.WINDOW_SIZE)
        self._fill_history = deque(maxlen=self.HISTORY_SIZE)
        self._total_ops = 0
        self._ops_since_split = 999
        self._ops_since_merge = 999

    def record_insert(self):
        self._ops.append(self.INSERT); self._total_ops += 1
        self._ops_since_split += 1; self._ops_since_merge += 1

    def record_delete(self):
        self._ops.append(self.DELETE); self._total_ops += 1
        self._ops_since_split += 1; self._ops_since_merge += 1

    def record_search(self):
        self._ops.append(self.SEARCH); self._total_ops += 1

    def record_split(self): self._ops_since_split = 0
    def record_merge(self): self._ops_since_merge = 0

    def snapshot_fill(self, current_keys: int):
        ratio = current_keys / self.max_keys if self.max_keys > 0 else 0.0
        self._fill_history.append(ratio)

    def get_features(self, current_keys, parent_keys=None, left_sib_keys=None, right_sib_keys=None):
        w = len(self._ops) or 1
        ins = sum(1 for op in self._ops if op == self.INSERT)
        dlt = sum(1 for op in self._ops if op == self.DELETE)
        srch= sum(1 for op in self._ops if op == self.SEARCH)
        curr = current_keys / self.max_keys if self.max_keys > 0 else 0.0
        par  = (parent_keys / self.max_keys) if parent_keys is not None else 0.5
        left = (left_sib_keys / self.max_keys) if left_sib_keys is not None else 0.5
        right= (right_sib_keys / self.max_keys) if right_sib_keys is not None else 0.5
        depth_n = min(self.depth / 20.0, 1.0)
        split_n = min(self._ops_since_split / 500.0, 1.0)
        merge_n = min(self._ops_since_merge / 500.0, 1.0)
        return [curr, ins/w, dlt/w, srch/w, self._compute_trend(), par, left, right, depth_n, split_n, merge_n]

    def _compute_trend(self):
        h = list(self._fill_history); n = len(h)
        if n < 3: return 0.0
        xm = (n-1)/2.0; ym = sum(h)/n
        num = sum((i-xm)*(h[i]-ym) for i in range(n))
        den = sum((i-xm)**2 for i in range(n))
        return num/den if den != 0 else 0.0

    @property
    def total_ops(self): return self._total_ops
