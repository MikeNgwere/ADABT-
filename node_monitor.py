"""
ADABT - Node Monitor
Tracks per-node operation history and extracts features for the fill-rate predictor.
"""

from collections import deque
import time


class NodeMonitor:
    """
    Attached to every B-Tree node. Records a sliding window of operations
    and computes features for the fill-rate predictor.

    Features extracted:
        1.  current_fill_ratio       - keys / max_keys
        2.  insert_rate              - inserts / window size
        3.  delete_rate              - deletes / window size
        4.  search_rate              - searches / window size
        5.  fill_trend               - linear slope of recent fill ratios
        6.  parent_fill_ratio        - parent node fill (0.5 if root)
        7.  left_sibling_fill_ratio  - left sibling fill (0.5 if none)
        8.  right_sibling_fill_ratio - right sibling fill (0.5 if none)
        9.  depth                    - level in tree (root = 0)
        10. time_since_last_split    - ops since last split on this node
        11. time_since_last_merge    - ops since last merge on this node
    """

    WINDOW_SIZE = 100       # sliding window of last N operations
    HISTORY_SIZE = 20       # fill ratio snapshots for trend calculation

    # Operation type codes
    INSERT = 0
    DELETE = 1
    SEARCH = 2

    def __init__(self, max_keys: int, depth: int = 0):
        self.max_keys = max_keys            # 2t - 1
        self.depth = depth

        # Sliding operation window
        self._ops = deque(maxlen=self.WINDOW_SIZE)

        # Fill ratio history for trend
        self._fill_history = deque(maxlen=self.HISTORY_SIZE)

        # Time counters (in op units)
        self._total_ops = 0
        self._ops_since_split = 999
        self._ops_since_merge = 999

    # ------------------------------------------------------------------ #
    # Event recording                                                      #
    # ------------------------------------------------------------------ #

    def record_insert(self):
        self._ops.append(self.INSERT)
        self._total_ops += 1
        self._ops_since_split += 1
        self._ops_since_merge += 1

    def record_delete(self):
        self._ops.append(self.DELETE)
        self._total_ops += 1
        self._ops_since_split += 1
        self._ops_since_merge += 1

    def record_search(self):
        self._ops.append(self.SEARCH)
        self._total_ops += 1
        self._ops_since_split += 1
        self._ops_since_merge += 1

    def record_split(self):
        self._ops_since_split = 0

    def record_merge(self):
        self._ops_since_merge = 0

    def snapshot_fill(self, current_keys: int):
        """Call after every structural change to log the fill ratio."""
        ratio = current_keys / self.max_keys if self.max_keys > 0 else 0.0
        self._fill_history.append(ratio)

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    def get_features(
        self,
        current_keys: int,
        parent_keys: int = None,
        left_sib_keys: int = None,
        right_sib_keys: int = None,
    ) -> list:
        """
        Returns a feature vector of length 11.
        All values normalised to [0, 1] where possible.
        """
        w = len(self._ops) or 1  # avoid division by zero

        insert_count = sum(1 for op in self._ops if op == self.INSERT)
        delete_count = sum(1 for op in self._ops if op == self.DELETE)
        search_count = sum(1 for op in self._ops if op == self.SEARCH)

        current_fill = current_keys / self.max_keys if self.max_keys > 0 else 0.0
        insert_rate  = insert_count / w
        delete_rate  = delete_count / w
        search_rate  = search_count / w

        fill_trend = self._compute_fill_trend()

        parent_fill = (
            (parent_keys / self.max_keys) if parent_keys is not None else 0.5
        )
        left_fill = (
            (left_sib_keys / self.max_keys) if left_sib_keys is not None else 0.5
        )
        right_fill = (
            (right_sib_keys / self.max_keys) if right_sib_keys is not None else 0.5
        )

        # Normalise depth (cap at 20 levels)
        depth_norm = min(self.depth / 20.0, 1.0)

        # Normalise time-since events (cap at 500 ops)
        split_norm = min(self._ops_since_split / 500.0, 1.0)
        merge_norm = min(self._ops_since_merge / 500.0, 1.0)

        return [
            current_fill,
            insert_rate,
            delete_rate,
            search_rate,
            fill_trend,
            parent_fill,
            left_fill,
            right_fill,
            depth_norm,
            split_norm,
            merge_norm,
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_fill_trend(self) -> float:
        """
        Linear slope of recent fill ratios using least-squares.
        Positive = filling up, negative = emptying.
        Returns 0.0 if not enough history.
        """
        h = list(self._fill_history)
        n = len(h)
        if n < 3:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = sum(h) / n
        numerator   = sum((i - x_mean) * (h[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0

    @property
    def total_ops(self) -> int:
        return self._total_ops
