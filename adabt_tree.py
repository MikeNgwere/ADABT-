"""ADABT — AI-Driven Adaptive B-Tree. AI active on DELETE only."""
from node_monitor import NodeMonitor
from fill_rate_predictor import FillRatePredictor

class BTreeNode:
    def __init__(self, t, leaf=False, depth=0):
        self.t = t; self.keys = []; self.children = []; self.leaf = leaf; self.depth = depth
        self.monitor = NodeMonitor(max_keys=2*t-1, depth=depth)

    @property
    def n(self): return len(self.keys)
    @property
    def is_full(self): return self.n >= 2*self.t - 1
    @property
    def fill_ratio(self):
        mx = 2*self.t - 1
        return self.n / mx if mx > 0 else 0.0

class ADABT:
    _PREDICT_INTERVAL = 10
    _MERGE_RISK_CEIL  = 0.50

    def __init__(self, t=3, predictor=None, enable_ai=True):
        if t < 2: raise ValueError("t must be >= 2")
        self.t = t; self.root = BTreeNode(t=t, leaf=True, depth=0)
        self.predictor = predictor or FillRatePredictor()
        self.enable_ai = enable_ai
        self._proactive_splits=0; self._reactive_splits=0
        self._proactive_merges=0; self._reactive_merges=0
        self._total_inserts=0;    self._total_deletes=0; self._total_searches=0

    def search(self, key): self._total_searches += 1; return self._search(self.root, key)

    def insert(self, key):
        self._total_inserts += 1
        if self.root.is_full:
            old = self.root; new = BTreeNode(t=self.t, leaf=False, depth=0)
            new.children.append(old); self._inc_depths(old)
            self._split_child(new, 0); self.root = new; self._reactive_splits += 1
        self._insert_nf(self.root, key)

    def delete(self, key):
        self._total_deletes += 1
        if not self.search(key): return
        self._delete(self.root, key)
        if len(self.root.keys)==0 and not self.root.leaf:
            self.root = self.root.children[0]

    def _search(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]: i += 1
        if i < len(node.keys) and key == node.keys[i]: return (node, i)
        if node.leaf: return None
        return self._search(node.children[i], key)

    def _insert_nf(self, node, key):
        if node.leaf:
            i = len(node.keys)-1; node.keys.append(None)
            while i >= 0 and key < node.keys[i]: node.keys[i+1]=node.keys[i]; i -= 1
            node.keys[i+1] = key; return
        i = len(node.keys)-1
        while i >= 0 and key < node.keys[i]: i -= 1
        i += 1; child = node.children[i]
        if child.is_full:
            self._split_child(node, i); self._reactive_splits += 1
            if key > node.keys[i]: i += 1
            child = node.children[i]
        self._insert_nf(child, key)

    def _split_child(self, parent, i):
        t = self.t; full = parent.children[i]; new = BTreeNode(t=t, leaf=full.leaf, depth=full.depth)
        mid = full.keys[t-1]; new.keys=full.keys[t:]; full.keys=full.keys[:t-1]
        if not full.leaf: new.children=full.children[t:]; full.children=full.children[:t]
        parent.children.insert(i+1, new); parent.keys.insert(i, mid)

    def _delete(self, node, key):
        t = self.t; i = 0
        while i < len(node.keys) and key > node.keys[i]: i += 1
        if i < len(node.keys) and node.keys[i]==key:
            if node.leaf: node.keys.pop(i)
            else: self._del_internal(node, i)
            return
        if node.leaf: return
        child = node.children[i]
        if self.enable_ai and child.n > t-1 and child.monitor.total_ops % self._PREDICT_INTERVAL==0 and child.fill_ratio <= self._MERGE_RISK_CEIL:
            feats = self._feats(child, node, i)
            if self.predictor.should_merge(feats): self._proactive_merges += 1
        if child.n < t:
            self._fix(node, i); self._reactive_merges += 1
            child = node.children[min(i, len(node.children)-1)]
        self._delete(child, key)

    def _del_internal(self, node, i):
        t = self.t; lc = node.children[i]; rc = node.children[i+1]
        if len(lc.keys) >= t:
            pred = self._pred(lc); node.keys[i]=pred; self._delete(lc, pred)
        elif len(rc.keys) >= t:
            succ = self._succ(rc); node.keys[i]=succ; self._delete(rc, succ)
        else:
            self._merge(node, i); self._delete(node.children[i], node.keys[i] if i < len(node.keys) else lc.keys[-1])

    def _fix(self, parent, i):
        t = self.t; child = parent.children[i]
        hl = i>0 and len(parent.children[i-1].keys)>=t
        hr = i<len(parent.children)-1 and len(parent.children[i+1].keys)>=t
        if hl:
            sib = parent.children[i-1]; child.keys.insert(0, parent.keys[i-1])
            parent.keys[i-1]=sib.keys.pop()
            if not sib.leaf: child.children.insert(0, sib.children.pop())
        elif hr:
            sib = parent.children[i+1]; child.keys.append(parent.keys[i])
            parent.keys[i]=sib.keys.pop(0)
            if not sib.leaf: child.children.append(sib.children.pop(0))
        elif i>0: self._merge(parent, i-1)
        elif i<len(parent.children)-1: self._merge(parent, i)

    def _merge(self, parent, i):
        l=parent.children[i]; r=parent.children[i+1]; sep=parent.keys.pop(i)
        parent.children.pop(i+1); l.keys.append(sep); l.keys.extend(r.keys)
        if not l.leaf: l.children.extend(r.children)

    def _pred(self, node):
        while not node.leaf: node=node.children[-1]
        return node.keys[-1]

    def _succ(self, node):
        while not node.leaf: node=node.children[0]
        return node.keys[0]

    def _inc_depths(self, node):
        node.depth+=1; node.monitor.depth=node.depth
        for c in node.children: self._inc_depths(c)

    def _feats(self, node, parent, i):
        mk=2*self.t-1; cf=node.n/mk if mk>0 else 0.5; pf=parent.n/mk if mk>0 else 0.5
        lf=(parent.children[i-1].n/mk) if i>0 else 0.5
        rf=(parent.children[i+1].n/mk) if i<len(parent.children)-1 else 0.5
        dn=min(node.depth/20.0,1.0)
        return [cf,0.05,0.60,0.05,cf-pf,pf,lf,rf,dn,0.8,0.2]

    def stats(self):
        ts=self._proactive_splits+self._reactive_splits
        tm=self._proactive_merges+self._reactive_merges
        return {"total_inserts":self._total_inserts,"total_deletes":self._total_deletes,
                "total_searches":self._total_searches,
                "proactive_splits":self._proactive_splits,"reactive_splits":self._reactive_splits,
                "proactive_merges":self._proactive_merges,"reactive_merges":self._reactive_merges,
                "overflow_elimination":round(self._proactive_splits/ts,4) if ts>0 else 0.0,
                "underflow_elimination":round(self._proactive_merges/tm,4) if tm>0 else 0.0}
