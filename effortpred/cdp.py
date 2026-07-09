"""CDP_1 — Conditional Distribution Prediction (Zahavi, Felner, Burch &
Holte, JAIR 2010, Eq. 5-6).

    N~_i(v)   = sum_{vp=0}^{d-(i-1)} N~_{i-1}(vp) * b_{i-1} * p(v|vp)
    N~_0(v)   = 1 if v == h(start) else 0
    CDP_1(s,d) = sum_{i=0}^{d} sum_{v=0}^{d-i} N~_i(v)

Restricting vp <= d-(i-1) at every step ensures (even for inconsistent
heuristics) that a node is counted at level i only if ALL its ancestors were
expanded — the property KRE lacks. Branching is passed explicitly and is
exact for the pancake (b_root = n-1, b_rest = n-2); the paper's sampled
b_vp is unnecessary here.

Matrix convention: cond[v, vp] = p(v | vp), columns normalized (Task 5).
"""

import numpy as np


def cdp1_predict(h_start, bound, cond, b_root, b_rest):
    """Predicted number of nodes one bounded DFS iteration expands."""
    H = cond.shape[0]
    if h_start > bound:
        return 0.0
    if h_start >= H:
        # Silent 0.0 here would drop the root term of Eq. 6 (an infidelity
        # to CDP_1) — fail loudly instead. Unreachable in our experiments
        # (matrices are built with h_max = n >= max heuristic value); flagged
        # by the 2026-07-03 cross-model (Codex) audit.
        raise ValueError(
            f"h_start={h_start} outside conditional-matrix support (H={H}); "
            "build the matrix with a larger h_max")

    prev = np.zeros(H)      # N~_{i-1}(v)
    prev[h_start] = 1.0
    total = 1.0             # level 0: the root, fertile since h_start <= bound

    for i in range(1, bound + 1):
        vp_cap = bound - (i - 1)              # fertile parents: vp <= d-(i-1)
        parents = prev.copy()
        if vp_cap + 1 < H:
            parents[vp_cap + 1:] = 0.0
        b = b_root if i == 1 else b_rest
        cur = cond @ (parents * b)            # N~_i(v) for all v (generated)
        expanded_cap = bound - i              # expanded at level i: v <= d-i
        total += float(cur[: expanded_cap + 1].sum())
        prev = cur

    return total
