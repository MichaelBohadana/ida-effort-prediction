"""The KRE formula (Korf, Reid & Edelkamp 2001), as taught in Lecture 5:

    N(b, d, P) = sum_{i=0}^{d} N_i * P(d - i)

where N_i is the number of depth-i nodes of the brute-force tree and P is the
heuristic-value distribution. We use the EXACT per-depth tree sizes from the
blank-position recurrence (tree_size.py) instead of the b^i approximation,
so the only approximations left are (a) P itself and (b) KRE's modelling
assumptions. KRE deliberately ignores h(start) — its known weakness on single
start states, which is the hook of this project.

Predicts the number of nodes ONE bounded DFS iteration expands (the same
quantity count_expansions() measures).
"""

from .tree_size import nodes_per_depth


def kre_predict(state, n, bound, dist):
    ni = nodes_per_depth(state.index(0), n, bound)
    return float(sum(ni[i] * dist.P(bound - i) for i in range(bound + 1)))
