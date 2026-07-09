"""Instance-measured 1-step transition features (Phase 3c).

CDP_1 uses *space-averaged* parent→child heuristic transitions (the
conditional matrix p(v|vp) is estimated over random states).  These probe
features instead measure the ACTUAL transitions from a specific start state
in one step — the root's children have g=1, so each child's f-value is
1 + h(child).

PROBE_FEATURE_NAMES — 12 features, 6 per heuristic (gap, rand):
  probe_{h}_min_child   — min  h(child) over all root children
  probe_{h}_mean_child  — mean h(child)
  probe_{h}_max_child   — max  h(child)
  probe_{h}_n_improving — #children with h(child) < h(parent)
  probe_{h}_n_worsening — #children with h(child) > h(parent)
  probe_{h}_n_fertile   — #children whose f = 1 + h(child) <= bound
                          (they would be expanded by IDA* in this iteration)

The root has no parent, so ALL n-1 flips (k=2..n) are generated — this
matches the node-counter semantics that also counts all root children.

GAP is consistent (changes by at most 1 per flip), so
  probe_gap_max_child  <= gap_h(state) + 1
  probe_gap_min_child  >= gap_h(state) - 1
(verified in tests).  rand_h is inconsistent: it selects between gap_h and
gap2_h by crc32(state), so a single flip can cause a jump > 1 (also verified
in tests).
"""

from .pancake import flip, gap_h, rand_h

# Exact order required (tests and probe_eval.py rely on it).
PROBE_FEATURE_NAMES = [
    f"probe_{h}_{s}"
    for h in ("gap", "rand")
    for s in ("min_child", "mean_child", "max_child",
              "n_improving", "n_worsening", "n_fertile")
]  # 12 names


def extract_probe_features(state: tuple, bound: int) -> dict:
    """Compute all 12 probe features for *state* under *bound*.

    Parameters
    ----------
    state:
        A pancake permutation (tuple of ints).
    bound:
        The current IDA* bound (depth limit).

    Returns
    -------
    dict mapping each name in PROBE_FEATURE_NAMES to its value.
    """
    n = len(state)
    # ALL flips from the root (no parent, no pruning).
    children = [flip(state, k) for k in range(2, n + 1)]

    result = {}
    for h_name, h_fn in (("gap", gap_h), ("rand", rand_h)):
        hp = h_fn(state)
        child_hs = [h_fn(child) for child in children]

        n_children = len(child_hs)
        result[f"probe_{h_name}_min_child"] = min(child_hs)
        result[f"probe_{h_name}_mean_child"] = sum(child_hs) / n_children
        result[f"probe_{h_name}_max_child"] = max(child_hs)
        result[f"probe_{h_name}_n_improving"] = sum(1 for h in child_hs if h < hp)
        result[f"probe_{h_name}_n_worsening"] = sum(1 for h in child_hs if h > hp)
        # Fertile: the child's f = g(1) + h(child) must not exceed the bound.
        result[f"probe_{h_name}_n_fertile"] = sum(1 for h in child_hs if 1 + h <= bound)

    return result
