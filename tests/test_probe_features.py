"""Tests for effortpred.probe_features (Phase 3c).

Written BEFORE the implementation (TDD).  The four test cases check:
  1. Hand-computed values on a small goal state (n=5, bound=2).
  2. GAP consistency: max/min child gap_h can move by at most 1.
  3. rand_h inconsistency witness: a jump > 1 in some direction exists.
  4. n_fertile is monotone non-decreasing in the bound parameter.
"""

import random

import numpy as np
import pytest

from effortpred.pancake import flip, gap_h, rand_h
from effortpred.probe_features import PROBE_FEATURE_NAMES, extract_probe_features


# ---------------------------------------------------------------------------
# Test 1 — hand case: n=5 goal state (0,1,2,3,4), bound=2
# ---------------------------------------------------------------------------

def test_hand_case_goal_n5_bound2():
    """Verify every probe feature for the goal state of n=5 with bound=2.

    Parent state: (0, 1, 2, 3, 4)   gap_h = 0   rand_h = 0

    ALL four flips (k=2..5) — the root has no parent, so all are valid:
      flip(s, 2) = (1, 0, 2, 3, 4)   gap_h=1   rand_h=1
      flip(s, 3) = (2, 1, 0, 3, 4)   gap_h=1   rand_h=0
      flip(s, 4) = (3, 2, 1, 0, 4)   gap_h=1   rand_h=0
      flip(s, 5) = (4, 3, 2, 1, 0)   gap_h=1   rand_h=0

    GAP child values: [1, 1, 1, 1]  —  hp = 0
      min_child     = 1
      mean_child    = 1.0
      max_child     = 1
      n_improving   = 0   (no child_h < 0)
      n_worsening   = 4   (all child_h > 0)
      n_fertile     = 4   (all: 1 + 1 = 2 <= bound=2)

    RAND child values: [1, 0, 0, 0]  —  hp = 0
      min_child     = 0
      mean_child    = 0.25
      max_child     = 1
      n_improving   = 0   (no child_h < 0)
      n_worsening   = 1   (only flip-2 gives rand_h=1 > 0)
      n_fertile     = 4   (1+1=2<=2  and  1+0=1<=2  — all qualify)
    """
    state = (0, 1, 2, 3, 4)
    bound = 2
    f = extract_probe_features(state, bound)

    # --- GAP features ---
    assert f["probe_gap_min_child"] == 1
    assert abs(f["probe_gap_mean_child"] - 1.0) < 1e-9
    assert f["probe_gap_max_child"] == 1
    assert f["probe_gap_n_improving"] == 0   # goal has gap_h=0; no child < 0
    assert f["probe_gap_n_worsening"] == 4
    assert f["probe_gap_n_fertile"] == 4

    # --- RAND features ---
    assert f["probe_rand_min_child"] == 0
    assert abs(f["probe_rand_mean_child"] - 0.25) < 1e-9
    assert f["probe_rand_max_child"] == 1
    assert f["probe_rand_n_improving"] == 0  # goal rand_h=0; no child < 0
    assert f["probe_rand_n_worsening"] == 1
    assert f["probe_rand_n_fertile"] == 4


def test_probe_feature_names():
    """PROBE_FEATURE_NAMES must be exactly 12 strings in the specified order."""
    expected = [
        f"probe_{h}_{s}"
        for h in ("gap", "rand")
        for s in ("min_child", "mean_child", "max_child",
                  "n_improving", "n_worsening", "n_fertile")
    ]
    assert PROBE_FEATURE_NAMES == expected
    assert len(PROBE_FEATURE_NAMES) == 12


def test_hand_case_keys_match_names():
    """extract_probe_features must return exactly the keys listed in PROBE_FEATURE_NAMES."""
    state = (0, 1, 2, 3, 4)
    f = extract_probe_features(state, bound=2)
    assert set(f.keys()) == set(PROBE_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Test 2 — GAP consistency: max/min child gap_h shifts by at most 1
# ---------------------------------------------------------------------------

def test_gap_consistency_property():
    """For 200 random 12-pancake states: probe_gap_max_child <= gap_h(state)+1
    and probe_gap_min_child >= gap_h(state)-1.

    This follows directly from GAP being a consistent heuristic whose value
    changes by at most 1 per flip (each flip changes exactly one adjacency).
    """
    rng = random.Random(2024)
    n = 12
    states = [tuple(rng.sample(range(n), n)) for _ in range(200)]

    for state in states:
        f = extract_probe_features(state, bound=10)
        hp = gap_h(state)
        assert f["probe_gap_max_child"] <= hp + 1, (
            f"state={state}: probe_gap_max_child={f['probe_gap_max_child']} > hp+1={hp+1}")
        assert f["probe_gap_min_child"] >= hp - 1, (
            f"state={state}: probe_gap_min_child={f['probe_gap_min_child']} < hp-1={hp-1}")


# ---------------------------------------------------------------------------
# Test 3 — rand_h inconsistency witness: jump > 1 in at least one state
# ---------------------------------------------------------------------------

def test_rand_inconsistency_witness():
    """Over 200 random 12-pancake states, at least one must have a child
    whose rand_h differs from the parent rand_h by more than 1 — proving
    rand_h is genuinely inconsistent (the GAP consistency property does NOT
    hold for rand_h).

    We check via min/max child values rather than counting features.
    """
    rng = random.Random(2025)
    n = 12
    states = [tuple(rng.sample(range(n), n)) for _ in range(200)]

    found = False
    for state in states:
        f = extract_probe_features(state, bound=10)
        hp = rand_h(state)
        if f["probe_rand_max_child"] - hp > 1 or hp - f["probe_rand_min_child"] > 1:
            found = True
            break

    assert found, (
        "Expected at least one state where rand_h max/min child differs by >1 "
        "from parent (inconsistency). None found in 200 samples — check rand_h."
    )


# ---------------------------------------------------------------------------
# Test 4 — n_fertile is monotone non-decreasing in bound
# ---------------------------------------------------------------------------

def test_n_fertile_monotone_in_bound():
    """For a fixed state, n_fertile (both gap and rand variants) must be
    non-decreasing as bound increases: a larger bound makes strictly at-least
    as many children fertile (child f = g(=1) + h, fertile iff f <= bound).
    """
    rng = random.Random(2026)
    n = 12
    # Test 20 random states over bounds 0..15
    states = [tuple(rng.sample(range(n), n)) for _ in range(20)]

    for state in states:
        prev_fertile_gap = -1
        prev_fertile_rand = -1
        for bound in range(0, 16):
            f = extract_probe_features(state, bound=bound)
            assert f["probe_gap_n_fertile"] >= prev_fertile_gap, (
                f"state={state}, bound={bound}: gap n_fertile decreased from "
                f"{prev_fertile_gap} to {f['probe_gap_n_fertile']}"
            )
            assert f["probe_rand_n_fertile"] >= prev_fertile_rand, (
                f"state={state}, bound={bound}: rand n_fertile decreased from "
                f"{prev_fertile_rand} to {f['probe_rand_n_fertile']}"
            )
            prev_fertile_gap = f["probe_gap_n_fertile"]
            prev_fertile_rand = f["probe_rand_n_fertile"]
