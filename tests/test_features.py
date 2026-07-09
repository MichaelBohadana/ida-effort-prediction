from effortpred.features import FEATURE_NAMES, extract_features
from effortpred.puzzle import goal_state


def test_goal_features():
    f = extract_features(goal_state(4), 4, bound=6)
    assert set(f.keys()) == set(FEATURE_NAMES)
    assert f["h_manhattan"] == 0
    assert f["bound"] == 6
    assert f["gap"] == 6
    assert f["misplaced"] == 0
    assert f["inversions"] == 0
    assert f["linear_conflict_pairs"] == 0
    assert f["blank_row"] == 0 and f["blank_col"] == 0
    assert f["blank_degree"] == 2          # corner cell
    assert f["tile_md_max"] == 0.0


def test_hand_case_features():
    # 3x3: swap tiles 1 and 2 -> state (0, 2, 1, 3..8)
    s = (0, 2, 1, 3, 4, 5, 6, 7, 8)
    f = extract_features(s, 3, bound=10)
    assert f["h_manhattan"] == 2           # each swapped tile is 1 away
    assert f["gap"] == 8
    assert f["misplaced"] == 2
    assert f["inversions"] == 1            # tile order: 2,1,3,...,8
    assert f["linear_conflict_pairs"] == 1
    assert f["blank_degree"] == 2


def test_no_nans_and_stable_order():
    f = extract_features((0, 2, 1, 3, 4, 5, 6, 7, 8), 3, bound=10)
    row = [f[k] for k in FEATURE_NAMES]
    assert all(v == v for v in row)        # no NaNs
    assert len(FEATURE_NAMES) == 12
