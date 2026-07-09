from effortpred.pancake import gap_h, gap2_h, pancake_goal, rand_h
from effortpred.pancake_features import (
    PANCAKE_FEATURE_NAMES, extract_pancake_features,
)


def test_goal_features():
    f = extract_pancake_features(pancake_goal(12), bound=4)
    assert set(f.keys()) == set(PANCAKE_FEATURE_NAMES)
    assert f["h_gap"] == 0 and f["h_gap2"] == 0 and f["h_rand"] == 0
    assert f["bound"] == 4
    assert f["slack_gap"] == 4 and f["slack_gap2"] == 4 and f["slack_rand"] == 4
    assert f["plate_gap"] == 0
    assert f["num_fixed_suffix"] == 12
    assert f["inversions"] == 0
    assert f["max_displacement"] == 0.0
    assert f["first_gap_pos"] == 12          # sentinel: no gap found


def test_hand_case():
    # (1, 0, 2, 3, 4): one gap (0,2); plate fine; suffix 2,3,4 fixed (3 long)
    s = (1, 0, 2, 3, 4)
    f = extract_pancake_features(s, bound=3)
    assert f["h_gap"] == 1
    assert f["h_gap2"] == 0                  # both gap adjacencies involve 0/1
    assert f["h_rand"] in (0, 1)             # one of the two, crc32-selected
    assert f["h_rand"] == rand_h(s)          # and consistent with the module
    assert f["slack_gap"] == 2
    assert f["plate_gap"] == 0
    assert f["num_fixed_suffix"] == 3
    assert f["inversions"] == 1
    assert f["first_gap_pos"] == 1           # gap between positions 1 and 2
    assert f["max_displacement"] == 1.0
    assert len(PANCAKE_FEATURE_NAMES) == 13
