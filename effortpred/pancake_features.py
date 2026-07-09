"""Static features of a pancake state for effort prediction. All three
heuristic values (gap_h, gap2_h, rand_h) are legitimate observable features
regardless of which heuristic the search itself used."""

import numpy as np

from .pancake import gap2_h, gap_h, rand_h

PANCAKE_FEATURE_NAMES = [
    "h_gap", "h_gap2", "h_rand", "bound",
    "slack_gap", "slack_gap2", "slack_rand",
    "plate_gap", "num_fixed_suffix", "inversions",
    "max_displacement", "mean_displacement", "first_gap_pos",
]


def extract_pancake_features(state, bound):
    n = len(state)
    hg = gap_h(state)
    hg2 = gap2_h(state)
    hr = rand_h(state)
    plate_gap = int(state[n - 1] != n - 1)

    fixed = 0
    for i in range(n - 1, -1, -1):
        if state[i] == i:
            fixed += 1
        else:
            break

    inversions = sum(
        1 for i in range(n) for j in range(i + 1, n) if state[i] > state[j]
    )
    disp = np.array([abs(state[i] - i) for i in range(n)], dtype=float)

    first_gap = n                              # sentinel: no gap
    for i in range(n - 1):
        if abs(state[i] - state[i + 1]) > 1:
            first_gap = i
            break
    if first_gap == n and plate_gap:
        first_gap = n - 1

    return {
        "h_gap": hg,
        "h_gap2": hg2,
        "h_rand": hr,
        "bound": bound,
        "slack_gap": bound - hg,
        "slack_gap2": bound - hg2,
        "slack_rand": bound - hr,
        "plate_gap": plate_gap,
        "num_fixed_suffix": fixed,
        "inversions": inversions,
        "max_displacement": float(disp.max()),
        "mean_displacement": float(disp.mean()),
        "first_gap_pos": first_gap,
    }
