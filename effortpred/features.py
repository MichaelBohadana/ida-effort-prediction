"""Static state features for effort prediction.

Deliberately excluded: walk_length (generation metadata — not observable for
an arbitrary state; would leak difficulty).
"""

import numpy as np

from .heuristic import linear_conflict_pairs, manhattan, md_table
from .puzzle import neighbors_table

FEATURE_NAMES = [
    "h_manhattan", "bound", "gap",
    "linear_conflict_pairs", "misplaced", "inversions",
    "blank_row", "blank_col", "blank_degree",
    "tile_md_max", "tile_md_mean", "tile_md_std",
]


def extract_features(state, n, bound):
    md = md_table(n)
    h = manhattan(state, n)
    b = state.index(0)
    tiles = [t for t in state if t != 0]
    inversions = sum(
        1
        for i in range(len(tiles))
        for j in range(i + 1, len(tiles))
        if tiles[i] > tiles[j]
    )
    tile_mds = np.array(
        [md[tile][cell] for cell, tile in enumerate(state) if tile != 0],
        dtype=float,
    )
    return {
        "h_manhattan": h,
        "bound": bound,
        "gap": bound - h,
        "linear_conflict_pairs": linear_conflict_pairs(state, n),
        "misplaced": int((tile_mds > 0).sum()),
        "inversions": inversions,
        "blank_row": b // n,
        "blank_col": b % n,
        "blank_degree": len(neighbors_table(n)[b]),
        "tile_md_max": float(tile_mds.max()),
        "tile_md_mean": float(tile_mds.mean()),
        "tile_md_std": float(tile_mds.std()),
    }
