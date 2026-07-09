import random

import numpy as np

from effortpred.conditional import (
    sample_conditional_matrix, exact_conditional_matrix,
)
from effortpred.pancake import gap_h, rand_h


def test_exact_matrix_properties_consistent():
    M = exact_conditional_matrix(7, gap_h, h_max=7)
    # every column with mass sums to 1
    cols = M.sum(axis=0)
    for vp in range(8):
        assert cols[vp] == 0 or abs(cols[vp] - 1.0) < 1e-12
    # CONSISTENCY signature: gap_h changes by at most 1 per move, so
    # p(v|vp) = 0 whenever |v - vp| > 1 (the matrix is tridiagonal)
    for v in range(8):
        for vp in range(8):
            if abs(v - vp) > 1:
                assert M[v, vp] == 0.0
    # explicit orientation guard: the matrix must NOT be symmetric — a
    # transposed (row-normalized) matrix would slip past the tridiagonal check
    assert not np.allclose(M, M.T), "matrix should not be symmetric — orientation check"


def test_exact_matrix_properties_inconsistent():
    M = exact_conditional_matrix(7, rand_h, h_max=7)
    cols = M.sum(axis=0)
    for vp in range(8):
        assert cols[vp] == 0 or abs(cols[vp] - 1.0) < 1e-12
    # INCONSISTENCY signature: some mass beyond the tridiagonal
    off = sum(M[v, vp] for v in range(8) for vp in range(8) if abs(v - vp) > 1)
    assert off > 0.01


def test_sampled_matches_exact():
    exact = exact_conditional_matrix(7, gap_h, h_max=7)
    est = sample_conditional_matrix(7, gap_h, 100_000, random.Random(50), h_max=7)
    # compare only columns that exist in both; entrywise tolerance
    for vp in range(8):
        if exact[:, vp].sum() > 0 and est[:, vp].sum() > 0:
            assert np.max(np.abs(exact[:, vp] - est[:, vp])) < 0.03


def test_sampling_deterministic():
    a = sample_conditional_matrix(6, gap_h, 5_000, random.Random(51), h_max=6)
    b = sample_conditional_matrix(6, gap_h, 5_000, random.Random(51), h_max=6)
    assert np.array_equal(a, b)
