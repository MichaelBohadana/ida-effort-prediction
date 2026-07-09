"""Exact brute-force tree sizes and the KRE formula for the pancake puzzle.

The parent-pruned pancake tree has UNIFORM branching (n-1 at the root, n-2
below), so N_i has a closed form and — unlike the sliding puzzle — does not
depend on the start state at all. KRE therefore issues literally one
prediction per (n, bound): the purest form of its start-state blindness.
"""


def nodes_per_depth_pancake(n, max_depth):
    result = [1]
    if max_depth >= 1:
        result.append(n - 1)
    for _ in range(2, max_depth + 1):
        result.append(result[-1] * (n - 2))
    return result


def kre_predict_pancake(n, bound, dist):
    ni = nodes_per_depth_pancake(n, bound)
    return float(sum(ni[i] * dist.P(bound - i) for i in range(bound + 1)))
