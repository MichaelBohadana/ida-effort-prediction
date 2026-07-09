# The core library (`effortpred/`)

The `effortpred/` package holds all the reusable logic. The scripts in
`scripts/` import from here and add only command-line handling and file output.
This page walks through the modules by role. For the concepts behind them, read
[overview.md](overview.md) first.

The modules fall into five groups: the two puzzle domains, the node counters
that produce labels, the analytic predictors, the feature builders, and the
shared model and metric code.

## Domains

These define the state spaces, the moves, and the heuristics.

- **`puzzle.py`** is the generic NxN sliding-tile puzzle. A state is a tuple of
  length `n*n`, where `state[cell]` is the tile at that cell and `0` is the
  blank. It gives the legal moves, the neighbor tables, and random-state
  generation. The 8-puzzle (used for exhaustive checks) and the 15-puzzle are
  both instances of this.
- **`heuristic.py`** is the Manhattan-distance heuristic for the sliding puzzle,
  plus a linear-conflict pair count that is used only as a feature. Manhattan
  distance here is admissible and consistent.
- **`pancake.py`** is the pancake domain. A state is a permutation of `0..n-1`,
  a move flips the first `k` items, and parent pruning forbids repeating the
  last flip. It also defines the three pancake heuristics: `gap_h` (GAP),
  `gap2_h` (GAP-2), and `rand_h` (the inconsistent RAND). The tests check all
  three exhaustively on the 7-pancake.

## Node counters (the label source)

A label is the true node count for one bounded pass. These modules compute it by
running the bounded DFS directly.

- **`count.py`** counts expansions for the sliding puzzle. It has a slow
  reference version (obviously correct, used as ground truth in tests) and a
  fast version. Both count the fertile nodes of the parent-pruned tree with no
  goal stop.
- **`pancake_count.py`** is the same counter for the pancake. Its heuristic is
  pluggable, so the one function labels data under GAP and under the
  inconsistent RAND. Under an inconsistent heuristic a fertile node can be
  unreachable because an ancestor was pruned, and the DFS handles that
  naturally.

## Analytic predictors

The KRE and CDP1 formulas, plus the tree-size and distribution helpers they
need.

- **`tree_size.py`** computes the exact per-depth node counts of the sliding
  puzzle's parent-pruned tree from a recurrence over blank positions. This
  replaces the usual `b^i` approximation with the true count.
- **`distribution.py`** estimates `P(x)`, the distribution of heuristic values,
  which KRE needs. It samples uniform random states and also offers an
  equilibrium-reweighted variant.
- **`kre.py`** is the KRE formula for the sliding puzzle. It sums the per-depth
  tree sizes weighted by the value distribution. It deliberately ignores the
  start state's heuristic value, which is its known weak point.
- **`pancake_tree.py`** gives the exact tree sizes and the KRE formula for the
  pancake. Because pancake branching is uniform, KRE issues literally one
  prediction per bound.
- **`conditional.py`** estimates the conditional matrices `p(v | vp)`, the
  probability that a child has heuristic value `v` given its parent has value
  `vp`. It samples with incoming-flip exclusion so the tallied children match
  the parent-pruned tree.
- **`cdp.py`** is the CDP1 recursion. It propagates the conditional value
  distribution down the levels and sums the expanded nodes. Capping the parent
  value at each level is what lets it stay correct under inconsistent
  heuristics.

## Feature builders (for the learned models)

These turn a state and a bound into the numeric vectors the regressors read.

- **`features.py`** builds the static features for the 15-puzzle: heuristic
  value, bound, slack, inversions, linear conflicts, misplaced tiles, the
  blank's row, column, and degree, and tile-distance statistics. It explicitly
  leaves out generation metadata so nothing about how a state was made can leak.
- **`pancake_features.py`** builds the 13 static features for the pancake: all
  three heuristic values and their slacks, plus structural features like the
  plate gap, the solved-suffix length, inversions, and displacement. All three
  heuristic values are legitimate inputs no matter which heuristic guided the
  search.
- **`probe_features.py`** builds the one-step lookahead features. It expands the
  start state once and summarizes the children's heuristic values (min, mean,
  max, and how many improve or worsen). Unlike CDP1's space-averaged
  transitions, these are measured on the actual start state. They cost one
  expansion, so they are a declared operating point, not a free input.

## Models and metrics

Shared machinery used by every experiment.

- **`models.py`** holds the leakage-safe data split (70/15/15 grouped by state,
  so no state lands on two sides of a split), the two simple baselines, the GBM,
  and the MLP. The target everywhere is `log10(nodes)`.
- **`metrics.py`** holds the evaluation metrics in `log10` space (RMSE, median
  factor error, the fraction within a factor of two, and Spearman rank
  correlation) and the cluster-bootstrap standard errors, which resample whole
  states rather than rows.
- **`__init__.py`** is empty and just marks the package.
