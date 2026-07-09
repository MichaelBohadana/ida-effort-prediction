# Core library reference (`effortpred/`)

File-by-file reference for the `effortpred/` package, with the internal
mechanisms of each module. The scripts in `scripts/` import these functions and
add argument parsing and file output. The tests in `tests/` gate each module
(see [tests.md](tests.md)).

Six roles: the sliding-tile domain, the pancake domain, the node counters that
produce labels, the analytic predictors, the feature extractors, and the shared
models and metrics.

## Sliding-tile domain

### `puzzle.py`
Generic NxN sliding-tile puzzle. A state is a tuple of length `n*n` where
`state[cell]` is the tile at that cell and `0` is the blank. Cells are indexed
row-major, so `cell = row * n + col`. The goal is `(0, 1, ..., n*n-1)`, which
puts the goal cell of tile `t` at cell `t`.

- `goal_state(n)`, `neighbors_table(n)`: the goal tuple and, for each cell, its
  adjacent cells.
- `apply_move(state, to_cell)`, `successors(state, n)`: slide the tile at
  `to_cell` into the blank, and enumerate all one-move successors.
- `permutation_parity(perm)`, `is_solvable(state, n)`: the parity-based
  solvability rule. A sliding-tile state is solvable if and only if its
  permutation parity matches the blank-row parity condition. Checked against
  breadth-first reachability in the tests.
- `random_walk_state(n, length, rng)`: a state reached by a random walk from the
  goal. Used only by the pilot generator.
- `random_solvable_state(n, rng)`: a uniformly random solvable state. This is the
  state source for the 15-puzzle generalization dataset.

### `heuristic.py`
Manhattan distance and a linear-conflict feature.

- `md_table(n)`: a table where `md_table(n)[tile][cell]` is the Manhattan
  distance of `tile` from its goal cell when it sits at `cell`. The fast counter
  uses this table for incremental updates. The blank row is all zeros.
- `manhattan(state, n)`: sum of per-tile Manhattan distances. Admissible and
  consistent on this domain.
- `linear_conflict_pairs(state, n)`: count of tile pairs that share a goal row or
  column but are in reversed order. Used as a feature, not added to the
  heuristic.

## Pancake domain

### `pancake.py`
The pancake puzzle and its three heuristics. A state is a tuple permutation of
`0..n-1` with goal `(0, 1, ..., n-1)`.

- `flip(state, k)`: reverse the first `k` elements. Implemented as
  `state[k-1::-1] + state[k:]`.
- `successors_pancake(state, prev_k)`: the pairs `(k, flip(state, k))` for
  `k = 2..n` excluding `prev_k`. Passing `prev_k=0` disables pruning (root).
- `dual_state(state)`: the inverse permutation, used to test dual evaluation.
- `gap_h(state)`: GAP. Counts adjacent positions whose values differ by more than
  one, plus one for the bottom pancake if it is not the largest (the virtual
  plate). A single flip changes exactly one adjacency, so GAP changes by at most
  one per move, which makes it consistent and admissible.
- `gap2_h(state)`: GAP that ignores any adjacency involving pancake 0 or 1.
  Strictly weaker, still consistent and admissible.
- `rand_h(state)`: returns `gap_h(state)` or `gap2_h(state)`, selected by
  `zlib.crc32(bytes(state)) & 1`. CRC32 is used instead of Python's `hash`, which
  is salted per process. Each component is admissible, so RAND is admissible, but
  the values consulted along a path can jump by more than one, so RAND is
  inconsistent. The self-dual lemma (GAP equals its own dual) is why dual
  evaluation cannot produce inconsistency here and random selection is used
  instead.
- `random_pancake_state(n, rng)`: a uniformly random permutation. The state source
  for the pancake datasets.
- `exact_pancake_distribution(n, h_fn)`, `estimate_pancake_distribution(n, h_fn,
  n_samples, rng)`: the exact (enumerated) and sampled heuristic-value
  distributions. KRE on the pancake reads the sampled one.

## Node counters (label source)

The label of a dataset row is the exact number of nodes one bounded pass expands.
It is computed by a bounded depth-first search over the parent-pruned tree, not
estimated. A node is counted (expanded) when `f = g + h <= bound`. The search
does not stop at the goal, so it counts every fertile node.

### `count.py`
Bounded-DFS counter for the sliding puzzle. Two implementations that must agree.

- `count_expansions_reference(state, n, bound)`: the slow ground truth. It
  recomputes Manhattan distance in full at every node and identifies the parent by
  comparing whole states. Used only in the tests.
- `count_expansions(state, n, bound, cap=None)`: the fast counter used for all
  generation. It mutates one board array in place, tracks the blank cell, and
  updates the heuristic incrementally: moving `tile` from cell `to` into the blank
  at `blank` changes the heuristic by `md[tile][blank] - md[tile][to]`, so a child
  is entered only when `g + 1 + h + dh <= bound`. Parent pruning is done by
  skipping the cell the blank just came from (`prev`). It returns
  `(count, censored)`. If `cap` is given and the running count reaches it, the
  search raises `CapExceeded` and returns `(cap, True)`.

### `pancake_count.py`
Bounded-DFS counter for the pancake, same semantics.

- `count_expansions_pancake(state, bound, h_fn, cap=None)`: the depth-first search
  iterates over flips `k = 2..n`, skips `k == prev_k` (parent pruning), and enters
  a child when `g + 1 + h_fn(child) <= bound`. The heuristic `h_fn` is a
  parameter, so the same counter labels the consistent `gap_h` set and the
  inconsistent `rand_h` set. Under an inconsistent heuristic a fertile node can be
  unreachable because an ancestor exceeded the bound and was never expanded, and
  the search counts only the reachable fertile nodes it actually reaches. It
  returns `(count, censored)` with the same cap behavior as `count.py`.

## Analytic predictors

The analytic predictors compute an expected node count from value statistics,
without running the search. They read distributions sampled from random states,
which are separate from the labeled rows.

### `tree_size.py`
Exact per-depth node counts of the sliding puzzle's parent-pruned tree.

- `nodes_per_depth(start_blank, n, max_depth)`: the number of nodes at each depth.
  The tree's shape depends only on the blank trajectory, so the count is
  propagated over aggregates keyed by `(current blank cell, previous blank cell)`,
  which costs `O(cells^2 * depth)` with exact integer arithmetic. This replaces the
  usual `b^i` approximation with the true `N_i`.
- `blank_cell_equilibrium(n, depth=200)`: the long-run fraction of tree nodes with
  the blank at each cell, averaged over two consecutive depths to damp the parity
  oscillation. Used by the reweighted value distribution.

### `distribution.py`
The heuristic-value distribution `P(x) = Pr[h <= x]` that KRE needs.

- `HDistribution`: wraps a cumulative array. `P(x)` returns `0` below zero, the
  stored cumulative value inside the array, and `1` past its end.
- `estimate_distribution(n, n_samples, rng, weights_by_blank_cell=None)`: samples
  uniformly random solvable states, records each heuristic value and blank cell,
  and builds the cumulative distribution. With `weights_by_blank_cell`, samples are
  reweighted so the blank-cell mass matches the tree's equilibrium (the variant
  from the original KRE paper).
- `exact_distribution_8puzzle()`: the exact distribution by enumerating all
  solvable 8-puzzle states. Used to check the sampled one and to drive the KRE gate.

### `kre.py`
The KRE formula for the sliding puzzle.

- `kre_predict(state, n, bound, dist)`: computes `sum over i of N_i * P(bound - i)`,
  where `N_i` comes from `nodes_per_depth` and `P` from the distribution. The
  formula uses only the start state's blank position (for the tree shape) and the
  global value distribution, so it does not read `h(start)`. This is its known
  weak point on single start states.

### `conditional.py`
The conditional matrices `p(v | vp)` that CDP needs, where `v` is a child's value
and `vp` is its parent's value.

- `sample_conditional_matrix(n, h_fn, n_samples, rng, h_max)`: for each sampled
  state it fixes a random incoming flip (the move that hypothetically produced the
  parent) and tallies the heuristic values of the children of the other flips.
  Excluding the incoming flip makes the tallied children match the parent-pruned
  tree, which is the correction the paper otherwise gets from its two-step model.
  Columns are normalized.
- `exact_conditional_matrix(n, h_fn, h_max)`: the same matrix by full enumeration
  over states, incoming flips, and children. Used to check the sampled one.
- `H_MAX_PANCAKE(n)`: returns `n`, the largest possible GAP value, which sizes the
  matrix.

### `cdp.py`
The CDP1 recursion (Zahavi et al., 2010, Eq. 5 and 6).

- `cdp1_predict(h_start, bound, cond, b_root, b_rest)`: propagates a level
  occupancy vector down the tree. It starts with all mass on `h_start` at level 0.
  At each level `i` it caps the parent values at `bound - (i-1)` (so a node is
  counted only if all its ancestors were expanded, the property KRE lacks under
  inconsistency), multiplies by the branching factor (`b_root` at the first level,
  `b_rest` afterward), applies the conditional matrix by `cond @ (parents * b)`, and
  adds the mass at values within `bound - i`. Branching is passed in and is exact on
  the pancake. If `h_start` falls outside the matrix support the function raises,
  rather than silently dropping the root term.

## Feature extractors (learned models)

Each extractor turns a state and a bound into the numeric inputs the regressors
read. The extractors are pure functions of the state and bound, so they can be
computed before any search and cannot leak generation metadata.

### `pancake_features.py`
- `PANCAKE_FEATURE_NAMES`, `extract_pancake_features(state, bound)`: the 13 static
  pancake features. These are the three heuristic values (`h_gap`, `h_gap2`,
  `h_rand`), the bound, the three slacks (`bound - h`), the plate-gap indicator,
  the solved-suffix length, inversions, max and mean displacement, and the
  first-gap position. All three heuristic values are valid inputs regardless of
  which heuristic guided the search, because each is a function of the state alone.

### `features.py`
- `FEATURE_NAMES`, `extract_features(state, n, bound)`: the 15-puzzle static
  features: the Manhattan value, the bound, the slack, linear-conflict pairs,
  misplaced tiles, inversions, the blank's row, column, and degree, and
  tile-distance statistics. The blank's degree equals the root's number of legal
  moves, which carries the non-uniform branching.

### `probe_features.py`
- `PROBE_FEATURE_NAMES`, `extract_probe_features(state, bound)`: the 12 one-step
  probe features (6 per heuristic). It expands the state once and summarizes the
  children's heuristic values (min, mean, max, count improving, count worsening,
  and the fertile-children count) for GAP and for RAND. These are measured on the
  actual state, unlike CDP1's space-averaged transitions, so they cost the `n-1`
  child evaluations of one expansion and are a declared operating point rather than
  a free input.

## Models and metrics

### `models.py`
Data split, baselines, and the two learned models. The target everywhere is
`log10(nodes)`.

- `split_by_state(df, seed=0)`: a 70/15/15 train, validation, test split grouped by
  the `state` column, using `GroupShuffleSplit`, so all rows of one state fall on
  the same side of every boundary.
- `MeanBaseline`: predicts the mean training effort.
- `GapBaseline`: predicts the mean training effort of rows with the same slack
  (`bound - h`). The strong floor.
- `fit_gbm(X_train, y_train, seed=0)`: fits scikit-learn's
  `HistGradientBoostingRegressor` with internal early stopping.
- `fit_mlp(X_train, y_train, X_val, y_val, seed=0, ...)`: fits a two-layer
  (64 units each) ReLU network on standardized features with Adam and early
  stopping on the validation set.

### `metrics.py`
- `eval_log10_predictions(y_true_log10, y_pred_log10)`: returns the RMSE in
  `log10` space, the median factor error, the fraction of predictions within a
  factor of two, and the Spearman rank correlation.
- `cluster_bootstrap_se(states, y_true, y_pred, metric_key, n_boot=1000,
  seed=0)`: standard errors from a bootstrap that resamples whole states, because
  rows of the same state at different bounds are correlated.

### `__init__.py`
Empty. Marks the directory as a package.
