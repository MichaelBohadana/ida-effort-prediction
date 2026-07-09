# Core library reference (`effortpred/`)

File-by-file reference for the `effortpred/` package. Every module is listed
with its public functions and classes and their role. The scripts in `scripts/`
import from these modules and add command-line handling and file output (see
[scripts.md](scripts.md)).

The modules group into six roles: sliding-tile domain, pancake domain, node
counters, analytic predictors, feature extractors, and models and metrics.

## Sliding-tile domain

### `puzzle.py`
Generic NxN sliding-tile puzzle. A state is a tuple of length `n*n` where
`state[cell]` is the tile at that cell and `0` is the blank. Cells are indexed
row-major. The goal is `(0, 1, ..., n*n-1)`.

- `goal_state(n)`: the goal tuple for an NxN board.
- `neighbors_table(n)`: for each cell, the list of adjacent cells.
- `apply_move(state, to_cell)`: slide the tile at `to_cell` into the blank.
- `successors(state, n)`: all states reachable in one move.
- `permutation_parity(perm)`: parity of a permutation, used by the solvability
  test.
- `is_solvable(state, n)`: whether a state can reach the goal.
- `random_walk_state(n, length, rng)`: a state produced by a random walk from
  the goal. Used by the pilot 15-puzzle generator only.
- `random_solvable_state(n, rng)`: a uniformly random solvable state. This is
  the state source for the 15-puzzle generalization dataset.

### `heuristic.py`
Manhattan-distance heuristic for the sliding puzzle, plus a linear-conflict
count used only as a feature.

- `md_table(n)`: lookup table of the Manhattan distance of each tile from its
  goal cell for every cell.
- `manhattan(state, n)`: the Manhattan-distance heuristic. Admissible and
  consistent on this domain.
- `linear_conflict_pairs(state, n)`: count of tile pairs that share a goal row
  or column but are reversed. A feature, not part of the heuristic.

## Pancake domain

### `pancake.py`
The pancake puzzle domain and its three heuristics. A state is a tuple
permutation of `0..n-1`. The goal is `(0, 1, ..., n-1)`.

- `pancake_goal(n)`: the goal permutation.
- `flip(state, k)`: reverse the first `k` elements (the pancake move).
- `successors_pancake(state, prev_k)`: children under parent pruning (all flips
  except the one that repeats the previous flip).
- `dual_state(state)`: the inverse permutation, used to test dual evaluation.
- `gap_h(state)`: the GAP heuristic (consistent, admissible). Counts adjacencies
  whose sizes differ by more than one, including the plate below the stack.
- `gap2_h(state)`: GAP that ignores the two smallest pancakes. Strictly weaker,
  still consistent.
- `rand_h(state)`: the inconsistent heuristic. Returns `gap_h` or `gap2_h`
  chosen by a checksum of the state, so values can jump by more than one along a
  path.
- `random_pancake_state(n, rng)`: a uniformly random permutation. This is the
  state source for the pancake datasets.
- `exact_pancake_distribution(n, h_fn)`: the exact heuristic-value distribution
  by full enumeration (small `n`).
- `estimate_pancake_distribution(n, h_fn, n_samples, rng)`: the sampled
  heuristic-value distribution. KRE and CDP read this.

## Node counters (label source)

The label for a row is the true number of nodes one bounded pass expands. These
counters run that bounded depth-first search directly.

### `count.py`
Bounded-DFS expansion counter for the sliding puzzle.

- `CapExceeded`: exception raised when the count passes the node cap.
- `count_expansions_reference(state, n, bound)`: slow, obviously correct
  version. Ground truth for the tests only.
- `count_expansions(state, n, bound, cap=None)`: fast version used for
  generation. Counts fertile nodes of the parent-pruned tree, no goal stop.

### `pancake_count.py`
Bounded-DFS expansion counter for the pancake, with the same semantics as
`count.py`.

- `CapExceeded`: same cap exception for the pancake counter.
- `count_expansions_pancake(state, bound, h_fn, cap=None)`: counts fertile nodes
  under a pluggable heuristic `h_fn`, so it labels both the consistent `gap_h`
  set and the inconsistent `rand_h` set. Under an inconsistent heuristic a
  fertile node can be unreachable because an ancestor was pruned, which the DFS
  handles directly.

## Analytic predictors

### `tree_size.py`
Exact per-depth node counts of the sliding puzzle's parent-pruned tree.

- `nodes_per_depth(start_blank, n, max_depth)`: exact node count at each depth
  from a recurrence over blank positions. Replaces the `b^i` approximation.
- `blank_cell_equilibrium(n, depth=200)`: the equilibrium distribution of the
  blank cell, used by the reweighted value distribution.

### `distribution.py`
The heuristic-value distribution `P(x)` that KRE needs.

- `HDistribution`: holds a cumulative distribution and answers `P(x)` queries.
- `estimate_distribution(n, n_samples, rng, weights_by_blank_cell=None)`:
  estimate `P(x)` from sampled states, with an optional equilibrium reweighting.
- `exact_distribution_8puzzle()`: the exact distribution by full enumeration,
  used to check the sampled one.

### `kre.py`
The KRE formula for the sliding puzzle.

- `kre_predict(state, n, bound, dist)`: sum of per-depth tree sizes weighted by
  the value distribution. Ignores the start state's heuristic value by design,
  so it issues one prediction per bound.

### `pancake_tree.py`
Exact tree sizes and the KRE formula for the pancake.

- `nodes_per_depth_pancake(n, max_depth)`: closed-form node counts (`n-1` at the
  root, `n-2` below).
- `kre_predict_pancake(n, bound, dist)`: KRE on the pancake. Because branching is
  uniform, it returns one prediction per bound with no per-state variation.

### `conditional.py`
The conditional matrices `p(v | vp)` that CDP needs, where `v` is a child value
and `vp` its parent's value.

- `H_MAX_PANCAKE(n)`: the largest possible GAP value, sizing the matrix.
- `sample_conditional_matrix(n, h_fn, n_samples, rng, h_max)`: estimate the
  matrix by sampling, excluding the incoming flip so the tallied children match
  the parent-pruned tree.
- `exact_conditional_matrix(n, h_fn, h_max)`: the exact matrix by enumeration,
  used to check the sampled one.

### `cdp.py`
The CDP1 recursion (Zahavi et al., 2010, Eq. 5 and 6).

- `cdp1_predict(h_start, bound, cond, b_root, b_rest)`: propagate the conditional
  value distribution down the levels and sum the expanded nodes. Capping the
  parent value at each level keeps it correct under inconsistent heuristics.
  Branching is passed in and is exact on the pancake.

## Feature extractors (learned models)

Each extractor turns a state and a bound into the numeric inputs the regressors
read.

### `pancake_features.py`
- `PANCAKE_FEATURE_NAMES`: the 13 static pancake feature names.
- `extract_pancake_features(state, bound)`: builds the 13 features: the three
  heuristic values (`h_gap`, `h_gap2`, `h_rand`), the bound, the three slacks,
  the plate-gap indicator, the solved-suffix length, inversions, max and mean
  displacement, and the first-gap position. All three heuristic values are valid
  inputs regardless of which heuristic guided the search.

### `features.py`
- `FEATURE_NAMES`: the 15-puzzle static feature names.
- `extract_features(state, n, bound)`: builds the 15-puzzle features: Manhattan
  value, bound, slack, linear-conflict pairs, misplaced tiles, inversions, the
  blank's row, column, and degree, and tile-distance statistics. Generation
  metadata is excluded so nothing about how a state was made can leak.

### `probe_features.py`
- `PROBE_FEATURE_NAMES`: the 12 probe feature names (6 per heuristic).
- `extract_probe_features(state, bound)`: expands the state once and summarizes
  the children's heuristic values (min, mean, max, count improving, count
  worsening, and fertile-children count) for GAP and for RAND. These are
  measured on the actual state, unlike CDP1's space-averaged transitions, so
  they cost one expansion and are a declared operating point.

## Models and metrics

### `models.py`
Data split, baselines, and the two learned models. The target everywhere is
`log10(nodes)`.

- `split_by_state(df, seed=0)`: a 70/15/15 train, validation, test split grouped
  by the `state` column, so no state appears on two sides of any boundary.
- `MeanBaseline`: predicts the mean training effort.
- `GapBaseline`: predicts the mean training effort of rows with the same slack
  (`bound - h`). The strong floor.
- `fit_gbm(X_train, y_train, seed=0)`: fit the histogram gradient-boosted trees.
- `fit_mlp(X_train, y_train, X_val, y_val, seed=0, ...)`: fit the two-layer MLP
  with early stopping on the validation set.

### `metrics.py`
Evaluation in `log10` space.

- `eval_log10_predictions(y_true_log10, y_pred_log10)`: returns RMSE, median
  factor error, the fraction within a factor of two, and Spearman rank
  correlation.
- `cluster_bootstrap_se(states, y_true, y_pred, metric_key, n_boot=1000,
  seed=0)`: standard errors from a bootstrap that resamples whole states, not
  rows.

### `__init__.py`
Empty. Marks the directory as a package.
