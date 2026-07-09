# Test suite reference (`tests/`)

The tests are correctness gates. Each label, distribution, and predictor is
checked before it is used, exhaustively wherever the state space is small enough
to enumerate. The gates back the "Correctness Checks" section of the paper: the
claims that GAP is consistent and self-dual, that RAND is inconsistent, that the
tree-size recurrence is exact, and that the KRE and CDP implementations match
their source equations all rest on the tests below.

Run the full suite with `.venv/bin/pytest`. Every test file targets one library
module. Gates marked exhaustive enumerate an entire state space; the others use
hand-computed cases or fixed random samples.

## Domain gates

### `tests/test_puzzle.py`
Validates the sliding-tile domain in `puzzle.py`.

- `test_solvability_rule_exhaustive_8puzzle`: the permutation-parity solvability
  rule (`is_solvable`) is checked against true breadth-first reachability from the
  goal, over all `9!` permutations of the 8-puzzle. This is the authority test for
  the rule: 181,440 of the 362,880 permutations are reachable, and the parity rule
  must agree on every one.
- `test_goal_and_neighbors_3x3`, `test_apply_move`, `test_successors_counts`:
  hand cases for the goal state, the move mechanics, and the successor counts.
- `test_random_states_solvable`: states from `random_solvable_state` are always
  solvable.

### `tests/test_heuristic.py`
Validates the Manhattan heuristic in `heuristic.py`.

- `test_manhattan_admissible_exhaustive_8puzzle`: `manhattan(s) <= true optimal
  distance` for all 181,440 reachable 8-puzzle states. Admissibility is the
  precondition for every label in the sliding-tile domain.
- `test_manhattan_changes_by_exactly_one_per_move`: one tile slide changes the
  heuristic by exactly one, which establishes the f-parity used to restrict the
  15-puzzle to even offsets.
- `test_incremental_delta_matches_recompute`: the incremental delta-table update
  used inside the fast counter (`count.py`) agrees with a full recompute. This
  guards the optimization that makes generation feasible.
- `test_manhattan_hand_cases`, `test_linear_conflict_hand_cases`: hand cases for
  the heuristic and the linear-conflict feature.

### `tests/test_pancake.py`
Validates the pancake domain and its three heuristics in `pancake.py`. This file
holds the gates that justify the inconsistent-heuristic design.

- `test_gap_admissible_and_consistent_exhaustive_n7`: over all 5,040 states of the
  7-pancake, GAP is admissible and consistent (its value changes by at most one
  across any move).
- `test_gap2_and_rand_admissible_exhaustive_and_rand_inconsistent_n7`: over all
  5,040 states, GAP-2 is admissible and strictly weaker than GAP, RAND is
  admissible, and RAND is inconsistent (a move exists where the consulted values
  jump by more than one). This is the gate that certifies the controlled
  inconsistent heuristic.
- `test_gap_is_self_dual_exhaustive_n7`: `gap_h(dual(s)) == gap_h(s)` for every
  permutation. This is the lemma that refuted the dual-evaluation design: dual
  evaluation cannot make GAP inconsistent, so RAND uses random selection instead.
- `test_exact_distribution_n7_properties`,
  `test_sampled_distribution_matches_exact_n7`: the sampled heuristic-value
  distribution matches the exact enumerated one.
- `test_flip_hand_cases`, `test_successors_parent_pruning`, `test_dual_state`,
  `test_gap_hand_cases`, `test_gap2_hand_cases`,
  `test_all_permutations_reachable_n7`: mechanics and hand cases.

## Label gates (node counters)

### `tests/test_count.py`
Validates the 15-puzzle counter in `count.py`.

- `test_fast_matches_reference_8puzzle`, `test_fast_matches_reference_15puzzle_small`:
  the fast counter (`count_expansions`) equals the slow reference counter
  (`count_expansions_reference`) on many cases. The reference recomputes the
  heuristic in full and identifies the parent by comparing whole states, so it is
  independent of the fast counter's incremental logic.
- `test_odd_bound_equals_even_bound_below`: because Manhattan f-values share the
  parity of `h(start)`, an odd bound expands the same nodes as the even bound one
  below. This confirms the even-offset restriction adds no bias.
- `test_monotone_in_bound`: the count never decreases as the bound rises.
- `test_bound_below_h_is_zero`, `test_goal_bound_zero`: boundary cases.
- `test_cap_censors`: reaching the cap raises `CapExceeded` and returns the
  censored flag.

### `tests/test_pancake_count.py`
Validates the pancake counter in `pancake_count.py`.

- `test_matches_reference_consistent`, `test_matches_reference_inconsistent`: the
  fast counter equals an independent reference under both `gap_h` and the
  inconsistent `rand_h`. The inconsistent case matters because a fertile node can
  be unreachable, and the counter must still count only the reachable fertile
  nodes.
- `test_odd_offsets_do_add_nodes_sometimes`: an anti-regression check. Unlike the
  15-puzzle, the pancake has no f-parity structure (GAP can change by zero across a
  flip), so odd offsets do add nodes. This is why the pancake datasets step offsets
  by one.
- `test_cap_censors`, `test_goal_bound_zero`, `test_bound_below_h_is_zero`:
  boundary and cap cases.

## Analytic-predictor gates

### `tests/test_tree_size.py`
Validates the exact tree-size recurrence in `tree_size.py`.

- `test_recurrence_matches_enumeration`: the recurrence (`nodes_per_depth`) equals
  an explicit parent-pruned enumeration, which is independent of the recurrence.
  This certifies the exact `N_i` that KRE uses in place of the `b^i`
  approximation.
- `test_hand_case`, `test_equilibrium_fractions`: a hand case and the blank-cell
  equilibrium used by the reweighted distribution.

### `tests/test_pancake_tree.py`
Validates the closed-form pancake tree sizes and the pancake KRE formula in
`pancake_tree.py`.

- `test_closed_form_matches_enumeration`, `test_hand_case`: the closed-form node
  counts (`n-1` at the root, `n-2` below) equal explicit enumeration.
- `test_kre_trivial_heuristic_is_brute_force`: with a trivial heuristic, KRE
  returns the full brute-force tree size, a case with a known answer.
- `test_kre_pancake_large_set_gate`: the end-to-end pancake KRE gate. With an
  exact `P(x)`, the ratio of mean predicted to mean actual effort stays near one,
  which guards against off-by-one and indexing errors in the formula.

### `tests/test_distribution.py`
Validates the heuristic-value distribution in `distribution.py`.

- `test_sampler_matches_exact_8puzzle`: the sampled `P(x)` from
  `estimate_distribution` agrees with the exact enumerated `P(x)` on the 8-puzzle.
- `test_exact_8puzzle_properties`, `test_hdistribution_lookup`,
  `test_reweighted_distribution_valid`: the cumulative lookup and the reweighted
  variant are valid distributions.

### `tests/test_kre.py`
Validates the KRE formula in `kre.py`.

- `test_trivial_heuristic_gives_brute_force_size`: with `h == 0` (so `P(x) == 1`),
  KRE returns the full brute-force tree size within the bound. This checks the
  formula against a case with a known answer.
- `test_kre_ballpark_on_8puzzle_large_set`: the end-to-end gate. With an exact
  `P`, KRE's ratio of mean predicted to mean actual effort on the 8-puzzle sits
  near one (about 0.98). The comment marks this gate as not to be weakened.

### `tests/test_conditional.py`
Validates the conditional matrices in `conditional.py`.

- `test_sampled_matches_exact`: the sampled matrix `p(v | vp)` agrees with the
  exact enumerated matrix.
- `test_exact_matrix_properties_consistent`,
  `test_exact_matrix_properties_inconsistent`: the columns are normalized, and the
  consistent heuristic produces a banded matrix (values move by at most one) while
  the inconsistent one does not.
- `test_sampling_deterministic`: sampling is reproducible under a fixed seed.

### `tests/test_cdp.py`
Validates the CDP1 recursion in `cdp.py`.

- `test_cdp_gate_consistent`: with exact conditional matrices under the consistent
  GAP heuristic, CDP predicts per state more accurately than KRE. This reproduces
  the paper's finding.
- `test_cdp_gate_inconsistent`: under the inconsistent RAND heuristic, KRE
  overestimates (it counts fertile nodes the search never reaches) while CDP1 stays
  calibrated. This is the gate for CDP's advantage under inconsistency.
- `test_degenerate_heuristic_gives_brute_force`: with `h == 0` the recursion
  returns the brute-force size, a case with a known answer.
- `test_bound_zero`, `test_out_of_support_h_start_raises`: boundary cases. The last
  one confirms the recursion fails loudly if `h(start)` falls outside the matrix
  support, rather than silently dropping the root term.

## Feature and model gates

### `tests/test_pancake_features.py` and `tests/test_features.py`
Validate the feature extractors. They check the goal-state feature values, one
hand-computed non-goal state, the exact set and order of the feature names, and
that no feature is NaN. Stable names and order are required so the trained models
and the committed importance tables line up.

### `tests/test_probe_features.py`
Validates the one-step probe features in `probe_features.py`.

- `test_gap_consistency_property`: over 200 random 12-pancake states, a child's
  probed GAP value stays within one of the parent's, matching GAP's consistency.
- `test_rand_inconsistency_witness`: over the same states, at least one child under
  RAND jumps by more than one, witnessing inconsistency at the probe level.
- `test_n_fertile_monotone_in_bound`: the fertile-children count rises with the
  bound.
- `test_hand_case_goal_n5_bound2`, `test_probe_feature_names`,
  `test_hand_case_keys_match_names`: hand values, the exact 12 names, and matching
  keys.

### `tests/test_metrics.py`
Validates the metrics in `metrics.py`: perfect predictions give zero error, a
one-decade offset gives the expected RMSE and factor error, and the cluster
bootstrap standard error is positive and reproducible.

### `tests/test_models.py`
Validates the models in `models.py`.

- `test_split_by_state_no_leakage`: no state appears on two sides of the
  train, validation, or test boundary. This is the check that prevents leakage
  across the bounds of one state.
- `test_gap_baseline_table`: the slack-table baseline returns the correct
  per-slack means.
- `test_gbm_and_mlp_beat_mean_baseline_on_synthetic_data`: both learned models fit
  a synthetic signal better than the mean baseline.

### `tests/test_smoke.py`
`test_import`: the package imports. A minimal guard against a broken install.
