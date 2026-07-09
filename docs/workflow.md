# Workflow and data pipeline

This page traces the data pipeline from state generation to final tables, names
the exact functions and files at each step, and gives the commands to reproduce
every result. For the per-file reference, see [core-library.md](core-library.md)
and [scripts.md](scripts.md).

## Pipeline stages

```
  STAGE 1  prepare inputs

  start states
     random_pancake_state / random_solvable_state
     -> generate_pancake.py / generate_tiles_uniform.py
     -> count_expansions_pancake / count_expansions   (bounded DFS)
     -> label CSV   (columns: state, bound, nodes, censored)

  random states
     estimate_distribution.py / sample_conditional_matrix
     -> P(x)      (KRE reads this)
     -> p(v|vp)   (CDP1 reads this)

  STAGE 2  train and evaluate

  label CSV
     -> train_eval_pancake.py
          split_by_state          70/15/15, grouped by state
          extract_pancake_features build model inputs
          fit_gbm / fit_mlp       train the learned models
          kre_predict_pancake / cdp1_predict  run the analytic predictors
          eval_log10_predictions  score every method on the same test rows
     -> summary CSVs in results/

  STAGE 3  plot

  summary CSVs
     -> make_report_figures.py
     -> figure PDFs
```

## Data generation pipeline (explicit)

This section answers where each piece of data comes from.

### Pancake domain and states
The pancake domain is defined in `effortpred/pancake.py`. States are permutations
of `0..n-1`. Start states for the datasets are drawn by `random_pancake_state(n,
rng)`, which returns a uniformly random permutation. The generator
`scripts/generate_pancake.py` calls this function once per requested state.

### 15-puzzle domain and states
The sliding-tile domain is defined in `effortpred/puzzle.py`. For the paper's
generalization study, start states are uniformly random solvable states from
`random_solvable_state(n, rng)`, called by
`scripts/generate_tiles_uniform.py`. The earlier pilot generator
`scripts/generate_data.py` instead used `random_walk_state`, which is why the
uniform-state study replaced it.

### Node counting (the labels)
The label of a row is the true number of nodes one bounded pass expands. It is
computed by a bounded depth-first search, not estimated.

- Pancake labels: `count_expansions_pancake(state, bound, h_fn, cap)` in
  `effortpred/pancake_count.py`. The heuristic `h_fn` is `gap_h` for the
  consistent set and `rand_h` for the inconsistent set.
- 15-puzzle labels: `count_expansions(state, n, bound, cap)` in
  `effortpred/count.py`, using Manhattan distance.

Both counters count the fertile nodes of the parent-pruned tree (nodes with
`f = g + h <= bound`), with no goal stop. A slow reference counter
(`count_expansions_reference`) checks the fast one in the tests.

### From states to a labeled dataset
For each start state, a generator produces one row per offset. The bound is
`h(state) + offset`. Each row stores the state, the bound, the node count, and a
censored flag. A row is censored when the count passes the cap; censored rows are
dropped from analysis and counted. The result is a `*labels*.csv` file, which is
the only place the true node counts live.

### Distributions for the analytic predictors
KRE and CDP do not run the search. They read distributions sampled from random
states, separate from the labeled rows.

- KRE reads `P(x)`, the heuristic-value distribution. On the pancake it comes
  from `estimate_pancake_distribution`. On the 15-puzzle it comes from
  `scripts/estimate_distribution.py`, saved as `.npy`.
- CDP1 reads the conditional matrix `p(v | vp)` from `sample_conditional_matrix`.

## Feature extraction
The learned models never see the raw state. `train_eval_pancake.py` calls
`extract_pancake_features(state, bound)` to build the 13-feature vector for each
row (or the 2-feature minimal vector under `--feature-set minimal`). The
15-puzzle scripts use `extract_features`. The optional probe tier adds
`extract_probe_features`. Feature definitions are in [core-library.md](core-library.md).

## Model training
`split_by_state` partitions the rows 70/15/15 grouped by state. `fit_gbm` trains
the gradient-boosted trees, and `fit_mlp` trains the MLP with early stopping on
the validation split. The analytic predictors are computed on the same test rows
by `kre_predict_pancake` and `cdp1_predict`. Every method is scored by
`eval_log10_predictions`, with standard errors from `cluster_bootstrap_se`.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Python 3.11 or newer is required. All experiments run on the CPU.

## Correctness checks

```
.venv/bin/pytest
```

The suite verifies each component exhaustively where the state space is small:
the 8-puzzle solvability rule against breadth-first reachability over all `9!`
states, the pancake heuristics over all 5,040 states of the 7-pancake (including
that GAP is self-dual and RAND is inconsistent), the tree-size recurrence against
explicit enumeration, and the KRE and CDP code against the equations in their
papers.

## Reproduction commands

All randomness is seeded. Label generation is the only slow stage: labeling the
12-pancake GAP set runs a bounded search on 1,500 states and takes on the order
of an hour on a multi-core CPU. Every later stage runs in seconds to a few
minutes. Each stage below matches a section of the report and writes to
`results/`.

### Stage A: main pancake comparison
Produces the headline table and the equal-information ablation on the 12-pancake.

```
.venv/bin/python scripts/generate_pancake.py --heuristic gap  --offsets 0 1 2 3 --cap 20000000
.venv/bin/python scripts/generate_pancake.py --heuristic rand --offsets 0 1     --cap 2000000
.venv/bin/python scripts/train_eval_pancake.py
.venv/bin/python scripts/train_eval_pancake.py --feature-set minimal
```

### Stage B: depth study (10-pancake)

```
.venv/bin/python scripts/generate_pancake.py --heuristic gap  --n 10 --n-states 1000 --offsets 0 1 2 3 4 --cap 20000000 --out results/pancake_labels_gap_n10.csv
.venv/bin/python scripts/generate_pancake.py --heuristic rand --n 10 --n-states 1000 --offsets 0 1 2 3   --cap 20000000 --out results/pancake_labels_rand_n10.csv
.venv/bin/python scripts/train_eval_pancake.py --n 10 --data-suffix _n10
.venv/bin/python scripts/train_eval_pancake.py --n 10 --data-suffix _n10 --feature-set minimal
```

### Stage C: extrapolation

```
.venv/bin/python scripts/extrapolation_test.py
.venv/bin/python scripts/extrapolation_tiers.py
```

### Stage D: tail errors

```
.venv/bin/python scripts/tail_errors.py
```

### Stage E: learning curves and offline costs

```
.venv/bin/python scripts/learning_curves.py --reps 10 --out-prefix results/learning_curve10
.venv/bin/python scripts/cost_ledger.py
```

### Stage F: probe tier

```
.venv/bin/python scripts/probe_eval.py
.venv/bin/python scripts/probe_no_fertile.py
```

### Stage G: feature importance and latency

```
.venv/bin/python scripts/night_extras.py
```

### Stage H: split robustness

```
.venv/bin/python scripts/robustness_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9 --out results/robustness_seeds10.csv
```

### Stage I: 15-puzzle generalization

```
.venv/bin/python scripts/estimate_distribution.py
.venv/bin/python scripts/generate_tiles_uniform.py
.venv/bin/python scripts/tiles_generalization_eval.py
```

### Stage J: figures
Depends on stages A, B, D, E, and G.

```
.venv/bin/python scripts/make_report_figures.py
```

## Reading the numbers without rerunning
The frozen summary CSVs are committed in `results/`. See
[data-and-results.md](data-and-results.md) for which file backs which table.
