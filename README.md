# effortpred: Predicting IDA\* Search Effort

This repository contains the code and frozen result tables for the study
*Predicting IDA\* Search Effort: Learned Models vs. KRE and CDP1*.

Iterative-deepening A\* (IDA\*) runs a sequence of depth-first iterations, each
bounded by a cost threshold. The number of nodes a single bounded iteration
expands varies by orders of magnitude across start states at the same bound.
This project predicts that node count for a given start state, admissible
heuristic, and cost bound, before any node is generated. We train two offline
supervised regressors, a histogram gradient-boosted tree ensemble (GBM) and a
small multilayer perceptron (MLP), on static features of the start state, and
compare them head to head against two analytic predictors from the search
literature: the KRE formula (Korf, Reid, and Edelkamp, 2001) and the one-step
Conditional Distribution Prediction model, CDP1 (Zahavi et al., 2010).

The evaluation covers the pancake puzzle under a consistent heuristic (GAP) and
a controlled inconsistent heuristic (random selection between GAP and GAP-2),
and a generalization study on the 15-puzzle, whose non-uniform branching tests
whether the approach carries beyond the uniform-branching pancake.

## Repository layout

```
effortpred/     Core library: puzzle domains, heuristics, node counters,
                the KRE and CDP predictors, features, and models.
scripts/        Command-line entry points for every stage of the pipeline
                (dataset generation, training, evaluation, and plotting).
tests/          Correctness gates, including exhaustive checks on the
                8-puzzle, 15-puzzle recurrence, and 7-pancake heuristics.
results/        Frozen summary CSVs that back the tables in the paper.
                Regenerable artifacts (raw label datasets, sampled
                distributions, figures, logs) are not tracked; see below.
pyproject.toml  Package metadata and dependencies.
```

## Requirements

- Python 3.11 or newer.
- The dependencies declared in `pyproject.toml`: NumPy, pandas, scikit-learn,
  SciPy, Matplotlib, and PyTorch. All experiments run on the CPU only; no GPU
  is required.

## Installation

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

This installs the `effortpred` package in editable mode together with the test
dependencies. All commands below use the interpreter inside `.venv` so that no
global environment is touched.

## Verifying correctness

Every component is checked before its outputs are used, exhaustively wherever
the state space is small enough to enumerate.

```
.venv/bin/pytest
```

The suite runs in a few minutes. It verifies, among other properties, the
8-puzzle solvability rule against BFS reachability over all 9! permutations,
the admissibility of Manhattan distance over all reachable 8-puzzle states, the
15-puzzle tree-size recurrence against explicit enumeration, and, over all
5,040 states of the 7-pancake, that GAP is admissible and consistent, GAP-2 is
strictly weaker, the random-selection heuristic is admissible and inconsistent,
and GAP is self-dual. The KRE and CDP implementations are checked against the
equations of their source papers.

## Reproducing the paper

All randomness is seeded, so every command below is deterministic and
reproduces the frozen numbers. The dataset generation stage is the only
expensive part: labeling the 12-pancake GAP set runs a bounded search on each
of 1,500 start states and takes on the order of an hour on a multi-core CPU.
Every stage downstream of generation completes in seconds to a few minutes.

The `results/` directory already holds the frozen summary CSV for every table,
so the reported numbers can be inspected directly without rerunning the
pipeline. Rerun the stages below to regenerate them from scratch.

### 1. Main pancake comparison

Produces the headline comparison and the equal-information ablation on the
12-pancake, under both heuristics.

```
# Generate the labeled datasets (the expensive stage).
.venv/bin/python scripts/generate_pancake.py --heuristic gap  --offsets 0 1 2 3 --cap 20000000
.venv/bin/python scripts/generate_pancake.py --heuristic rand --offsets 0 1     --cap 2000000

# Full 13-feature models plus KRE and CDP1 baselines.
.venv/bin/python scripts/train_eval_pancake.py

# Equal-information ablation: restrict the learned models to CDP1's exact
# inputs (the active heuristic value and the bound).
.venv/bin/python scripts/train_eval_pancake.py --feature-set minimal
```

Outputs: `results/pancake_eval_{gap,rand}.csv` and the corresponding
`_minimal.csv` files, which back the headline and ablation tables.

### 2. Depth study (10-pancake)

The smaller board reaches deeper offsets within the node cap, giving a longer
depth range for the per-offset view.

```
.venv/bin/python scripts/generate_pancake.py --heuristic gap  --n 10 --n-states 1000 --offsets 0 1 2 3 4 --cap 20000000 --out results/pancake_labels_gap_n10.csv
.venv/bin/python scripts/generate_pancake.py --heuristic rand --n 10 --n-states 1000 --offsets 0 1 2 3   --cap 20000000 --out results/pancake_labels_rand_n10.csv
.venv/bin/python scripts/train_eval_pancake.py --n 10 --data-suffix _n10
.venv/bin/python scripts/train_eval_pancake.py --n 10 --data-suffix _n10 --feature-set minimal
```

Outputs: `results/pancake_eval_{gap,rand}_n10*.csv` and the `_by_offset.csv`
files behind the depth figure.

### 3. Growth-ratio extrapolation

Trains on shallow offsets and predicts an unseen deeper offset, with the test
states also held out (double holdout).

```
.venv/bin/python scripts/extrapolation_test.py     # results/extrapolation_test.csv
.venv/bin/python scripts/extrapolation_tiers.py    # results/extrapolation_tiers.csv (per-tier addendum)
```

### 4. Tail errors

```
.venv/bin/python scripts/tail_errors.py            # results/tail_errors.csv
```

### 5. Learning curves and offline costs

```
.venv/bin/python scripts/learning_curves.py --reps 10 --out-prefix results/learning_curve10
.venv/bin/python scripts/cost_ledger.py            # results/cost_ledger.csv
```

Outputs: `results/learning_curve10.csv` (sample efficiency) and
`results/cost_ledger.csv` (offline preparation cost of each predictor family).

### 6. Probe tier

Adds one-step-lookahead statistics of the child heuristic values to the minimal
set, and the companion ablation that removes the fertile-children count.

```
.venv/bin/python scripts/probe_eval.py             # results/probe_eval.csv
.venv/bin/python scripts/probe_no_fertile.py       # results/probe_no_fertile.csv
```

### 7. Feature importance and prediction latency

```
.venv/bin/python scripts/night_extras.py
```

Outputs: `results/mlp_importance_{gap,rand}.csv` (permutation importance) and
`results/prediction_latency.csv`.

### 8. Split robustness

Repeats the main evaluation over ten split seeds with the analytic predictors'
samplers held fixed, so the across-seed spread reflects only the partition.

```
.venv/bin/python scripts/robustness_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9 --out results/robustness_seeds10.csv
```

### 9. Generalization to the 15-puzzle

```
# Sampled heuristic-value distribution used by KRE.
.venv/bin/python scripts/estimate_distribution.py

# Uniform-state labels (the expensive stage for this domain).
.venv/bin/python scripts/generate_tiles_uniform.py

# Learned models vs KRE on the 15-puzzle.
.venv/bin/python scripts/tiles_generalization_eval.py
```

Output: `results/tiles_generalization_eval.csv`.

### 10. Regenerating the paper figures

The figure script reads frozen CSVs and one raw label dataset (the headline
scatter is drawn from the individual instances of the 12-pancake GAP set), so
run stages 1, 2, 4, 5, and 7 first, then:

```
.venv/bin/python scripts/make_report_figures.py
```

## Notes on reproducibility

- All seeds (dataset generation, train/validation/test splits, and the analytic
  predictors' samplers) are fixed in the scripts and can be overridden with the
  `--seed` flags.
- The train, validation, and test split is grouped by start state, so no state
  appears on two sides of any boundary.
- The target is the base-10 logarithm of the node count. Rows that hit the node
  cap are censored: they are dropped from analysis and counted.
- Only the frozen summary CSVs are tracked in `results/`. Raw label datasets,
  sampled distributions (`.npy`), figures, and run logs are regenerated by the
  commands above and are excluded by `.gitignore`.

## Citation

If you use this code, please cite the accompanying report:

> M. Bohadana and N. Klein. Predicting IDA\* Search Effort: Learned Models vs.
> KRE and CDP1. Course 237-2-5513 (Search in Artificial Intelligence),
> Ben-Gurion University of the Negev.
