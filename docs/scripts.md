# Scripts reference (`scripts/`)

File-by-file reference for the command-line entry points. Each script imports
logic from [the core library](core-library.md) and adds argument parsing and
file output. For the order to run them in, see [workflow.md](workflow.md).

Run every script through the virtual environment, for example:

```
.venv/bin/python scripts/train_eval_pancake.py --help
```

All scripts take `--seed` and write output under `results/`. The defaults
reproduce the reported numbers.

## Calibration (pilots)

### `calibrate_pancake.py`
Reports node count, wall time, and censoring per offset on the pancake, for both
heuristics. Used to pick the `--offsets` and `--cap` for generation. Output:
printed table. Does not need rerunning to reproduce frozen data.

### `calibrate.py`
Same pilot for the 15-puzzle. Output: printed table.

## Data generation

### `generate_pancake.py`
Builds the pancake datasets. For each uniformly random state (from
`random_pancake_state`) and each offset, sets `bound = h(state) + offset` and
calls `count_expansions_pancake` to get the label. Writes one row per (state,
bound). Key flags: `--heuristic {gap,rand}`, `--n` (default 12), `--n-states`,
`--offsets`, `--cap`, `--out`. Output: a `pancake_labels_*.csv` file.

### `generate_tiles_uniform.py`
Builds the 15-puzzle generalization dataset from uniformly random solvable
states (`random_solvable_state`), using Manhattan distance and
`count_expansions`. Flags: `--n-states`, `--offsets`, `--cap`, `--out`,
`--calibrate`. Output: `results/tiles_uniform_labels.csv`.

### `generate_data.py`
The pilot 15-puzzle generator using random-walk states (`random_walk_state`).
Superseded by `generate_tiles_uniform.py` for the paper. Output:
`results/labels.csv`.

## Distribution estimation

### `estimate_distribution.py`
Samples the 15-puzzle heuristic-value distribution and saves it as
`results/hdist_overall.npy` and an equilibrium-reweighted
`results/hdist_equilibrium.npy`. KRE on the 15-puzzle reads these.

## Training and evaluation

### `train_eval_pancake.py`
The main pancake experiment. Reads `results/pancake_labels_{heuristic}{suffix}.csv`,
splits by state, trains the GBM and MLP, runs KRE and CDP1 on the same test
states, and writes the comparison tables. Flags: `--heuristics`, `--n`,
`--data-suffix` (points at the `n=10` files), `--feature-set {full,minimal}`
(minimal restricts the learned models to CDP1's inputs). Output:
`results/pancake_eval_*.csv` and the `_by_offset.csv` variants.

### `train_eval.py`
The pilot 15-puzzle experiment. Reads `results/labels.csv` and the distribution
`.npy`, trains the learned models, and compares against KRE. Output:
`results/eval_table.csv` and figures.

### `tiles_generalization_eval.py`
The 15-puzzle generalization study on uniform states. Reads
`results/tiles_uniform_labels.csv` and `results/hdist_overall.npy`, compares the
learned models against KRE, and breaks error down by blank-cell class (corner,
edge, interior). Flags: `--data`, `--dist`, `--out`, `--fig`. Output:
`results/tiles_generalization_eval.csv`.

### `kre_sanity.py`
Large-set KRE check on the 15-puzzle. Reads the distribution `.npy` files and
reports the ratio of mean predicted to mean actual effort at bounds near the
mean heuristic value. Output: printed report.

## Results-chapter analyses

### `extrapolation_test.py`
Trains on shallow offsets and predicts an unseen deeper offset, with the test
states also held out. Output: `results/extrapolation_test.csv`.

### `extrapolation_tiers.py`
Repeats the extrapolation test for the minimal, full, and probe feature tiers
and in the reverse direction, with cluster-bootstrap standard errors. Output:
`results/extrapolation_tiers.csv`.

### `tail_errors.py`
Retrains the canonical models on the 12-pancake datasets and reports per-row
`log10` error, the 90th and 95th percentile factor error per method, and how
often each underpredicts by more than tenfold. Output: `results/tail_errors.csv`.

### `learning_curves.py`
Measures accuracy against the number of training states, over repeated samples,
using a fixed canonical split so results compare to the headline evaluation.
Flags: `--reps`, `--sizes`, `--feature-sets`, `--out-prefix`. For the paper use
`--reps 10 --out-prefix results/learning_curve10`. Output: a learning-curve CSV
and PNGs.

### `cost_ledger.py`
Measures the offline preparation cost of each predictor family: label generation
and model training for the learned models, distribution and conditional-matrix
sampling for the analytic ones. Output: `results/cost_ledger.csv`.

### `night_extras.py`
Two additive analyses. First, MLP permutation importance by shuffling each
feature column on the test set and measuring the RMSE increase, written to
`results/mlp_importance_{gap,rand}.csv` and `.png`. Second, per-prediction
latency for all methods, written to `results/prediction_latency.csv`.

### `probe_eval.py`
Evaluates the four feature tiers (minimal, full, probe combinations) with GBM and
MLP against KRE and CDP1. Output: `results/probe_eval.csv`.

### `probe_no_fertile.py`
Repeats the probe evaluation with the fertile-children count removed, to show the
probe gain does not depend on that one feature. Output:
`results/probe_no_fertile.csv`.

### `robustness_seeds.py`
Repeats the main evaluation over several split seeds, with the analytic
predictors' samplers held fixed, to check that the method ranking does not depend
on the split. Flags: `--seeds`, `--out`. For the paper use
`--seeds 0 1 2 3 4 5 6 7 8 9 --out results/robustness_seeds10.csv`. Output: a
robustness CSV.

### `gbm_grouped_check.py`
Diagnostic. Reruns the GBM under a strict group-disjoint early-stopping protocol
to confirm the published GBM numbers are unaffected by scikit-learn's internal
row-level split. Output: printed deltas.

## Figures

### `make_report_figures.py`
Reads the frozen result CSVs (and `results/pancake_labels_gap.csv` for the
scatter plot) and writes the five report figures as vector PDFs. Run the analyses
it depends on first (see [workflow.md](workflow.md)). Output: figure PDFs.
