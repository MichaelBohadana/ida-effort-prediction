# Scripts reference (`scripts/`)

File-by-file reference for the command-line entry points, with the mechanics of
the data generators. Each script imports logic from
[the core library](core-library.md) and adds argument parsing and file output.
For the run order see [workflow.md](workflow.md). For the correctness gates see
[tests.md](tests.md).

Run each script through the virtual environment, for example
`.venv/bin/python scripts/train_eval_pancake.py --help`. Every script takes
`--seed` and writes under `results/`. The defaults reproduce the reported numbers.

## Data generation

### `generate_pancake.py`
Builds the pancake label datasets. Mechanics:

1. Sampling. It draws `--n-states` uniformly random permutations from
   `random_pancake_state` (default seed 2), deduplicating through a `set`, so each
   start state is distinct.
2. Job construction. Each state becomes a job tuple `(state, heuristic_name,
   offsets, cap)`. All parameters travel inside the tuple so multiprocessing
   workers under the macOS spawn start method never read module-level globals.
3. Labeling. For each state, `label_one` computes `h0 = h_fn(state)` once, then for
   each offset sets `bound = h0 + offset` and calls
   `count_expansions_pancake(state, bound, h_fn, cap)` to get the exact node count
   and the censored flag. It also calls `extract_pancake_features(state, bound)`
   and writes the features into the same row.
4. Output. One row per (state, offset) with columns `state` (space-joined),
   `heuristic`, `nodes`, `censored`, and the 13 pancake features. Rows are streamed
   from `pool.imap_unordered`. The script prints the total row count and the
   censored count.

Key flags: `--heuristic {gap,rand}` (selects `gap_h` or `rand_h`), `--n` (pancake
size, default 12), `--n-states`, `--offsets`, `--cap`, `--out`. Default output:
`results/pancake_labels_{heuristic}.csv`.

### `generate_tiles_uniform.py`
Builds the 15-puzzle generalization dataset. Mechanics:

1. Sampling. It draws `--n-states` uniformly random solvable states from
   `random_solvable_state(4, rng)` (default seed 3), deduplicated.
2. Offset validation. Every offset must be even. Manhattan distance changes by
   exactly one per move, so an odd offset above `h(start)` adds no fertile nodes.
   An odd offset raises an error.
3. Labeling. For each state, `label_one` sets `bound = manhattan(state) + offset`
   and calls `count_expansions(state, 4, bound, cap)`, then
   `extract_features(state, 4, bound)`. Rows carry `state`, `nodes`, `censored`,
   and the 15-puzzle features.
4. Calibration mode. With `--calibrate` it runs 30 states over offsets 0 to 10,
   writes a `_calibrate.csv`, and prints a per-offset table (median and max nodes,
   censored percent) plus the largest even offset that stays within the cap.

Key flags: `--n-states`, `--offsets` (even, default 0 2 4 6 8), `--cap` (default
2e7), `--out`, `--calibrate`. Default output:
`results/tiles_uniform_labels.csv`.

### `generate_data.py`
The pilot 15-puzzle generator using random-walk states (`random_walk_state`)
instead of uniform states. Superseded by `generate_tiles_uniform.py` for the
paper, because random-walk states would confound KRE's per-state error. Output:
`results/labels.csv`.

### `calibrate_pancake.py`, `calibrate.py`
Pilot runs that report node count, wall time, and censoring per offset (pancake
and 15-puzzle) so the offset range and cap can be chosen. Output: printed tables.
They do not need rerunning to reproduce frozen data.

## Distribution estimation

### `estimate_distribution.py`
Samples the 15-puzzle heuristic-value distribution and saves
`results/hdist_overall.npy` (uniform states) and `results/hdist_equilibrium.npy`
(reweighted by the blank-cell equilibrium). KRE on the 15-puzzle reads these. The
pancake distributions are sampled inline by the pancake evaluators, not here.

## Training and evaluation

### `train_eval_pancake.py`
The main pancake experiment. It reads
`results/pancake_labels_{heuristic}{suffix}.csv`, splits by state with
`split_by_state`, extracts features, trains the GBM and MLP, and runs KRE
(`kre_predict_pancake`) and CDP1 (`cdp1_predict`) on the same held-out test rows.
It samples the value distribution and the conditional matrix inline (200,000
states each by default). Flags: `--heuristics`, `--n`, `--data-suffix` (points at
the `n=10` files), `--feature-set {full,minimal}` (minimal restricts the learned
models to CDP1's two inputs, the active heuristic value and the bound). Output:
`results/pancake_eval_*.csv` and the `_by_offset.csv` variants.

### `train_eval.py`
The pilot 15-puzzle experiment. Reads `results/labels.csv` and the distribution
`.npy`, trains the learned models, and compares against KRE. Output:
`results/eval_table.csv` and figures.

### `tiles_generalization_eval.py`
The 15-puzzle generalization study on uniform states. Reads
`results/tiles_uniform_labels.csv` and `results/hdist_overall.npy`, compares the
learned models against KRE, and breaks the error down by blank-cell class (corner,
edge, interior). Flags: `--data`, `--dist`, `--out`, `--fig`. Output:
`results/tiles_generalization_eval.csv`.

### `kre_sanity.py`
Large-set KRE check on the 15-puzzle. Reads the distribution `.npy` files and
reports the ratio of mean predicted to mean actual effort at bounds near the mean
heuristic value. Output: printed report.

## Results-chapter analyses

### `extrapolation_test.py`
Trains on shallow offsets and predicts an unseen deeper offset, with the test
states also held out (double holdout). Output: `results/extrapolation_test.csv`.

### `extrapolation_tiers.py`
Repeats the extrapolation test for the minimal, full, and probe feature tiers and
in the reverse direction, with cluster-bootstrap standard errors. Output:
`results/extrapolation_tiers.csv`.

### `tail_errors.py`
Retrains the canonical models on the 12-pancake datasets and reports per-row
`log10` error, the 90th and 95th percentile factor error per method, and the rate
of underprediction beyond tenfold. Output: `results/tail_errors.csv`.

### `learning_curves.py`
Measures accuracy against the number of training states, over repeated samples,
on a fixed canonical split so the numbers compare to the headline evaluation.
Flags: `--reps`, `--sizes`, `--feature-sets`, `--out-prefix`. The paper uses
`--reps 10 --out-prefix results/learning_curve10`. Output: a learning-curve CSV
and PNGs.

### `cost_ledger.py`
Measures the offline preparation cost of each predictor family: label generation
and model training for the learned models, distribution and conditional-matrix
sampling for the analytic ones. Output: `results/cost_ledger.csv`.

### `night_extras.py`
Two additive analyses. First, MLP permutation importance, by shuffling each
feature column on the test set and measuring the RMSE increase, written to
`results/mlp_importance_{gap,rand}.csv` and `.png`. Second, per-prediction latency
for all methods, written to `results/prediction_latency.csv`.

### `probe_eval.py`
Evaluates the feature tiers (minimal, full, and probe combinations) with GBM and
MLP against KRE and CDP1. Output: `results/probe_eval.csv`.

### `probe_no_fertile.py`
Repeats the probe evaluation with the fertile-children count removed, to show the
probe gain does not depend on that one feature. Output:
`results/probe_no_fertile.csv`.

### `robustness_seeds.py`
Repeats the main evaluation over several split seeds with the analytic
predictors' samplers held fixed, to confirm the method ranking is stable across
splits. Flags: `--seeds`, `--out`. The paper uses
`--seeds 0 1 2 3 4 5 6 7 8 9 --out results/robustness_seeds10.csv`. Output: a
robustness CSV.

### `gbm_grouped_check.py`
Diagnostic. Reruns the GBM under a strict group-disjoint early-stopping protocol
to confirm the published GBM numbers are unaffected by scikit-learn's internal
row-level validation split. Output: printed deltas.

## Figures

### `make_report_figures.py`
Reads the frozen result CSVs (and `results/pancake_labels_gap.csv` for the scatter
plot) and writes the five report figures as vector PDFs. Run the analyses it
depends on first (see [workflow.md](workflow.md)). Output: figure PDFs.
