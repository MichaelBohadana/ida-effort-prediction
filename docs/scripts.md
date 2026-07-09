# The scripts (`scripts/`)

Each script is a command-line entry point for one stage of the study. They
import the logic from [the core library](core-library.md) and add argument
parsing and file output. This page says what each script does and its main
options. For the order to run them in, see [workflow.md](workflow.md).

Every script is run through the project's virtual environment, for example:

```
.venv/bin/python scripts/train_eval_pancake.py --help
```

All scripts take a `--seed` and write their output under `results/`. The
defaults reproduce the numbers in the paper, so most runs need no flags.

## Data generation

- **`generate_pancake.py`** labels pancake states. For each state and offset it
  runs the bounded counter and records the node count. Main options:
  `--heuristic {gap,rand}`, `--n` (pancake size, default 12), `--n-states`,
  `--offsets`, `--cap` (the node cap for censoring), and `--out`.
- **`generate_data.py`** does the same for the 15-puzzle using random-walk start
  states. This is the earlier pilot; the paper's 15-puzzle study uses uniform
  states instead (the next script).
- **`generate_tiles_uniform.py`** labels 15-puzzle states sampled uniformly at
  random. This is the dataset behind the generalization study. It can also run a
  `--calibrate` pass to pick the offset range and cap.
- **`calibrate_pancake.py`** and **`calibrate.py`** are small pilot runs that
  report node counts across offsets, so you can pick an offset range and cap
  that finish without censoring. They inform the flags for the generation
  scripts and do not need to be rerun to reproduce frozen data.

## Distribution estimation

- **`estimate_distribution.py`** samples the 15-puzzle heuristic-value
  distribution `P(x)` and saves it (`results/hdist_overall.npy` and an
  equilibrium-reweighted variant). KRE on the 15-puzzle reads these.

## Training and evaluation

- **`train_eval_pancake.py`** is the main pancake experiment. It trains the GBM
  and the MLP, runs the KRE and CDP1 baselines, and writes the comparison
  tables. Key options: `--heuristics`, `--n`, `--data-suffix` (to point at the
  `n=10` datasets), and `--feature-set {full,minimal}`, where `minimal`
  restricts the learned models to CDP1's exact inputs for the equal-information
  test.
- **`train_eval.py`** is the 15-puzzle version of the pilot, comparing the
  learned models against KRE. It reads the label file and the distribution
  `.npy`.
- **`tiles_generalization_eval.py`** is the 15-puzzle generalization study on
  uniform states. It reports the learned models against KRE and breaks the error
  down by blank-cell class (corner, edge, interior).
- **`kre_sanity.py`** runs a large-set KRE check on the 15-puzzle and reports the
  ratio of mean predicted to mean actual effort.

## Analyses in the results chapter

- **`extrapolation_test.py`** trains on shallow offsets and predicts an unseen
  deeper offset, with the test states also held out.
- **`extrapolation_tiers.py`** repeats the extrapolation test for every feature
  tier and also in the reverse direction, with bootstrap standard errors.
- **`tail_errors.py`** reports the worst-case error (90th and 95th percentile
  factor error) and how often each predictor underpredicts by more than tenfold.
- **`learning_curves.py`** measures accuracy against the number of training
  states. Use `--reps 10 --out-prefix results/learning_curve10` for the version
  in the paper.
- **`cost_ledger.py`** measures the offline preparation cost of each predictor
  family (label generation and training versus distribution and matrix
  sampling).
- **`night_extras.py`** produces the permutation-importance tables
  (`mlp_importance_{gap,rand}.csv`) and the per-prediction latency
  (`prediction_latency.csv`).
- **`probe_eval.py`** evaluates the one-step probe tier against the static
  tiers.
- **`probe_no_fertile.py`** repeats the probe evaluation without the
  fertile-children count, to show the probe gain does not depend on that one
  feature.
- **`robustness_seeds.py`** repeats the main evaluation over several split
  seeds. Use `--seeds 0 1 2 3 4 5 6 7 8 9 --out results/robustness_seeds10.csv`
  for the ten-seed version.
- **`gbm_grouped_check.py`** is a small diagnostic that confirms the grouped
  split behaves as intended.

## Figures

- **`make_report_figures.py`** reads the frozen result CSVs (and one raw pancake
  label file for the scatter plot) and writes the report figures as PDFs. Run
  the analyses it depends on first; see [workflow.md](workflow.md).
