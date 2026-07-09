# Workflow: how the pipeline fits together

This page is the end-to-end guide. It shows how the pieces connect and gives the
exact commands to reproduce the study. If you read one document to understand
the project, read this one. For the ideas behind each step, see
[overview.md](overview.md).

## The pipeline in one picture

Every experiment follows the same three stages: prepare the inputs, train and
evaluate, then plot. The learned models and the analytic predictors share the
same held-out test states, so the comparison is fair.

```
  STAGE 1  prepare inputs

  start states (uniform random)
     -> generate_pancake.py  (runs the bounded DFS counter on each state)
     -> label CSV  (columns: state, bound, node count)

  random states
     -> estimate_distribution.py  -> P(x)       (KRE reads this)
     -> conditional sampling      -> p(v|vp)    (CDP1 reads this)

  STAGE 2  train and evaluate

  label CSV
     -> train_eval_pancake.py
          . split 70/15/15 by state
          . train GBM and MLP on the train split
          . run KRE and CDP1 on the same held-out test states
          . score everyone in log10 space
     -> summary CSVs in results/

  STAGE 3  plot

  summary CSVs
     -> make_report_figures.py
     -> figure PDFs
```

The key idea: the label CSV is the ground truth (real node counts), and every
predictor is judged against it on the same test states. The learned models learn
from the training rows. The analytic predictors do not train, but they still
need an offline sampling pass to estimate their distributions.

## What connects to what

- The **label CSV** is produced by a generation script and read by the
  evaluation script. It is the only place the true node counts live.
- The **distribution files** (`P(x)` and the conditional matrices) are what let
  KRE and CDP1 predict without running the search. They are sampled from random
  states, separate from the labeled rows.
- The **summary CSVs** in `results/` are the scored comparison tables. These are
  committed to the repository, so you can read the paper's numbers without
  rerunning anything.
- The **figures** are drawn only from the summary CSVs (and one label file for
  the scatter), so they are cheap to regenerate once the CSVs exist.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

This installs the `effortpred` package and the test tools into a local virtual
environment. Python 3.11 or newer is required. Everything runs on the CPU.

## Check correctness first

Before trusting any result, run the test suite. It verifies each component
exhaustively where the state space is small enough.

```
.venv/bin/pytest
```

This checks, among other things, the 8-puzzle solvability rule against a
breadth-first search over all `9!` states, the pancake heuristics over all 5,040
states of the 7-pancake (including that GAP is self-dual and RAND is
inconsistent), the tree-size recurrence against explicit enumeration, and the
KRE and CDP code against the equations in their papers.

## Reproducing the study

All randomness is seeded, so every command is deterministic. The only slow part
is label generation: labeling the 12-pancake GAP set runs a bounded search on
1,500 start states and takes on the order of an hour on a multi-core CPU.
Everything after generation runs in seconds to a few minutes.

The stages below match the sections of the report. You can run just the ones you
care about. Each writes its output to `results/`.

### Stage A: main pancake comparison

Produces the headline table and the equal-information ablation on the
12-pancake, under both heuristics.

```
# Generate the labeled datasets (the slow step).
.venv/bin/python scripts/generate_pancake.py --heuristic gap  --offsets 0 1 2 3 --cap 20000000
.venv/bin/python scripts/generate_pancake.py --heuristic rand --offsets 0 1     --cap 2000000

# Full 13-feature models plus the KRE and CDP1 baselines.
.venv/bin/python scripts/train_eval_pancake.py

# Equal-information ablation: restrict the learned models to CDP1's exact
# inputs (the active heuristic value and the bound).
.venv/bin/python scripts/train_eval_pancake.py --feature-set minimal
```

### Stage B: depth study (10-pancake)

The smaller board reaches deeper offsets within the node cap.

```
.venv/bin/python scripts/generate_pancake.py --heuristic gap  --n 10 --n-states 1000 --offsets 0 1 2 3 4 --cap 20000000 --out results/pancake_labels_gap_n10.csv
.venv/bin/python scripts/generate_pancake.py --heuristic rand --n 10 --n-states 1000 --offsets 0 1 2 3   --cap 20000000 --out results/pancake_labels_rand_n10.csv
.venv/bin/python scripts/train_eval_pancake.py --n 10 --data-suffix _n10
.venv/bin/python scripts/train_eval_pancake.py --n 10 --data-suffix _n10 --feature-set minimal
```

### Stage C: extrapolation to an unseen depth

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

The figure script reads frozen CSVs and one raw label dataset (the scatter plot
is drawn from the individual 12-pancake GAP instances), so run stages A, B, D,
E, and G first, then:

```
.venv/bin/python scripts/make_report_figures.py
```

## A shortcut for checking the numbers

If you only want to confirm the reported numbers, you do not need to run
anything. The frozen summary CSVs are already in `results/`. See
[data-and-results.md](data-and-results.md) for which file backs which table.
