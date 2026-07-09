# effortpred: Predicting IDA\* Search Effort

This repository predicts the number of nodes one bounded IDA\* pass expands, from
static features of the start state, before the search runs. It compares two
learned regressors, a gradient-boosted tree ensemble (GBM) and a small neural
network (MLP), against two analytic predictors from the search literature, the
KRE formula and the one-step Conditional Distribution Prediction model (CDP1).
The evaluation covers the pancake puzzle under a consistent heuristic (GAP) and a
controlled inconsistent heuristic (RAND), and the 15-puzzle under Manhattan
distance.

Result summary: the learned models are the most accurate on per-instance
prediction in every tested setting, while CDP1 remains the most accurate at
extrapolating to deeper bounds under the consistent heuristic. The two approaches
divide the work.

## Requirements

- Python 3.11 or newer.
- The dependencies in `pyproject.toml`: NumPy, pandas, scikit-learn, SciPy,
  Matplotlib, and PyTorch. All experiments run on the CPU.

## Setup and checks

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The test suite verifies each component, exhaustively where the state space is
small enough to enumerate.

## Main reproduction command

The main pancake comparison, after the slow generation step:

```
.venv/bin/python scripts/generate_pancake.py --heuristic gap  --offsets 0 1 2 3 --cap 20000000
.venv/bin/python scripts/generate_pancake.py --heuristic rand --offsets 0 1     --cap 2000000
.venv/bin/python scripts/train_eval_pancake.py
```

The frozen summary tables are already in `results/`, so the reported numbers can
be read without rerunning the pipeline.

## Documentation

Full documentation is in [`docs/`](docs/):

- [overview.md](docs/overview.md): the prediction task, the predictors, and the
  two domains.
- [workflow.md](docs/workflow.md): the data pipeline (state generation, node
  counting, dataset construction, feature extraction, model training) and the
  full reproduction commands.
- [core-library.md](docs/core-library.md): file-by-file reference for the
  `effortpred/` package.
- [scripts.md](docs/scripts.md): file-by-file reference for the `scripts/`
  entry points, including the generator mechanics.
- [tests.md](docs/tests.md): the correctness gates in `tests/`, file-by-file,
  and what each proves.
- [data-and-results.md](docs/data-and-results.md): the datasets and the map from
  each committed CSV to the table it backs.

## Repository layout

```
effortpred/     Core library: domains, heuristics, node counters, the KRE and
                CDP predictors, feature extractors, models, and metrics.
scripts/        Command-line entry points for each pipeline stage.
tests/          Correctness gates, exhaustive on the small state spaces.
results/        Frozen summary CSVs that back the tables in the report.
docs/           The documentation listed above.
pyproject.toml  Package metadata and dependencies.
```

## Citation

If you use this code, please cite the accompanying report:

> M. Bohadana and N. Klein. Predicting IDA\* Search Effort: Learned Models vs.
> KRE and CDP1. Course 237-2-5513 (Search in Artificial Intelligence),
> Ben-Gurion University of the Negev.
