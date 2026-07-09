# effortpred: Predicting IDA\* Search Effort

Can you predict how much work a search will take before you run it?

IDA\* (iterative-deepening A\*) explores a problem in a series of bounded passes.
The number of nodes a single pass expands swings by orders of magnitude from one
start state to another, even at the same bound. This project predicts that
number ahead of time, from features of the start state.

We train two learned models, a gradient-boosted tree ensemble (GBM) and a small
neural network (MLP), and compare them head to head against two analytic
predictors from the search literature: the KRE formula and the one-step
Conditional Distribution Prediction model (CDP1). The comparison runs on the
pancake puzzle, under a consistent heuristic and a controlled inconsistent one,
and on the 15-puzzle.

The short version of the finding: the learned models are the most accurate at
predicting a single instance, while CDP1 stays ahead at extrapolating to deeper
bounds under a consistent heuristic. The two approaches divide the work.

## Quick start

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Python 3.11 or newer is required. Everything runs on the CPU. The test suite
checks every component, exhaustively where the state space is small.

To reproduce the main pancake comparison (the slow generation step aside):

```
.venv/bin/python scripts/generate_pancake.py --heuristic gap  --offsets 0 1 2 3 --cap 20000000
.venv/bin/python scripts/generate_pancake.py --heuristic rand --offsets 0 1     --cap 2000000
.venv/bin/python scripts/train_eval_pancake.py
```

You do not have to run anything to read the reported numbers. The frozen result
tables are already in `results/`.

## Documentation

The full documentation lives in [`docs/`](docs/):

- **[overview.md](docs/overview.md)**: the prediction task, the four predictors,
  and the two domains, in plain terms. Start here for the ideas.
- **[workflow.md](docs/workflow.md)**: how the pipeline fits together and the
  exact commands to reproduce every result. Start here to run things.
- **[core-library.md](docs/core-library.md)**: a tour of the `effortpred/`
  package, module by module.
- **[scripts.md](docs/scripts.md)**: what each script in `scripts/` does.
- **[data-and-results.md](docs/data-and-results.md)**: the datasets and which
  committed CSV backs which table.

## Repository layout

```
effortpred/     Core library: domains, heuristics, node counters, the KRE
                and CDP predictors, features, models, and metrics.
scripts/        Command-line entry points for each stage of the pipeline.
tests/          Correctness gates, exhaustive on the small state spaces.
results/        Frozen summary CSVs that back the tables in the report.
docs/           The documentation linked above.
pyproject.toml  Package metadata and dependencies.
```

## Citation

If you use this code, please cite the accompanying report:

> M. Bohadana and N. Klein. Predicting IDA\* Search Effort: Learned Models vs.
> KRE and CDP1. Course 237-2-5513 (Search in Artificial Intelligence),
> Ben-Gurion University of the Negev.
