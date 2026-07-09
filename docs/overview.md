# Overview: task, predictors, and domains

Factual summary of what the project predicts and what it compares. For the code,
see [core-library.md](core-library.md). For the pipeline, see
[workflow.md](workflow.md).

## Prediction task

IDA\* runs a series of depth-first passes, each with a cost bound. A pass expands
every node whose estimated total cost `f = g + h` is within the bound, where `g`
is the depth and `h` is the heuristic estimate. Nodes that pass this test are
fertile. The number of fertile nodes one bounded pass expands varies by orders of
magnitude across start states at the same bound.

The task is: given a start state, an admissible heuristic, and a bound, predict
the number of nodes one bounded pass expands. All predictors work in `log10` of
the node count.

The counted tree is the parent-pruned brute-force tree: the children of a node
are its moves minus the one that undoes the previous move, with no other
duplicate detection. The count does not stop at the goal. This is the estimand
that the analytic predictors KRE and CDP were defined for.

## Predictors compared

### Learned models (proposed method)
Two regressors trained offline on labeled examples, then applied to new states in
constant time before any node is generated:

- GBM: scikit-learn histogram gradient-boosted regression trees.
- MLP: a two-layer neural network (64 units per layer) in PyTorch.

Both read a static feature vector computed from the start state and the bound.
Implementation: `fit_gbm` and `fit_mlp` in `models.py`; features in
`pancake_features.py` and `features.py`.

### Analytic predictors (literature baselines)
- KRE (Korf, Reid, and Edelkamp, 2001): predicts effort from the brute-force tree
  size and the overall heuristic-value distribution. Issues one prediction per
  bound, the same for every start state, and assumes a consistent heuristic.
  Implementation: `kre.py`, `pancake_tree.py`.
- CDP1 (Zahavi et al., 2010): the one-step Conditional Distribution Prediction
  model. Conditions on how the heuristic value changes from a parent to its
  children, giving a per-state prediction that handles inconsistent heuristics. It
  reads heuristic values only, not features of the state. Implementation:
  `cdp.py`, `conditional.py`.

CDP1 is the strong baseline. The learned models are designed to read state
features that CDP1's value-only model cannot express.

### Simple baselines (floor)
- Mean: predict the average training effort (`MeanBaseline`).
- Slack table: predict the average training effort of rows with the same slack
  `bound - h` (`GapBaseline`). This is the strong floor, since slack drives most
  of the effort.

## Domains and heuristics

### Pancake puzzle (main comparison)
State: a permutation of `0..n-1`. Move: flip the first `k` items. Branching is
uniform (`n-1` at the root, `n-2` below), so the tree size has a closed form. The
main experiments use `n=12`, the depth study uses `n=10`, and the exhaustive
correctness checks use `n=7`. Domain code: `pancake.py`.

Heuristics on the pancake:

- GAP (consistent): counts adjacent pancakes whose sizes differ by more than one,
  including the plate below the stack. Changes by at most one per move.
- GAP-2: GAP ignoring the two smallest pancakes. Strictly weaker, still
  consistent.
- RAND (inconsistent): returns GAP or GAP-2 by a checksum of the state. Each part
  is admissible, so RAND is admissible, but values can jump by more than one along
  a path.

An earlier design tried to build the inconsistent heuristic by dual evaluation.
That fails on the pancake because GAP is self-dual: the dual value equals the
original. The tests verify this for all 5,040 states of the 7-pancake. RAND uses
random selection instead.

### 15-puzzle (generalization study)
A 4x4 grid with one blank and Manhattan distance as the heuristic. Branching is
non-uniform: the blank has 2, 3, or 4 moves depending on its position, so the tree
shape varies within one search. The comparison here is against KRE only, because
handling non-uniform branching in CDP needs machinery that is out of scope. Domain
code: `puzzle.py`, `heuristic.py`.

## Result summary

On per-instance accuracy the learned models are the most accurate in every tested
setting. CDP1 remains the most accurate at extrapolating to an unseen deeper bound
under the consistent heuristic, the regime its transition model targets. Under the
inconsistent heuristic the learned models lead on accuracy, and the GBM also leads
on extrapolation. An equal-information test, which restricts the learned models to
CDP1's exact inputs, shows the learned advantage survives without the extra
features.
