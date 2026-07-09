# Overview: the problem and the predictors

This page explains what the project predicts and what it compares, in plain
terms. For the code, see [core-library.md](core-library.md). For how to run
everything, see [workflow.md](workflow.md).

## The prediction task

IDA\* (iterative-deepening A\*) solves a search problem by running a series of
depth-first passes. Each pass has a cost limit called the bound. A pass expands
every node whose estimated total cost `f = g + h` stays within the bound, where
`g` is the depth and `h` is the heuristic estimate to the goal. Nodes that pass
this test are called fertile.

The number of nodes one bounded pass expands swings by orders of magnitude from
one start state to another, even at the same bound. If you could predict that
number before the search runs, you could budget time, pick between algorithms,
or skip a problem you cannot afford.

That is the task here. Given a start state, an admissible heuristic, and a
bound, predict how many nodes one bounded IDA\* pass expands. Because the counts
span many orders of magnitude, every predictor works in `log10` of the node
count.

The tree we count is the parent-pruned brute-force tree: the children of a node
are all its moves except the one that undoes the move just made. There is no
other duplicate detection. This matches the memory model of IDA\* and the exact
setting the analytic predictors below were built for. The count does not stop at
the goal.

## The predictors we compare

The study puts two learned models against two analytic formulas, with two
simple baselines underneath to mark the floor.

### Learned models (the proposed approach)

We train two regressors offline on labeled examples, then use them to predict
for new states in constant time, before any node is generated.

- **GBM**: scikit-learn histogram gradient-boosted regression trees.
- **MLP**: a small two-layer neural network (64 units per layer) in PyTorch.

Both read a static feature vector computed from the start state and the bound.
The features describe the state itself (heuristic values, slacks, disorder,
local structure), which is information the analytic formulas cannot read.

### Analytic predictors (the baselines from the literature)

- **KRE** (Korf, Reid, and Edelkamp, 2001). Predicts effort from the size of
  the brute-force tree and the overall distribution of heuristic values. It
  issues one prediction per bound, the same for every start state, and it
  assumes a consistent heuristic.
- **CDP1** (Zahavi et al., 2010). The one-step Conditional Distribution
  Prediction model. It conditions on how the heuristic value changes from a
  parent to its children, which gives a per-state prediction and handles
  inconsistent heuristics. It still reads only heuristic values, not the state.

CDP1 is the strong baseline. The whole point of the learned models is to read
features of the state that CDP1's value-only model cannot express.

### Simple baselines (the floor)

- **Mean**: always predict the average training effort. The weak floor.
- **Slack table**: predict the average effort of training rows with the same
  slack `bound - h`. The strong floor, since slack already drives most of the
  effort.

A learned model that cannot beat the slack table has learned nothing beyond
the slack.

## The two domains and their heuristics

### Pancake puzzle (the main comparison)

A state is a permutation of `0..n-1`. A move flips the first `k` items. The goal
is the sorted order. Branching is uniform (`n-1` at the root, `n-2` below), so
the tree size has a closed form. The main experiments use `n=12`, the depth
study uses `n=10`, and all exhaustive correctness checks use `n=7`, which is
small enough to enumerate every state.

We use two kinds of heuristic on the pancake:

- **GAP** (consistent): counts adjacent pancakes whose sizes differ by more than
  one, including the plate below the bottom pancake. Each flip changes exactly
  one adjacency, so GAP changes by at most one per move.
- **GAP-2**: GAP that ignores the two smallest pancakes. Strictly weaker, still
  consistent.
- **RAND** (inconsistent): returns GAP or GAP-2 depending on a checksum of the
  state. Each part is admissible, so RAND is admissible, but the value can jump
  by more than one along a path, which makes it inconsistent. This is the
  controlled inconsistent heuristic the study needs.

We first tried to build the inconsistent heuristic by dual evaluation, a
standard trick. It does not work on the pancake, because GAP is self-dual: the
dual value always equals the original. The tests verify this for all 5,040
states of the 7-pancake. That is why RAND uses random selection instead.

### 15-puzzle (the generalization study)

A 4x4 grid of tiles with one blank. The heuristic is Manhattan distance. Unlike
the pancake, the branching is not uniform: the blank has 2, 3, or 4 moves
depending on where it sits. This tests whether the learned approach still works
when the tree shape varies within one search. On this domain we compare against
KRE only, because handling non-uniform branching well in CDP needs machinery
that is out of scope here.

## The headline finding

The two families split the work. On per-instance accuracy the learned models
are the most accurate in every setting tested. CDP1 stays the best at
extrapolating to an unseen deeper bound under the consistent heuristic, the
regime its transition model was built for. Under the inconsistent heuristic the
learned models lead on accuracy, and the GBM also leads on extrapolation. An
equal-information test, which restricts the learned models to CDP1's exact
inputs, shows the learned advantage survives even without the extra features.
