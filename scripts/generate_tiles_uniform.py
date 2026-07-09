"""Labeled dataset for the 15-puzzle (n=4): uniformly random solvable states.

Generates (state, bound) -> (nodes, censored) rows using Manhattan distance
and the parent-pruned IDA* tree counter. All parameters travel inside job
tuples for macOS spawn safety.

f-parity: Manhattan h changes by exactly ±1 per tile slide, so
f = g + h can only change by 0 or 2 at each step. IDA* bounds that differ
from h(start) by an odd amount add no new nodes. Default offsets are all even.

Post-calibration recommended invocation used offsets 0 2 4 6 8 10 (pass via --offsets; argparse default remains 0-8).

Usage:
  # Calibration (30 states, offsets 0 2 4 6 8 10, then prints advice):
  python scripts/generate_tiles_uniform.py --calibrate

  # Main generation (1500 states, default offsets 0 2 4 6 8):
  python scripts/generate_tiles_uniform.py

  # Override offsets after calibration:
  python scripts/generate_tiles_uniform.py --offsets 0 2 4 6 8 10
"""

import argparse
import csv
import multiprocessing as mp
import random

from effortpred.count import count_expansions
from effortpred.features import FEATURE_NAMES, extract_features
from effortpred.heuristic import manhattan
from effortpred.puzzle import random_solvable_state


def label_one(job):
    """job = (state_tuple, offsets_tuple, cap_int) — all data in the tuple
    so multiprocessing.Pool spawn mode (macOS default) never touches globals."""
    state, offsets, cap = job
    n = 4  # 15-puzzle
    h0 = manhattan(state, n)
    rows = []
    for off in offsets:
        bound = h0 + off
        nodes, censored = count_expansions(state, n, bound, cap=cap)
        feats = extract_features(state, n, bound)
        rows.append({
            "state": " ".join(map(str, state)),
            "nodes": nodes,
            "censored": int(censored),
            **{k: feats[k] for k in FEATURE_NAMES},
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Generate 15-puzzle effort labels (uniform states).")
    ap.add_argument("--n-states", type=int, default=1500,
                    help="number of states (main run)")
    ap.add_argument("--offsets", type=int, nargs="+",
                    default=[0, 2, 4, 6, 8],
                    help="even offsets added to h(start) for bound")
    ap.add_argument("--cap", type=int, default=20_000_000,
                    help="node cap; censored=1 when hit")
    ap.add_argument("--seed", type=int, default=3,
                    help="RNG seed (deterministic deduplicated sample)")
    ap.add_argument("--out", default="results/tiles_uniform_labels.csv",
                    help="output CSV path")
    ap.add_argument("--workers", type=int,
                    default=max(1, mp.cpu_count() - 2),
                    help="parallel workers")
    ap.add_argument("--calibrate", action="store_true",
                    help="calibration run: 30 states, offsets 0 2 4 6 8 10, "
                         "prints per-offset table then advice")
    args = ap.parse_args()

    if args.calibrate:
        n_states = 30
        offsets = [0, 2, 4, 6, 8, 10]
        out = args.out.replace(".csv", "_calibrate.csv")
    else:
        n_states = args.n_states
        offsets = args.offsets
        out = args.out

    # Validate all offsets are even (f-parity constraint)
    for off in offsets:
        if off % 2 != 0:
            raise ValueError(f"offset {off} is odd — sliding-tile f-parity "
                             "requires even offsets with Manhattan heuristic")

    rng = random.Random(args.seed)
    seen, states = set(), []
    while len(states) < n_states:
        s = random_solvable_state(4, rng)
        if s in seen:
            continue
        seen.add(s)
        states.append(s)

    print(f"Generated {len(states)} unique states (seed={args.seed})")
    print(f"offsets={offsets}, cap={args.cap:,}, workers={args.workers}")

    jobs = [(s, tuple(offsets), args.cap) for s in states]
    fieldnames = ["state", "nodes", "censored"] + FEATURE_NAMES

    n_rows, n_censored = 0, 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        with mp.Pool(args.workers) as pool:
            for i, rows in enumerate(
                    pool.imap_unordered(label_one, jobs, chunksize=4)):
                for row in rows:
                    n_rows += 1
                    n_censored += row["censored"]
                    w.writerow(row)
                if (i + 1) % 10 == 0 or (i + 1) == len(jobs):
                    print(f"  {i + 1}/{len(jobs)} states done "
                          f"(censored so far: {n_censored})", flush=True)

    print(f"\nwrote {n_rows} rows to {out}")
    print(f"censored: {n_censored} / {n_rows} "
          f"({100 * n_censored / max(n_rows, 1):.1f}%) — REPORT this")

    # Sanity checks
    expected_rows = n_states * len(offsets)
    if n_rows != expected_rows:
        print(f"WARNING: expected {expected_rows} rows, got {n_rows}")
    else:
        print(f"Sanity OK: {n_rows} = {n_states} states × {len(offsets)} offsets")

    if args.calibrate:
        _print_calibration_table(out, offsets, args.cap)


def _print_calibration_table(csv_path, offsets, cap):
    """Read back the calibration CSV and print a decision table."""
    import statistics

    import pandas as pd

    df = pd.read_csv(csv_path)
    # gap == offset for the start state (bound = h0 + offset, features extracted
    # at start state so h_manhattan == h0 and gap = bound - h_manhattan = offset)
    print("\n=== CALIBRATION TABLE (30 states) ===")
    print(f"{'offset':>8} {'n_rows':>8} {'censored':>9} "
          f"{'med_nodes':>14} {'max_nodes':>14} {'cen%':>6}")
    print("-" * 64)
    for off in offsets:
        sub = df[df["gap"] == off]
        if len(sub) == 0:
            continue
        nodes_list = sub["nodes"].tolist()
        cen = int(sub["censored"].sum())
        med = statistics.median(nodes_list)
        mx = max(nodes_list)
        cen_pct = 100 * cen / len(sub)
        print(f"{off:>8} {len(sub):>8} {cen:>9} "
              f"{med:>14,.0f} {mx:>14,.0f} {cen_pct:>5.1f}%")
    print("-" * 64)

    # Decision rule: largest even offset with censored <= 1
    eligible = []
    for off in offsets:
        sub = df[df["gap"] == off]
        cen = int(sub["censored"].sum()) if len(sub) > 0 else 9999
        if cen <= 1:
            eligible.append(off)

    if eligible:
        max_ok = max(eligible)
        chosen = list(range(0, max_ok + 1, 2))
        print(f"\nDecision: max even offset with censored <= 1 is {max_ok}")
        print(f"  => use --offsets {' '.join(map(str, chosen))}")
        print(f"  => expect {1500 * len(chosen):,} rows in the main run")
    else:
        print(f"\nWARNING: all offsets exceed censored threshold. "
              f"Consider reducing cap or offsets.")
        chosen = [0, 2, 4, 6, 8]
        print(f"  => falling back to default --offsets {' '.join(map(str, chosen))}")

    print(f"\nMain generation command:")
    print(f"  python scripts/generate_tiles_uniform.py "
          f"--offsets {' '.join(map(str, chosen))}")


if __name__ == "__main__":
    main()
