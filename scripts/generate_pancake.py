"""Labeled datasets for the n-pancake (default 12): (uniform state, bound) -> nodes,
for the consistent gap_h and the inconsistent rand_h. Uniform states (not
random walks) deliberately remove Phase 1's walk-vs-uniform confound.
All parameters travel inside job tuples (macOS spawn safety)."""

import argparse
import csv
import multiprocessing as mp
import random

from effortpred.pancake import gap_h, rand_h, random_pancake_state
from effortpred.pancake_count import count_expansions_pancake
from effortpred.pancake_features import (
    PANCAKE_FEATURE_NAMES, extract_pancake_features,
)

HEURISTICS = {"gap": gap_h, "rand": rand_h}


def label_one(job):
    state, heuristic_name, offsets, cap = job
    h_fn = HEURISTICS[heuristic_name]
    h0 = h_fn(state)
    rows = []
    for off in offsets:
        bound = h0 + off
        nodes, censored = count_expansions_pancake(state, bound, h_fn, cap=cap)
        feats = extract_pancake_features(state, bound)
        rows.append({
            "state": " ".join(map(str, state)),
            "heuristic": heuristic_name,
            "nodes": nodes,
            "censored": int(censored),
            **feats,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heuristic", choices=["gap", "rand"], required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n-states", type=int, default=1500)
    ap.add_argument("--offsets", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--cap", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()
    out = args.out or f"results/pancake_labels_{args.heuristic}.csv"

    rng = random.Random(args.seed)
    seen, states = set(), []
    while len(states) < args.n_states:
        s = random_pancake_state(args.n, rng)
        if s in seen:
            continue
        seen.add(s)
        states.append(s)

    jobs = [(s, args.heuristic, tuple(args.offsets), args.cap) for s in states]
    fieldnames = ["state", "heuristic", "nodes", "censored"] + PANCAKE_FEATURE_NAMES

    n_rows, n_censored = 0, 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        with mp.Pool(args.workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(label_one, jobs, chunksize=4)):
                for row in rows:
                    n_rows += 1
                    n_censored += row["censored"]
                    w.writerow(row)
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(jobs)} states done", flush=True)

    print(f"wrote {n_rows} rows to {out}; censored: {n_censored} — REPORT this")


if __name__ == "__main__":
    main()
