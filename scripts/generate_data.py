"""Full label generation.

Bounds are h(start) + EVEN offsets only (f-parity: odd increments add
nothing — tested in tests/test_count.py). All run parameters travel inside
each job tuple, so multiprocessing workers never read module-level state
(macOS spawn would resurrect defaults and silently ignore CLI overrides).
"""

import argparse
import csv
import multiprocessing as mp
import random

from effortpred.count import count_expansions
from effortpred.features import FEATURE_NAMES, extract_features
from effortpred.heuristic import manhattan
from effortpred.puzzle import random_walk_state

N = 4
WALK_MIN, WALK_MAX = 10, 70


def sample_unique_states(n_states, seed):
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n_states:
        length = rng.randint(WALK_MIN, WALK_MAX)
        s = random_walk_state(N, length, rng)
        if s in seen:
            continue
        seen.add(s)
        out.append((s, length))
    return out


def label_one(job):
    state, walk_length, offsets, cap = job
    h0 = manhattan(state, N)
    rows = []
    for off in offsets:
        bound = h0 + off
        nodes, censored = count_expansions(state, N, bound, cap=cap)
        feats = extract_features(state, N, bound)
        rows.append({
            "state": " ".join(map(str, state)),
            "walk_length": walk_length,
            "nodes": nodes,
            "censored": int(censored),
            **feats,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=2000)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--cap", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/labels.csv")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    assert all(off % 2 == 0 for off in args.offsets), "offsets must be even (f-parity)"

    jobs = [(s, wl, tuple(args.offsets), args.cap)
            for s, wl in sample_unique_states(args.n_states, args.seed)]
    fieldnames = ["state", "walk_length", "nodes", "censored"] + FEATURE_NAMES

    n_rows, n_censored = 0, 0
    with open(args.out, "w", newline="") as f:
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

    print(f"wrote {n_rows} rows to {args.out}; "
          f"censored rows: {n_censored} — REPORT this number, never hide it")


if __name__ == "__main__":
    main()
