"""Offline cost ledger for the effort-prediction pipeline (Freeze-Blocker 3+4).

Measures and assembles:

  (a) estimate_pancake_distribution(12, gap_h, 200k) wall-time
  (b) sample_conditional_matrix(12, gap_h, 200k) wall-time
  (c) fit_gbm / fit_mlp training time on gap_n12 (full canonical split)
  (d) Label-generation cost: time a 20-state sample at gap n12 offsets 0-3
      cap 2e7 (single-threaded); extrapolate to 1500 states.
      (Repo ledger: full 1500-state gap n12 regeneration took ~13 min on this
       machine with workers.)
  (e) Implied minimal label cost at 50-state and 400-state learning-curve
      sufficiency points (single-threaded extrapolation from 20-state sample).

  [Deliverable 4] Node-equivalent latency:
      Measure pancake counter throughput (nodes/sec) on a ~1e5-node instance.
      Convert per-prediction latencies (MLP=0.5µs, GBM=10µs, KRE=2µs,
      CDP_1=16µs, probe_features=17.8µs) into equivalent node expansions.

Output: results/cost_ledger.csv  +  a readable printed table.
"""

import random
import sys
import os
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effortpred.cdp import cdp1_predict
from effortpred.conditional import H_MAX_PANCAKE, sample_conditional_matrix
from effortpred.models import fit_gbm, fit_mlp, split_by_state
from effortpred.pancake import (
    estimate_pancake_distribution,
    gap_h,
    random_pancake_state,
)
from effortpred.pancake_count import count_expansions_pancake
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES

N = 12
N_SAMPLES = 200_000
LABEL_GEN_STATES = 20
LABEL_OFFSETS = [0, 1, 2, 3]
LABEL_CAP = int(2e7)
FULL_N_STATES = 1500
FULL_REGEN_MIN = 13.0  # minutes, unverified estimate from session timestamps (log has no timing lines)

# Known per-prediction latencies (µs) from existing measurement
LATENCIES_US = {
    "mlp": 0.5,
    "gbm": 10.0,
    "kre": 2.0,
    "cdp1": 16.0,
    "probe_features": 17.8,
}


def timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def fmt_time(s):
    if s >= 60:
        return f"{s/60:.1f} min"
    return f"{s:.2f} s"


def main():
    rows = []

    # ------------------------------------------------------------------
    # (a) estimate_pancake_distribution(12, gap_h, 200k)
    # ------------------------------------------------------------------
    print("(a) Timing estimate_pancake_distribution(12, gap_h, 200k) ...")
    rng_a = random.Random(100)
    dist, t_dist = timed(estimate_pancake_distribution, N, gap_h, N_SAMPLES, rng_a)
    print(f"    -> {fmt_time(t_dist)}")
    rows.append(
        {
            "item": "dist_sampling",
            "description": "estimate_pancake_distribution(12, gap_h, 200k)",
            "wall_time_s": round(t_dist, 3),
            "note": "KRE analytic offline cost",
        }
    )

    # ------------------------------------------------------------------
    # (b) sample_conditional_matrix(12, gap_h, 200k)
    # ------------------------------------------------------------------
    print("(b) Timing sample_conditional_matrix(12, gap_h, 200k) ...")
    rng_b = random.Random(200)
    cond, t_cond = timed(
        sample_conditional_matrix, N, gap_h, N_SAMPLES, rng_b, h_max=H_MAX_PANCAKE(N)
    )
    print(f"    -> {fmt_time(t_cond)}")
    rows.append(
        {
            "item": "cond_sampling",
            "description": "sample_conditional_matrix(12, gap_h, 200k)",
            "wall_time_s": round(t_cond, 3),
            "note": "CDP_1 analytic offline cost",
        }
    )

    # ------------------------------------------------------------------
    # (c) fit_gbm / fit_mlp training time on gap_n12 full
    # ------------------------------------------------------------------
    print("(c) Timing GBM + MLP training on gap_n12 (full canonical split) ...")
    df = pd.read_csv("results/pancake_labels_gap.csv")
    df = df[df["censored"] == 0].copy()
    df["y"] = np.log10(df["nodes"].astype(float))
    train, val, test = split_by_state(df, seed=0)
    X_tr = train[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_tr = train["y"].values
    X_va = val[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_va = val["y"].values

    _, t_gbm = timed(fit_gbm, X_tr, y_tr, seed=0)
    print(f"    GBM -> {fmt_time(t_gbm)}")
    rows.append(
        {
            "item": "gbm_training",
            "description": "fit_gbm on gap_n12 full (n_train={})".format(len(X_tr)),
            "wall_time_s": round(t_gbm, 3),
            "note": "learned pipeline offline cost (training only)",
        }
    )

    _, t_mlp = timed(fit_mlp, X_tr, y_tr, X_va, y_va, seed=0)
    print(f"    MLP -> {fmt_time(t_mlp)}")
    rows.append(
        {
            "item": "mlp_training",
            "description": "fit_mlp on gap_n12 full (n_train={})".format(len(X_tr)),
            "wall_time_s": round(t_mlp, 3),
            "note": "learned pipeline offline cost (training only)",
        }
    )

    # ------------------------------------------------------------------
    # (d) Label-generation cost: 20-state sample, single-threaded
    # ------------------------------------------------------------------
    print(
        f"(d) Timing label generation for {LABEL_GEN_STATES} states "
        f"at gap n12 offsets {LABEL_OFFSETS} cap {LABEL_CAP:.0e} (single-threaded) ..."
    )
    rng_gen = random.Random(42)
    seen, sample_states = set(), []
    while len(sample_states) < LABEL_GEN_STATES:
        s = random_pancake_state(N, rng_gen)
        if s not in seen:
            seen.add(s)
            sample_states.append(s)

    def gen_labels_20():
        results = []
        for s in sample_states:
            h0 = gap_h(s)
            for off in LABEL_OFFSETS:
                bound = h0 + off
                cnt, cens = count_expansions_pancake(s, bound, gap_h, cap=LABEL_CAP)
                results.append((cnt, cens))
        return results

    _, t_gen20 = timed(gen_labels_20)
    print(f"    20 states -> {fmt_time(t_gen20)}")

    # Extrapolate to 1500 states (linear scale; actual run used workers)
    t_gen_full_extrap = t_gen20 * FULL_N_STATES / LABEL_GEN_STATES
    print(
        f"    Extrapolated to {FULL_N_STATES} states (single-threaded): "
        f"{fmt_time(t_gen_full_extrap)}"
    )
    print(
        f"    Repo ledger (parallel workers): full gap n12 regen ~{FULL_REGEN_MIN:.0f} min"
    )

    rows.append(
        {
            "item": "label_gen_20states",
            "description": f"gen labels {LABEL_GEN_STATES} states gap n12 off={LABEL_OFFSETS} cap 2e7 (single-threaded)",
            "wall_time_s": round(t_gen20, 3),
            "note": "direct measurement",
        }
    )
    rows.append(
        {
            "item": "label_gen_full_extrap",
            "description": f"extrapolated to {FULL_N_STATES} states (single-threaded, linear)",
            "wall_time_s": round(t_gen_full_extrap, 1),
            "note": "extrapolation from 20-state sample",
        }
    )
    rows.append(
        {
            "item": "label_gen_full_actual",
            "description": f"full gap n12 regen (repo ledger, parallel workers)",
            "wall_time_s": round(FULL_REGEN_MIN * 60, 0),
            "note": "unverified estimate from session timestamps (gen_gap20m.log has no timing lines)",
        }
    )

    # ------------------------------------------------------------------
    # (e) Minimal label cost at learning-curve sufficiency points
    # ------------------------------------------------------------------
    for n_states, label in [(50, "50-state"), (400, "400-state")]:
        t_extrap = t_gen20 * n_states / LABEL_GEN_STATES
        rows.append(
            {
                "item": f"label_gen_{n_states}states",
                "description": f"implied label cost at {n_states} states (single-threaded extrap.)",
                "wall_time_s": round(t_extrap, 1),
                "note": f"learning-curve sufficiency point ({label})",
            }
        )
        print(
            f"    Implied label cost at {n_states} states: {fmt_time(t_extrap)}"
        )

    # ------------------------------------------------------------------
    # [Deliverable 4] Node-equivalent latency
    # ------------------------------------------------------------------
    print("\n[D4] Measuring pancake counter throughput on a ~1e5-node instance ...")

    # Find a state + bound that gives roughly 1e5 nodes
    rng_thr = random.Random(7)
    target_nodes = 1e5
    best_state, best_bound, best_nodes = None, None, 0

    # Try states until we find one with ~1e5 nodes
    for _ in range(200):
        s = random_pancake_state(N, rng_thr)
        h0 = gap_h(s)
        for off in range(1, 6):
            bound = h0 + off
            cnt, cens = count_expansions_pancake(s, bound, gap_h, cap=int(5e5))
            if not cens and 5e4 <= cnt <= 5e5:
                if best_state is None or abs(cnt - target_nodes) < abs(best_nodes - target_nodes):
                    best_state, best_bound, best_nodes = s, bound, cnt

    if best_state is None:
        # Fallback: use whatever we found
        s = random_pancake_state(N, rng_thr)
        h0 = gap_h(s)
        best_state, best_bound = s, h0 + 3
        best_nodes, _ = count_expansions_pancake(best_state, best_bound, gap_h)

    print(f"    Benchmark instance: state={str(best_state[:4])+'...'}  "
          f"bound={best_bound}  nodes={best_nodes:,}")

    # Time multiple runs to get stable throughput
    N_RUNS = 5
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        count_expansions_pancake(best_state, best_bound, gap_h)
        times.append(time.perf_counter() - t0)

    median_t = float(np.median(times))
    throughput = best_nodes / median_t
    print(f"    Median time per run: {median_t*1000:.1f} ms")
    print(f"    Throughput: {throughput:,.0f} nodes/sec")

    rows.append(
        {
            "item": "counter_throughput",
            "description": f"pancake DFS counter throughput (n={N}, {best_nodes:,} nodes, median of {N_RUNS} runs)",
            "wall_time_s": round(median_t, 4),
            "note": f"{throughput:,.0f} nodes/sec",
        }
    )

    # Convert latencies to node equivalents
    print(f"\n[D4] Per-prediction latency -> node equivalents")
    print(f"     (throughput = {throughput:,.0f} nodes/sec)")
    print(f"     node_equiv = latency_µs / (1e6 / throughput)")
    print(f"     {'Method':<20} {'Latency (µs)':>14} {'Node-equiv':>14}")
    print(f"     {'-'*50}")

    node_equiv_rows = []
    for method, lat_us in LATENCIES_US.items():
        # time per node expansion in µs = 1e6 / throughput
        time_per_node_us = 1e6 / throughput
        node_equiv = lat_us / time_per_node_us
        node_equiv_rows.append(
            {
                "item": f"node_equiv_{method}",
                "description": f"{method} per-prediction latency in node-expansions",
                "wall_time_s": lat_us * 1e-6,
                "note": f"{node_equiv:.1f} node-equiv (lat={lat_us}µs, tput={throughput:.0f} n/s)",
            }
        )
        print(f"     {method:<20} {lat_us:>14.1f} {node_equiv:>14.1f}")

    rows.extend(node_equiv_rows)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_df = pd.DataFrame(rows)
    out_path = "results/cost_ledger.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n\nWrote {out_path}")

    # Pretty-print summary table
    print("\n" + "=" * 80)
    print("COST LEDGER SUMMARY")
    print("=" * 80)

    categories = [
        ("ANALYTIC offline (dist/cond sampling, gap_h n=12)", ["dist_sampling", "cond_sampling"]),
        ("LEARNED pipeline training (gap_n12)", ["gbm_training", "mlp_training"]),
        ("Label generation (gap n12 offsets 0-3 cap 2e7)", [
            "label_gen_20states", "label_gen_full_extrap",
            "label_gen_full_actual", "label_gen_50states", "label_gen_400states",
        ]),
        ("Node-equivalent latency", [
            f"node_equiv_{m}" for m in LATENCIES_US
        ]),
    ]

    for category, items in categories:
        print(f"\n  {category}")
        for item in items:
            sub = out_df[out_df["item"] == item]
            if sub.empty:
                continue
            r = sub.iloc[0]
            t = r["wall_time_s"]
            if item.startswith("node_equiv"):
                display = r["note"]
            else:
                display = fmt_time(float(t))
                if r["note"]:
                    display += f"  [{r['note']}]"
            print(f"    {r['description']:<65} {display}")

    # Bottom line: learned vs analytic offline cost comparison
    print("\n" + "-" * 80)
    print("BOTTOM LINE: Learned vs Analytic offline cost")
    print("-" * 80)

    t_analytic = t_dist + t_cond
    t_labels_full = t_gen_full_extrap
    t_labels_400 = t_gen20 * 400 / LABEL_GEN_STATES
    t_labels_50 = t_gen20 * 50 / LABEL_GEN_STATES
    t_train = t_gbm + t_mlp
    t_learned_full = t_labels_full + t_train
    t_learned_400 = t_labels_400 + t_train
    t_learned_50 = t_labels_50 + t_train

    print(f"  Analytic (dist + cond)          : {fmt_time(t_analytic)}")
    print(f"  Learned full ({FULL_N_STATES} states)      : {fmt_time(t_learned_full)} "
          f"(labels {fmt_time(t_labels_full)} + training {fmt_time(t_train)})")
    print(f"  Learned at 400 states           : {fmt_time(t_learned_400)} "
          f"(labels {fmt_time(t_labels_400)} + training {fmt_time(t_train)})")
    print(f"  Learned at 50 states            : {fmt_time(t_learned_50)} "
          f"(labels {fmt_time(t_labels_50)} + training {fmt_time(t_train)})")

    rows.append({
        "item": "summary_analytic_total",
        "description": "analytic offline total (dist + cond)",
        "wall_time_s": round(t_analytic, 3),
        "note": "KRE + CDP_1 ready",
    })
    rows.append({
        "item": "summary_learned_full",
        "description": f"learned offline total (labels {FULL_N_STATES} states + training)",
        "wall_time_s": round(t_learned_full, 1),
        "note": "single-threaded extrapolation",
    })
    rows.append({
        "item": "summary_learned_400",
        "description": "learned offline total (labels 400 states + training)",
        "wall_time_s": round(t_learned_400, 1),
        "note": "single-threaded extrapolation",
    })
    rows.append({
        "item": "summary_learned_50",
        "description": "learned offline total (labels 50 states + training)",
        "wall_time_s": round(t_learned_50, 1),
        "note": "single-threaded extrapolation",
    })

    # Overwrite with final rows
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
