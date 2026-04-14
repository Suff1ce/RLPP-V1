#!/usr/bin/env python3
"""
Phase 2.5 — Python reference seed sweep vs exported F1 bundle (same window as C++ replay).

Uses rlpp_bundle_replay_numpy.py (NumPy-only, no SciPy) so it matches the C++ pipeline math.
NumPy RNG differs from std::mt19937 — compare mismatch-rate and calibration *distributions*, not per-seed equality with C++.

Example:
  python phase2_seed_sweep_bundle_reference.py --bundle D:/rlpp_f1_bundle_rat01 \\
    --valid-col-start 0 --valid-col-count 100000 --seed-start 0 --seed-count 10
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from rlpp_bundle_replay_numpy import (
    load_bundle_and_decoder_maps,
    run_deterministic_replay,
    run_sampled_replay,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed sweep (sampled) vs F1 bundle references")
    ap.add_argument(
        "--bundle",
        type=str,
        default=os.environ.get("RLPP_F1_BUNDLE", "D:/rlpp_f1_bundle_rat01"),
        help="Directory with exported CSVs (same layout as C++ load_f1_bundle + decoder_*.csv)",
    )
    ap.add_argument("--tau-bins", type=float, default=150.0, help="Exponential history tau (bins)")
    ap.add_argument("--valid-col-start", type=int, default=0)
    ap.add_argument(
        "--valid-col-count",
        type=int,
        default=100000,
        help="Number of valid columns to record; use -1 for all remaining after start",
    )
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument(
        "--skip-deterministic-check",
        action="store_true",
        help="Skip the no-RNG deterministic replay sanity check (should match logits ref to ~1e-14)",
    )
    args = ap.parse_args()

    b = load_bundle_and_decoder_maps(args.bundle)

    if not args.skip_deterministic_check:
        print("[py] Deterministic replay (bundle downstream spikes, no RNG)...")
        dmet = run_deterministic_replay(
            upstream=b["upstream_spikes"],
            downstream_spikes_ref=b["downstream_spikes_ref"],
            generator_W1=b["generator_W1"],
            generator_W2=b["generator_W2"],
            sorted_indices_1based=b["sorted_indices_1based"],
            decoder_W1=b["decoder_W1"],
            decoder_b1=b["decoder_b1"],
            decoder_W2=b["decoder_W2"],
            decoder_b2=b["decoder_b2"],
            xoffset=b["xoffset"],
            gain=b["gain"],
            ymin=b["ymin"],
            decoder_logits_ref=b["decoder_logits_ref"],
            labels_ref=b["labels_ref"],
            tau_bins=args.tau_bins,
            valid_col_start=args.valid_col_start,
            valid_col_count=args.valid_col_count,
        )
        print(
            f"[py][det] max_abs_err={dmet.max_abs_err:.6e} mean_abs_err={dmet.mean_abs_err:.6e} "
            f"label_mismatches={dmet.label_mismatches}/{dmet.n_labels}"
        )
        if dmet.label_mismatches != 0 or dmet.max_abs_err > 1e-9:
            print(
                "[py] WARNING: deterministic Python path does not match bundle refs.",
                file=sys.stderr,
            )

    print(f"[py] Sampled replay: seeds {args.seed_start}..{args.seed_start + args.seed_count - 1}")
    rates = []
    cal_mean = []
    cal_max = []
    for s in range(args.seed_start, args.seed_start + args.seed_count):
        rng = np.random.default_rng(s)
        _, _, m = run_sampled_replay(
            upstream=b["upstream_spikes"],
            generator_W1=b["generator_W1"],
            generator_W2=b["generator_W2"],
            sorted_indices_1based=b["sorted_indices_1based"],
            decoder_W1=b["decoder_W1"],
            decoder_b1=b["decoder_b1"],
            decoder_W2=b["decoder_W2"],
            decoder_b2=b["decoder_b2"],
            xoffset=b["xoffset"],
            gain=b["gain"],
            ymin=b["ymin"],
            decoder_logits_ref=b["decoder_logits_ref"],
            labels_ref=b["labels_ref"],
            tau_bins=args.tau_bins,
            valid_col_start=args.valid_col_start,
            valid_col_count=args.valid_col_count,
            rng=rng,
        )
        rate = m.label_mismatches / max(1, m.n_labels)
        rates.append(rate)
        cal_mean.append(m.mean_abs_spike_calib)
        cal_max.append(m.max_abs_spike_calib)
        # max_abs_err vs frozen decoder_logits_ref is ~O(1) in sampled mode (different spikes vs export).
        print(
            f"[py][seed={s}] mismatches={m.label_mismatches}/{m.n_labels} rate={rate:.5f} "
            f"(logits_vs_det_ref_max_abs={m.max_abs_err:.3e}, ignore for sampled)"
        )
        print(
            f"  Spike-rate calibration: mean(|mean(s)-mean(p)|)={m.mean_abs_spike_calib:.8f} "
            f"max={m.max_abs_spike_calib:.8f}"
        )

    if rates:
        print(
            f"[py] Summary over {len(rates)} seeds: mismatch_rate mean={np.mean(rates):.5f} "
            f"std={np.std(rates):.5f} "
            f"cal_mean mean={np.mean(cal_mean):.8f} cal_max mean={np.mean(cal_max):.8f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
