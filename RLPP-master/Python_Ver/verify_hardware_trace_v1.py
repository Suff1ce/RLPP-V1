#!/usr/bin/env python3
"""
Verify C++ hardware_trace_v1.bin against bundle decoder_logits_ref (same valid window as replay).

Layout must match C++ pack_rlpp_hardware_frame_v1() / hardware_io_contract_v1.txt.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

MAGIC_LE = 0x31504C52  # 'RLP1' on little-endian host
VERSION = 1


def load_csv_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def parse_trace(path: str, k_expected: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    frames: list[tuple[int, int, int, int, np.ndarray]] = []
    frame_bytes = 32 + 8 * k_expected
    while off < len(data):
        if off + frame_bytes > len(data):
            raise ValueError(
                f"Truncated trace at byte {off}: need {frame_bytes} bytes, have {len(data) - off}"
            )
        magic, ver, flags, seq, tbin, valid, lab, k = struct.unpack_from("<IHHQiiii", data, off)
        off += 32
        if magic != MAGIC_LE:
            raise ValueError(f"Bad magic at frame {len(frames)}: {magic:#x} expected {MAGIC_LE:#x}")
        if ver != VERSION:
            raise ValueError(f"Bad version {ver}")
        if k != k_expected:
            raise ValueError(f"Frame K={k} expected K={k_expected}")
        logits = np.frombuffer(data[off : off + 8 * k], dtype="<f8").copy()
        off += 8 * k
        frames.append((int(seq), int(tbin), int(valid), int(lab), logits))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, help="Path to hardware_trace_v1.bin")
    ap.add_argument("--bundle", required=True, help="Bundle directory with decoder_logits_ref.csv")
    ap.add_argument("--valid-col-start", type=int, default=0)
    ap.add_argument(
        "--valid-col-count",
        type=int,
        default=-1,
        help="-1 = all columns from start to end of reference",
    )
    ap.add_argument("--tol", type=float, default=1e-8, help="Max abs diff per logit")
    args = ap.parse_args()

    ref_path = os.path.join(args.bundle, "decoder_logits_ref.csv")
    if not os.path.isfile(ref_path):
        print(f"Missing {ref_path}", file=sys.stderr)
        return 1
    ref = load_csv_matrix(ref_path)
    if ref.ndim != 2:
        raise SystemExit("decoder_logits_ref must be 2D")
    k, t_valid = ref.shape[0], ref.shape[1]

    if args.valid_col_start < 0 or args.valid_col_start > t_valid:
        print("valid_col_start out of range", file=sys.stderr)
        return 1
    remaining = t_valid - args.valid_col_start
    out_cols = remaining if args.valid_col_count < 0 else min(args.valid_col_count, remaining)
    if out_cols < 0:
        print("valid_col_count out of range", file=sys.stderr)
        return 1

    ref_slice = ref[:, args.valid_col_start : args.valid_col_start + out_cols]

    try:
        frames = parse_trace(args.trace, k)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    if len(frames) != out_cols:
        print(
            f"Frame count {len(frames)} != replay window columns {out_cols}",
            file=sys.stderr,
        )
        return 1

    max_abs = 0.0
    label_mism = 0
    for i in range(out_cols):
        seq, _tbin, valid, lab, log = frames[i]
        if seq != i:
            print(f"Warning: frame {i} sequence={seq} expected {i}", file=sys.stderr)
        if valid != 1:
            print(f"Frame {i}: expected valid=1, got {valid}", file=sys.stderr)
            return 1
        ref_col = ref_slice[:, i]
        d = np.abs(log - ref_col)
        max_abs = max(max_abs, float(d.max()))
        pred = int(np.argmax(log) + 1)
        ref_lab = int(np.argmax(ref_col) + 1)
        if pred != ref_lab:
            label_mism += 1

    print(f"frames={len(frames)} K={k} max_abs_err={max_abs:.6e} label_mismatches={label_mism}")
    if label_mism != 0 or max_abs > args.tol:
        print("VERIFY FAIL", file=sys.stderr)
        return 1
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
