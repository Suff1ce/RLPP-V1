"""
Export decodingModel_01_para.mat / decodingModel_02_para.mat to CSV for C++ (load_decoding_nn_params_csv).

When the .mat files are absent (not checked into the repo), writes deterministic synthetic
weights with the same tensor layout so emulator_real parity tests can run.

Run:
  python Python_Ver/export_decoding_params_to_csv.py
  python Python_Ver/export_decoding_params_to_csv.py --out C++/Ver/Phase_3/loader_emulator_case

Optional:
  --synthetic   force synthetic export even if .mat exists
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _save_vec(path: str, a: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    v = np.asarray(a, dtype=float).reshape(-1, 1)
    np.savetxt(path, v, delimiter=",", fmt="%.17g")


def _save_mat(path: str, a: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, np.asarray(a, dtype=float), delimiter=",", fmt="%.17g")


def _save_scalar(path: str, x: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, np.array([[float(x)]], dtype=float), delimiter=",", fmt="%.17g")


def export_mat_to_prefix(mat_path: str, path_prefix: str) -> None:
    from scipy.io import loadmat

    params = loadmat(mat_path)
    x1_step1 = params["x1_step1"]
    xoffset = np.asarray(x1_step1["xoffset"].item(), dtype=float).flatten()
    gain = np.asarray(x1_step1["gain"].item(), dtype=float).flatten()
    ymin = float(np.asarray(x1_step1["ymin"].item()).reshape(-1)[0])
    b1 = np.asarray(params["b1"], dtype=float).flatten()
    IW1_1 = np.asarray(params["IW1_1"], dtype=float)
    b2 = np.asarray(params["b2"], dtype=float).flatten()
    LW2_1 = np.asarray(params["LW2_1"], dtype=float)

    _save_vec(path_prefix + "_xoffset.csv", xoffset)
    _save_vec(path_prefix + "_gain.csv", gain)
    _save_scalar(path_prefix + "_ymin.csv", ymin)
    _save_vec(path_prefix + "_b1.csv", b1)
    _save_mat(path_prefix + "_IW1_1.csv", IW1_1)
    _save_vec(path_prefix + "_b2.csv", b2)
    _save_mat(path_prefix + "_LW2_1.csv", LW2_1)


def write_synthetic_prefix(path_prefix: str, seed: int, input_dim: int, hidden: int, nclass: int) -> None:
    rng = np.random.default_rng(seed)
    xoffset = rng.random(input_dim)
    gain = rng.standard_normal(input_dim) * 0.1
    ymin = float(rng.standard_normal())
    b1 = rng.standard_normal(hidden)
    IW1_1 = rng.standard_normal((hidden, input_dim))
    b2 = rng.standard_normal(nclass)
    LW2_1 = rng.standard_normal((nclass, hidden))

    _save_vec(path_prefix + "_xoffset.csv", xoffset)
    _save_vec(path_prefix + "_gain.csv", gain)
    _save_scalar(path_prefix + "_ymin.csv", ymin)
    _save_vec(path_prefix + "_b1.csv", b1)
    _save_mat(path_prefix + "_IW1_1.csv", IW1_1)
    _save_vec(path_prefix + "_b2.csv", b2)
    _save_mat(path_prefix + "_LW2_1.csv", LW2_1)


def _clear_decoding_param_csvs(out_dir: str) -> None:
    for name in os.listdir(out_dir):
        if name.startswith("decoding01_") or name.startswith("decoding02_"):
            try:
                os.remove(os.path.join(out_dir, name))
            except OSError:
                pass


def _decoding_param_csvs_ok(out_dir: str, input_dim_01: int, input_dim_02: int) -> bool:
    for pre, dim in (("decoding01", input_dim_01), ("decoding02", input_dim_02)):
        px = os.path.join(out_dir, pre + "_xoffset.csv")
        if not os.path.isfile(px):
            return False
        xo = np.loadtxt(px, delimiter=",", ndmin=1)
        if int(np.asarray(xo).size) != dim:
            return False
        for suffix in ("_gain.csv", "_ymin.csv", "_b1.csv", "_IW1_1.csv", "_b2.csv", "_LW2_1.csv"):
            if not os.path.isfile(os.path.join(out_dir, pre + suffix)):
                return False
    return True


def ensure_decoding_param_csvs(out_dir: str) -> None:
    """Ensure decoding01_* / decoding02_* CSVs exist: prefer MATLAB exports next to decodingModel_*.py."""
    decoding_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decoding")
    m01 = os.path.join(decoding_dir, "decodingModel_01_para.mat")
    m02 = os.path.join(decoding_dir, "decodingModel_02_para.mat")

    if os.path.isfile(m01) and os.path.isfile(m02):
        export_mat_to_prefix(m01, os.path.join(out_dir, "decoding01"))
        export_mat_to_prefix(m02, os.path.join(out_dir, "decoding02"))
        return

    # Synthetic fallback (no .mat): Ny=8, his=2 -> 24 inputs for both (placeholder nets).
    input_dim = 24
    if _decoding_param_csvs_ok(out_dir, input_dim, input_dim):
        return
    _clear_decoding_param_csvs(out_dir)
    hidden = 18
    nclass = 3
    write_synthetic_prefix(os.path.join(out_dir, "decoding01"), seed=9001, input_dim=input_dim, hidden=hidden, nclass=nclass)
    write_synthetic_prefix(os.path.join(out_dir, "decoding02"), seed=9002, input_dim=input_dim, hidden=hidden, nclass=nclass)


def ensemble_ny_his_for_input_dim(input_dim: int) -> tuple[int, int]:
    """
    Pick (Ny, his) so (his+1)*Ny == input_dim (same stacking as emulator_real.py).
    Matches bundled decodingModel_01 (122) and decodingModel_02 (183) layouts.
    """
    if input_dim == 122:
        return 61, 1
    if input_dim == 183:
        return 61, 2
    if input_dim == 24:
        return 8, 2
    for his in range(0, 32):
        if input_dim % (his + 1) != 0:
            continue
        ny = input_dim // (his + 1)
        if ny >= 1:
            return ny, his
    raise ValueError(f"ensemble_ny_his_for_input_dim: cannot factor input_dim={input_dim}")


def decoding_nn_forward_numpy(ensemble: np.ndarray, path_prefix: str) -> np.ndarray:
    """Same math as decodingModel_01.py forward, weights loaded from CSV."""
    xoffset = np.loadtxt(path_prefix + "_xoffset.csv", delimiter=",", ndmin=1)
    gain = np.loadtxt(path_prefix + "_gain.csv", delimiter=",", ndmin=1)
    ymin = float(np.loadtxt(path_prefix + "_ymin.csv", delimiter=","))
    b1 = np.loadtxt(path_prefix + "_b1.csv", delimiter=",", ndmin=1)
    IW1_1 = np.loadtxt(path_prefix + "_IW1_1.csv", delimiter=",")
    b2 = np.loadtxt(path_prefix + "_b2.csv", delimiter=",", ndmin=1)
    LW2_1 = np.loadtxt(path_prefix + "_LW2_1.csv", delimiter=",")

    xoffset = np.asarray(xoffset, dtype=float).flatten()
    gain = np.asarray(gain, dtype=float).flatten()
    b1 = np.asarray(b1, dtype=float).flatten()
    b2 = np.asarray(b2, dtype=float).flatten()

    Xp = (ensemble - xoffset[:, None]) * gain[:, None] + ymin
    a1 = np.tanh(IW1_1 @ Xp + b1[:, None])
    n2 = LW2_1 @ a1 + b2[:, None]
    nmax = np.max(n2, axis=0, keepdims=True)
    ex = np.exp(n2 - nmax)
    den = np.sum(ex, axis=0, keepdims=True)
    den[den == 0] = 1.0
    return ex / den


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "C++_Ver", "Phase_3", "decoding_params"),
        help="Directory for decoding01_* / decoding02_* CSV files (consumed by C++ trained decoder emulator)",
    )
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic weights")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if args.synthetic:
        for f in (
            "decoding01_xoffset.csv",
            "decoding01_gain.csv",
            "decoding01_ymin.csv",
            "decoding01_b1.csv",
            "decoding01_IW1_1.csv",
            "decoding01_b2.csv",
            "decoding01_LW2_1.csv",
            "decoding02_xoffset.csv",
            "decoding02_gain.csv",
            "decoding02_ymin.csv",
            "decoding02_b1.csv",
            "decoding02_IW1_1.csv",
            "decoding02_b2.csv",
            "decoding02_LW2_1.csv",
        ):
            fp = os.path.join(out_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    ensure_decoding_param_csvs(out_dir)
    print("Wrote decoding params CSVs to", out_dir)


if __name__ == "__main__":
    main()
