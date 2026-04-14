"""
Export an end-to-end Phase 3 "runner" parity case:
  (Python) DataLoader -> applynets_priori (fixed u01) -> emulator_real(...) -> reward smoothing -> getgradient -> update

This is meant to verify the *runner-style* loop (variable batch sizes/time indexes) in C++ without relying on matching RNG
implementations across languages. All stochastic draws needed for parity are exported as CSVs.

Run from repo root:
  python Python_Ver/export_phase3_run_case.py --model 01
  python Python_Ver/export_phase3_run_case.py --model 02
  python Python_Ver/export_phase3_run_case.py --model manual

Writes:
  C++_Ver/Phase_3/run_case_decoding01/
  C++_Ver/Phase_3/run_case_decoding02/
  C++_Ver/Phase_3/run_case_manual/
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_decoding_params_to_csv import ensure_decoding_param_csvs
from model.applynets_priori import applynets_priori
from model.getgradient import getgradient
from utils.DataLoader import DataLoader
from decoding.emulator_real import emulator_real


def _save_csv(path: str, a: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, a, delimiter=",", fmt="%.17g")


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("01", "02", "manual"), default="01")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Source dataset used by C++ runner by default.
    case_dir = os.path.join(root, "C++_Ver", "Phase_3", "loader_emulator_case")
    if not os.path.isdir(case_dir):
        raise RuntimeError(
            "Missing loader_emulator_case. Run: python Python_Ver/export_loader_emulator_reference.py"
        )

    inputEnsemble = np.loadtxt(os.path.join(case_dir, "inputEnsemble.csv"), delimiter=",")
    M1_truth = np.loadtxt(os.path.join(case_dir, "M1_truth.csv"), delimiter=",")
    Actions = np.loadtxt(os.path.join(case_dir, "Actions.csv"), delimiter=",").astype(int).reshape(-1)
    Trials = np.loadtxt(os.path.join(case_dir, "Trials.csv"), delimiter=",").astype(int).reshape(-1)

    # Ensure decoding CSV params exist (from .mat when present).
    dec_params_dir = os.path.join(root, "C++_Ver", "Phase_3", "decoding_params")
    ensure_decoding_param_csvs(dec_params_dir)

    if args.model == "01":
        out_dir = os.path.join(root, "C++_Ver", "Phase_3", "run_case_decoding01")
        py_model = "decodingModel_01"
        decoder_prefix = os.path.join(dec_params_dir, "decoding01")
    elif args.model == "02":
        out_dir = os.path.join(root, "C++_Ver", "Phase_3", "run_case_decoding02")
        py_model = "decodingModel_02"
        decoder_prefix = os.path.join(dec_params_dir, "decoding02")
    else:
        out_dir = os.path.join(root, "C++_Ver", "Phase_3", "run_case_manual")
        py_model = "decodingModel_manual"
        decoder_prefix = ""

    os.makedirs(out_dir, exist_ok=True)
    _write_text(os.path.join(out_dir, "model_name.txt"), py_model + "\n")
    _write_text(os.path.join(out_dir, "case_dir.txt"), os.path.normpath(case_dir).replace("\\", "/") + "\n")
    if decoder_prefix:
        _write_text(
            os.path.join(out_dir, "decoder_prefix.txt"),
            os.path.normpath(decoder_prefix).replace("\\", "/"),
        )

    # ----- Runner / training hyperparams (match C++ defaults where reasonable) -----
    episodes = int(args.episodes)
    H = 20
    priori_m, priori_n = 1.0, 1.0

    epsilon = 0.3
    max_episode_for_schedule = 1000
    discount_factor = 0.9
    discount_length = 5

    # DataLoader opt (note: Python DataLoader shuffles trainTrials when cursor==1)
    K = int(np.max(Trials))
    opt = {
        "Mode": "train",
        "DataLoaderCursor": 1,
        "batchSize": min(3, Actions.size),
        "discountLength": discount_length,
        "trainTrials": np.arange(1, K + 1),
        "NumberOfTrainTrials": K,
    }

    # ----- Network size inferred from dataset (same as C++ runner) -----
    feat = int(inputEnsemble.shape[0])
    Nx = feat + 1

    # Decoder-driven Ny/his for real trained decoders; manual uses Ny=8, his=2 in Python examples.
    if py_model == "decodingModel_01":
        Ny, his = 61, 1
    elif py_model == "decodingModel_02":
        Ny, his = 61, 2
    else:
        Ny, his = 8, 2

    # Fixed indexes permutation (exported; used by emulator).
    np.random.seed(2026)
    indexes = np.random.permutation(np.arange(1, Ny + 1)).astype(int)

    # Weights init (exported; avoids cross-language RNG matching).
    np.random.seed(1337)
    W1 = 2.0 * np.random.rand(H, Nx) - 1.0
    W2 = 2.0 * np.random.rand(Ny, H + 1) - 1.0

    _save_csv(os.path.join(out_dir, "W1_init.csv"), W1)
    _save_csv(os.path.join(out_dir, "W2_init.csv"), W2)
    _save_csv(os.path.join(out_dir, "indexes.csv"), indexes.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "his.csv"), np.array([[his]], dtype=np.float64))
    _save_csv(
        os.path.join(out_dir, "reward_params.csv"),
        np.array([[epsilon, max_episode_for_schedule, discount_factor, discount_length]], dtype=np.float64),
    )
    _save_csv(os.path.join(out_dir, "priori_params.csv"), np.array([[priori_m, priori_n]], dtype=np.float64))
    _save_csv(os.path.join(out_dir, "train_params.csv"), np.array([[episodes]], dtype=np.float64))

    # Export base dataset too (for debugging / inspection).
    _save_csv(os.path.join(out_dir, "inputEnsemble.csv"), inputEnsemble)
    _save_csv(os.path.join(out_dir, "M1_truth.csv"), M1_truth)
    _save_csv(os.path.join(out_dir, "Actions.csv"), Actions.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "Trials.csv"), Trials.reshape(-1, 1))

    # ----- Episode loop -----
    for ep in range(episodes):
        # Make DataLoader deterministic per-episode by fixing RNG used in permutation when cursor==1.
        # This is *not* attempting to match C++ RNG; instead we export batches and verify C++ against them.
        np.random.seed(9000 + ep)
        batchInput, batchM1Truth, batchActions, opt = DataLoader(inputEnsemble, M1_truth, Actions, Trials, opt)
        NumOfSamples = int(batchInput.shape[1])
        inputUnit = np.vstack([batchInput, np.ones((1, NumOfSamples))])

        # Fixed u01 for applynets_priori
        np.random.seed(10000 + ep)
        u01 = np.random.rand(Ny, NumOfSamples)

        orig_rand = np.random.rand

        def _rand_override(*shape):  # type: ignore[override]
            if shape != (Ny, NumOfSamples):
                return orig_rand(*shape)
            return u01.copy()

        np.random.rand = _rand_override  # type: ignore[assignment]
        try:
            p, hidden_u, spk = applynets_priori(inputUnit, W1, W2, NumOfSamples, ep, priori_m, priori_n)
        finally:
            np.random.rand = orig_rand  # type: ignore[assignment]

        success, suc_rate, motor_perform, _ensemble = emulator_real(
            spk, batchActions.astype(int), indexes, his, py_model
        )

        # Reward smoothing (RLPP.py)
        n_motor1 = np.sum(motor_perform == 1) + 1
        n_motor2 = np.sum(motor_perform == 2) + 1
        n_motor3 = np.sum(motor_perform == 3) + 1
        n_max = max(n_motor1, n_motor2, n_motor3)
        inner_reward = (motor_perform == 1) * (n_max / n_motor1 - 1) + (motor_perform == 2) * (
            n_max / n_motor2 - 1
        ) + (motor_perform == 3) * (n_max / n_motor3 - 1)

        reward = success + epsilon * (1 - ep / max_episode_for_schedule) * inner_reward
        temp = reward.copy()
        temp[np.isnan(reward)] = 0
        discount_filter = discount_factor ** np.arange(discount_length - 1, -1, -1)
        discount_filter /= discount_length
        full_conv = np.convolve(temp, discount_filter, mode="full")
        smoothed = full_conv[-len(reward) :]
        smoothed[~np.isnan(reward)] = (smoothed[~np.isnan(reward)] - np.mean(smoothed[~np.isnan(reward)])) / np.std(
            smoothed[~np.isnan(reward)]
        )
        smoothed[np.isnan(reward)] = 0

        dW2, dW1 = getgradient(smoothed, p, spk, hidden_u, inputUnit, W2, NumOfSamples)

        lr = 0.7 * (1 - ep / max_episode_for_schedule) + 0.5
        W2 = W2 + lr * dW2
        W1 = W1 + lr * dW1

        # ----- Export per-episode artifacts -----
        _save_csv(os.path.join(out_dir, f"batchInput_ep{ep}.csv"), batchInput)
        _save_csv(os.path.join(out_dir, f"batchM1Truth_ep{ep}.csv"), batchM1Truth.astype(np.float64))
        _save_csv(os.path.join(out_dir, f"batchActions_ep{ep}.csv"), batchActions.reshape(-1, 1))
        _save_csv(os.path.join(out_dir, f"inputUnit_ep{ep}.csv"), inputUnit)
        _save_csv(os.path.join(out_dir, f"u01_ep{ep}.csv"), u01)

        _save_csv(os.path.join(out_dir, f"p_ep{ep}.csv"), p)
        _save_csv(os.path.join(out_dir, f"hidden_ep{ep}.csv"), hidden_u)
        _save_csv(os.path.join(out_dir, f"spk_ep{ep}.csv"), spk.astype(np.float64))
        _save_csv(os.path.join(out_dir, f"success_ep{ep}.csv"), success.reshape(1, -1))
        _save_csv(os.path.join(out_dir, f"motor_perform_ep{ep}.csv"), motor_perform.reshape(-1, 1))
        _save_csv(os.path.join(out_dir, f"suc_rate_ep{ep}.csv"), np.array([[suc_rate]], dtype=np.float64))

        _save_csv(os.path.join(out_dir, f"smoothed_ep{ep}.csv"), smoothed.reshape(1, -1))
        _save_csv(os.path.join(out_dir, f"dW2_ep{ep}.csv"), dW2)
        _save_csv(os.path.join(out_dir, f"dW1_ep{ep}.csv"), dW1)
        _save_csv(os.path.join(out_dir, f"W2_after_ep{ep}.csv"), W2)
        _save_csv(os.path.join(out_dir, f"W1_after_ep{ep}.csv"), W1)

    print("Wrote", out_dir)


if __name__ == "__main__":
    main()

