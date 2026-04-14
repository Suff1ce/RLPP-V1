"""
Export Phase 3 full RLPP loop parity:
  applynets_priori (u01) -> emulator_real(decodingModel_01 | decodingModel_02)
  -> RLPP reward -> getgradient -> weight update

Uses Ny=61 with his=1 (122-dim decoder 01) or his=2 (183-dim decoder 02).
Requires decodingModel_*_para.mat under Python_Ver/decoding/ (or synthetic CSV fallback).

Run from repo root:
  python Python_Ver/export_phase3_full_loop_trained_decoder_case.py --model 01
  python Python_Ver/export_phase3_full_loop_trained_decoder_case.py --model 02

Writes:
  C++_Ver/Phase_3/full_loop_decoding01_case/   or   full_loop_decoding02_case/
  plus decoder_prefix.txt (path for C++ load_decoding_nn_params_csv)
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
    parser.add_argument("--model", choices=("01", "02"), default="01", help="decodingModel_01 (his=1) or _02 (his=2)")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dec_params_dir = os.path.join(root, "C++_Ver", "Phase_3", "decoding_params")
    ensure_decoding_param_csvs(dec_params_dir)

    if args.model == "01":
        his = 1
        out_dir = os.path.join(root, "C++_Ver", "Phase_3", "full_loop_decoding01_case")
        py_model = "decodingModel_01"
        prefix_rel = os.path.join(dec_params_dir, "decoding01")
    else:
        his = 2
        out_dir = os.path.join(root, "C++_Ver", "Phase_3", "full_loop_decoding02_case")
        py_model = "decodingModel_02"
        prefix_rel = os.path.join(dec_params_dir, "decoding02")

    prefix_posix = os.path.normpath(prefix_rel).replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)
    _write_text(os.path.join(out_dir, "decoder_prefix.txt"), prefix_posix)
    _write_text(os.path.join(out_dir, "model_name.txt"), py_model + "\n")

    np.random.seed(123)
    hidden = 6
    ny = 61
    feat = 8
    n = 13
    episodes = 5

    W1 = 2.0 * np.random.rand(hidden, feat + 1) - 1.0
    W2 = 2.0 * np.random.rand(ny, hidden + 1) - 1.0
    X = np.random.rand(feat, n)
    input_unit = np.vstack([X, np.ones((1, n))])

    np.random.seed(789)
    batch_actions = np.random.randint(0, 4, size=n).astype(int)
    indexes = np.random.permutation(np.arange(1, ny + 1))

    epsilon = 1.0
    max_episode = 100
    discount_factor = 0.98
    discount_length = 20
    priori_m, priori_n = 0.2, 1.5
    data_is_simulations = False

    _save_csv(os.path.join(out_dir, "input_unit.csv"), input_unit)
    _save_csv(os.path.join(out_dir, "W1_init.csv"), W1)
    _save_csv(os.path.join(out_dir, "W2_init.csv"), W2)
    _save_csv(os.path.join(out_dir, "batchActions.csv"), batch_actions.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "indexes.csv"), indexes.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "his.csv"), np.array([[his]], dtype=np.float64))
    _save_csv(
        os.path.join(out_dir, "reward_params.csv"),
        np.array([[epsilon, max_episode, discount_factor, discount_length]], dtype=np.float64),
    )
    _save_csv(os.path.join(out_dir, "priori_params.csv"), np.array([[priori_m, priori_n]], dtype=np.float64))
    _save_csv(
        os.path.join(out_dir, "train_params.csv"),
        np.array([[episodes, 1.0 if data_is_simulations else 0.0]], dtype=np.float64),
    )

    for ep in range(episodes):
        np.random.seed(1000 + ep)
        u01 = np.random.rand(ny, n)

        orig_rand = np.random.rand

        def _rand_override(*shape):  # type: ignore[override]
            if shape != (ny, n):
                return orig_rand(*shape)
            return u01.copy()

        np.random.rand = _rand_override  # type: ignore[assignment]
        try:
            p, hidden_u, spk = applynets_priori(input_unit, W1, W2, n, ep, priori_m, priori_n)
        finally:
            np.random.rand = orig_rand  # type: ignore[assignment]

        success, suc_rate, motor_perform, ensemble = emulator_real(
            spk, batch_actions, indexes, his, py_model
        )

        n_motor1 = np.sum(motor_perform == 1) + 1
        n_motor2 = np.sum(motor_perform == 2) + 1
        n_motor3 = np.sum(motor_perform == 3) + 1
        n_max = max(n_motor1, n_motor2, n_motor3)
        inner_reward = (motor_perform == 1) * (n_max / n_motor1 - 1) + (motor_perform == 2) * (
            n_max / n_motor2 - 1
        ) + (motor_perform == 3) * (n_max / n_motor3 - 1)

        reward = success + epsilon * (1 - ep / max_episode) * inner_reward
        temp = reward.copy()
        temp[np.isnan(reward)] = 0
        discount_filter = discount_factor ** np.arange(discount_length - 1, -1, -1)
        discount_filter /= discount_length
        full_conv = np.convolve(temp, discount_filter, mode="full")
        smoothed = full_conv[-len(reward) :]
        smoothed[~np.isnan(reward)] = (
            smoothed[~np.isnan(reward)] - np.mean(smoothed[~np.isnan(reward)])
        ) / np.std(smoothed[~np.isnan(reward)])
        smoothed[np.isnan(reward)] = 0

        dW2, dW1 = getgradient(smoothed, p, spk, hidden_u, input_unit, W2, n)

        if data_is_simulations:
            lr = 0.1 * (1 - ep / max_episode) + 0.5
        else:
            lr = 0.7 * (1 - ep / max_episode) + 0.5

        W2 = W2 + lr * dW2
        W1 = W1 + lr * dW1

        _save_csv(os.path.join(out_dir, f"u01_ep{ep}.csv"), u01)
        _save_csv(os.path.join(out_dir, f"p_ep{ep}.csv"), p)
        _save_csv(os.path.join(out_dir, f"hidden_ep{ep}.csv"), hidden_u)
        _save_csv(os.path.join(out_dir, f"spk_ep{ep}.csv"), spk.astype(np.float64))
        _save_csv(os.path.join(out_dir, f"success_ep{ep}.csv"), success.reshape(1, -1))
        _save_csv(os.path.join(out_dir, f"motor_perform_ep{ep}.csv"), motor_perform.reshape(-1, 1))
        _save_csv(os.path.join(out_dir, f"smoothed_ep{ep}.csv"), smoothed.reshape(1, -1))
        _save_csv(os.path.join(out_dir, f"dW2_ep{ep}.csv"), dW2)
        _save_csv(os.path.join(out_dir, f"dW1_ep{ep}.csv"), dW1)
        _save_csv(os.path.join(out_dir, f"W2_after_ep{ep}.csv"), W2)
        _save_csv(os.path.join(out_dir, f"W1_after_ep{ep}.csv"), W1)

    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
