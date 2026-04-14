"""
Export a Phase 3 RLPP case with:
  applynets_priori (Bernoulli via fixed u01) -> emulator_real(decodingModel_manual)
  -> RLPP reward smoothing -> getgradient -> weight update

Matches Python_Ver/training/RLPP.py structure for one batch (no DataLoader shuffle complexity).

Run from repo root:
  python Python_Ver/export_phase3_full_loop_emulator_case.py

Writes:
  C++_Ver/Phase_3/full_loop_emulator_case/
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.applynets_priori import applynets_priori
from model.getgradient import getgradient
from decoding.emulator_real import emulator_real


def _save_csv(path: str, a: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, a, delimiter=",", fmt="%.17g")


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "C++_Ver", "Phase_3", "full_loop_emulator_case")
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(123)
    hidden = 6
    ny = 8  # M1 neurons; must match decodingModel_manual ensemble layout (Ny=8, his=2 -> 24)
    feat = 8
    n = 13
    episodes = 5
    his = 2

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
            spk, batch_actions, indexes, his, "decodingModel_manual"
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
