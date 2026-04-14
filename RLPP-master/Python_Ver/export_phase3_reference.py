"""
Generate CSV reference tensors for C++ Phase 3 RL parity (run from repo root):

  python Python_Ver/export_phase3_reference.py

Writes C++_Ver/Phase_3/testdata/*.csv (reinforcement learning path only; no supervised reference).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.applynets_priori import applynets_priori
from model.getgradient import getgradient


def _save_csv(path: str, a: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, a, delimiter=",", fmt="%.17g")


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "C++_Ver", "Phase_3", "testdata")
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(42)
    hidden = 5
    ny = 3
    feat = 7
    n = 11
    input_dim = feat + 1

    W1 = 2.0 * np.random.rand(hidden, input_dim) - 1.0
    W2 = 2.0 * np.random.rand(ny, hidden + 1) - 1.0
    batch_input = np.random.rand(feat, n)
    input_unit = np.vstack([batch_input, np.ones((1, n))])

    _save_csv(os.path.join(out_dir, "W1.csv"), W1)
    _save_csv(os.path.join(out_dir, "W2.csv"), W2)
    _save_csv(os.path.join(out_dir, "input_unit.csv"), input_unit)

    np.random.seed(11)
    episode = 3
    max_ep = 100
    priori_m, priori_n = 0.2, 1.5
    p_rl, hidden_rl, spk_rl = applynets_priori(
        input_unit, W1, W2, n, episode, priori_m, priori_n
    )

    motor_perform = np.random.randint(1, 4, size=n)
    success = np.random.rand(n)
    success[success < 0.15] = np.nan

    opt_eps = 1.0
    discount_factor = 0.98
    discount_length = 20

    n_motor1 = np.sum(motor_perform == 1) + 1
    n_motor2 = np.sum(motor_perform == 2) + 1
    n_motor3 = np.sum(motor_perform == 3) + 1
    n_max = max(n_motor1, n_motor2, n_motor3)
    inner_reward = (motor_perform == 1) * (n_max / n_motor1 - 1) + (motor_perform == 2) * (
        n_max / n_motor2 - 1
    ) + (motor_perform == 3) * (n_max / n_motor3 - 1)

    reward = success + opt_eps * (1 - episode / max_ep) * inner_reward
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

    wd1_rl, wd2_rl = getgradient(smoothed, p_rl, spk_rl, hidden_rl, input_unit, W2, n)

    _save_csv(os.path.join(out_dir, "rl_p_output.csv"), p_rl)
    _save_csv(os.path.join(out_dir, "rl_hidden_unit.csv"), hidden_rl)
    _save_csv(os.path.join(out_dir, "rl_spk.csv"), spk_rl)
    _save_csv(os.path.join(out_dir, "rl_smoothed_reward.csv"), smoothed.reshape(1, -1))
    _save_csv(os.path.join(out_dir, "expected_rl_delta1.csv"), wd1_rl)
    _save_csv(os.path.join(out_dir, "expected_rl_delta2.csv"), wd2_rl)
    _save_csv(os.path.join(out_dir, "motor_perform.csv"), motor_perform.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "success.csv"), success.reshape(1, -1))

    meta = f"episode={episode}\nmax_episode={max_ep}\nepsilon={opt_eps}\n"
    meta += f"discount_factor={discount_factor}\ndiscount_length={discount_length}\n"
    meta += f"priori_m={priori_m}\npriori_n={priori_n}\n"
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(meta)
    rp = np.array([[episode, max_ep, opt_eps, discount_factor, discount_length]], dtype=np.float64)
    _save_csv(os.path.join(out_dir, "reward_params.csv"), rp)

    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
