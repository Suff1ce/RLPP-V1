"""
Export Python reference outputs for DataLoader + emulator_simu + emulator_real(decodingModel_manual),
so C++ can match exactly.

Run from repo root:
  python Python_Ver/export_loader_emulator_reference.py

Writes:
  C++_Ver/Phase_3/loader_emulator_case/
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.DataLoader import DataLoader
from decoding.emulator_simu import emulator_simu
from decoding.emulator_real import emulator_real
from export_decoding_params_to_csv import (
    decoding_nn_forward_numpy,
    ensure_decoding_param_csvs,
    ensemble_ny_his_for_input_dim,
)


def _save_csv(path: str, a: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, a, delimiter=",", fmt="%.17g")


def _build_ensemble_emulator_real(spikes: np.ndarray, indexes: np.ndarray, his: int) -> np.ndarray:
    """Same stacking as decoding/emulator_real.py."""
    sorted_indices = np.argsort(indexes)
    spikes = spikes[sorted_indices, :]
    m1num, time_length = spikes.shape
    ensemble = np.zeros(((his + 1) * m1num, time_length))
    for i in range(his + 1):
        ensemble[i * m1num : (i + 1) * m1num, :] = np.hstack(
            [np.zeros((m1num, i)), spikes[:, :-i] if i > 0 else spikes]
        )
    return ensemble


def _emulator_real_from_forward(
    spikes: np.ndarray,
    motor_expect: np.ndarray,
    indexes: np.ndarray,
    his: int,
    path_prefix: str,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    ensemble = _build_ensemble_emulator_real(spikes, indexes, his)
    y = decoding_nn_forward_numpy(ensemble, path_prefix)
    motor_perform = np.argmax(y, axis=0) + 1
    success = (motor_perform == motor_expect).astype(float)
    success[motor_expect == 0] = np.nan
    rate = np.nansum(success) / np.sum(~np.isnan(success))
    return success, rate, motor_perform, ensemble


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "C++_Ver", "Phase_3", "loader_emulator_case")
    os.makedirs(out_dir, exist_ok=True)

    # decodingModel_01/02 weights as CSV (from .mat if present, else synthetic for parity harness)
    ensure_decoding_param_csvs(out_dir)

    # ---------------- DataLoader case ----------------
    np.random.seed(202)
    feat = 5
    ny = 3
    T = 40
    inputEnsemble = np.random.rand(feat, T)
    M1_truth = (np.random.rand(ny, T) > 0.7).astype(float)
    Actions = np.random.randint(0, 4, size=T).astype(int)
    # Trials in 1..K with some repeats
    Trials = np.random.randint(1, 7, size=T).astype(int)

    opt = {
        "Mode": "train",
        "DataLoaderCursor": 1,
        "batchSize": 3,
        "discountLength": 5,
        "trainTrials": np.array([1, 2, 3, 4, 5, 6]),
        "NumberOfTrainTrials": 6,
        "testTrials": np.array([2, 5, 6]),
    }

    # Fix RNG so permutation is deterministic
    np.random.seed(999)
    # Capture internal indices exactly (mirror DataLoader.py)
    opt_shadow = dict(opt)
    opt_shadow["trainTrials"] = np.random.permutation(opt_shadow["trainTrials"])
    start = opt_shadow["DataLoaderCursor"]
    stop = min(opt_shadow["NumberOfTrainTrials"], opt_shadow["DataLoaderCursor"] + opt_shadow["batchSize"] - 1)
    opt_shadow["DataLoaderCursor"] = (stop % opt_shadow["NumberOfTrainTrials"]) + 1
    trialIndexes = opt_shadow["trainTrials"][start - 1 : stop]
    time_indexes = np.concatenate([np.where(Trials == t)[0] for t in trialIndexes])
    offsets = np.arange(-opt_shadow["discountLength"], 1)
    time_indexes = np.unique(time_indexes[:, None] + offsets).astype(int)

    batchInput, batchM1, batchAct, opt2 = DataLoader(inputEnsemble, M1_truth, Actions, Trials, opt)

    _save_csv(os.path.join(out_dir, "inputEnsemble.csv"), inputEnsemble)
    _save_csv(os.path.join(out_dir, "M1_truth.csv"), M1_truth)
    _save_csv(os.path.join(out_dir, "Actions.csv"), Actions.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "Trials.csv"), Trials.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "opt_trainTrials.csv"), opt2["trainTrials"].reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "opt_meta.csv"), np.array([[opt2["DataLoaderCursor"], opt2["batchSize"], opt2["discountLength"], opt2["NumberOfTrainTrials"]]], dtype=float))
    _save_csv(os.path.join(out_dir, "trialIndexes.csv"), trialIndexes.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "time_indexes.csv"), time_indexes.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "batchInput.csv"), batchInput)
    _save_csv(os.path.join(out_dir, "batchM1_truth.csv"), batchM1)
    _save_csv(os.path.join(out_dir, "batchActions.csv"), batchAct.reshape(-1, 1))

    # ---------------- emulator_simu case ----------------
    np.random.seed(303)
    Ny = 2
    T2 = 25
    spikes = (np.random.rand(Ny, T2) > 0.8).astype(float)
    motorExpect = np.random.randint(0, 4, size=T2).astype(int)
    indexes = np.array([10, 3])  # arbitrary order to trigger argsort
    his = 6
    success, rate, motorPerform = emulator_simu(spikes, motorExpect, indexes, his, "decodingModel_simulation")

    _save_csv(os.path.join(out_dir, "simu_spikes.csv"), spikes)
    _save_csv(os.path.join(out_dir, "simu_motorExpect.csv"), motorExpect.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "simu_indexes.csv"), indexes.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "simu_his.csv"), np.array([[his]], dtype=float))
    _save_csv(os.path.join(out_dir, "simu_success.csv"), success.reshape(1, -1))
    _save_csv(os.path.join(out_dir, "simu_motorPerform.csv"), motorPerform.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "simu_rate.csv"), np.array([[rate]], dtype=float))

    # ---------------- emulator_real manual case ----------------
    np.random.seed(404)
    Ny3 = 8
    T3 = 18
    spikes3 = (np.random.rand(Ny3, T3) > 0.75).astype(float)
    motorExpect3 = np.random.randint(0, 4, size=T3).astype(int)
    indexes3 = np.random.permutation(np.arange(1, Ny3 + 1))
    his3 = 2
    success3, rate3, motorPerform3, ensemble3 = emulator_real(spikes3, motorExpect3, indexes3, his3, "decodingModel_manual")

    _save_csv(os.path.join(out_dir, "real_spikes.csv"), spikes3)
    _save_csv(os.path.join(out_dir, "real_motorExpect.csv"), motorExpect3.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real_indexes.csv"), indexes3.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real_his.csv"), np.array([[his3]], dtype=float))
    _save_csv(os.path.join(out_dir, "real_success.csv"), success3.reshape(1, -1))
    _save_csv(os.path.join(out_dir, "real_motorPerform.csv"), motorPerform3.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real_rate.csv"), np.array([[rate3]], dtype=float))
    _save_csv(os.path.join(out_dir, "real_ensemble.csv"), ensemble3)

    # ---------------- emulator_real decodingModel_01 / 02 (CSV weights; dims differ: 122 vs 183 inputs) ----------------
    p01 = os.path.join(out_dir, "decoding01")
    p02 = os.path.join(out_dir, "decoding02")

    def _input_dim(prefix: str) -> int:
        xo = np.loadtxt(prefix + "_xoffset.csv", delimiter=",", ndmin=1)
        return int(np.asarray(xo).size)

    id1 = _input_dim(p01)
    id2 = _input_dim(p02)
    Ny_01, his_01 = ensemble_ny_his_for_input_dim(id1)
    Ny_02, his_02 = ensemble_ny_his_for_input_dim(id2)
    T4 = 18

    np.random.seed(606)
    spikes_01 = (np.random.rand(Ny_01, T4) > 0.75).astype(float)
    np.random.seed(607)
    spikes_02 = (np.random.rand(Ny_02, T4) > 0.75).astype(float)
    np.random.seed(608)
    motor_expect4 = np.random.randint(0, 4, size=T4).astype(int)
    np.random.seed(609)
    indexes_01 = np.random.permutation(np.arange(1, Ny_01 + 1))
    np.random.seed(610)
    indexes_02 = np.random.permutation(np.arange(1, Ny_02 + 1))

    success41, rate41, motor41, ensemble41 = _emulator_real_from_forward(
        spikes_01, motor_expect4, indexes_01, his_01, p01
    )
    success42, rate42, motor42, ensemble42 = _emulator_real_from_forward(
        spikes_02, motor_expect4, indexes_02, his_02, p02
    )

    _save_csv(os.path.join(out_dir, "real01_spikes.csv"), spikes_01)
    _save_csv(os.path.join(out_dir, "real01_motorExpect.csv"), motor_expect4.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real01_indexes.csv"), indexes_01.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real01_his.csv"), np.array([[his_01]], dtype=float))
    _save_csv(os.path.join(out_dir, "real01_success.csv"), success41.reshape(1, -1))
    _save_csv(os.path.join(out_dir, "real01_motorPerform.csv"), motor41.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real01_rate.csv"), np.array([[rate41]], dtype=float))
    _save_csv(os.path.join(out_dir, "real01_ensemble.csv"), ensemble41)

    _save_csv(os.path.join(out_dir, "real02_spikes.csv"), spikes_02)
    _save_csv(os.path.join(out_dir, "real02_motorExpect.csv"), motor_expect4.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real02_indexes.csv"), indexes_02.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real02_his.csv"), np.array([[his_02]], dtype=float))
    _save_csv(os.path.join(out_dir, "real02_success.csv"), success42.reshape(1, -1))
    _save_csv(os.path.join(out_dir, "real02_motorPerform.csv"), motor42.reshape(-1, 1))
    _save_csv(os.path.join(out_dir, "real02_rate.csv"), np.array([[rate42]], dtype=float))
    _save_csv(os.path.join(out_dir, "real02_ensemble.csv"), ensemble42)

    print("Wrote", out_dir)


if __name__ == "__main__":
    main()

