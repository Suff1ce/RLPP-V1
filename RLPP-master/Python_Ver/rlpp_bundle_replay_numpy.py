"""
Offline bundle replay in pure NumPy — mirrors C++ RLPPInference + replay_runner semantics.

- ExponentialHistoryEncoder matches C++ ExponentialHistoryEncoder (Phase_1).
- Generator: TwoLayerMLP with bias as last column (same as applynets-style weights).
- Spikes: rng.random(shape) <= p (same rule as applynets: rand <= p_output).
- Decoder: decodingModel01_forward from CSV mapminmax + IW1_1 / LW2_1 (same as decoding_model01.cpp).

RNG: NumPy's PCG64/MT19937 differs from std::mt19937 — compare distributions, not bit-identical streams.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def load_csv_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def load_csv_vector(path: str) -> np.ndarray:
    m = load_csv_matrix(path)
    if m.ndim == 0:
        return np.array([float(m)])
    if m.ndim == 1:
        # Single-column CSV often loads as shape (N,) not (N, 1)
        return m.reshape(-1)
    if m.size == 1:
        return np.array([float(m.reshape(-1)[0])])
    if m.shape[0] == 1:
        return m.reshape(-1)
    if m.shape[1] == 1:
        return m.reshape(-1)
    raise ValueError(f"Expected vector CSV at {path}, got shape {m.shape}")


def load_csv_scalar(path: str) -> float:
    m = load_csv_matrix(path)
    if m.size != 1:
        raise ValueError(f"Expected single value in {path}")
    return float(m.reshape(-1)[0])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def mlp_forward_probs(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """x: [feature_dim], W1: [hidden x feature_dim+1], W2: [Ny x hidden+1] -> probs [Ny]"""
    fd = W1.shape[1] - 1
    xb = np.concatenate([x, np.ones(1, dtype=x.dtype)])
    h = sigmoid(W1 @ xb)
    hb = np.concatenate([h, np.ones(1, dtype=x.dtype)])
    logits = W2 @ hb
    return sigmoid(logits)


def decoding_model01_forward_col(
    ensemble: np.ndarray,
    xoffset: np.ndarray,
    gain: np.ndarray,
    ymin: float,
    IW1_1: np.ndarray,
    b1: np.ndarray,
    LW2_1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """ensemble: [input_dim] single column; returns softmax probs [num_labels]."""
    x = ensemble.reshape(-1)
    if x.shape[0] != xoffset.shape[0]:
        raise ValueError("ensemble dim vs mapminmax")
    xp = (x - xoffset) * gain + ymin
    n1 = IW1_1 @ xp + b1
    a1 = np.tanh(n1)
    n2 = LW2_1 @ a1 + b2
    n2 = n2 - np.max(n2)
    ex = np.exp(n2)
    s = ex.sum()
    if s == 0.0:
        s = 1.0
    return ex / s


def apply_sorted_indices_1based(v: np.ndarray, sorted_indices_1based: np.ndarray) -> np.ndarray:
    """Match C++ apply_sorted_indices_1based: out[i] = v[sorted_indices[i]-1]."""
    n = v.size
    if sorted_indices_1based.size != n:
        raise ValueError("sorted_indices length mismatch")
    out = np.empty(n, dtype=v.dtype)
    for i in range(n):
        src = int(sorted_indices_1based[i]) - 1
        if src < 0 or src >= n:
            raise ValueError("sorted index out of range")
        out[i] = v[src]
    return out


class ExponentialHistoryEncoder:
    """Port of C++ ExponentialHistoryEncoder (strict online)."""

    def __init__(self, num_inputs: int, history_length: int, tau_bins: float):
        if num_inputs <= 0 or history_length <= 0 or tau_bins <= 0:
            raise ValueError("bad encoder dims")
        self.Nx = num_inputs
        self.H = history_length
        self.tau = float(tau_bins)
        self._spikes: list[list[int]] = [[] for _ in range(num_inputs)]
        self._h_spike_time = [-1] * num_inputs
        self._last_observed = 0
        self._last_encoded = 0

    def observe_bin(self, spikes: np.ndarray, time_bin: int) -> None:
        if spikes.size != self.Nx:
            raise ValueError("spikes size")
        if time_bin <= 0:
            raise ValueError("time_bin 1-based positive")
        if time_bin < self._last_observed:
            raise ValueError("time nondecreasing")
        for i in range(self.Nx):
            if spikes[i] != 0.0:
                self._observe_spike(i, time_bin)
        self._last_observed = time_bin

    def _observe_spike(self, neuron_index: int, time_bin: int) -> None:
        sp = self._spikes[neuron_index]
        if sp and time_bin < sp[-1]:
            raise ValueError("spike times must be nondecreasing")
        sp.append(time_bin)
        if self._h_spike_time[neuron_index] < 0 and len(sp) == self.H:
            self._h_spike_time[neuron_index] = time_bin
        while len(sp) > self.H:
            sp.pop(0)

    def start_time_bin(self) -> int:
        start = -1
        for i in range(self.Nx):
            if self._h_spike_time[i] < 0:
                return -1
            cand = self._h_spike_time[i] + 1
            if cand > start:
                start = cand
        return start

    def can_encode(self, current_time_bin: int) -> bool:
        if current_time_bin <= 0:
            return False
        st = self.start_time_bin()
        if st < 0:
            return False
        return current_time_bin >= st

    def encode(self, current_time_bin: int) -> np.ndarray:
        if current_time_bin <= 0:
            raise ValueError("encode time")
        if current_time_bin < self._last_encoded:
            raise ValueError("encode nondecreasing")
        if not self.can_encode(current_time_bin):
            raise ValueError("encode before Start")
        feats = np.zeros(self.Nx * self.H, dtype=np.float64)
        for neuron in range(self.Nx):
            spikes = self._spikes[neuron]
            n = len(spikes)
            num_rel = n if n < self.H else self.H
            start_idx = n - num_rel
            offset = neuron * self.H
            for j in range(num_rel):
                st = spikes[start_idx + j]
                delta_t = current_time_bin - st
                if delta_t < 0:
                    raise ValueError("future spike")
                feats[offset + j] = np.exp(-float(delta_t) / self.tau)
        self._last_encoded = current_time_bin
        return feats


class DecoderHistoryBuffer:
    """Port of C++ DecoderHistoryBuffer.flatten_for_python_decoder order."""

    def __init__(self, num_outputs: int, num_lags: int):
        if num_outputs <= 0 or num_lags <= 0:
            raise ValueError("bad decoder history")
        self.Ny = num_outputs
        self.Hd = num_lags
        self.history = np.zeros((num_outputs, num_lags), dtype=np.float64)

    def push(self, spikes: np.ndarray) -> None:
        if spikes.size != self.Ny:
            raise ValueError("push size")
        if self.Hd > 1:
            self.history[:, 0 : self.Hd - 1] = self.history[:, 1:self.Hd]
        self.history[:, self.Hd - 1] = spikes

    def flatten_for_python_decoder(self) -> np.ndarray:
        flat = np.zeros(self.Ny * self.Hd, dtype=np.float64)
        k = 0
        for col in range(self.Hd - 1, -1, -1):
            for row in range(self.Ny):
                flat[k] = self.history[row, col]
                k += 1
        return flat


@dataclass
class ReplayMetrics:
    max_abs_err: float
    mean_abs_err: float
    label_mismatches: int
    n_labels: int
    mean_abs_spike_calib: float
    max_abs_spike_calib: float


def run_sampled_replay(
    upstream: np.ndarray,
    generator_W1: np.ndarray,
    generator_W2: np.ndarray,
    sorted_indices_1based: np.ndarray,
    decoder_W1: np.ndarray,
    decoder_b1: np.ndarray,
    decoder_W2: np.ndarray,
    decoder_b2: np.ndarray,
    xoffset: np.ndarray,
    gain: np.ndarray,
    ymin: float,
    decoder_logits_ref: np.ndarray,
    labels_ref: np.ndarray,
    tau_bins: float,
    valid_col_start: int,
    valid_col_count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, ReplayMetrics]:
    """
    Returns recorded logits [K x out_cols], recorded labels [out_cols], metrics vs sliced ref.
    """
    t_rows, nx = upstream.shape
    fd = generator_W1.shape[1] - 1
    h_hist = fd // nx
    if h_hist * nx != fd:
        raise ValueError("feature_dim not divisible by Nx")

    ny = generator_W2.shape[0]
    drows = decoder_W1.shape[1]
    if drows % ny != 0:
        raise ValueError("decoder W1 cols vs Ny")
    num_lags = drows // ny

    t_valid = decoder_logits_ref.shape[1]
    num_labels = decoder_logits_ref.shape[0]

    if labels_ref.size != t_valid:
        raise ValueError("labels_ref length")
    if valid_col_start < 0 or valid_col_start > t_valid:
        raise ValueError("valid_col_start")
    remaining = t_valid - valid_col_start
    out_cols = remaining if valid_col_count < 0 else min(valid_col_count, remaining)
    if out_cols < 0:
        raise ValueError("valid_col_count")

    enc = ExponentialHistoryEncoder(nx, h_hist, tau_bins)
    dec_hist = DecoderHistoryBuffer(ny, num_lags)

    logits_rec = np.zeros((num_labels, out_cols), dtype=np.float64)
    labels_rec = np.zeros(out_cols, dtype=np.int32)

    probs_dump = np.zeros((ny, out_cols), dtype=np.float64)
    spikes_dump = np.zeros((ny, out_cols), dtype=np.float64)

    col = 0
    rec_col = 0
    for t_idx in range(t_rows):
        t = t_idx + 1
        u = upstream[t_idx, :].astype(np.float64, copy=False)
        enc.observe_bin(u, t)
        if not enc.can_encode(t):
            continue
        if col >= t_valid:
            raise RuntimeError("replay produced too many valid cols")

        x = enc.encode(t)
        p = mlp_forward_probs(x, generator_W1, generator_W2)
        rnd = rng.random(ny)
        s = (rnd <= p).astype(np.float64)
        s_sorted = apply_sorted_indices_1based(s, sorted_indices_1based) if sorted_indices_1based.size else s
        dec_hist.push(s_sorted)
        ens = dec_hist.flatten_for_python_decoder()
        y = decoding_model01_forward_col(ens, xoffset, gain, ymin, decoder_W1, decoder_b1, decoder_W2, decoder_b2)
        lab = int(np.argmax(y) + 1)

        record = (col >= valid_col_start) and (rec_col < out_cols)
        if record:
            logits_rec[:, rec_col] = y
            labels_rec[rec_col] = lab
            probs_dump[:, rec_col] = p
            spikes_dump[:, rec_col] = s_sorted
            rec_col += 1
            if rec_col >= out_cols:
                break
        col += 1  # C++: increments only when not breaking from filled window

    if rec_col != out_cols:
        raise RuntimeError(f"replay produced {rec_col} cols, expected {out_cols}")

    ref_slice = decoder_logits_ref[:, valid_col_start : valid_col_start + out_cols]
    diff = np.abs(logits_rec - ref_slice)
    ref_labels_slice = labels_ref[valid_col_start : valid_col_start + out_cols].astype(int)
    mism = int(np.sum(labels_rec != ref_labels_slice))

    mean_p = probs_dump.mean(axis=1)
    mean_s = spikes_dump.mean(axis=1)
    gap = np.abs(mean_s - mean_p)

    metrics = ReplayMetrics(
        max_abs_err=float(np.max(diff)),
        mean_abs_err=float(np.mean(diff)),
        label_mismatches=mism,
        n_labels=out_cols,
        mean_abs_spike_calib=float(np.mean(gap)),
        max_abs_spike_calib=float(np.max(gap)),
    )
    return logits_rec, labels_rec, metrics


def run_deterministic_replay(
    upstream: np.ndarray,
    downstream_spikes_ref: np.ndarray,
    generator_W1: np.ndarray,
    generator_W2: np.ndarray,
    sorted_indices_1based: np.ndarray,
    decoder_W1: np.ndarray,
    decoder_b1: np.ndarray,
    decoder_W2: np.ndarray,
    decoder_b2: np.ndarray,
    xoffset: np.ndarray,
    gain: np.ndarray,
    ymin: float,
    decoder_logits_ref: np.ndarray,
    labels_ref: np.ndarray,
    tau_bins: float,
    valid_col_start: int,
    valid_col_count: int,
) -> ReplayMetrics:
    """Drive decoder from bundle downstream_spikes_ref columns (no RNG). Should match C++ deterministic replay."""
    t_rows, nx = upstream.shape
    fd = generator_W1.shape[1] - 1
    h_hist = fd // nx
    if h_hist * nx != fd:
        raise ValueError("feature_dim not divisible by Nx")

    ny = generator_W2.shape[0]
    drows = decoder_W1.shape[1]
    num_lags = drows // ny

    t_valid = decoder_logits_ref.shape[1]
    if downstream_spikes_ref.shape != (ny, t_valid):
        raise ValueError("downstream_spikes_ref shape vs T_valid")
    if labels_ref.size != t_valid:
        raise ValueError("labels_ref length")
    if valid_col_start < 0 or valid_col_start > t_valid:
        raise ValueError("valid_col_start out of range")
    remaining = t_valid - valid_col_start
    out_cols = remaining if valid_col_count < 0 else min(valid_col_count, remaining)
    if out_cols < 0:
        raise ValueError("valid_col_count out of range")

    enc = ExponentialHistoryEncoder(nx, h_hist, tau_bins)
    dec_hist = DecoderHistoryBuffer(ny, num_lags)

    logits_rec = np.zeros((decoder_logits_ref.shape[0], out_cols), dtype=np.float64)
    labels_rec = np.zeros(out_cols, dtype=np.int32)

    col = 0
    rec_col = 0
    for t_idx in range(t_rows):
        t = t_idx + 1
        u = upstream[t_idx, :].astype(np.float64, copy=False)
        enc.observe_bin(u, t)
        if not enc.can_encode(t):
            continue
        if col >= t_valid:
            raise RuntimeError("replay produced too many valid cols")

        x = enc.encode(t)
        p = mlp_forward_probs(x, generator_W1, generator_W2)
        s = downstream_spikes_ref[:, col].astype(np.float64, copy=False)
        s_sorted = apply_sorted_indices_1based(s, sorted_indices_1based) if sorted_indices_1based.size else s
        dec_hist.push(s_sorted)
        ens = dec_hist.flatten_for_python_decoder()
        y = decoding_model01_forward_col(ens, xoffset, gain, ymin, decoder_W1, decoder_b1, decoder_W2, decoder_b2)
        lab = int(np.argmax(y) + 1)

        record = (col >= valid_col_start) and (rec_col < out_cols)
        if record:
            logits_rec[:, rec_col] = y
            labels_rec[rec_col] = lab
            rec_col += 1
            if rec_col >= out_cols:
                break
        col += 1

    if rec_col != out_cols:
        raise RuntimeError(f"deterministic replay produced {rec_col} cols, expected {out_cols}")

    ref_slice = decoder_logits_ref[:, valid_col_start : valid_col_start + out_cols]
    diff = np.abs(logits_rec - ref_slice)
    ref_labels_slice = labels_ref[valid_col_start : valid_col_start + out_cols].astype(int)
    mism = int(np.sum(labels_rec != ref_labels_slice))

    return ReplayMetrics(
        max_abs_err=float(np.max(diff)),
        mean_abs_err=float(np.mean(diff)),
        label_mismatches=mism,
        n_labels=out_cols,
        mean_abs_spike_calib=0.0,
        max_abs_spike_calib=0.0,
    )


def load_bundle_and_decoder_maps(bundle_dir: str) -> dict:
    p = lambda n: os.path.join(bundle_dir, n)
    return {
        "upstream_spikes": load_csv_matrix(p("upstream_spikes.csv")),
        "generator_W1": load_csv_matrix(p("generator_W1.csv")),
        "generator_W2": load_csv_matrix(p("generator_W2.csv")),
        "decoder_W1": load_csv_matrix(p("decoder_W1.csv")),
        "decoder_b1": load_csv_vector(p("decoder_b1.csv")),
        "decoder_W2": load_csv_matrix(p("decoder_W2.csv")),
        "decoder_b2": load_csv_vector(p("decoder_b2.csv")),
        "decoder_logits_ref": load_csv_matrix(p("decoder_logits_ref.csv")),
        "labels_ref": load_csv_vector(p("labels_ref.csv")).astype(np.int32),
        "downstream_spikes_ref": load_csv_matrix(p("downstream_spikes_ref.csv")),
        "sorted_indices_1based": load_csv_vector(p("sortedIndices.csv")).astype(np.int64),
        "xoffset": load_csv_vector(p("decoder_xoffset.csv")),
        "gain": load_csv_vector(p("decoder_gain.csv")),
        "ymin": load_csv_scalar(p("decoder_ymin.csv")),
    }
