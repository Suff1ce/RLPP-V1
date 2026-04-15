## RLPP C++ implementation

This document describes the **C++_Ver** implementation: what each stage does, how the C++ executables fit together, and how to run training + export a bundle that can be loaded for **real-time inference**.

### High-level architecture (what “stages” mean in C++)

The C++ code is organized by “phases” that mirror the MATLAB/Python pipeline, but with a pragmatic split:

- **Phase 1 (inference + parity + replay)**: a self-contained “bundle loader + online state machine” implementation intended to match the MATLAB/Python reference bundle format.
- **Phase 3 (offline RL training loop)**: end-to-end RLPP training over pre-exported CSV datasets, including a runner that can export a **Phase-1 compatible inference bundle** at the end.

If you only want to *run* a trained model in real time, Phase 1 is what you care about. If you want to *train* a generator and then deploy it for real-time inference, you will use Phase 3 + export + Phase 1.

### Repository layout (C++)

- `C++_Ver/Include/`: shared headers (encoder, inference, bundle structs, etc.)
- `C++_Ver/Phase_1/`: inference / replay / parity checks / UDP scaffold
- `C++_Ver/Phase_3/`: offline RL training + verifiers + CSV-based runner
- `C++_Ver/RLPP_All/`: optional umbrella launcher that dispatches to the other executables
- `C++_Ver/External/eigen/`: Eigen dependency (vendored)

### What is a “bundle”?

Phase 1 loads a directory of CSV files (an “F1 bundle”) via `load_f1_bundle()` and related helpers. The bundle contains:

- **Generator weights**: `generator_W1.csv`, `generator_W2.csv` (two-layer MLP with sigmoid)
- **Decoder weights + normalization**: `decoder_W1/b1/W2/b2.csv` and `decoder_xoffset/gain/ymin.csv`
- **Sorted indices**: `sortedIndices.csv` (1-based channel permutation used before decoder history)
- **Reference matrices**: `*_ref.csv` used for parity / shape checks / replay tools

The **real-time UDP inference** executable (`RLPP_realtime_udp_infer`) requires:

- The generator weights (`generator_W1.csv`, `generator_W2.csv`)
- Decoder weights (`decoder_W1/b1/W2/b2.csv`) and mapminmax (`decoder_xoffset/gain/ymin.csv`)
- `sortedIndices.csv`
- Enough reference matrices to infer dimensions (notably `upstream_spikes.csv`, `encoder_features_ref.csv`, `decoder_features_ref.csv`, `decoder_logits_ref.csv`)

Phase 3’s export step produces such a directory automatically.

---

## Stage-by-stage: what each phase does

### Phase 1: online inference + parity + replay (`C++_Ver/Phase_1/`)

**Core components**

- **Exponential encoder** (`ExponentialHistoryEncoder`): turns streaming upstream activity into a feature vector of size \(Nx \times H\) using an exponential history filter.
- **Generator MLP** (`TwoLayerMLP`): maps encoder features → per-channel spike probabilities (`Ny` outputs).
- **Decoder history buffer** (`DecoderHistoryBuffer`): builds a lagged feature vector of size \(Ny \times \text{num\_lags}\) (typically \(\text{his}+1\)).
- **Movement decoder** (`decodingModel01_forward`): maps lagged spikes → label probabilities; uses `decoder_xoffset/gain/ymin` (mapminmax) and the decoder weights.
- **Inference state machine** (`RLPPInference`): glues encoder + generator + history + decoder into a per-bin `step()`.

**Executables**

- `RLPP_Phase_1.exe`  
  A CLI wrapper for parity checks, deterministic online tests, and replays against a bundle directory.

  Run help:
  - `RLPP_Phase_1.exe --help`

- `RLPP_realtime_udp_infer.exe`  
  A real-time scaffold: one thread receives UDP frames, another runs fixed-period inference ticks, then sends a packed downstream frame over UDP.

  Run help:
  - `RLPP_realtime_udp_infer.exe --help`

- `RLPP_hardware_udp_loopback.exe`  
  A simple loopback tool for the packed frame format (useful for verifying transport/parsing).

**Typical Phase 1 workflows**

- **Parity / sanity checks** on a known bundle:
  - `RLPP_Phase_1.exe --bundle <BUNDLE_DIR> --mode parity`
- **Replay** a window deterministically (uses reference downstream spikes):
  - `RLPP_Phase_1.exe --bundle <BUNDLE_DIR> --mode replay-det`
- **Run the UDP inference scaffold**:
  - `RLPP_realtime_udp_infer.exe --bundle-dir <BUNDLE_DIR> --ticks 2000`

### Phase 3: offline RL training + export (`C++_Ver/Phase_3/`)

Phase 3 contains:

- **Training math parity** (`RLPP_Phase_3.exe`): checks RL gradients and smoothed reward vs Python-exported CSV reference.
- **Episode-loop parity** (`RLPP_Phase_3_Trainer.exe`): runs an end-to-end episode update against a Python-exported deterministic “training_case”.
- **End-to-end runner** (`RLPP_Phase_3_Run.exe`): runs the RL loop over dataset CSVs (DataLoader → generator forward → emulator → reward → gradients → weight update) and can **export a Phase-1-compatible bundle** after training.

The runner (`RLPP_Phase_3_Run`) expects a **case directory** containing at least:

- `inputEnsemble.csv`
- `M1_truth.csv`
- `Actions.csv`
- `Trials.csv`

And optionally:

- `opt_trainTrials.csv`
- `opt_meta.csv` (cursor/batch/discount metadata)

At the end of training you can export:

- `--export-bundle-dir <DIR>`
- Optionally set encoder geometry: `--export-inference-nx <Nx>` and `--export-encoder-h <H>`  
  These must satisfy: **`Nx * H == inputEnsemble.rows()`**.

---

## Build instructions (Windows, Visual Studio generator)

The repo uses CMake. Each phase has its own `CMakeLists.txt` with its own build directory.

### Build Phase 1

From repo root:

```powershell
cmake -S C++_Ver/Phase_1 -B C++_Ver/Phase_1/build
cmake --build C++_Ver/Phase_1/build --config Release
```

Outputs:

- `C++_Ver/Phase_1/build/Release/RLPP_Phase_1.exe`
- `C++_Ver/Phase_1/build/Release/RLPP_realtime_udp_infer.exe`
- `C++_Ver/Phase_1/build/Release/RLPP_hardware_udp_loopback.exe`

### Build Phase 3

```powershell
cmake -S C++_Ver/Phase_3 -B C++_Ver/Phase_3/build
cmake --build C++_Ver/Phase_3/build --config Release
```

Outputs include:

- `C++_Ver/Phase_3/build/Release/RLPP_Phase_3_Run.exe`
- `C++_Ver/Phase_3/build/Release/RLPP_Phase_3.exe`
- `C++_Ver/Phase_3/build/Release/RLPP_Phase_3_Trainer.exe`

### Optional: umbrella launcher

`C++_Ver/RLPP_All` builds `RLPP_All.exe`, which spawns the other executables so you can run:

```powershell
cmake -S C++_Ver/RLPP_All -B C++_Ver/RLPP_All/build
cmake --build C++_Ver/RLPP_All/build --config Release
```

Then:

- `RLPP_All.exe verify-all`
- `RLPP_All.exe phase3-run ...`
- `RLPP_All.exe realtime-udp-infer ...`

---

## Running on a newly prepared dataset (end-to-end)

This section describes the intended flow:

1. Prepare your dataset as CSVs for `RLPP_Phase_3_Run`
2. Run offline training (Phase 3)
3. Export a Phase-1 bundle
4. Run real-time inference (Phase 1 UDP scaffold) using the exported bundle

### 1) Prepare a Phase 3 “case directory”

Create a directory, e.g. `D:/my_rlpp_case/`, with:

- **`inputEnsemble.csv`**: shape **`[feat x T]`**  
  Feature rows are the encoder features used during training. In the Phase 3 runner, the bias term is appended internally (so the generator sees `feat + 1` inputs).

- **`M1_truth.csv`**: shape **`[Ny x T]`**  
  This is the recorded downstream (M1) activity used by the emulator / reward pathway.

- **`Actions.csv`**: vector length **`T`** (either `T x 1` or `1 x T`)
- **`Trials.csv`**: vector length **`T`** (either `T x 1` or `1 x T`)

Optional (recommended if you want to mirror Python’s loader cursor behavior exactly):

- **`opt_trainTrials.csv`**: vector of trial IDs to use for training
- **`opt_meta.csv`**: 1x4 metadata: `cursor, batchSize, discountLength, NumberOfTrainTrials`

Notes:

- **Index conventions**: where the pipeline expects “sorted indices”, the C++ code uses **1-based indices** (MATLAB-style).
- **Decoder**: the runner uses `--decoder-prefix` to load the movement decoder (trained separately) when running in `--data-index Real` mode.

### 2) Offline training with Phase 3 runner

Example:

```powershell
& C++_Ver/Phase_3/build/Release/RLPP_Phase_3_Run.exe `
  --case-dir D:/my_rlpp_case `
  --episodes 50 `
  --model decodingModel_01 `
  --decoder-prefix d:/RLPP-master/C++_Ver/Phase_3/decoding_params/decoding01 `
  --seed 2026
```

### 3) Export a Phase-1 inference bundle (automated)

Add `--export-bundle-dir` to the same Phase 3 run:

```powershell
& C++_Ver/Phase_3/build/Release/RLPP_Phase_3_Run.exe `
  --case-dir D:/my_rlpp_case `
  --episodes 50 `
  --model decodingModel_01 `
  --decoder-prefix d:/RLPP-master/C++_Ver/Phase_3/decoding_params/decoding01 `
  --export-bundle-dir D:/my_exported_bundle
```

Encoder geometry is inferred as:

- default: `Nx = feat`, `H = 1`

If your Phase 1 inference should use a specific \(Nx\) and history length \(H\) (so features are interpreted as an \(Nx \times H\) encoder grid), set:

```powershell
  --export-inference-nx <Nx> --export-encoder-h <H>
```

Constraint:

- **`Nx * H == inputEnsemble.rows()`**

The export directory will also include a `phase3_export_manifest.txt` that records the inferred dimensions and the decoder prefix used.

### 4) Run real-time inference using the exported bundle

Quick load/smoke test (no external UDP sender required; it will tick using zero upstream bins if nothing arrives):

```powershell
& C++_Ver/Phase_1/build/Release/RLPP_realtime_udp_infer.exe `
  --bundle-dir D:/my_exported_bundle `
  --ticks 2000
```

Real-time IO options:

- `--udp-in-port` (default 46000)
- `--udp-out-host` / `--udp-out-port` (default 127.0.0.1:45123)
- `--bin-ms` inference period (default 5ms)
- `--tau-bins` encoder time constant in bins (default 150)

---

## Tips / common gotchas

- **Dimension mismatches**:
  - Phase 1 infers `Nx` from `upstream_spikes.csv` columns and infers `H` from `encoder_features_ref.csv` rows.
  - Phase 1 infers `num_lags` from `decoder_features_ref.csv` rows divided by `Ny`.
  - Decoder normalization vectors must be length `Ny * num_lags`.

- **Decoder prefixes**:
  - Phase 3 expects the decoder CSVs at:
    - `<prefix>_xoffset.csv`, `<prefix>_gain.csv`, `<prefix>_ymin.csv`, `<prefix>_IW1_1.csv`, `<prefix>_b1.csv`, `<prefix>_LW2_1.csv`, `<prefix>_b2.csv`
  - The Phase 3 export copies these into the bundle as:
    - `decoder_xoffset.csv`, `decoder_gain.csv`, `decoder_ymin.csv`, `decoder_W1.csv`, `decoder_b1.csv`, `decoder_W2.csv`, `decoder_b2.csv`

- **Reproducibility**:
  - Phase 3 runner is stochastic (generator sampling), so exact trajectories differ by seed.
  - For strict parity checks against the Python exporters, use the dedicated Phase 3 verifier executables.

- **Using Phase 1 tools on an exported bundle**:
  - You can run `RLPP_Phase_1.exe --bundle <exported_bundle> --mode parity-online` to sanity-check the exported directory and run an online deterministic test.

---

## What’s next?

If you plan to run truly real-time with a device/DAQ, you’ll typically:

- Replace or adapt the UDP sender to emit `upstream_frame_v1` packets (see `Include/upstream_frame_v1.hpp`).
- Consume the packed downstream `hardware_frame_v1` (see `Include/hardware_frame_v1.hpp`) on the receiver side.
