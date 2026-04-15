## RLPP C++ quickstart

This is a **minimal** guide for running the C++ pipeline. For project background and dataset/science context, see `README.md`.

### What you get in C++

- **Phase 3**: offline RL training runner (**`RLPP_Phase_3_Run`**) that can **export** a Phase-1 compatible bundle.
- **Phase 1**: inference + replay tools (**`RLPP_Phase_1`**) and a **real-time UDP scaffold** (**`RLPP_realtime_udp_infer`**) that loads the exported bundle.

---

## Build (Windows / CMake)

### Phase 3 (offline training)

```powershell
cmake -S C++_Ver/Phase_3 -B C++_Ver/Phase_3/build
cmake --build C++_Ver/Phase_3/build --config Release --target RLPP_Phase_3_Run
```

### Phase 1 (inference + realtime UDP)

```powershell
cmake -S C++_Ver/Phase_1 -B C++_Ver/Phase_1/build
cmake --build C++_Ver/Phase_1/build --config Release --target RLPP_Phase_1 RLPP_realtime_udp_infer
```

---

## Run on your own dataset

### 1) Prepare a Phase 3 case directory

Create a directory (e.g. `D:/my_rlpp_case/`) with the required CSVs:

- `inputEnsemble.csv` (shape `[feat x T]`)
- `M1_truth.csv` (shape `[Ny x T]`)
- `Actions.csv` (vector length `T`)
- `Trials.csv` (vector length `T`)

Optional: `opt_trainTrials.csv`, `opt_meta.csv` (if you want to control loader cursor/batch settings exactly).

### 2) Train offline + export a Phase 1 bundle

```powershell
& C++_Ver/Phase_3/build/Release/RLPP_Phase_3_Run.exe `
  --case-dir D:/my_rlpp_case `
  --episodes 50 `
  --model decodingModel_01 `
  --decoder-prefix d:/RLPP-master/C++_Ver/Phase_3/decoding_params/decoding01 `
  --export-bundle-dir D:/my_exported_bundle
```

If your Phase 1 encoder geometry should be interpreted as \(Nx \times H\), pass both and ensure:

- `Nx * H == inputEnsemble.rows()`

```powershell
  --export-inference-nx 61 --export-encoder-h 2
```

### 3) Smoke test Phase 1 on the exported bundle

```powershell
& C++_Ver/Phase_1/build/Release/RLPP_Phase_1.exe `
  --bundle D:/my_exported_bundle `
  --mode parity
```

### 4) Run real-time UDP inference (loads the same bundle)

```powershell
& C++_Ver/Phase_1/build/Release/RLPP_realtime_udp_infer.exe `
  --bundle-dir D:/my_exported_bundle `
  --ticks 2000
```

Common realtime flags:

- `--udp-in-port 46000` (upstream input)
- `--udp-out-host 127.0.0.1 --udp-out-port 45123` (downstream output)
- `--bin-ms 5` (tick rate)
- `--tau-bins 150` (encoder time constant)

---

## Notes (only the non-obvious bits)

- The exported bundle directory also includes a `phase3_export_manifest.txt` to help debug dimension issues.
- Decoder CSV naming in Phase 3 uses a **prefix** like `.../decoding01` and expects files like `decoding01_xoffset.csv`, `decoding01_IW1_1.csv`, etc.

