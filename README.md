# HS-3D-DIC

High-speed stereo digital image correlation (Stereo-DIC) for full-field 3D vibration
measurement of blades: 3D shape, 3D displacement time series, natural frequencies,
amplitude/phase and operating deflection shapes.

Python package: `stereodic`. Technical design: [docs/StereoDIC_Technical_Design.md](docs/StereoDIC_Technical_Design.md).

## Status

| Milestone | Scope | State |
| --- | --- | --- |
| M0 | Repo skeleton, config, types, TIFF reader, HDF5 store, synthetic speckle | done (Linux GPU check pending) |
| M1 | 2D DIC core (B-spline, IC-GN, Numba), benchmarks B01/B03 | done |
| M2 | Calibration (ChArUco), 2nd-order IC-GN, stereo matching (SIFT + RG-DIC), triangulation, reference shape, B04 | done (real-data E1/E2 pending) |
| M3 | Temporal DIC, CUDA backend, frequencies | next |
| M4 | Harmonic fit, ODS, FDD, mode shapes | — |

## Machines and workflow

- **Mac**: development, unit tests, small datasets, visualization (`numpy` / `numba` backends).
- **Linux + RTX A6000**: full datasets, CUDA backend.
- Code is synced with git only. Edit on the Mac, `git pull` on Linux; never leave
  uncommitted changes on Linux. Data and results never go into git.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
# Mac
uv sync

# Linux with NVIDIA GPU (CUDA 12.x driver)
uv sync --extra cuda

# optional 3D visualization (PyVista)
uv sync --extra viz
```

Machine-specific paths:

```bash
cp configs/local.yaml.example configs/local.yaml   # then edit data_root / results_root
```

Environment variables `STEREODIC_DATA_ROOT` and `STEREODIC_RESULTS_ROOT` override `local.yaml`.

First-time setup on the Linux GPU machine:

```bash
git clone https://github.com/WangHaoZJU/HS-3D-DIC.git
cd HS-3D-DIC
uv sync --extra cuda
cp configs/local.yaml.example configs/local.yaml   # set data_root to local NVMe
uv run pytest -m "cuda or not cuda"                # includes GPU tests
```

Afterwards, update with `git pull && uv sync --extra cuda`.

Check the environment and available compute backends:

```bash
uv run python scripts/check_env.py
```

## Running an experiment

1. Copy `configs/experiments/example.yaml` to `configs/experiments/<name>.yaml` and edit paths,
   the ChArUco board (`calibration.board`), ROI / POI mask and reference frame.
2. Calibrate (writes `calibration_file` + `.json` report + residual plot):

```bash
uv run python scripts/calibrate.py --config configs/experiments/<name>.yaml -v
```

3. Reference stage: POI grid on the left reference frame, stereo matching, 3D reference shape:

```bash
uv run python scripts/run_dic.py --config configs/experiments/<name>.yaml --stage stereo -v
```

Temporal DIC, displacement and vibration analysis follow in M3/M4.

## Tests

```bash
uv run pytest              # default: skips slow tests; CUDA tests auto-skip without a GPU
uv run pytest -m slow      # slow tests only
uv run pytest -m cuda      # GPU tests (Linux)
```

## Benchmarks

```bash
uv run python -m benchmarks.b01_subpixel_bias [--quick] [--diameter 4]   # sub-pixel bias curve
uv run python -m benchmarks.b03_subset_size [--quick] [--interp 5]       # error vs subset size, throughput
uv run python -m benchmarks.b04_stereo_s4 [--quick]                      # S4: stereo shape + 3D rigid motion
```

Results (CSV/PNG) go to `results/benchmarks/` (git-ignored).

## Configuration

- `configs/defaults/dic.yaml` — default DIC parameters.
- `configs/experiments/<name>.yaml` — one file per experiment; overrides any default key.
- `configs/local.yaml` — machine paths (git-ignored).

## Layout

```text
stereodic/
  core/        types, config, status codes, provenance
  io/          image sequences (TIFF), HDF5 result store
  synth/       independent synthetic renderer: speckle, stereo scenes, ChArUco boards (ground truth)
  calibration/ camera model, ChArUco detection, mono/stereo calibration, report
  dic/         B-spline interpolation, shape functions (0/1/2), IC-GN, SIFT/RG-DIC stereo matching, backends
  recon/       triangulation, normals and local frames, reference shape
  pipeline.py  stage orchestration (calibrate, stereo)
benchmarks/    synthetic accuracy and throughput benchmarks
scripts/       command-line entry points
tests/         unit / integration / backends (cuda)
docs/          technical design
```
