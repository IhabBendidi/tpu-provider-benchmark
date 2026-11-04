# Getting started

Commands below run from the repository root. TPU launchers require an existing TPU allocation.

## Install the backend environments

Use Python 3.12 and separate environments: the backends depend on different PyTorch and libtpu versions. Run commands below from the repository root, on a TPU worker with the allocation's required device access and topology environment already configured.

```bash
python3.12 -m venv .venv-xla
.venv-xla/bin/pip install -r requirements/xla.txt

python3.12 -m venv .venv-torchtpu
.venv-torchtpu/bin/pip install -r requirements/torchtpu.txt
```

The reference stacks are XLA 2.9.0 / PyTorch 2.9.0 / libtpu 0.0.21 and TorchTPU 0.2.0 / PyTorch 2.14.1 / libtpu 0.0.51. Requirements use CPU PyTorch wheels; TPU execution comes from the selected TPU backend. For the original worker layout, `bash scripts/setup/bootstrap-worker.sh /tmp/tpu-benchmark` installs both isolated environments with pinned uv. It does not change system drivers.

Launchers default to `.venv-xla/bin/python` or `.venv-torchtpu/bin/python`. Override with `--python /absolute/path/to/python`, `TPU_XLA_PYTHON`, or `TPU_TORCHTPU_PYTHON`. `TPU_LAUNCH_PYTHON` selects only the standard-library launch dispatcher. Real execution refuses a missing backend interpreter; TPU initialization must succeed and never falls back to CPU.

## Run cached ViT or MLP training

Configs under `configs/` cover one, four, and sixteen chips for both workloads. Their batch sizes and schedules are explicit in TOML; the reference ViT configs use global batch **512** and per-rank microbatch **32**, preserving the original TPU study. Change the configuration explicitly to run a different batch protocol. Legacy `gpu_count` and `gpu_only` names come from the frozen shared schema and denote TPU chip count and cached-input execution here.

Freeze the same CPU model initialization once and reuse it across backend environments:

```bash
.venv-torchtpu/bin/python src/tpu_benchmark/prepare_initial_state.py \
  --config configs/vit-4chip.toml \
  --output /tmp/vit-4chip-initial.npz

launchers/torchtpu-ddp-defer.sh \
  --config configs/vit-4chip.toml \
  --initial-state /tmp/vit-4chip-initial.npz \
  --output /tmp/vit-torchtpu-smoke --nproc-per-node 4

launchers/xla.sh \
  --config configs/vit-4chip.toml \
  --initial-state /tmp/vit-4chip-initial.npz \
  --output /tmp/vit-xla-full --nproc-per-node 4 \
  --measurement-kind full
```

The initialization command also writes an adjacent `.rng.npy` file; keep both together. Use fresh output directories to preserve results.

The default `--measurement-kind smoke` adds `--preflight`: five optimizer updates for correctness and startup checks. It does **not** measure sustained throughput. `--measurement-kind full` runs the TOML schedule: reference configs specify 600 seconds of warmup, three 1,200-second measurements, and 5,000 fixed-job updates. `--measurement-kind full --single-trial` selects the original optimization-study schedule of 30 seconds warmup and one measurement, omitting the fixed job. Keep these result types separate.

Inspect any launch command without installing PyTorch or accessing TPUs:

```bash
launchers/torchtpu-functional-adamw.sh --dry-run \
  --config configs/vit-4chip.toml \
  --initial-state /tmp/vit-4chip-initial.npz \
  --output /tmp/preview --nproc-per-node 4
```

TorchTPU uses `torch.distributed.run` for multiple processes. XLA uses `torch_xla.launch` to start the visible local devices; `--nproc-per-node` is a requested topology description, while allocation/device visibility controls the actual XLA process count. The benchmark validates actual world size against the TOML config.

For multiple hosts, invoke the same launcher on every host with matching `--nnodes`, `--nproc-per-node`, `--master-addr`, and `--master-port`, and a distinct `--node-rank`. Set the TPU topology environment appropriate to the allocated slice. The launcher preserves that environment. A launcher does not reserve TPU resources or configure cloud networking.

## Run RPE1 convergence

Supply an external prepared RPE1 package and worker-local staged arrays. The committed selection JSON pins the original dataset manifest and validation-selected learning rate. Package preparation/staging helpers are retained in `src/tpu_benchmark/baseline/src/gpu_pilot/rpe1_data.py`; the original [RPE1 protocol](https://github.com/IhabBendidi/Cloud-GPU-Provider-Benchmark/tree/main/data/rpe1-convergence-v1) describes the data package.

```bash
launchers/rpe1-torchtpu.sh \
  --package /data/rpe1-package --local /tmp/rpe1-local \
  --selection configs/rpe1-selection.json \
  --output /tmp/rpe1-torchtpu --phase final --seed-base 1000
```

Use `rpe1-xla.sh` for XLA. These launchers run one independent seed per invocation, without DDP. Allocate and expose one chip per process through the backend's allocation environment. The runner defaults to 150 epochs, batch 512, FP32. The retained historical selection document records a 30-epoch final campaign; pass `--epochs 30` to reproduce that duration. Explicit smoke limits are available through the underlying runner's `--phase smoke`, `--max-train-steps`, and `--max-eval-cells` options. `src/tpu_benchmark/rpe1_phase_audit.py` records time attribution.
