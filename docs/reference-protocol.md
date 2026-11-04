> Historical protocol notes from the source GPU/TPU repository. Cluster names,
> service status, and result paths describe the original campaign, not this
> standalone checkout. Datasets and collected results are not bundled here.
> Use the [root README](../README.md) for current setup and launcher instructions.
> Paths and commands below intentionally retain their historical spelling.

# PyTorch/XLA versus TorchTPU on the Slurm TPU cluster

This is an isolated TPU experiment. The original `src/`, configurations, GPU
environments, and archived GPU results are unchanged. `baseline/` contains a
copy of the original source and configurations at commit
`d5e2eab59f8ffabc1daecb81ccb8098b5cddc4b8`; `baseline/provenance.json` pins every
copied file by SHA-256. The new runner imports model/data definitions from that
copy, not from the active GPU checkout.

## What works unchanged

The original main runner selects CUDA when available and otherwise CPU
(`baseline/src/gpu_pilot/run.py:85`). Its provider BF16 profile explicitly
requires CUDA (`resolve_precision`). Installing a TPU backend alone does not
change that selection. `native_probe.py` executed those unchanged functions on
a TPU worker: device `cpu`, CUDA unavailable, BF16 rejected. An FP32 CPU smoke
would not be a TPU benchmark.

The MLP, ViT, deterministic CPU datasets, and storage codecs are reusable.
The original Zarr training pipeline additionally uses CUDA streams/events,
NCCL, and GPU communication hooks. RPE1 has a configurable device but its
timing, telemetry, checkpoints, and precision policy still require a TPU audit.
The NVIDIA memory probe and telemetry cannot measure TPUs.

## Historical cluster discovered on 2026-10-28

- Controller: `tputrial-controller`; partition: `tpu`; maximum job: 12 hours.
- Workers: `tputrial-tpuworkers-[0-3]`, four v6e chips per worker, 16 total.
- Slice feature: `v6e_4x4`; each worker advertises 90 CPUs and 715,280 MiB RAM.
- Worker device access uses `/dev/vfio/0` through `/dev/vfio/3`.
- Controller and workers have independent filesystems. The checkout is not
  shared. Worker installations live under `/tmp/gpu-pilot-tpu-1694368683`.
- Worker metadata confirms project `rivercell-science`, zone `us-east5-a`.
  The actual hourly cost is unknown; zero placeholders are not free compute.

## Two explicitly separate software stacks

| Stack | PyTorch | Backend | libtpu | Python |
|---|---|---|---|---|
| PyTorch/XLA | 2.9.0+cpu | torch-xla 2.9.0 | 0.0.21 | 3.12.16 |
| TorchTPU | 2.14.1+cpu | torch-tpu 0.2.0 | 0.0.51 | 3.12.16 |

Both use NumPy 2.2.6. Full environment freezes are collected with results.
These releases do not provide an identical PyTorch/libtpu pairing: interpret
results as a software-stack comparison. PyTorch/XLA is deprecated upstream;
it is retained here specifically as the requested comparison baseline.

Sources checked during implementation:

- [TorchTPU](https://github.com/google-pytorch/torch_tpu)
- [TorchTPU execution modes](https://github.com/google-pytorch/torch_tpu/blob/main/docs/torchtpu_execution_model.md)
- [TorchTPU distributed setup](https://github.com/google-pytorch/torch_tpu/blob/main/docs/distributed_support.md)
- [PyTorch/XLA 2.9 API guide](https://github.com/pytorch/xla/blob/v2.9.0/API_GUIDE.md)
- [PyTorch/XLA deprecation notice](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html)

TorchTitan and vLLM are not required for these MLP/ViT workloads. Introducing
LLM training or serving would create additional benchmarks, not a faithful
port of the existing workloads.

## Implemented scope and measurement protocol

`benchmark.py` runs synthetic transcriptomics MLP and imaging ViT training:

- Original model definitions, seeds, dataset generation and held-out split.
- BF16 provider configurations retain MLP 8192→4096→8192 and ViT-B/16.
- Identical batch size, accumulation and unfused AdamW on both stacks.
- XLA lazy execution with an execution boundary after each microbatch.
- TorchTPU compiles model and loss with `torch.compile(backend="tpu",
  fullgraph=True)`; the optimizer remains eager. This is a first port, not a
  claim of optimal TorchTPU performance.
- Default explicit gradient averaging once per optimizer update. An optional
  TorchTPU DDP path uses 25 MiB gradient buckets and `no_sync()` for accumulation.
- Shared initial CPU tensors loaded from an NPZ fixture. Equal seeds alone
  proved insufficient: the two PyTorch versions initialized ViT differently.
  `prepare_initial_state.py` produces the shared fixture using PyTorch 2.14.1.
- Hashes record initial parameters, the first CPU batch, the configuration,
  and, in newer releases, adapter source files. The comparison rejects mismatches.
- The masked MLP loss uses a fixed-shape masked sum/count; CPU tests verify its
  value and gradients against the original boolean-indexed loss.
- Device synchronization is included in timing. Distributed throughput uses
  the slowest rank's elapsed time. No CPU fallback is permitted for TPU modes.
- Warmup includes initialization/compilation work and is reported separately.
  Compilation caches may already be warm; warmup is not labeled cold compilation.
- Trials continue the warmed model. Historical GPU trials reset their models;
  these new trial results must not be merged into GPU comparisons as if their
  protocols were identical. The copied TOML timing block is provenance only;
  this runner uses its explicit CLI step/repetition arguments.
- Validation preserves partial batches and avoids distributed sampler padding.
  Smoke tests bound validation to two batches per rank and label that scope.
- GPU-only input mode retains its legacy config name but means cached TPU
  inputs here. Procedural pipeline mode is available; it is not a substitute
  for the archived physical-disk Zarr benchmark.

The 1/4/16-chip configuration files use global batch 512 and microbatch 32.
They target the original standard-batch workload, not the newer 1024-per-GPU
imaging capacity campaign. A chip is not equivalent to a GPU for cost or scaling.
The copied input schema still has legacy `gpu_*` provider keys; actual backend,
device and package identity are recorded separately in the new result schema.

Not yet ported: the bounded Zarr scratch/S3 training pipeline,
frozen-checkpoint inference, checkpoint timing, hardware utilization/peak memory,
and the GPU capacity search. The snapshot retains those original modules for
reference; their presence does not imply TPU support. No full-provider ranking
or convergence claim should be inferred from smoke tests.

## Launching and logout persistence

`bootstrap-worker.sh` installs both private environments on each worker. Python
uses uv-managed 3.12; `LD_LIBRARY_PATH` must include that Python's `lib` directory
for the XLA extension. TorchTPU backend autoload must be enabled in its process.
XLA's Slurm launch supplies TPU metadata explicitly. Do not set
`CLOUD_TPU_TASK_ID=0` for every XLA child: the spawn launcher must assign ranks.

Submit from this checkout:

```sh
python3 experiments/tpu/submit.py --phase provider-smoke \
  --receipt results/tpu-20261028/receipt.json
python3 experiments/tpu/submit.py --phase distributed-smoke \
  --receipt results/tpu-20261028/receipt.json
python3 experiments/tpu/submit.py --phase multihost-smoke \
  --receipt results/tpu-20261028/receipt.json
python3 experiments/tpu/submit.py --phase multihost-pilot \
  --receipt results/tpu-20261028/receipt.json
```

The workers must be idle for staging, or supply `--after-job JOB_ID` to stage
inside an existing allocation covering the required workers and queue after it.
Source archives are streamed to each worker without a shared-filesystem assumption.
Each submission stages an immutable source release, then submits via `sbatch`. Individual smoke tests
have five-minute timeouts; the batch job is bounded to 25 minutes. Failure of
either backend makes the batch fail while retaining both logs.

The `multihost-pilot` phase instead uses ten warmup updates and three trials of
1,000 updates, followed by the complete holdout. Each workload/backend has a
20-minute limit; the complete job has a 90-minute limit. Results label this as a
fixed-work pilot, not a reproduction of the historical GPU full schedule.
Both the allocation and multi-host job steps explicitly request
`--gres=tpu:v6e:4 --layout=v6e=4x4`. Without the step resource request, Slurm's
device cgroup hides the TPU devices and both runtimes fail during discovery.

User lingering is enabled. `tpu-results-collector.timer` is enabled in the
controller's user systemd manager and runs every two minutes. It collects finished
receipt-listed jobs under `results/tpu-20261028/collected/`, retaining failed
attempts and dependency freezes. It then writes all-rank comparison reports to
`results/tpu-20261028/reports/`; missing ranks and unequal initialization invalidate
the affected comparison. It retries when a worker is busy. The unit
templates are in `scripts/systemd/` and contain this checkout's absolute path.

```sh
squeue -u "$USER"
systemctl --user status tpu-results-collector.timer
journalctl --user -u tpu-results-collector.service --since today
```

Slurm jobs and collection continue after logout. This does not make the chat
agent an autonomous background service. Worker `/tmp` data is not guaranteed
to survive VM replacement; successful collection creates controller-side copies.
Do not terminate workers before collection. No cloud resources, IAM permissions,
or global package installations were changed by this port.

## Research-friendly communication and execution changes

The cluster launch establishes the 4×4 ICI topology once. TorchTPU lowers
collectives through XLA; it does not use NCCL for TPU communication. Its default
separate collective executables limit compute/communication overlap. DDP can
group gradients, but the benefit depends on the model and compilation cost.
Compilation handshake is an additional candidate when collectives are captured
in compiled graphs. It requires coordinated compilation; rank-specific
recompilation needs dispatch-stage checking. An unconditional `spmd_safe`
annotation is inappropriate for arbitrary research models with rank-dependent
control flow. These mechanisms are described in the
[distributed guide](https://github.com/google-pytorch/torch_tpu/blob/main/docs/distributed_support.md).

`TORCH_TPU_DEFER_AND_FUSE=1` is a launcher-level experiment. It can combine small
eager operations around the already-compiled model, including optimizer work,
without rewriting model layers. Compilation can cost more initially, so we
record warmup separately. Host reads and synchronization interrupt deferred
execution; production training can log less frequently, but the historical
comparison deliberately retains the archived runner's host reads and fences.
Persistent compilation caches are another candidate, not yet configured here;
changed architectures/shapes still require new compilation. See the
[execution guide](https://github.com/google-pytorch/torch_tpu/blob/main/docs/torchtpu_execution_model.md).

Keep these controls in the backend/launcher and run a short correctness check
after model changes. This is reusable infrastructure, not a guarantee of equal
speedups for every architecture. The [API policy](https://github.com/google-pytorch/torch_tpu/blob/main/docs/public_apis.md)
labels TPU eager-mode, topology, and optimizer helpers experimental; the tested
environment is pinned to torch-tpu 0.2.0. No custom attention, Pallas kernels,
model-specific sharding, or experimental replacement optimizer is required for
the changes tested here. DDP remains optional because it regressed the MLP.

`multihost-optimize` tests four TorchTPU variants with identical weights/data:
default, DDP, defer-and-fuse, and DDP plus defer-and-fuse. The collector verifies
all ranks and writes `optimization_variants` in the job report. These are short
pilot measurements with bounded validation; they are not convergence results.

## Alignment to existing AWS H100 records

No new H100 run is requested or scheduled. `historical_benchmark.py` uses the
archived runner's protocol: 600 seconds warmup, three 1200-second trials, and a
separate 5000-update fixed job. Each phase resets model and AdamW. The original
loaders retain their sampler order and validation drop/padding rules. Every
microbatch extracts its loss to the host; every update synchronizes execution
and performs the original three separate MAX timing reductions. A CPU regression
test matches the original fixed-update validation result and verifies phase resets.

The historical modes run XLA, default TorchTPU, and TorchTPU with defer-and-fuse.
The shorter `historical-smoke`/`multihost-historical-smoke` modes only check
execution. The full modes use the TOML timing block; they do not use the pilot's
fixed-step timing options. A full job has a six-hour limit, with two hours per
backend variant. Partial phases are retained if execution fails.

| TPU full mode | Archived AWS source | Devices each | Per-device batch | Global batch | Accumulation |
|---|---|---:|---:|---:|---:|
| `historical-full` (MLP) | `tx-1gpu-6-full-gpu_only` | 1 | 32 | 512 | 16 |
| `multihost-historical-full` (ViT) | `imaging-16gpu-21-full-gpu_only` | 16 | 32 | 512 | 1 |
| `multihost-ddp-historical-full` (ViT; DDP / DDP+fusion) | `imaging-16gpu-21-full-gpu_only` | 16 | 32 | 512 | 1 |

Slurm rejected an eight-chip `v6e=2x4` allocation with the current feature
configuration. The archived eight-H100 results remain reference data, not a
matched comparison to sixteen TPUs. The newer 1024-per-GPU capacity runs are
excluded. Equal device counts are not equal power, price, memory, or chip capability.

`historical_report.py` checks all ranks, per-phase initialization/input hashes,
device count, seed, precision, global and per-device batch, accumulation, dataset
settings, trial lengths and fixed-update count before calculating an H100 ratio.
It refuses smoke records or mismatched settings. Archived H100 records contain
no initialization/input hashes, so bitwise historical identity cannot be proven.
PyTorch/XLA's older torch/libtpu versions, backend math, and optimizer implementation
remain explicit qualifications. Matching timing and work does not establish
numerical quality equivalence. Check the 5000-update validation results separately.

`vit-capacity` separately tests per-chip batches 32 and 1024 in fresh one-chip
processes, for XLA, TorchTPU, and TorchTPU with fusion. Each candidate performs
five forward/backward/AdamW updates, without activation checkpointing. Each
process has a ten-minute timeout and the overall job has a 75-minute limit.
Memory exhaustion is retained as a capacity outcome; unrelated errors/timeouts
remain unclassified failures. This is neither a maximum-batch search nor a
distributed throughput test. `capacity_candidates` appears in the collected
job report, with memory API diagnostics where supported.

## Local verification

Use a separate CPU environment with PyTorch, NumPy, Pydantic, psutil, pytest and
ruff. `cpu-test` is an explicit test mode, never an implicit TPU fallback.

```sh
python -m pytest -q experiments/tpu/test_adapters.py
python experiments/tpu/benchmark.py --backend cpu-test \
  --config experiments/tpu/baseline/config/benchmark.synthetic.local.toml \
  --output /tmp/tpu-cpu-check --warmup-steps 2 --steps 2 --repetitions 1 \
  --validation-batches 2
```

Every output directory must be new. Live receipts, failures, successful smoke
records, and matched comparison reports are in `results/tpu-20261028/`.

## RPE1 convergence campaign

`rpe1_train.py` ports the frozen trainer while importing its original model,
NumPy initialization, evaluation and Pearson Delta implementation. The prepared
package is already checked in: `stage_rpe1.py` verifies its shards and reconstructed
expression checksum on all workers. NumPy stays pinned; h5py 3.16.0 and SciPy 1.18.1
are added for package loading, with separate RPE1 dependency freezes.

This is **16 independent single-chip runs**, seeds 1000–1015, batch **512 per
chip**, no accumulation, and no DDP. Each seed trains 150 epochs at the already
selected learning rate 0.003. Full runs retain all training and heldout cells,
including the final 333-cell training batch. No hyperparameter resweep or early
stopping is introduced. The full package/split manifest is
`ac1f6774ed55694df9086e8eca8132dd34b7a428449934b158a9bb2db8a121cf`.

Both backends use FP32 tensors and explicit `highest` matrix multiplication
precision. XLA uses `torch_xla.backends.set_mat_mul_precision`; TorchTPU uses
`torch.set_float32_matmul_precision`. TPU's highest mode is still not a promise
of bitwise CUDA arithmetic. TorchTPU compiles the model forward and enables
defer-and-fuse for surrounding operations. Original CPU sampling and metrics are
preserved; a regression test matches the original CPU learning curve. The TPU
smoke has not run at submission, so hardware compatibility remains to be verified.

The smoke phase runs one epoch with 20 training steps and bounded evaluation on
both stacks. The separate full jobs require its Slurm success before starting;
both use all cells and 150 epochs. The collector retains CSV curves and original
threshold crossings, checks all sixteen seeds, and compares epoch-150 scores
and times against the existing H100 campaign. Compilation is included in the
setup, training or evaluation time where incurred, not separately isolated yet.
NVIDIA telemetry and its allocator cap do not apply; legacy GPU memory fields
are explicitly unmeasured zeros. Initial/best/final NPZ model files remain
worker-local and are excluded from evidence archives and Git publication.

### RPE1 evaluation on XLA

`rpe1_evaluation.py` preserves the frozen evaluator but uses `torch.no_grad()`
for both TPU backends. XLA 2.9 fails under `torch.inference_mode()` with an
inference-tensor version-counter error. The original GPU evaluator is unchanged.
The RPE1 smoke phase now runs a complete epoch and all evaluation cells to
cover final partial batches. On job 106, all sixteen seeds per backend passed,
with matching paired initial weights and training order. Full convergence uses
150 epochs, batch 512 per chip, and independent seeds 1000–1015.

`submit.py --dependency afterok:JOB` can gate a run on smoke success separately
from `--after-job JOB`, which selects an active allocation for code staging.
