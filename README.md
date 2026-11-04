# TPU Provider Benchmark

Reproducible training benchmarks for **PyTorch/XLA** and **TorchTPU**, with dedicated launchers for backend and optimization comparisons.

The suite includes cached-input Vision Transformer and masked-expression MLP training, RPE1 convergence, profiling, capacity probes, and Slurm orchestration. Model and data definitions are preserved in a frozen, hash-verified snapshot from [Cloud-GPU-Provider-Benchmark](https://github.com/IhabBendidi/Cloud-GPU-Provider-Benchmark).

## Quick start

Use Python 3.12. Install each TPU backend in its own environment:

```bash
python3.12 -m venv .venv-torchtpu
.venv-torchtpu/bin/pip install -r requirements/torchtpu.txt
```

Preview a launch without accessing a TPU or installing a backend:

```bash
launchers/torchtpu-ddp-defer.sh --dry-run \
  --config configs/vit-4chip.toml \
  --initial-state /tmp/vit-initial.npz \
  --output /tmp/vit-smoke --nproc-per-node 4
```

For real execution, first prepare shared initial weights and configure device visibility inside your TPU allocation. Follow [Getting started](docs/getting-started.md) for complete setup and training commands.

## Documentation

| Guide | Contents |
| --- | --- |
| [Getting started](docs/getting-started.md) | Isolated environments, initial weights, ViT/MLP and RPE1 commands |
| [Launchers](docs/launchers.md) | All 15 launchers and their exact optimization compositions |
| [Benchmark protocols](docs/protocols.md) | Batch sizes, precision, timing, comparison boundaries |
| [Slurm operations](docs/slurm.md) | Immutable releases, staging, collection and reporting |
| [Known limitations](docs/known-limitations.md) | Performance caveats and hardware validation status |
| [Development](CONTRIBUTING.md) | CPU checks and reproducibility requirements |

## Repository layout

```text
configs/                   Workload and convergence configurations
docs/                      Setup, protocols and operational guides
launchers/                 Separate backend and optimization entrypoints
requirements/              Pinned CPU, XLA and TorchTPU environments
scripts/                   Setup, Slurm workers, CI and service templates
src/tpu_benchmark/         Runners, adapters, profilers and report tools
  baseline/                Frozen model/data dependencies and hash manifest
tests/                     CPU correctness, launcher and release checks
```

## Validation and scope

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/cpu.txt
make check
```

CPU checks validate numerical behavior, frozen source hashes, launcher plans, and release packaging. They do not establish TPU performance or backend compatibility. The reference training configs preserve global batch 512 and per-device microbatch 32; smoke and full results must remain separate.

This checkout contains code and configuration. Supply datasets, weights, and result archives separately. See [provenance.json](provenance.json) for source attribution and [LICENSE](LICENSE) for the MIT license.
