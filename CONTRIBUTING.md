# Development

Use Python 3.12 and keep CPU, XLA and TorchTPU dependencies in separate environments. Run commands from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/cpu.txt
make check
```

`make test` runs the CPU suite; `make lint` checks Python and shell syntax. Override the interpreter with `make PYTHON=/path/to/python check`. No check submits Slurm jobs or requires a TPU.

## Preserve benchmark meaning

Keep smoke, single-trial, diagnostic, and full-run results separate. Changes to loss, accumulation, sampling, precision, initialization, synchronization or timer boundaries require an explicitly versioned protocol and appropriate numerical validation. CPU correctness does not establish a TPU speedup.

The snapshot in `src/tpu_benchmark/baseline/` is frozen. Its provenance manifest verifies original file hashes; do not reformat or edit those files. Implement TPU-specific adapters in the adjacent source modules.

Python entrypoints remain executable files under `src/tpu_benchmark/`; launchers select the backend interpreter. Slurm submission maps the separated checkout into the established flat worker release. Update and run the release test whenever staging paths change.

## Continuous integration

`scripts/ci/cpu-tests.yml` is an optional GitHub Actions template. Copy it to `.github/workflows/cpu-tests.yml` using a credential permitted to publish workflows. It installs the CPU environment and runs the checks; it does not launch training.

## Changes and artifacts

Keep dataset files, checkpoints, raw results, logs, local credentials and environment directories out of commits. Record source/environment hashes alongside externally stored benchmark results. See [benchmark protocols](docs/protocols.md) and [known limitations](docs/known-limitations.md).
