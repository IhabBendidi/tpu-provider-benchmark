# Slurm operations and reporting

`src/tpu_benchmark/submit.py` stages immutable releases and submits the original bounded worker phases, including profiling, capacity, full benchmarks, and RPE1. Run `--help` for phase and variant choices. `--node`, `--nodes`, and `--partition` override the original cluster defaults. The existing multi-host worker scripts assume a **four-host v6e slice**, with four chips per host; adapt those scripts before using another topology. Worker environments live below `/tmp/gpu-pilot-tpu-<uid>` in this original workflow; pass that directory itself to `bootstrap-worker.sh`. Shared root filesystem access is not assumed. `stage_rpe1.py` requires an explicit `--package` path.

`collect.py`, `report.py`, `optimization_report.py`, `historical_report.py`, and `rpe1_report.py` process collected JSON/CSV evidence. Historical cross-provider reporting helpers retain references to the original study's result paths. Those archives are not bundled; optional reference readers report unavailable references. Provide the original archives when reproducing cross-provider reports. [Historical campaign notes](reference-protocol.md) preserves historical campaign notes, including original cluster paths, rather than describing a new measurement campaign.

The systemd collector template under `scripts/systemd/` is opt-in. Replace its placeholders for your checkout before installing it; nothing enables services automatically.

## Worker release layout

The checkout separates Python source, configs, requirements, and worker scripts. `submit.py` packages them into the existing flat worker release layout:

```text
release/
├── historical_benchmark.py
├── smoke-worker.sh
├── phase-worker.sh
├── single-trial-worker.sh
├── bootstrap-worker.sh
├── config/
├── requirements/
└── baseline/
```

Worker commands and relative paths inside that archive are preserved. The release test verifies the files and frozen SHA-256 manifest without submitting a job.

```bash
python src/tpu_benchmark/submit.py --help
python src/tpu_benchmark/stage_rpe1.py --help
python src/tpu_benchmark/collect.py --help
```

Use the backend interpreter where a command imports PyTorch. Study-specific reporting tools can require archives from the original repository; missing archives are not benchmark failures.
