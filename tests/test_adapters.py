import json
from pathlib import Path

import pytest
import torch
from backend import Backend
from workload import make_loader, masked_mse

from gpu_pilot.config import load_config

TPU = Path(__file__).resolve().parents[1] / "src/tpu_benchmark"


def test_masked_loss_and_gradient_match_original():
    torch.manual_seed(12)
    prediction = torch.randn(8, 32, requires_grad=True)
    target = torch.randn_like(prediction)
    mask = torch.rand_like(prediction) < 0.2
    original = ((prediction[mask] - target[mask]) ** 2).mean()
    replacement = masked_mse(prediction, target, mask)
    torch.testing.assert_close(original, replacement)
    a = torch.autograd.grad(original, prediction, retain_graph=True)[0]
    b = torch.autograd.grad(replacement, prediction)[0]
    torch.testing.assert_close(a, b)


def test_backend_rejects_implicit_fallback():
    with pytest.raises(ValueError):
        Backend("auto")


def test_validation_keeps_all_heldout_samples():
    config = load_config(TPU / "baseline/config/benchmark.synthetic.local.toml")
    backend = Backend("cpu-test")
    config.synthetic.micro_batch_size_per_rank = 33
    loader = make_loader(config, backend, validation=True)
    assert sum(len(batch[0]) for batch in loader) == config.synthetic.validation_sample_count


def test_baseline_snapshot_hashes():
    import hashlib

    root = TPU / "baseline"
    manifest = json.loads((root / "provenance.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_distributed_validation_has_no_duplicate_padding():
    config = load_config(TPU / "baseline/config/benchmark.synthetic.local.toml")
    backend = Backend("cpu-test")
    backend.world_size = 3
    indices = []
    for rank in range(3):
        backend.rank = rank
        indices.extend(make_loader(config, backend, validation=True).sampler)
    assert sorted(indices) == list(range(config.synthetic.validation_sample_count))


def test_comparison_rejects_unmatched_work():
    import copy

    from compare import compare

    record = {
        "status": "completed",
        "runtime": {"backend": "xla", "world_size": 1, "rank": 0},
        "config_sha256": "config",
        "initial_state_sha256": "weights",
        "first_cpu_batch_sha256": "input",
        "gradient_accumulation_steps": 1,
        "total_optimizer_updates": 8,
        "warmup_updates": 3,
        "trials": [{"optimizer_updates": 5, "samples_processed": 160}],
    }
    other = copy.deepcopy(record)
    other["runtime"]["backend"] = "torchtpu"
    other["first_cpu_batch_sha256"] = "different"
    with pytest.raises(ValueError, match="first_cpu_batch"):
        compare(record, other)
    other["first_cpu_batch_sha256"] = "input"
    other["trials"][0]["samples_processed"] = 320
    with pytest.raises(ValueError, match="samples_processed"):
        compare(record, other)


def test_historical_fixed_phase_matches_original_cpu(tmp_path):
    """Catch changes to RNG consumption, accumulation, resets and heldout sampling."""
    import argparse

    import numpy as np
    from historical_benchmark import run_phase
    from workload import make_model

    from gpu_pilot.run import DistributedContext, _run_trial

    config = load_config(TPU / "baseline/config/benchmark.synthetic.local.toml")
    config.run.mode = "gpu_only"
    config.synthetic.validation_sample_count = 131  # Exercise original drop_last.
    state = make_model(config).state_dict()
    fixture = tmp_path / "initial.npz"
    np.savez(fixture, **{k: v.numpy() for k, v in state.items()})
    np.save(fixture.with_suffix(".rng.npy"), torch.get_rng_state().numpy())
    args = argparse.Namespace(initial_state=fixture, communication="per-parameter")
    backend = Backend("cpu-test")
    torch.set_num_threads(1)
    adapted = run_phase(args, config, backend, "fixed_job", updates=3)
    original, _ = _run_trial(
        config,
        DistributedContext(0, 1, 0, torch.device("cpu")),
        seconds=None,
        updates=3,
        gpu_only=True,
        accumulation=2,
        precision="fp32",
    )
    assert adapted["samples_processed"] == original.samples_processed
    assert adapted["validation_loss"] == pytest.approx(original.validation_loss, abs=1e-6)
    repeated = run_phase(args, config, backend, "fixed_job_again", updates=3)
    for field in (
        "first_training_loss",
        "last_training_loss",
        "validation_loss",
        "first_cpu_batch_sha256",
        "initial_state_sha256",
    ):
        assert repeated[field] == adapted[field]


def test_historical_report_refuses_smoke_or_wrong_batch(tmp_path):
    from historical_report import compare_historical

    root = tmp_path
    config = load_config(TPU.parents[1] / "configs/mlp-1chip.toml").model_dump(mode="json")
    from historical_report import SOURCES

    reference = root / SOURCES[("synthetic_transcriptomics_mlp", 1)]
    reference.parent.mkdir(parents=True)
    reference.write_text(
        json.dumps(
            {
                "seed": config["run"]["seed"],
                "mode": "gpu_only",
                "effective_precision": "bf16",
                "global_batch_size": 512,
                "micro_batch_size_per_rank": 32,
                "gradient_accumulation_steps": 16,
                "generator_parameters": {},
                "trials": [{"throughput_samples_per_second": 1.0} for _ in range(3)],
                "fixed_job": {"optimizer_updates": 5000, "validation_loss": 0.1},
            }
        )
    )
    phases = [
        {
            "phase": name,
            "initial_state_sha256": "weights",
            "first_cpu_batch_sha256": "inputs",
            "optimizer_updates": 5000,
            "elapsed_seconds": 1200,
            "throughput_samples_per_second": 1.0,
            "validation_loss": 0.1,
        }
        for name in ("warmup", "measurement_1", "measurement_2", "measurement_3", "fixed_job")
    ]
    record = {
        "status": "completed",
        "measurement_kind": "full",
        "config": config,
        "runtime": {"world_size": 1},
        "adapter_sources": {},
        "phases": phases,
        "qualifications": [],
    }
    key = ("synthetic_transcriptomics_mlp", "xla", 0)
    records = {key: record}

    def result():
        return next(iter(compare_historical(records, root).values()))

    assert result()["status"] == "completed_with_aligned_historical_protocol"
    record["measurement_kind"] = "smoke"
    assert result()["status"] == "invalid_or_incomplete"
    record["measurement_kind"] = "full"
    config["synthetic"]["global_batch_size"] = 1024
    assert "global batch" in result()["reason"]
