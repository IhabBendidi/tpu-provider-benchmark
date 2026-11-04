"""Stage fences must preserve the ordinary training update."""

import copy

import torch
from backend import Backend
from phase_profile import update
from workload import LossModel


def test_segmented_update_preserves_parameters_and_gradients():
    torch.manual_seed(123)
    original = LossModel(
        torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4)),
        "synthetic_transcriptomics_mlp",
    )
    segmented = copy.deepcopy(original)
    batch = (torch.randn(3, 4), torch.randn(3, 4), torch.ones(3, 4, dtype=torch.bool))
    optimizers = [
        torch.optim.AdamW(m.parameters(), foreach=False, fused=False) for m in (original, segmented)
    ]
    for _ in range(3):
        for model, optimizer, split in zip(
            (original, segmented), optimizers, (False, True), strict=True
        ):
            timings = update(
                model,
                model,
                optimizer,
                batch,
                Backend("cpu-test"),
                2,
                "fp32",
                "per-parameter",
                segmented=split,
            )
            if split:
                assert set(timings) == {
                    "zero_grad",
                    "forward_and_loss",
                    "backward_including_ddp_if_enabled",
                    "host_loss_read",
                    "optimizer",
                }
                assert all(t >= 0 for t in timings.values())
    for a, b in zip(original.parameters(), segmented.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_timing_reduction_uses_float_for_integer_barrier(monkeypatch):
    backend = Backend("cpu-test")
    seen = []

    def reduce(value, operation):
        seen.append((value.dtype, operation))
        return value

    monkeypatch.setattr(backend, "reduce", reduce)
    assert backend.max_seconds(0) == 0.0
    assert seen == [(torch.float32, "max")]


def test_microstep_leaves_profile_scope(monkeypatch):
    from contextlib import contextmanager

    import phase_profile

    scopes = []

    @contextmanager
    def fake_stage(name, *args):
        scopes.append(name)
        try:
            yield
        finally:
            scopes.pop()

    backend = Backend("cpu-test")
    calls = []

    def microstep():
        assert not scopes
        calls.append(True)

    monkeypatch.setattr(phase_profile, "stage", fake_stage)
    monkeypatch.setattr(backend, "microstep", microstep)
    model = LossModel(torch.nn.Linear(4, 4), "synthetic_transcriptomics_mlp")
    optimizer = torch.optim.AdamW(model.parameters())
    batch = (torch.ones(2, 4), torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool))
    update(model, model, optimizer, batch, backend, 2, "fp32", "per-parameter", tracing=True)
    assert len(calls) == 2
