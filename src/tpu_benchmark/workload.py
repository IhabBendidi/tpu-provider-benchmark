"""Matched models and data from the immutable GPU source copy."""

import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

BASELINE = Path(__file__).resolve().parent / "baseline"
sys.path.insert(0, str(BASELINE / "src"))

from gpu_pilot.imaging import VisionTransformer  # noqa: E402
from gpu_pilot.imaging import _make_loader as image_loader  # noqa: E402
from gpu_pilot.run import DistributedContext, MaskedExpressionMlp  # noqa: E402
from gpu_pilot.run import _make_loader as expression_loader  # noqa: E402


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in values:
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def masked_mse(prediction, target, mask):
    # Boolean indexing in the GPU original creates a data-dependent shape.
    # This keeps the same masked mean and gradients using fixed-shape tensors.
    squared = (prediction.float() - target.float()).square()
    return torch.where(mask, squared, 0.0).sum() / mask.sum().clamp_min(1)


class LossModel(nn.Module):
    def __init__(self, model, workload):
        super().__init__()
        self.model, self.workload = model, workload

    def forward(self, *batch):
        prediction = self.model(batch[0])
        if self.workload == "synthetic_transcriptomics_mlp":
            return masked_mse(prediction, batch[1], batch[2])
        return nn.functional.cross_entropy(prediction.float(), batch[1])


def make_model(config):
    torch.manual_seed(config.run.seed)
    if config.synthetic is not None:
        model = MaskedExpressionMlp(config.synthetic.feature_count, config.synthetic.hidden_size)
    else:
        model = VisionTransformer(config.imaging)
    return LossModel(model, config.run.workload)


def make_loader(config, backend, validation=False):
    # A dedicated sampler RNG makes sample order independent of model init and
    # backend/compiler RNG consumption, for single-device and distributed runs.
    context = DistributedContext(backend.rank, backend.world_size, 0, torch.device("cpu"))
    original = (expression_loader if config.synthetic else image_loader)(
        config, context, validation
    )
    dataset = original.dataset
    sampler = DistributedSampler(
        dataset,
        num_replicas=backend.world_size,
        rank=backend.rank,
        shuffle=not validation,
        seed=config.run.seed,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_size=original.batch_size,
        sampler=range(backend.rank, len(dataset), backend.world_size) if validation else sampler,
        num_workers=0,
        drop_last=not validation,
    )


def validate(model, config, backend, max_batches=None):
    model.eval()
    numerator = torch.zeros((), device=backend.device)
    denominator = torch.zeros((), device=backend.device)
    batches = 0
    with torch.no_grad():
        for batch in make_loader(config, backend, validation=True):
            batch = tuple(x.to(backend.device) for x in batch)
            with backend.autocast(config.run.precision):
                prediction = model.model(batch[0])
                if config.synthetic:
                    count = batch[2].sum()
                    error = torch.where(batch[2], (prediction.float() - batch[1]).square(), 0).sum()
                else:
                    count = torch.tensor(len(batch[1]), device=backend.device)
                    error = nn.functional.cross_entropy(
                        prediction.float(), batch[1], reduction="sum"
                    )
            numerator += error
            denominator += count
            backend.sync()
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    numerator = backend.reduce(numerator)
    denominator = backend.reduce(denominator)
    backend.sync()
    result = {
        "loss": float((numerator / denominator).cpu()),
        "denominator": float(denominator.cpu()),
        "batches_per_rank": batches,
        "scope": "bounded_smoke" if max_batches else "full_holdout",
        "distributed_sampler_may_pad": False,
    }
    model.train()
    return result
