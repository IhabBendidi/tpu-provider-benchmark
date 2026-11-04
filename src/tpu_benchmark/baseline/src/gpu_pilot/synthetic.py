"""Deterministic domain-shaped synthetic transcriptomics data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


def _sample_seed(seed: int, sample_id: int) -> int:
    return (seed * 6_364_136_223_846_793_005 + sample_id) % (2**63 - 1)


@dataclass(frozen=True)
class SyntheticExpressionSpec:
    seed: int
    dataset_size: int
    feature_count: int
    latent_size: int
    masking_rate: float
    sparsity_rate: float
    sample_offset: int = 0


class SyntheticExpressionDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Indexed sparse, correlated expression-like vectors without stored samples."""

    def __init__(self, spec: SyntheticExpressionSpec) -> None:
        self.spec = spec
        weight_generator = torch.Generator().manual_seed(spec.seed)
        self.loadings = torch.randn(
            spec.latent_size, spec.feature_count, generator=weight_generator
        )
        self.loadings /= spec.latent_size**0.5

    def __len__(self) -> int:
        return self.spec.dataset_size

    def __getitem__(self, sample_id: int) -> tuple[Tensor, Tensor, Tensor]:
        generator = torch.Generator().manual_seed(
            _sample_seed(self.spec.seed, self.spec.sample_offset + sample_id)
        )
        latent = torch.randn(self.spec.latent_size, generator=generator)
        expression = torch.nn.functional.softplus(latent @ self.loadings)
        expression = expression * (
            torch.rand(self.spec.feature_count, generator=generator) >= self.spec.sparsity_rate
        )
        expression = torch.log1p(expression)
        mask = torch.rand(self.spec.feature_count, generator=generator) < self.spec.masking_rate
        return expression.masked_fill(mask, 0.0), expression, mask


def generator_parameters(spec: SyntheticExpressionSpec) -> dict[str, int | float | str | bool]:
    return {
        "dataset_size": spec.dataset_size,
        "feature_count": spec.feature_count,
        "latent_size": spec.latent_size,
        "masking_rate": spec.masking_rate,
        "sparsity_rate": spec.sparsity_rate,
        "sample_offset": spec.sample_offset,
        "storage": "indexed_on_the_fly",
    }
