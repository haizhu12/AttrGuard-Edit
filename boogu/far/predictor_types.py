"""Structured outputs produced by the semantic PRI predictor."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .regions import PRIRegionProbabilities


@dataclass(frozen=True, eq=False)
class PRIRegionPrediction:
    r"""Two hierarchical logits and their Primary--Response--Invariant simplex.

    ``primary_logits`` parameterize :math:`p(P)`, while ``response_logits``
    parameterize :math:`p(R\mid\neg P)`.  ``regions`` contains the explicit
    three-channel stick-breaking result in P/R/I order.
    """

    regions: PRIRegionProbabilities
    primary_logits: torch.Tensor = field(repr=False)
    response_logits: torch.Tensor = field(repr=False)

    def __post_init__(self) -> None:
        expected_shape = (
            self.regions.values.shape[0],
            1,
            self.regions.values.shape[2],
            self.regions.values.shape[3],
        )
        for name, logits in (
            ("primary_logits", self.primary_logits),
            ("response_logits", self.response_logits),
        ):
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if logits.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
            if not torch.is_floating_point(logits):
                raise TypeError(f"{name} must have a floating-point dtype")
            if logits.device != self.regions.values.device:
                raise ValueError(f"{name} must be on the probability tensor device")
            if not torch.isfinite(logits.detach()).all().item():
                raise ValueError(f"{name} must be finite")

    @property
    def probabilities(self) -> torch.Tensor:
        """Return the P/R/I tensor while retaining its autograd graph."""

        return self.regions.values
