"""Validated Primary--Response--Invariant region probability tensors."""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn.functional as F

_DEFAULT_TOLERANCE = 1e-4
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_PROBABILITY_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_SUPPORTED_PROBABILITY_DESCRIPTION = (
    "dense strided float16, bfloat16, float32, and float64 tensors"
)
_SIMPLEX_DENOMINATORS = {
    torch.float16: 2048,
    torch.bfloat16: 256,
}


def _normalize_tolerance(tolerance: Any) -> float:
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a real number")
    try:
        normalized = float(tolerance)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("tolerance must be finite and between 0 and 1") from error
    if not isfinite(normalized) or not 0 < normalized < 1:
        raise ValueError("tolerance must be finite and between 0 and 1")
    return normalized


def _validate_tensor_layout(values: Any) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if (
        values.layout != torch.strided
        or values.dtype not in _SUPPORTED_PROBABILITY_DTYPES
    ):
        raise TypeError(
            f"values only supports {_SUPPORTED_PROBABILITY_DESCRIPTION}"
        )
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("values must have shape [B, 3, H, W]")
    if values.shape[0] <= 0 or values.shape[2] <= 0 or values.shape[3] <= 0:
        raise ValueError("values must have positive batch and spatial dimensions")
    if not torch.isfinite(values).all().item():
        raise ValueError("values must be finite")


def _calculation_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in _LOW_PRECISION_DTYPES:
        return torch.float32
    return torch.float64


def _directional_rounding_budget(
    values: torch.Tensor,
    signed_sum_error: torch.Tensor,
) -> torch.Tensor:
    next_up = torch.nextafter(values, torch.full_like(values, float("inf")))
    next_down = torch.nextafter(values, torch.full_like(values, float("-inf")))
    upward_spacing = (next_up - values).abs()
    downward_spacing = (values - next_down).abs()
    upward_spacing = torch.where(
        torch.isfinite(upward_spacing),
        upward_spacing,
        downward_spacing,
    )
    downward_spacing = torch.where(
        torch.isfinite(downward_spacing),
        downward_spacing,
        upward_spacing,
    )
    relevant_spacing = torch.where(
        (signed_sum_error >= 0).unsqueeze(1),
        downward_spacing,
        upward_spacing,
    )
    return 0.5 * relevant_spacing.sum(
        dim=1,
        dtype=signed_sum_error.dtype,
    )


def _validate_probabilities(values: Any, tolerance: float) -> None:
    _validate_tensor_layout(values)
    if ((values < 0) | (values > 1)).any().item():
        raise ValueError("probabilities must lie in [0, 1] in every channel")
    calculation_dtype = _calculation_dtype(values.dtype)
    channel_sums = values.sum(dim=1, dtype=calculation_dtype)
    if (channel_sums <= 0).any().item():
        raise ValueError("probability channel sums must be strictly positive")
    signed_sum_error = channel_sums - 1
    rounding_budget = _directional_rounding_budget(
        values,
        signed_sum_error,
    )
    if (signed_sum_error.abs() > tolerance + rounding_budget).any().item():
        raise ValueError(
            "probabilities must sum to 1 along the region dimension "
            "within the explicit tolerance and component rounding budget"
        )


def _quantize_simplex_with_ste(
    normalized: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    denominator = _SIMPLEX_DENOMINATORS[output_dtype]
    scaled = normalized * denominator
    lower_quotas = torch.floor(scaled)
    remainders = scaled - lower_quotas
    remaining = (denominator - lower_quotas.sum(dim=1, keepdim=True)).to(torch.int64)
    remaining = remaining.clamp(min=0, max=normalized.shape[1])
    remainder_ranks = remainders.argsort(dim=1, descending=True).argsort(dim=1)
    increments = (remainder_ranks < remaining).to(normalized.dtype)
    quantized = (lower_quotas + increments) / denominator
    straight_through = normalized + (quantized - normalized).detach()
    return straight_through.to(output_dtype)


def _normalize_generated_probabilities(
    values: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    _validate_tensor_layout(values)
    calculation_dtype = _calculation_dtype(values.dtype)
    working_values = values.to(calculation_dtype)
    denominator = working_values.sum(dim=1, keepdim=True)
    if not torch.isfinite(denominator).all().item() or (denominator <= 0).any().item():
        raise ValueError(
            "probability normalization denominator must be finite and strictly positive"
        )

    normalized_working = working_values / denominator
    normalized = normalized_working.to(values.dtype)
    for _ in range(2):
        correction_values = normalized.to(calculation_dtype)
        signed_sum_error = correction_values.sum(dim=1, keepdim=True) - 1
        needs_correction = signed_sum_error.abs() > tolerance
        residual = -signed_sum_error
        can_absorb = (correction_values > 0) & (
            (residual >= 0) | (correction_values >= -residual)
        )
        candidates = torch.where(
            can_absorb,
            correction_values,
            torch.full_like(correction_values, float("inf")),
        )
        correction_channel = candidates.argmin(dim=1, keepdim=True)
        correction_mask = torch.zeros_like(correction_values).scatter(
            1,
            correction_channel,
            1,
        )
        correction = correction_mask * residual * needs_correction
        normalized = (correction_values + correction).to(values.dtype)

    if values.dtype in _LOW_PRECISION_DTYPES:
        remaining_error = normalized.sum(dim=1, dtype=torch.float64).sub(1).abs()
        needs_fallback = (remaining_error > tolerance).unsqueeze(1)
        simplex_values = _quantize_simplex_with_ste(
            normalized_working,
            values.dtype,
        )
        normalized = torch.where(needs_fallback, simplex_values, normalized)
    return normalized


@dataclass(frozen=True, eq=False)
class PRIRegionProbabilities:
    """Probabilities in primary/response/invariant channel order.

    ``direct`` and ``companion`` are read-only aliases retained as legacy terminology.
    """

    values: torch.Tensor = field(repr=False)
    tolerance: float = _DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        tolerance = _normalize_tolerance(self.tolerance)
        _validate_probabilities(self.values, tolerance)
        object.__setattr__(self, "tolerance", tolerance)

    @classmethod
    def from_probabilities(
        cls,
        values: torch.Tensor,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> "PRIRegionProbabilities":
        """Validate probabilities without transforming or detaching them."""

        return cls(values=values, tolerance=tolerance)

    @classmethod
    def _from_runtime_projection(
        cls,
        values: torch.Tensor,
        tolerance: float,
    ) -> "PRIRegionProbabilities":
        """Construct from tensor-only projection invariants without a host sync."""

        instance = object.__new__(cls)
        object.__setattr__(instance, "values", values)
        object.__setattr__(instance, "tolerance", _normalize_tolerance(tolerance))
        return instance

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> "PRIRegionProbabilities":
        """Convert three-channel logits to probabilities explicitly."""

        tolerance = _normalize_tolerance(tolerance)
        _validate_tensor_layout(logits)
        if logits.dtype in _LOW_PRECISION_DTYPES:
            probabilities = torch.softmax(logits.float(), dim=1).to(logits.dtype)
        else:
            probabilities = torch.softmax(logits, dim=1)
        probabilities = _normalize_generated_probabilities(
            probabilities,
            tolerance,
        )
        return cls(values=probabilities, tolerance=tolerance)

    @property
    def primary(self) -> torch.Tensor:
        return self.values[:, 0:1]

    @property
    def response(self) -> torch.Tensor:
        return self.values[:, 1:2]

    @property
    def invariant(self) -> torch.Tensor:
        return self.values[:, 2:3]

    @property
    def direct(self) -> torch.Tensor:
        """Alias for :attr:`primary` using legacy terminology."""

        return self.primary

    @property
    def companion(self) -> torch.Tensor:
        """Alias for :attr:`response` using legacy terminology."""

        return self.response

    def resize_like(self, reference: torch.Tensor) -> "PRIRegionProbabilities":
        """Resize to a reference tensor's spatial grid and renormalize."""

        if not isinstance(reference, torch.Tensor):
            raise TypeError("reference must be a torch.Tensor")
        if reference.ndim != 4 or any(dimension <= 0 for dimension in reference.shape):
            raise ValueError(
                "reference must have shape [B, C, H, W] with positive dimensions"
            )
        if reference.shape[0] != self.values.shape[0]:
            raise ValueError("reference batch size must match probability batch size")

        interpolation_values = (
            self.values.float()
            if self.values.dtype in _LOW_PRECISION_DTYPES
            else self.values
        )
        resized = F.interpolate(
            interpolation_values,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.values.dtype in _LOW_PRECISION_DTYPES:
            resized = resized.to(self.values.dtype)
        normalized = _normalize_generated_probabilities(
            resized,
            self.tolerance,
        )
        return type(self).from_probabilities(
            normalized,
            tolerance=self.tolerance,
        )

    def repeat_interleave(self, repeats: int) -> "PRIRegionProbabilities":
        """Repeat each probability-map sample consecutively along the batch."""

        if isinstance(repeats, bool) or not isinstance(repeats, Integral):
            raise TypeError("repeats must be an integer")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        repeated = self.values.repeat_interleave(int(repeats), dim=0)
        return type(self).from_probabilities(
            repeated,
            tolerance=self.tolerance,
        )

    def to(self, *args: Any, **kwargs: Any) -> "PRIRegionProbabilities":
        """Apply ``Tensor.to`` and validate the converted probabilities."""

        _validate_probabilities(self.values, self.tolerance)
        converted = self.values.to(*args, **kwargs)
        if converted.dtype != self.values.dtype:
            converted = _normalize_generated_probabilities(
                converted,
                self.tolerance,
            )
        return type(self).from_probabilities(
            converted,
            tolerance=self.tolerance,
        )


CausalRegionProbabilities = PRIRegionProbabilities
