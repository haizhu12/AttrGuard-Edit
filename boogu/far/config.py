"""Runtime configuration for flow-aware regional editing."""

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal

MAX_RESPONSE_KERNEL_SIZE = 4095
"""Largest supported odd response-filter kernel, bounding temporary memory."""


def _normalize_finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _normalize_invariant_strength(value: Any) -> float:
    normalized = _normalize_finite_real("invariant_strength", value)
    if not 0 <= normalized <= 1:
        raise ValueError("invariant_strength must be between 0 and 1")
    return normalized


def _normalize_kernel_sizes(name: str, values: Any) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of integers")
    if not values:
        raise ValueError(f"{name} must not be empty")

    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain only integers")
        kernel_size = int(value)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"{name} must contain only positive odd integers")
        if kernel_size > MAX_RESPONSE_KERNEL_SIZE:
            raise ValueError(
                f"{name} values must be at most {MAX_RESPONSE_KERNEL_SIZE}"
            )
        normalized.append(kernel_size)
    return tuple(normalized)


@dataclass(frozen=True)
class FARConfig:
    """Validated controls for flow-aware regional editing."""

    enabled: bool = False
    effect_strength: float = 0.5
    effect_low_pass_kernel: int = 5
    probability_tolerance: float = 1e-4
    invariant_strength: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")

        effect_strength = _normalize_finite_real(
            "effect_strength",
            self.effect_strength,
        )
        if not 0 <= effect_strength <= 1:
            raise ValueError("effect_strength must be between 0 and 1")
        object.__setattr__(self, "effect_strength", effect_strength)

        kernel = self.effect_low_pass_kernel
        if isinstance(kernel, bool) or not isinstance(kernel, Integral):
            raise TypeError("effect_low_pass_kernel must be an integer")
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("effect_low_pass_kernel must be a positive odd integer")
        if kernel > MAX_RESPONSE_KERNEL_SIZE:
            raise ValueError(
                f"effect_low_pass_kernel must be at most {MAX_RESPONSE_KERNEL_SIZE}"
            )
        object.__setattr__(self, "effect_low_pass_kernel", int(kernel))

        probability_tolerance = _normalize_finite_real(
            "probability_tolerance",
            self.probability_tolerance,
        )
        if not 0 < probability_tolerance < 1:
            raise ValueError("probability_tolerance must be between 0 and 1")
        object.__setattr__(self, "probability_tolerance", probability_tolerance)
        object.__setattr__(
            self,
            "invariant_strength",
            _normalize_invariant_strength(self.invariant_strength),
        )


@dataclass(frozen=True)
class ResponseTransportConfig:
    """Validated controls for timestep-conditioned response transport."""

    mode: Literal["fixed", "analytic_multiscale"] = "analytic_multiscale"
    kernel_sizes: tuple[int, ...] = (15, 7, 3)
    schedule_centers: tuple[float, ...] = (0.0, 0.5, 1.0)
    schedule_temperature: float = 0.25
    response_strength: float = 0.5
    support_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("mode must be a string")
        if self.mode not in ("fixed", "analytic_multiscale"):
            raise ValueError("mode must be 'fixed' or 'analytic_multiscale'")

        kernel_sizes = _normalize_kernel_sizes("kernel_sizes", self.kernel_sizes)
        if any(left <= right for left, right in pairwise(kernel_sizes)):
            raise ValueError("kernel_sizes must be strictly decreasing")
        object.__setattr__(self, "kernel_sizes", kernel_sizes)

        if not isinstance(self.schedule_centers, tuple):
            raise TypeError("schedule_centers must be a tuple of real numbers")
        if len(self.schedule_centers) != len(kernel_sizes):
            raise ValueError("schedule_centers length must match kernel_sizes length")
        centers = tuple(
            _normalize_finite_real("schedule_centers", center)
            for center in self.schedule_centers
        )
        if any(not 0 <= center <= 1 for center in centers):
            raise ValueError("schedule_centers must be between 0 and 1")
        if any(left >= right for left, right in pairwise(centers)):
            raise ValueError("schedule_centers must be strictly increasing")
        object.__setattr__(self, "schedule_centers", centers)

        if self.mode == "fixed" and len(kernel_sizes) != 1:
            raise ValueError(
                "fixed response transport requires exactly one kernel and center"
            )

        temperature = _normalize_finite_real(
            "schedule_temperature",
            self.schedule_temperature,
        )
        if temperature <= 0:
            raise ValueError("schedule_temperature must be positive")
        squared_temperature = temperature * temperature
        if squared_temperature == 0 or not isfinite(squared_temperature):
            raise ValueError(
                "schedule_temperature must have a finite, non-zero stable squared scale"
            )
        object.__setattr__(self, "schedule_temperature", temperature)

        response_strength = _normalize_finite_real(
            "response_strength",
            self.response_strength,
        )
        if not 0 <= response_strength <= 1:
            raise ValueError("response_strength must be between 0 and 1")
        object.__setattr__(self, "response_strength", response_strength)

        support_epsilon = _normalize_finite_real(
            "support_epsilon",
            self.support_epsilon,
        )
        if not 0 < support_epsilon < 1:
            raise ValueError("support_epsilon must be between 0 and 1")
        object.__setattr__(self, "support_epsilon", support_epsilon)
