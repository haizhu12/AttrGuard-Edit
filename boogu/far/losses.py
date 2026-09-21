"""Losses for the lightweight Primary--Response--Invariant predictor.

The objective in this module follows the registered two-timepoint training
formulation.  It deliberately operates on already-constructed PRI
probabilities; it neither applies a softmax nor silently projects malformed
inputs back onto the probability simplex.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import torch
from torch import nn


def _finite_real(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if nonnegative and normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


@dataclass(frozen=True)
class PRILossConfig:
    """Pre-registered coefficients and numerical constants for PRI training."""

    lambda_ce: float = 1.0
    lambda_dice: float = 1.0
    lambda_boundary: float = 0.1
    lambda_temporal: float = 0.2
    tau_mass: float = 1.0
    epsilon: float = 1e-6
    simplex_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        for name in (
            "lambda_ce",
            "lambda_dice",
            "lambda_boundary",
            "lambda_temporal",
        ):
            object.__setattr__(
                self,
                name,
                _finite_real(name, getattr(self, name), nonnegative=True),
            )
        tau_mass = _finite_real("tau_mass", self.tau_mass, nonnegative=True)
        if tau_mass <= 0:
            raise ValueError("tau_mass must be positive")
        object.__setattr__(self, "tau_mass", tau_mass)
        epsilon = _finite_real("epsilon", self.epsilon, nonnegative=True)
        tolerance = _finite_real(
            "simplex_tolerance",
            self.simplex_tolerance,
            nonnegative=True,
        )
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if tolerance <= 0:
            raise ValueError("simplex_tolerance must be positive")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "simplex_tolerance", tolerance)


@dataclass(frozen=True)
class PRILossSufficientStatistics:
    """Additive epoch statistics before batch-local normalization."""

    ce_numerator: torch.Tensor
    ce_denominator: torch.Tensor
    response_dice_numerator: torch.Tensor
    boundary_numerator: torch.Tensor
    temporal_numerator: torch.Tensor
    quality_denominator: torch.Tensor


@dataclass(frozen=True)
class PRILossReport:
    """Differentiable objective terms, or an explicit skipped-step result."""

    total: torch.Tensor | None
    region_ce: torch.Tensor | None
    response_dice: torch.Tensor | None
    boundary: torch.Tensor | None
    temporal_consistency: torch.Tensor | None
    skipped: bool
    valid_sample_count: int
    statistics: PRILossSufficientStatistics | None


def aggregate_pri_loss_reports(
    reports: list[PRILossReport] | tuple[PRILossReport, ...],
    config: PRILossConfig,
) -> PRILossReport:
    """Aggregate batch reports using their additive mathematical normalizers."""

    if not isinstance(reports, (list, tuple)):
        raise TypeError("reports must be a list or tuple")
    accumulator = PRILossEpochAccumulator(config)
    for report in reports:
        accumulator.update(report)
    return accumulator.compute()


def _report_from_statistics(
    statistics: PRILossSufficientStatistics | None,
    valid_sample_count: int,
    config: PRILossConfig,
) -> PRILossReport:
    if statistics is None or valid_sample_count == 0:
        return PRILossReport(
            total=None,
            region_ce=None,
            response_dice=None,
            boundary=None,
            temporal_consistency=None,
            skipped=True,
            valid_sample_count=0,
            statistics=None,
        )
    epsilon = config.epsilon
    region_ce = statistics.ce_numerator / (statistics.ce_denominator + epsilon)
    quality_denominator = statistics.quality_denominator + epsilon
    response_dice = statistics.response_dice_numerator / quality_denominator
    boundary = statistics.boundary_numerator / quality_denominator
    temporal_consistency = statistics.temporal_numerator / quality_denominator
    total = (
        config.lambda_ce * region_ce
        + config.lambda_dice * response_dice
        + config.lambda_boundary * boundary
        + config.lambda_temporal * temporal_consistency
    )
    return PRILossReport(
        total=total,
        region_ce=region_ce,
        response_dice=response_dice,
        boundary=boundary,
        temporal_consistency=temporal_consistency,
        skipped=False,
        valid_sample_count=valid_sample_count,
        statistics=statistics,
    )


class PRILossEpochAccumulator:
    """Incrementally detach and combine loss statistics without retaining graphs."""

    _NAMES = (
        "ce_numerator",
        "ce_denominator",
        "response_dice_numerator",
        "boundary_numerator",
        "temporal_numerator",
        "quality_denominator",
    )

    def __init__(self, config: PRILossConfig) -> None:
        if not isinstance(config, PRILossConfig):
            raise TypeError("config must be a PRILossConfig")
        self.config = config
        self._totals = {
            name: torch.zeros((), dtype=torch.float64) for name in self._NAMES
        }
        self.valid_sample_count = 0
        self.skipped_batch_count = 0

    def update(self, report: PRILossReport) -> None:
        if not isinstance(report, PRILossReport):
            raise TypeError("report must be a PRILossReport")
        if report.skipped:
            self.skipped_batch_count += 1
            return
        if report.statistics is None:
            raise ValueError("active loss reports must contain sufficient statistics")
        for name in self._NAMES:
            value = getattr(report.statistics, name)
            if value.numel() != 1 or not torch.isfinite(value.detach()).all().item():
                raise ValueError(f"loss statistic {name} must be one finite scalar")
            self._totals[name] += value.detach().to(device="cpu", dtype=torch.float64)
        self.valid_sample_count += report.valid_sample_count

    def compute(self) -> PRILossReport:
        statistics = None
        if self.valid_sample_count:
            statistics = PRILossSufficientStatistics(
                **{name: value.float() for name, value in self._totals.items()}
            )
        return _report_from_statistics(
            statistics,
            self.valid_sample_count,
            self.config,
        )


def _validate_label_inputs(
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[int, int, int]:
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if not isinstance(confidence, torch.Tensor):
        raise TypeError("confidence must be a torch.Tensor")
    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor")
    if target.ndim != 4 or target.shape[1] != 3:
        raise ValueError("target must have shape [B, 3, H, W]")
    batch, _, height, width = target.shape
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError("target dimensions must be positive")
    expected_map_shape = (batch, 1, height, width)
    if confidence.shape != expected_map_shape:
        raise ValueError("confidence must have shape [B, 1, H, W]")
    if valid.shape != expected_map_shape:
        raise ValueError("valid must have shape [B, 1, H, W]")
    if not torch.is_floating_point(target):
        raise TypeError("target must have a floating-point dtype")
    if not torch.is_floating_point(confidence):
        raise TypeError("confidence must have a floating-point dtype")
    if valid.dtype != torch.bool:
        raise TypeError("valid must have a boolean dtype")
    if target.device != confidence.device or target.device != valid.device:
        raise ValueError("target, confidence, and valid must share a device")
    return batch, height, width


def _validate_values_at_mask(
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    simplex_dim: int | None = None,
    tolerance: float = 1e-5,
) -> None:
    expanded_mask = mask.expand_as(values)
    selected = values[expanded_mask]
    if selected.numel() == 0:
        return
    if not torch.isfinite(selected).all().item():
        raise ValueError(f"{name} probabilities must be finite on valid pixels")
    if ((selected < 0) | (selected > 1)).any().item():
        raise ValueError(f"{name} probabilities must lie between 0 and 1")
    if simplex_dim is not None:
        sums = values.float().sum(dim=simplex_dim, keepdim=True)
        simplex_mask = mask.all(dim=simplex_dim, keepdim=True)
        selected_sums = sums[simplex_mask]
        ones = torch.ones_like(selected_sums)
        if not torch.allclose(selected_sums, ones, rtol=0.0, atol=tolerance):
            raise ValueError(f"{name} probabilities must sum to 1")


def _validated_confidence(
    confidence: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    selected = confidence[valid]
    if selected.numel() and not torch.isfinite(selected).all().item():
        raise ValueError("confidence must be finite on valid pixels")
    if selected.numel() and (selected < 0).any().item():
        raise ValueError("confidence must be non-negative on valid pixels")
    detached = confidence.detach().float()
    return torch.where(valid, detached, torch.zeros_like(detached))


def compute_pri_class_frequencies(
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    simplex_tolerance: float = 1e-4,
) -> torch.Tensor:
    """Compute frozen PRI frequencies on unaugmented, uncropped train labels.

    The caller is responsible for passing the complete eligible training-label
    set exactly once.  Invalid pixels are ignored before finite/simplex checks,
    so padding or registration-excluded storage values cannot leak into the
    statistics.
    """

    _validate_label_inputs(target, confidence, valid)
    tolerance = _finite_real(
        "simplex_tolerance",
        simplex_tolerance,
        nonnegative=True,
    )
    if tolerance <= 0:
        raise ValueError("simplex_tolerance must be positive")
    _validate_values_at_mask(
        "target",
        target,
        valid.expand(-1, 3, -1, -1),
        simplex_dim=1,
        tolerance=tolerance,
    )
    weight = _validated_confidence(confidence, valid)
    mass = weight.sum(dtype=torch.float32)
    if mass.item() <= 0:
        raise ValueError("class-frequency labels must have positive valid mass")

    safe_target = torch.where(
        valid.expand(-1, 3, -1, -1),
        target.detach().float(),
        torch.zeros_like(target, dtype=torch.float32),
    )
    numerator = (safe_target * weight).sum(dim=(0, 2, 3), dtype=torch.float32)
    return numerator / mass


def normalized_inverse_sqrt_class_weights(
    frequencies: torch.Tensor,
    *,
    frequency_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Convert fixed PRI frequencies to mean-one inverse-square-root weights."""

    if not isinstance(frequencies, torch.Tensor):
        raise TypeError("frequencies must be a torch.Tensor")
    if frequencies.shape != (3,):
        raise ValueError("frequencies must have shape [3]")
    if not torch.is_floating_point(frequencies):
        raise TypeError("frequencies must have a floating-point dtype")
    if not torch.isfinite(frequencies).all().item():
        raise ValueError("frequencies must be finite")
    if (frequencies < 0).any().item():
        raise ValueError("frequencies must be non-negative")
    if not torch.allclose(
        frequencies.float().sum(),
        torch.ones((), device=frequencies.device),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("frequencies must sum to 1")
    epsilon = _finite_real(
        "frequency_epsilon",
        frequency_epsilon,
        nonnegative=True,
    )
    if epsilon <= 0:
        raise ValueError("frequency_epsilon must be positive")

    inverse_sqrt = (frequencies.detach().float() + epsilon).rsqrt()
    return 3.0 * inverse_sqrt / inverse_sqrt.sum()


def compute_pri_class_weights(
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    frequency_epsilon: float = 1e-6,
    simplex_tolerance: float = 1e-4,
) -> torch.Tensor:
    """Compute one-time class weights from complete unaugmented train labels."""

    frequencies = compute_pri_class_frequencies(
        target,
        confidence,
        valid,
        simplex_tolerance=simplex_tolerance,
    )
    return normalized_inverse_sqrt_class_weights(
        frequencies,
        frequency_epsilon=frequency_epsilon,
    )


class PRILoss(nn.Module):
    """Fixed two-timepoint confidence-weighted PRI predictor objective."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        config: PRILossConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(class_weights, torch.Tensor):
            raise TypeError("class_weights must be a torch.Tensor")
        if class_weights.shape != (3,):
            raise ValueError("class_weights must have shape [3]")
        if not torch.is_floating_point(class_weights):
            raise TypeError("class_weights must have a floating-point dtype")
        if not torch.isfinite(class_weights).all().item():
            raise ValueError("class_weights must be finite")
        if (class_weights <= 0).any().item():
            raise ValueError("class_weights must be positive")
        if not torch.allclose(
            class_weights.float().sum(),
            torch.tensor(3.0, device=class_weights.device),
            rtol=0.0,
            atol=1e-4,
        ):
            raise ValueError("class_weights must sum to 3")
        self.config = PRILossConfig() if config is None else config
        if not isinstance(self.config, PRILossConfig):
            raise TypeError("config must be a PRILossConfig")
        self.register_buffer("class_weights", class_weights.detach().float().clone())

    @classmethod
    def from_training_labels(
        cls,
        target: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        *,
        config: PRILossConfig | None = None,
        frequency_epsilon: float = 1e-6,
    ) -> PRILoss:
        """Build the objective from one-time unaugmented training statistics."""

        objective_config = PRILossConfig() if config is None else config
        weights = compute_pri_class_weights(
            target,
            confidence,
            valid,
            frequency_epsilon=frequency_epsilon,
            simplex_tolerance=objective_config.simplex_tolerance,
        )
        return cls(weights, objective_config)

    def _validate_forward_inputs(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        q_pair: torch.Tensor,
        kappa_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = _validate_label_inputs(target, confidence, valid)
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("predictions must be a torch.Tensor")
        if predictions.shape != (batch, 2, 3, height, width):
            raise ValueError("predictions must have shape [B, 2, 3, H, W]")
        if not torch.is_floating_point(predictions):
            raise TypeError("predictions must have a floating-point dtype")
        if not isinstance(q_pair, torch.Tensor):
            raise TypeError("q_pair must be a torch.Tensor")
        if not isinstance(kappa_source, torch.Tensor):
            raise TypeError("kappa_source must be a torch.Tensor")
        if q_pair.shape != (batch,) or kappa_source.shape != (batch,):
            raise ValueError("q_pair and kappa_source must have shape [B]")
        if not torch.is_floating_point(q_pair) or not torch.is_floating_point(
            kappa_source
        ):
            raise TypeError("q_pair and kappa_source must be floating-point")
        common_device = target.device
        tensors = (predictions, q_pair, kappa_source, self.class_weights)
        if any(tensor.device != common_device for tensor in tensors):
            raise ValueError("all loss inputs and class_weights must share a device")
        for name, value in (
            ("q_pair", q_pair),
            ("kappa_source", kappa_source),
        ):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite")
            if ((value < 0) | (value > 1)).any().item():
                raise ValueError(f"{name} must lie between 0 and 1")
        if (kappa_source <= 0).any().item():
            raise ValueError("kappa_source must be greater than 0")
        return _validated_confidence(confidence, valid), valid

    @staticmethod
    def _skipped_report() -> PRILossReport:
        return PRILossReport(
            total=None,
            region_ce=None,
            response_dice=None,
            boundary=None,
            temporal_consistency=None,
            skipped=True,
            valid_sample_count=0,
            statistics=None,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        q_pair: torch.Tensor,
        kappa_source: torch.Tensor,
    ) -> PRILossReport:
        """Evaluate the registered objective on a same-seed timestep pair."""

        effective_confidence, valid = self._validate_forward_inputs(
            predictions,
            target,
            confidence,
            valid,
            q_pair,
            kappa_source,
        )
        sample_mass = effective_confidence.sum(dim=(1, 2, 3), dtype=torch.float32)
        valid_samples = sample_mass >= self.config.tau_mass
        valid_sample_count = int(valid_samples.sum().item())
        if valid_sample_count == 0:
            # This branch intentionally precedes probability checks and logs.
            return self._skipped_report()

        pixel_mask = valid & valid_samples[:, None, None, None]
        prediction_mask = pixel_mask[:, None].expand(-1, 2, 3, -1, -1)
        target_mask = pixel_mask.expand(-1, 3, -1, -1)
        _validate_values_at_mask(
            "predictions",
            predictions,
            prediction_mask,
            simplex_dim=2,
            tolerance=self.config.simplex_tolerance,
        )
        _validate_values_at_mask(
            "target",
            target,
            target_mask,
            simplex_dim=1,
            tolerance=self.config.simplex_tolerance,
        )

        probabilities = torch.where(
            prediction_mask,
            predictions.float(),
            torch.zeros_like(predictions, dtype=torch.float32),
        )
        labels = torch.where(
            target_mask,
            target.detach().float(),
            torch.zeros_like(target, dtype=torch.float32),
        )
        weight = torch.where(
            pixel_mask,
            effective_confidence,
            torch.zeros_like(effective_confidence),
        )[:, 0]
        epsilon = self.config.epsilon

        sample_quality = (
            q_pair.detach().float() * kappa_source.detach().float()
        )
        sample_quality = torch.where(
            valid_samples,
            sample_quality,
            torch.zeros_like(sample_quality),
        )
        quality_denominator = sample_quality.sum()
        outer_quality = sample_quality / (quality_denominator + epsilon)

        class_weights = self.class_weights.float()[None, :, None, None]
        ce_denominator = weight.sum() + epsilon
        ce_terms: list[torch.Tensor] = []
        dice_terms: list[torch.Tensor] = []
        boundary_terms: list[torch.Tensor] = []
        for timepoint in range(2):
            probability = probabilities[:, timepoint]
            ce_numerator = -(
                weight[:, None]
                * class_weights
                * labels
                * torch.log(probability + epsilon)
            ).sum()
            ce_terms.append(ce_numerator / ce_denominator)
            dice_terms.append(
                self._response_dice(probability, labels, weight, outer_quality, epsilon)
            )
            boundary_terms.append(
                self._boundary_loss(
                    probability,
                    labels,
                    weight,
                    pixel_mask[:, 0],
                    outer_quality,
                    epsilon,
                )
            )

        region_ce = 0.5 * (ce_terms[0] + ce_terms[1])
        response_dice = 0.5 * (dice_terms[0] + dice_terms[1])
        boundary = 0.5 * (boundary_terms[0] + boundary_terms[1])
        temporal_consistency = self._temporal_js(
            probabilities[:, 0],
            probabilities[:, 1],
            weight,
            outer_quality,
            epsilon,
        )
        total = (
            self.config.lambda_ce * region_ce
            + self.config.lambda_dice * response_dice
            + self.config.lambda_boundary * boundary
            + self.config.lambda_temporal * temporal_consistency
        )
        statistics = PRILossSufficientStatistics(
            ce_numerator=0.5 * (ce_terms[0] + ce_terms[1]) * ce_denominator,
            ce_denominator=weight.sum(),
            response_dice_numerator=response_dice * (quality_denominator + epsilon),
            boundary_numerator=boundary * (quality_denominator + epsilon),
            temporal_numerator=temporal_consistency * (quality_denominator + epsilon),
            quality_denominator=quality_denominator,
        )
        return PRILossReport(
            total=total,
            region_ce=region_ce,
            response_dice=response_dice,
            boundary=boundary,
            temporal_consistency=temporal_consistency,
            skipped=False,
            valid_sample_count=valid_sample_count,
            statistics=statistics,
        )

    @staticmethod
    def _response_dice(
        probability: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor,
        outer_quality: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        response_prediction = probability[:, 1]
        response_target = labels[:, 1]
        intersection = (weight * response_target * response_prediction).sum(
            dim=(1, 2)
        )
        target_mass = (weight * response_target).sum(dim=(1, 2))
        prediction_mass = (weight * response_prediction).sum(dim=(1, 2))
        per_sample = 1.0 - (2.0 * intersection + epsilon) / (
            target_mass + prediction_mass + epsilon
        )
        return (outer_quality * per_sample).sum()

    @staticmethod
    def _boundary_loss(
        probability: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor,
        valid: torch.Tensor,
        outer_quality: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        numerator = torch.zeros(
            probability.shape[0],
            device=probability.device,
            dtype=torch.float32,
        )
        denominator = torch.zeros_like(numerator)

        if probability.shape[-1] > 1:
            horizontal_valid = valid[:, :, :-1] & valid[:, :, 1:]
            horizontal_weight = torch.minimum(
                weight[:, :, :-1],
                weight[:, :, 1:],
            )
            horizontal_weight = torch.where(
                horizontal_valid,
                horizontal_weight,
                torch.zeros_like(horizontal_weight),
            )
            prediction_difference = (
                probability[:, :2, :, :-1] - probability[:, :2, :, 1:]
            )
            target_difference = labels[:, :2, :, :-1] - labels[:, :2, :, 1:]
            error = (prediction_difference - target_difference).abs().sum(dim=1)
            numerator = numerator + (horizontal_weight * error).sum(dim=(1, 2))
            denominator = denominator + horizontal_weight.sum(dim=(1, 2))

        if probability.shape[-2] > 1:
            vertical_valid = valid[:, :-1, :] & valid[:, 1:, :]
            vertical_weight = torch.minimum(
                weight[:, :-1, :],
                weight[:, 1:, :],
            )
            vertical_weight = torch.where(
                vertical_valid,
                vertical_weight,
                torch.zeros_like(vertical_weight),
            )
            prediction_difference = (
                probability[:, :2, :-1, :] - probability[:, :2, 1:, :]
            )
            target_difference = labels[:, :2, :-1, :] - labels[:, :2, 1:, :]
            error = (prediction_difference - target_difference).abs().sum(dim=1)
            numerator = numerator + (vertical_weight * error).sum(dim=(1, 2))
            denominator = denominator + vertical_weight.sum(dim=(1, 2))

        per_sample = numerator / (denominator + epsilon)
        return (outer_quality * per_sample).sum()

    @staticmethod
    def _temporal_js(
        first: torch.Tensor,
        second: torch.Tensor,
        weight: torch.Tensor,
        outer_quality: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        midpoint = 0.5 * (first + second)
        divergence = 0.5 * (
            first * torch.log((first + epsilon) / (midpoint + epsilon))
            + second * torch.log((second + epsilon) / (midpoint + epsilon))
        )
        divergence = divergence.sum(dim=1)
        per_sample = (weight * divergence).sum(dim=(1, 2)) / (
            weight.sum(dim=(1, 2)) + epsilon
        )
        return (outer_quality * per_sample).sum()


__all__ = [
    "PRILoss",
    "PRILossConfig",
    "PRILossEpochAccumulator",
    "PRILossReport",
    "PRILossSufficientStatistics",
    "aggregate_pri_loss_reports",
    "compute_pri_class_frequencies",
    "compute_pri_class_weights",
    "normalized_inverse_sqrt_class_weights",
]
