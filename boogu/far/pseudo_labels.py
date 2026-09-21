"""Deterministic soft pseudo-labels for Primary--Response--Invariant regions.

This module is an offline data-preparation component.  In particular, the
deterministic VAE posterior modes constructed here must never replace the
stochastic source reference latent used by the Boogu DiT inference path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, isclose, isfinite, log
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from .regions import PRIRegionProbabilities

_CUE_COUNT = 3
_OFFICIAL_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_OFFICIAL_IMAGENET_STD = (0.229, 0.224, 0.225)


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _real_triplet(
    name: str,
    values: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float, float, float]:
    if not isinstance(values, tuple) or len(values) != _CUE_COUNT:
        raise TypeError(f"{name} must be a three-element tuple")
    normalized = tuple(_finite_real(name, value) for value in values)
    if positive and any(value <= 0 for value in normalized):
        raise ValueError(f"{name} values must be positive")
    if nonnegative and any(value < 0 for value in normalized):
        raise ValueError(f"{name} values must be non-negative")
    return normalized  # type: ignore[return-value]


def _unit_interval(name: str, value: Any, *, exclude_zero: bool = False) -> float:
    normalized = _finite_real(name, value)
    lower_bound_satisfied = normalized > 0 if exclude_zero else normalized >= 0
    if not lower_bound_satisfied or normalized > 1:
        qualifier = "greater than 0 and" if exclude_zero else "between 0 and"
        raise ValueError(f"{name} must be {qualifier} 1")
    return normalized


def _validate_map(
    name: str,
    value: Any,
    *,
    channels: int = 1,
    require_single_sample: bool = True,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != channels:
        raise ValueError(f"{name} must have shape [B, {channels}, H, W]")
    if require_single_sample and value.shape[0] != 1:
        raise ValueError(f"{name} must contain exactly one single sample")
    if value.shape[0] <= 0 or value.shape[-2] <= 0 or value.shape[-1] <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    if not torch.is_floating_point(value) and value.dtype != torch.bool:
        raise TypeError(f"{name} must have a floating-point or boolean dtype")
    if torch.is_floating_point(value) and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    return value


def _validate_probability_map(name: str, value: Any) -> torch.Tensor:
    value = _validate_map(name, value)
    if ((value < 0) | (value > 1)).any().item():
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _validate_rgb_image(name: str, value: Any) -> torch.Tensor:
    value = _validate_map(name, value, channels=3)
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype")
    if ((value < 0) | (value > 1)).any().item():
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class AbsoluteChangeCalibration:
    """Auditable absolute gates estimated from held-out training no-edit pairs."""

    thresholds: tuple[float, float, float]
    temperatures: tuple[float, float, float]
    source_split: str
    source_kind: str
    pair_ids: tuple[str, ...]
    dataset_fingerprint: str
    estimator_id: str
    estimator_version: str

    def __post_init__(self) -> None:
        thresholds = _real_triplet("thresholds", self.thresholds, nonnegative=True)
        temperatures = _real_triplet(
            "temperatures",
            self.temperatures,
            positive=True,
        )
        if self.source_split != "train":
            raise ValueError("calibration must come from the training split")
        if self.source_kind != "no_edit":
            raise ValueError("calibration source_kind must identify no-edit pairs")
        if not isinstance(self.pair_ids, tuple) or not self.pair_ids:
            raise ValueError("calibration pair_ids must be a non-empty tuple")
        if any(not isinstance(pair_id, str) or not pair_id.strip() for pair_id in self.pair_ids):
            raise ValueError("calibration pair_ids must contain non-empty strings")
        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError("calibration pair_ids must be unique")
        if (
            not isinstance(self.dataset_fingerprint, str)
            or len(self.dataset_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.dataset_fingerprint.lower()
            )
        ):
            raise ValueError("dataset_fingerprint must be a SHA-256 hex digest")
        for name in ("estimator_id", "estimator_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "dataset_fingerprint", self.dataset_fingerprint.lower())

    def validate_non_overlap(self, sample_id: str) -> None:
        """Reject leakage between the calibration subset and edited examples."""

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if sample_id in self.pair_ids:
            raise ValueError("edited sample must not overlap calibration pair_ids")

    @property
    def fingerprint(self) -> str:
        """Return a stable digest for storing with every generated label."""

        payload = {
            "dataset_fingerprint": self.dataset_fingerprint,
            "estimator_id": self.estimator_id,
            "estimator_version": self.estimator_version,
            "pair_ids": sorted(self.pair_ids),
            "source_kind": self.source_kind,
            "source_split": self.source_split,
            "temperatures": self.temperatures,
            "thresholds": self.thresholds,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PRIPseudoLabelConfig:
    """Pre-registered hyperparameters for deterministic pseudo-label creation."""

    registration_threshold: float = 0.5
    min_valid_coverage: float = 0.5
    min_valid_pixels: int = 16
    robust_epsilon: float = 1e-6
    flat_evidence_delta: float = 1e-4
    cue_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    gaussian_sigma: float = 1.5
    gaussian_truncate: float = 3.0
    agreement_sigma: float = 0.25
    response_beta: float = 0.5
    response_supervision_weight: float = 0.0
    response_threshold: float = 0.25
    response_temperature: float = 0.1
    confidence_floor: float = 0.1
    min_total_valid_mass: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "registration_threshold",
            _unit_interval("registration_threshold", self.registration_threshold),
        )
        object.__setattr__(
            self,
            "min_valid_coverage",
            _unit_interval("min_valid_coverage", self.min_valid_coverage),
        )
        if isinstance(self.min_valid_pixels, bool) or not isinstance(
            self.min_valid_pixels,
            Integral,
        ):
            raise TypeError("min_valid_pixels must be an integer")
        if self.min_valid_pixels <= 0:
            raise ValueError("min_valid_pixels must be positive")
        object.__setattr__(self, "min_valid_pixels", int(self.min_valid_pixels))

        for name in (
            "robust_epsilon",
            "flat_evidence_delta",
            "gaussian_sigma",
            "gaussian_truncate",
            "agreement_sigma",
            "response_temperature",
        ):
            value = _finite_real(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

        weights = _real_triplet("cue_weights", self.cue_weights, nonnegative=True)
        if not isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("cue_weights must sum to 1")
        object.__setattr__(self, "cue_weights", weights)

        beta = _finite_real("response_beta", self.response_beta)
        if not 0 < beta <= 1:
            raise ValueError("response_beta must be greater than 0 and at most 1")
        object.__setattr__(self, "response_beta", beta)
        object.__setattr__(
            self,
            "response_supervision_weight",
            _unit_interval(
                "response_supervision_weight",
                self.response_supervision_weight,
            ),
        )
        object.__setattr__(
            self,
            "response_threshold",
            _finite_real("response_threshold", self.response_threshold),
        )
        object.__setattr__(
            self,
            "confidence_floor",
            _unit_interval("confidence_floor", self.confidence_floor, exclude_zero=True),
        )
        mass = _finite_real("min_total_valid_mass", self.min_total_valid_mass)
        if mass < 0:
            raise ValueError("min_total_valid_mass must be non-negative")
        object.__setattr__(self, "min_total_valid_mass", mass)


@dataclass(frozen=True)
class ChangeEvidence:
    """Pixel, deterministic-latent, and fixed-perceptual evidence maps."""

    pixel: torch.Tensor = field(repr=False)
    latent: torch.Tensor = field(repr=False)
    perceptual: torch.Tensor = field(repr=False)

    def __post_init__(self) -> None:
        maps = self.as_tuple()
        for name, value in zip(("pixel", "latent", "perceptual"), maps, strict=True):
            _validate_map(name, value)
            if (value < 0).any().item():
                raise ValueError(f"{name} evidence must be non-negative")
        if any(value.shape != maps[0].shape for value in maps[1:]):
            raise ValueError("all evidence maps must have identical shapes")
        if any(value.device != maps[0].device for value in maps[1:]):
            raise ValueError("all evidence maps must be on the same device")
        for name, value in zip(("pixel", "latent", "perceptual"), maps, strict=True):
            object.__setattr__(self, name, value.detach().clone())

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pixel, self.latent, self.perceptual


@dataclass(frozen=True)
class ResponseSupervision:
    """Optional Response target with a training-only, auditable provenance record."""

    values: torch.Tensor = field(repr=False)
    source_kind: str
    source_split: str
    provenance: str
    teacher_version: str | None = None
    prompt_template: str | None = None
    threshold_spec: str | None = None
    spatial_mapping_spec: str | None = None

    def __post_init__(self) -> None:
        values = _validate_probability_map("Response supervision", self.values)
        allowed_sources = {"human", "structured_metadata", "fixed_teacher"}
        if self.source_kind not in allowed_sources:
            raise ValueError(
                "source_kind must be human, structured_metadata, or fixed_teacher"
            )
        if self.source_split != "train":
            raise ValueError("Response supervision must come from the training split")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("provenance must be a non-empty string")
        if self.source_kind == "fixed_teacher":
            for name in (
                "teacher_version",
                "prompt_template",
                "threshold_spec",
                "spatial_mapping_spec",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{name} is required for fixed_teacher supervision"
                    )
        object.__setattr__(self, "values", values.detach().clone())


@dataclass(frozen=True)
class SpatialDecomposabilityAssessment:
    """Traceable output of a pre-registered spatial-supervision criterion."""

    decomposable: bool
    criterion_name: str
    criterion_version: str
    rationale: str
    threshold: float
    parameters: dict[str, Any]
    score: float
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.decomposable, bool):
            raise TypeError("decomposable must be a bool")
        for name in ("criterion_name", "criterion_version", "rationale"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "threshold", _finite_real("threshold", self.threshold))
        object.__setattr__(self, "score", _finite_real("score", self.score))
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a JSON-serializable dictionary")
        if any(not isinstance(key, str) or not key for key in self.parameters):
            raise ValueError("parameters must have non-empty string keys")
        try:
            parameters = json.loads(
                json.dumps(self.parameters, sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("parameters must be JSON-serializable and finite") from error
        if not isinstance(self.metrics, dict) or not self.metrics:
            raise ValueError("metrics must be a non-empty dictionary")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metrics must have non-empty string keys")
            metrics[name] = _finite_real(f"metrics[{name}]", value)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    @property
    def criterion_spec(self) -> dict[str, Any]:
        """Serializable pre-registered criterion specification."""

        return {
            "name": self.criterion_name,
            "version": self.criterion_version,
            "threshold": self.threshold,
            "parameters": json.loads(json.dumps(dict(self.parameters))),
        }


class SpatialCriterion(Protocol):
    """Injected registered criterion evaluated on fused spatial evidence."""

    def __call__(
        self,
        fused_change: torch.Tensor,
        primary_seed: torch.Tensor,
        registration_reliability: torch.Tensor,
        valid_positions: torch.Tensor,
    ) -> SpatialDecomposabilityAssessment: ...


class PseudoLabelRejectionReason(str, Enum):
    """Registered sample-level pseudo-label rejection reasons in priority order."""

    UNSUPPORTED_GEOMETRY = "unsupported_geometry"
    REGISTRATION_FAILED = "registration_failed"
    LOW_VALID_COVERAGE = "low_valid_coverage"
    INSUFFICIENT_VALID_PIXELS = "insufficient_valid_pixels"
    FLAT_EVIDENCE = "flat_evidence"
    NON_SPATIALLY_DECOMPOSABLE = "non_spatially_decomposable"
    LOW_TOTAL_MASS = "low_total_mass"


@dataclass(frozen=True)
class PRIPseudoLabelResult:
    """Accepted labels or one explicit rejection, never both and never neither."""

    rejection_reason: PseudoLabelRejectionReason | None
    calibration_fingerprint: str
    regions: PRIRegionProbabilities | None = field(default=None, repr=False)
    confidence: torch.Tensor | None = field(default=None, repr=False)
    registration_reliability: torch.Tensor | None = field(default=None, repr=False)
    valid_positions: torch.Tensor | None = field(default=None, repr=False)
    spatial_assessment: SpatialDecomposabilityAssessment | None = None
    evidence: ChangeEvidence | None = field(default=None, repr=False)
    normalized_evidence: ChangeEvidence | None = field(default=None, repr=False)
    evidence_spreads: torch.Tensor | None = field(default=None, repr=False)
    primary_seed: torch.Tensor | None = field(default=None, repr=False)
    response_conditional: torch.Tensor | None = field(default=None, repr=False)
    fused_change: torch.Tensor | None = field(default=None, repr=False)
    response_supervision: ResponseSupervision | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        accepted = self.rejection_reason is None and self.regions is not None
        rejected = self.rejection_reason is not None and self.regions is None
        if accepted == rejected:
            raise ValueError("result must be exactly one of accepted or rejected")
        if (
            not isinstance(self.calibration_fingerprint, str)
            or len(self.calibration_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.calibration_fingerprint)
        ):
            raise ValueError("calibration_fingerprint must be a SHA-256 hex digest")

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None

    @property
    def rejected(self) -> bool:
        return self.rejection_reason is not None


class VGG16PerceptualFeatures(nn.Module):
    """Frozen VGG-16 activations at registered torchvision feature indices."""

    feature_indices = (3, 8, 15)

    def __init__(self, features: nn.Sequential) -> None:
        super().__init__()
        if len(features) <= self.feature_indices[-1]:
            raise ValueError("VGG features must contain layer index 15")
        self.features = features
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "mean",
            torch.tensor(_OFFICIAL_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(_OFFICIAL_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        super().train(False)

    def train(self, mode: bool = True) -> VGG16PerceptualFeatures:
        """Keep this evidence extractor in evaluation mode unconditionally."""

        del mode
        super().train(False)
        return self

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _validate_rgb_image("image", image)
        hidden = (image.float() - self.mean) / self.std
        outputs = []
        with torch.no_grad():
            for index, layer in enumerate(self.features):
                hidden = layer(hidden)
                if index in self.feature_indices:
                    outputs.append(hidden.detach())
                if index == self.feature_indices[-1]:
                    break
        return tuple(outputs)


def build_vgg16_perceptual_extractor(
    *,
    factory: Callable[..., nn.Module] | None = None,
) -> VGG16PerceptualFeatures:
    """Build the fixed VGG-16 evidence network without resize or center crop."""

    from torchvision import models

    selected_factory = models.vgg16 if factory is None else factory
    backbone = selected_factory(weights=models.VGG16_Weights.IMAGENET1K_V1)
    features = getattr(backbone, "features", None)
    if not isinstance(features, nn.Sequential):
        raise TypeError("VGG factory must return an object with Sequential features")
    return VGG16PerceptualFeatures(features)


def _latent_affine_value(name: str, value: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(value, torch.Tensor):
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} tensor must have a floating-point dtype")
        if value.numel() != 1:
            raise ValueError(f"{name} tensor must be scalar")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must be finite")
        return value.reshape(())
    return _finite_real(name, value)


def deterministic_vae_posterior_mode(
    image: torch.Tensor,
    *,
    preprocess: Callable[[torch.Tensor], torch.Tensor],
    encode: Callable[[torch.Tensor], Any],
    shift_factor: float | torch.Tensor | None,
    scaling_factor: float | torch.Tensor | None,
) -> torch.Tensor:
    """Encode with posterior ``mode`` and apply ``(mode - shift) * scale``."""

    _validate_rgb_image("image", image)
    if not callable(preprocess) or not callable(encode):
        raise TypeError("preprocess and encode must be callable")
    shift = (
        None
        if shift_factor is None
        else _latent_affine_value("shift_factor", shift_factor)
    )
    scale = (
        None
        if scaling_factor is None
        else _latent_affine_value("scaling_factor", scaling_factor)
    )
    if isinstance(scale, torch.Tensor):
        if (scale <= 0).any().item():
            raise ValueError("scaling_factor must be positive")
    elif scale is not None and scale <= 0:
        raise ValueError("scaling_factor must be positive")

    with torch.no_grad():
        prepared = preprocess(image)
        if not isinstance(prepared, torch.Tensor):
            raise TypeError("VAE preprocess must return a torch.Tensor")
        encoded = encode(prepared)
        distribution = getattr(encoded, "latent_dist", encoded)
        mode = getattr(distribution, "mode", None)
        if not callable(mode):
            raise TypeError("VAE encoder output must expose a callable posterior mode")
        latent = mode()
        if not isinstance(latent, torch.Tensor):
            raise TypeError("VAE posterior mode must return a torch.Tensor")
        _validate_map(
            "VAE posterior mode",
            latent,
            channels=latent.shape[1] if latent.ndim == 4 else 1,
        )
        if isinstance(shift, torch.Tensor):
            shift = shift.to(device=latent.device, dtype=latent.dtype)
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=latent.device, dtype=latent.dtype)
        transformed = latent
        if shift is not None:
            transformed = transformed - shift
        if scale is not None:
            transformed = transformed * scale
    if not torch.isfinite(transformed).all().item():
        raise ValueError("transformed VAE posterior mode must be finite")
    return transformed.detach()


def compute_change_evidence(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    perceptual_extractor: Callable[[torch.Tensor], Sequence[torch.Tensor]],
) -> ChangeEvidence:
    """Compute pixel, latent, and equally weighted three-layer perceptual cues."""

    _validate_rgb_image("source_image", source_image)
    _validate_rgb_image("target_image", target_image)
    if source_image.shape != target_image.shape:
        raise ValueError("source_image and target_image must share the same canvas")
    _validate_map(
        "source_latent",
        source_latent,
        channels=source_latent.shape[1] if source_latent.ndim == 4 else 1,
    )
    _validate_map(
        "target_latent",
        target_latent,
        channels=target_latent.shape[1] if target_latent.ndim == 4 else 1,
    )
    if source_latent.shape != target_latent.shape:
        raise ValueError("deterministic source and target latents must have equal shapes")
    if not callable(perceptual_extractor):
        raise TypeError("perceptual_extractor must be callable")

    latent_size = source_latent.shape[-2:]
    with torch.no_grad():
        source = source_image.detach().float()
        target = target_image.detach().float()
        pixel_difference = (target - source).abs().mean(dim=1, keepdim=True)
        pixel = F.interpolate(pixel_difference, size=latent_size, mode="area")
        latent = (target_latent.detach().float() - source_latent.detach().float()).abs()
        latent = latent.mean(dim=1, keepdim=True)

        source_features = tuple(perceptual_extractor(source))
        target_features = tuple(perceptual_extractor(target))
        if len(source_features) != _CUE_COUNT or len(target_features) != _CUE_COUNT:
            raise ValueError("perceptual_extractor must return exactly three feature maps")
        perceptual = torch.zeros_like(pixel)
        for source_feature, target_feature in zip(
            source_features,
            target_features,
            strict=True,
        ):
            if source_feature.shape != target_feature.shape or source_feature.ndim != 4:
                raise ValueError("paired perceptual feature maps must have equal 4-D shapes")
            difference = target_feature.detach().float() - source_feature.detach().float()
            resized = F.interpolate(
                difference,
                size=latent_size,
                mode="bilinear",
                align_corners=False,
            )
            perceptual = perceptual + resized.abs().mean(dim=1, keepdim=True) / 3
    return ChangeEvidence(pixel=pixel, latent=latent, perceptual=perceptual)


def robust_normalize_cues(
    evidence: ChangeEvidence,
    valid_positions: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[ChangeEvidence, torch.Tensor]:
    """Normalize each cue by valid-domain Q50/Q95 statistics."""

    if not isinstance(evidence, ChangeEvidence):
        raise TypeError("evidence must be a ChangeEvidence instance")
    _validate_map("valid_positions", valid_positions)
    if valid_positions.dtype != torch.bool:
        raise TypeError("valid_positions must have boolean dtype")
    if valid_positions.shape != evidence.pixel.shape:
        raise ValueError("valid_positions shape must match evidence maps")
    if not valid_positions.any().item():
        raise ValueError("valid_positions must contain at least one valid pixel")
    epsilon = _finite_real("epsilon", epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    normalized_maps = []
    spreads = []
    for cue in evidence.as_tuple():
        valid_values = cue[valid_positions].float()
        q50 = torch.quantile(valid_values, 0.50)
        q95 = torch.quantile(valid_values, 0.95)
        spread = q95 - q50
        normalized_maps.append(((cue.float() - q50) / (spread + epsilon)).clamp(0, 1))
        spreads.append(spread)
    normalized = ChangeEvidence(*normalized_maps)
    return normalized, torch.stack(spreads)


def _reflect_indices(length: int, padding: int, device: torch.device) -> torch.Tensor:
    if length < 2:
        raise ValueError("reflection padding requires spatial dimensions of at least 2")
    positions = torch.arange(-padding, length + padding, device=device)
    period = 2 * (length - 1)
    wrapped = positions.remainder(period)
    return torch.where(wrapped <= length - 1, wrapped, period - wrapped).long()


def gaussian_primary_seed(
    primary_mask: torch.Tensor,
    latent_size: tuple[int, int],
    *,
    sigma: float,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Reflect-blur a binary Primary mask, then antialias it by area downsampling."""

    _validate_map("primary_mask", primary_mask)
    if not (((primary_mask == 0) | (primary_mask == 1)).all().item()):
        raise ValueError("primary_mask must be binary")
    height, width = _normalize_spatial_size(latent_size)
    sigma = _finite_real("sigma", sigma)
    truncate = _finite_real("truncate", truncate)
    if sigma <= 0 or truncate <= 0:
        raise ValueError("sigma and truncate must be positive")

    radius = max(1, ceil(sigma * truncate))
    coordinate = torch.arange(
        -radius,
        radius + 1,
        device=primary_mask.device,
        dtype=torch.float32,
    )
    kernel_1d = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_width = kernel_1d.numel()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(
        1,
        1,
        kernel_width,
        kernel_width,
    )
    mask = primary_mask.detach().float()
    row_indices = _reflect_indices(mask.shape[-2], radius, mask.device)
    column_indices = _reflect_indices(mask.shape[-1], radius, mask.device)
    padded = mask.index_select(-2, row_indices).index_select(-1, column_indices)
    blurred = F.conv2d(padded, kernel_2d)
    return F.interpolate(blurred, size=(height, width), mode="area").clamp(0, 1)


def cue_agreement(evidence: ChangeEvidence, *, sigma: float) -> torch.Tensor:
    """Return exp(-population-variance/sigma^2) across the three cues."""

    if not isinstance(evidence, ChangeEvidence):
        raise TypeError("evidence must be a ChangeEvidence instance")
    sigma = _finite_real("sigma", sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    stacked = torch.stack(evidence.as_tuple(), dim=0).float()
    population_variance = stacked.var(dim=0, unbiased=False)
    return torch.exp(-population_variance / sigma**2)


def response_conditional_probability(
    fused_change: torch.Tensor,
    agreement: torch.Tensor,
    config: PRIPseudoLabelConfig,
    *,
    response_supervision: ResponseSupervision | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute r=P(Response | not Primary) from response amplitude evidence."""

    _validate_probability_map("fused_change", fused_change)
    _validate_probability_map("agreement", agreement)
    if fused_change.shape != agreement.shape:
        raise ValueError("fused_change and agreement must have equal shapes")
    if not isinstance(config, PRIPseudoLabelConfig):
        raise TypeError("config must be a PRIPseudoLabelConfig")
    eta = config.response_supervision_weight
    if eta > 0:
        if not isinstance(response_supervision, ResponseSupervision):
            raise ValueError(
                "an auditable ResponseSupervision is required when its weight is positive"
            )
        if response_supervision.values.shape != agreement.shape:
            raise ValueError("response_supervision shape must match agreement")
        supervised = response_supervision.values.float()
    else:
        supervised = torch.zeros_like(agreement)
    amplitude = fused_change.float().pow(config.response_beta) * (
        eta * supervised + (1 - eta) * agreement.float()
    )
    return torch.sigmoid(
        (amplitude - config.response_threshold) / config.response_temperature
    )


def stick_breaking_labels(
    primary_probability: torch.Tensor,
    response_given_not_primary: torch.Tensor,
) -> PRIRegionProbabilities:
    """Construct exact P/R/I simplex labels without a second normalization."""

    _validate_probability_map("primary_probability", primary_probability)
    _validate_probability_map(
        "response_given_not_primary",
        response_given_not_primary,
    )
    if primary_probability.shape != response_given_not_primary.shape:
        raise ValueError("Primary and conditional Response shapes must match")
    remainder = 1 - primary_probability
    values = torch.cat(
        (
            primary_probability,
            remainder * response_given_not_primary,
            remainder * (1 - response_given_not_primary),
        ),
        dim=1,
    )
    return PRIRegionProbabilities.from_probabilities(values)


def label_confidence(
    registration_reliability: torch.Tensor,
    regions: PRIRegionProbabilities,
    agreement: torch.Tensor,
    *,
    q_pair: float,
    source_confidence: float,
    confidence_floor: float,
) -> torch.Tensor:
    """Compute entropy-aware soft confidence without clearing invalid positions."""

    _validate_probability_map("registration_reliability", registration_reliability)
    _validate_probability_map("agreement", agreement)
    if not isinstance(regions, PRIRegionProbabilities):
        raise TypeError("regions must be PRIRegionProbabilities")
    if registration_reliability.shape != agreement.shape:
        raise ValueError("registration_reliability and agreement shapes must match")
    if regions.values.shape[0] != 1 or regions.values.shape[-2:] != agreement.shape[-2:]:
        raise ValueError("region and confidence map shapes must match")
    q_pair = _unit_interval("q_pair", q_pair)
    source_confidence = _unit_interval(
        "source_confidence",
        source_confidence,
        exclude_zero=True,
    )
    confidence_floor = _unit_interval(
        "confidence_floor",
        confidence_floor,
        exclude_zero=True,
    )

    probabilities = regions.values.float()
    entropy_terms = torch.where(
        probabilities > 0,
        probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.zeros_like(probabilities),
    )
    entropy = -entropy_terms.sum(dim=1, keepdim=True)
    entropy_factor = confidence_floor + (1 - confidence_floor) * (
        1 - entropy / log(3)
    )
    return (
        registration_reliability.float()
        * q_pair
        * agreement.float()
        * source_confidence
        * entropy_factor
    )


def _normalize_spatial_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("latent_size must be a two-element tuple")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise TypeError("latent_size values must be integers")
    height, width = (int(item) for item in value)
    if height <= 0 or width <= 0:
        raise ValueError("latent_size values must be positive")
    return height, width


def _rejected(
    reason: PseudoLabelRejectionReason,
    *,
    registration: torch.Tensor,
    valid: torch.Tensor,
    calibration_fingerprint: str,
    assessment: SpatialDecomposabilityAssessment | None = None,
    evidence: ChangeEvidence | None = None,
    normalized: ChangeEvidence | None = None,
    spreads: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    primary: torch.Tensor | None = None,
    response: torch.Tensor | None = None,
    fused_change: torch.Tensor | None = None,
    response_supervision: ResponseSupervision | None = None,
) -> PRIPseudoLabelResult:
    return PRIPseudoLabelResult(
        rejection_reason=reason,
        calibration_fingerprint=calibration_fingerprint,
        registration_reliability=registration,
        valid_positions=valid,
        spatial_assessment=assessment,
        evidence=evidence,
        normalized_evidence=normalized,
        evidence_spreads=spreads,
        confidence=confidence,
        primary_seed=primary,
        response_conditional=response,
        fused_change=fused_change,
        response_supervision=response_supervision,
    )


def generate_pri_pseudo_labels(
    *,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    primary_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_id: str,
    geometry_supported: bool,
    registration_succeeded: bool,
    latent_size: tuple[int, int],
    q_pair: float,
    source_confidence: float,
    calibration: AbsoluteChangeCalibration,
    config: PRIPseudoLabelConfig,
    spatial_criterion: SpatialCriterion,
    vae_preprocess: Callable[[torch.Tensor], torch.Tensor],
    vae_encode: Callable[[torch.Tensor], Any],
    vae_shift_factor: float | torch.Tensor | None,
    vae_scaling_factor: float | torch.Tensor | None,
    perceptual_extractor: Callable[[torch.Tensor], Sequence[torch.Tensor]],
    response_supervision: ResponseSupervision | torch.Tensor | None = None,
) -> PRIPseudoLabelResult:
    """Generate one offline label using the registered rejection order.

    The order is unsupported geometry, registration failure, low coverage,
    insufficient valid pixels, flat evidence, non-spatial evidence, and finally
    low total valid mass.  Geometry and registration are explicit pair metadata;
    the spatial criterion runs only after all three evidence cues are fused.
    """

    _validate_rgb_image("source_image", source_image)
    _validate_rgb_image("target_image", target_image)
    if source_image.shape != target_image.shape:
        raise ValueError("source and target must be a same-canvas single sample")
    _validate_map("primary_mask", primary_mask)
    _validate_map("valid_mask", valid_mask)
    if primary_mask.shape[-2:] != source_image.shape[-2:] or valid_mask.shape[-2:] != source_image.shape[-2:]:
        raise ValueError("images, primary_mask, and valid_mask must share a canvas")
    if not (((primary_mask == 0) | (primary_mask == 1)).all().item()):
        raise ValueError("primary_mask must be binary")
    _validate_probability_map("valid_mask", valid_mask)
    if not isinstance(calibration, AbsoluteChangeCalibration):
        raise TypeError("calibration must be an explicit AbsoluteChangeCalibration")
    if not isinstance(config, PRIPseudoLabelConfig):
        raise TypeError("config must be a PRIPseudoLabelConfig")
    if not isinstance(geometry_supported, bool):
        raise TypeError("geometry_supported must be a bool")
    if not isinstance(registration_succeeded, bool):
        raise TypeError("registration_succeeded must be a bool")
    calibration.validate_non_overlap(sample_id)
    q_pair = _unit_interval("q_pair", q_pair)
    source_confidence = _unit_interval(
        "source_confidence",
        source_confidence,
        exclude_zero=True,
    )
    height, width = _normalize_spatial_size(latent_size)

    registration = F.interpolate(
        valid_mask.detach().float(),
        size=(height, width),
        mode="area",
    ).clamp(0, 1)
    valid = registration >= config.registration_threshold
    if not geometry_supported:
        return _rejected(
            PseudoLabelRejectionReason.UNSUPPORTED_GEOMETRY,
            registration=registration,
            valid=valid,
            calibration_fingerprint=calibration.fingerprint,
        )
    if not registration_succeeded:
        return _rejected(
            PseudoLabelRejectionReason.REGISTRATION_FAILED,
            registration=registration,
            valid=valid,
            calibration_fingerprint=calibration.fingerprint,
        )
    coverage = valid.float().mean().item()
    if coverage < config.min_valid_coverage:
        return _rejected(
            PseudoLabelRejectionReason.LOW_VALID_COVERAGE,
            registration=registration,
            valid=valid,
            calibration_fingerprint=calibration.fingerprint,
        )
    if int(valid.sum().item()) < config.min_valid_pixels:
        return _rejected(
            PseudoLabelRejectionReason.INSUFFICIENT_VALID_PIXELS,
            registration=registration,
            valid=valid,
            calibration_fingerprint=calibration.fingerprint,
        )

    source_latent = deterministic_vae_posterior_mode(
        source_image,
        preprocess=vae_preprocess,
        encode=vae_encode,
        shift_factor=vae_shift_factor,
        scaling_factor=vae_scaling_factor,
    )
    target_latent = deterministic_vae_posterior_mode(
        target_image,
        preprocess=vae_preprocess,
        encode=vae_encode,
        shift_factor=vae_shift_factor,
        scaling_factor=vae_scaling_factor,
    )
    if source_latent.shape[-2:] != (height, width):
        raise ValueError("latent_size must exactly match deterministic VAE output")
    evidence = compute_change_evidence(
        source_image,
        target_image,
        source_latent,
        target_latent,
        perceptual_extractor,
    )
    normalized, spreads = robust_normalize_cues(
        evidence,
        valid,
        epsilon=config.robust_epsilon,
    )
    if torch.all(spreads < config.flat_evidence_delta).item():
        return _rejected(
            PseudoLabelRejectionReason.FLAT_EVIDENCE,
            registration=registration,
            valid=valid,
            calibration_fingerprint=calibration.fingerprint,
            evidence=evidence,
            normalized=normalized,
            spreads=spreads,
        )

    absolute_gate = torch.zeros_like(evidence.pixel, dtype=torch.float32)
    normalized_mix = torch.zeros_like(evidence.pixel, dtype=torch.float32)
    for weight, cue, normalized_cue, threshold, temperature in zip(
        config.cue_weights,
        evidence.as_tuple(),
        normalized.as_tuple(),
        calibration.thresholds,
        calibration.temperatures,
        strict=True,
    ):
        absolute_gate = absolute_gate + weight * torch.sigmoid(
            (cue.float() - threshold) / temperature
        )
        normalized_mix = normalized_mix + weight * normalized_cue.float()
    fused_change = (absolute_gate * normalized_mix).clamp(0, 1)
    primary = gaussian_primary_seed(
        primary_mask,
        (height, width),
        sigma=config.gaussian_sigma,
        truncate=config.gaussian_truncate,
    )
    if not callable(spatial_criterion):
        raise TypeError("spatial_criterion must be callable")
    assessment = spatial_criterion(
        fused_change.detach().clone(),
        primary.detach().clone(),
        registration.detach().clone(),
        valid.detach().clone(),
    )
    if not isinstance(assessment, SpatialDecomposabilityAssessment):
        raise TypeError(
            "spatial_criterion must return an auditable "
            "SpatialDecomposabilityAssessment"
        )
    if not assessment.decomposable:
        return _rejected(
            PseudoLabelRejectionReason.NON_SPATIALLY_DECOMPOSABLE,
            registration=registration,
            valid=valid,
            calibration_fingerprint=calibration.fingerprint,
            assessment=assessment,
            evidence=evidence,
            normalized=normalized,
            spreads=spreads,
            primary=primary,
            fused_change=fused_change,
        )
    agreement = cue_agreement(normalized, sigma=config.agreement_sigma)
    response = response_conditional_probability(
        fused_change,
        agreement,
        config,
        response_supervision=response_supervision,
    )
    regions = stick_breaking_labels(primary, response)
    confidence = label_confidence(
        registration,
        regions,
        agreement,
        q_pair=q_pair,
        source_confidence=source_confidence,
        confidence_floor=config.confidence_floor,
    )
    total_valid_mass = confidence[valid].sum().item()
    if total_valid_mass < config.min_total_valid_mass:
        return _rejected(
            PseudoLabelRejectionReason.LOW_TOTAL_MASS,
            registration=registration,
            valid=valid,
            calibration_fingerprint=calibration.fingerprint,
            assessment=assessment,
            evidence=evidence,
            normalized=normalized,
            spreads=spreads,
            confidence=confidence,
            primary=primary,
            response=response,
            fused_change=fused_change,
            response_supervision=(
                response_supervision
                if isinstance(response_supervision, ResponseSupervision)
                else None
            ),
        )
    return PRIPseudoLabelResult(
        rejection_reason=None,
        calibration_fingerprint=calibration.fingerprint,
        regions=regions,
        confidence=confidence,
        registration_reliability=registration,
        valid_positions=valid,
        spatial_assessment=assessment,
        evidence=evidence,
        normalized_evidence=normalized,
        evidence_spreads=spreads,
        primary_seed=primary,
        response_conditional=response,
        fused_change=fused_change,
        response_supervision=(
            response_supervision
            if isinstance(response_supervision, ResponseSupervision)
            else None
        ),
    )


__all__ = [
    "AbsoluteChangeCalibration",
    "ChangeEvidence",
    "PRIPseudoLabelConfig",
    "PRIPseudoLabelResult",
    "PseudoLabelRejectionReason",
    "ResponseSupervision",
    "SpatialCriterion",
    "SpatialDecomposabilityAssessment",
    "VGG16PerceptualFeatures",
    "build_vgg16_perceptual_extractor",
    "compute_change_evidence",
    "cue_agreement",
    "deterministic_vae_posterior_mode",
    "gaussian_primary_seed",
    "generate_pri_pseudo_labels",
    "label_confidence",
    "response_conditional_probability",
    "robust_normalize_cues",
    "stick_breaking_labels",
]
