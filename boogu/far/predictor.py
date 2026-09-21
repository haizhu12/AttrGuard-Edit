"""Lightweight two-logit predictor for semantic PRI region probabilities."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .predictor_types import PRIRegionPrediction
from .regions import PRIRegionProbabilities

_MASKED_MEAN_EPSILON = 1e-6


def masked_qwen_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool valid Qwen tokens into a detached FP32 instruction vector.

    The cacheable result is the frozen Qwen representation before the
    predictor's trainable instruction projection.  An all-masked sample is a
    data error rather than an instruction with a silently fabricated zero
    representation.
    """

    if not isinstance(hidden, torch.Tensor):
        raise TypeError("hidden must be a torch.Tensor")
    if hidden.ndim != 3 or any(dimension <= 0 for dimension in hidden.shape):
        raise ValueError("hidden must have shape [B, L, D] with positive dimensions")
    if not torch.is_floating_point(hidden):
        raise TypeError("hidden must have a floating-point dtype")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("mask shape must match hidden's [B, L] dimensions")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have a boolean dtype")
    if mask.device != hidden.device:
        raise ValueError("mask and hidden must be on the same device")

    detached_hidden = hidden.detach()
    if not torch.isfinite(detached_hidden).all().item():
        raise ValueError("hidden must be finite")
    valid_counts = mask.sum(dim=1)
    if (valid_counts == 0).any().item():
        raise ValueError("every sample must contain at least one valid token")

    weights = mask.detach().to(dtype=torch.float32).unsqueeze(-1)
    pooled = (detached_hidden.to(dtype=torch.float32) * weights).sum(dim=1)
    denominator = valid_counts.detach().to(dtype=torch.float32).unsqueeze(-1)
    pooled = pooled / (denominator + _MASKED_MEAN_EPSILON)
    return pooled.detach()


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


class _DepthwiseSeparableBlock(nn.Module):
    """Bias-free depthwise/pointwise convolution followed by GN and SiLU."""

    def __init__(self, channels: int, num_groups: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups, channels)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.depthwise(values)
        values = self.pointwise(values)
        values = self.norm(values)
        return self.activation(values)


class SemanticPRIPredictor(nn.Module):
    """Predict semantic Primary/Response/Invariant probabilities.

    The network consumes detached features from the frozen Boogu backbone.  It
    emits two hierarchical logits, not three independent class logits: the
    first estimates Primary probability and the second estimates Response
    conditional on not being Primary.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        instruction_dim: int = 4096,
        hidden_channels: int = 64,
        num_groups: int = 8,
        validate_input_values: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = _positive_integer("latent_channels", latent_channels)
        self.instruction_dim = _positive_integer("instruction_dim", instruction_dim)
        self.hidden_channels = _positive_integer("hidden_channels", hidden_channels)
        self.num_groups = _positive_integer("num_groups", num_groups)
        if not isinstance(validate_input_values, bool):
            raise TypeError("validate_input_values must be a bool")
        self.validate_input_values = validate_input_values
        if self.hidden_channels % self.num_groups != 0:
            raise ValueError("hidden_channels must be divisible by num_groups")

        self.input_channels = 3 * self.latent_channels + 3
        self.instruction_projection = nn.Linear(
            self.instruction_dim,
            self.hidden_channels,
            bias=True,
        )
        self.stem = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.feature_blocks = nn.ModuleList(
            [
                _DepthwiseSeparableBlock(self.hidden_channels, self.num_groups)
                for _ in range(3)
            ]
        )
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.film_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.hidden_channels,
                    2 * self.hidden_channels,
                    bias=True,
                )
                for _ in range(3)
            ]
        )
        for film_layer in self.film_layers:
            nn.init.zeros_(film_layer.weight)
            nn.init.zeros_(film_layer.bias)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=1,
                    bias=False,
                )
                for _ in range(2)
            ]
        )
        self.output_head = nn.Conv2d(
            self.hidden_channels,
            2,
            kernel_size=1,
            bias=True,
        )

    @staticmethod
    def _detached_floating_tensor(
        name: str,
        value: Any,
        *,
        ndim: int,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions")
        if any(dimension <= 0 for dimension in value.shape):
            raise ValueError(f"{name} must have positive dimensions")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must have a floating-point dtype")
        return value.detach()

    @staticmethod
    def _values_are_valid(
        named_values: tuple[tuple[str, torch.Tensor], ...],
        progress: torch.Tensor,
        hint: torch.Tensor | None,
        reliability: torch.Tensor | None,
    ) -> None:
        """Validate values at a strict public or explicit trusted boundary.

        Full tensor reductions synchronize accelerator execution, so the
        predictor keeps them out of its private per-denoising-step hot path.
        Public ``forward`` always performs this validation.
        """

        for name, value in named_values:
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite")
        if (progress < 0).any().item() or (progress > 1).any().item():
            raise ValueError("progress values must lie in [0, 1]")
        if hint is not None and (
            (hint < 0).any().item() or (hint > 1).any().item()
        ):
            raise ValueError("primary_hint values must lie in [0, 1]")
        if reliability is not None and (
            (reliability < 0).any().item() or (reliability > 1).any().item()
        ):
            raise ValueError("hint_reliability values must lie in [0, 1]")

    @staticmethod
    def _require_device(
        name: str,
        value: torch.Tensor,
        expected_device: torch.device,
    ) -> None:
        if value.device != expected_device:
            raise ValueError(f"{name} must be on device {expected_device}")

    def _validate_inputs(
        self,
        source_latent_grid: Any,
        current_latent: Any,
        delta_velocity: Any,
        pooled_instruction: Any,
        progress: Any,
        primary_hint: Any,
        hint_reliability: Any,
        *,
        validate_input_values: bool | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        source = self._detached_floating_tensor(
            "source_latent_grid", source_latent_grid, ndim=4
        )
        current = self._detached_floating_tensor(
            "current_latent", current_latent, ndim=4
        )
        delta = self._detached_floating_tensor(
            "delta_velocity", delta_velocity, ndim=4
        )
        instruction = self._detached_floating_tensor(
            "pooled_instruction", pooled_instruction, ndim=2
        )
        normalized_progress = self._detached_floating_tensor(
            "progress", progress, ndim=1
        )

        batch_size = current.shape[0]
        if current.shape[1] != self.latent_channels:
            raise ValueError(
                f"current_latent must have {self.latent_channels} channels"
            )
        if min(current.shape[-2:]) < 4:
            raise ValueError("current_latent height and width must each be at least 4")
        if source.shape[:2] != (batch_size, self.latent_channels):
            raise ValueError(
                "source_latent_grid batch and channel dimensions must match "
                "current_latent"
            )
        if tuple(delta.shape) != tuple(current.shape):
            raise ValueError("delta_velocity shape must match current_latent")
        if tuple(instruction.shape) != (batch_size, self.instruction_dim):
            raise ValueError(
                "pooled_instruction must have shape "
                f"[{batch_size}, {self.instruction_dim}]"
            )
        if tuple(normalized_progress.shape) != (batch_size,):
            raise ValueError(f"progress must have shape [{batch_size}]")
        expected_device = next(self.parameters()).device
        for name, value in (
            ("source_latent_grid", source),
            ("current_latent", current),
            ("delta_velocity", delta),
            ("pooled_instruction", instruction),
            ("progress", normalized_progress),
        ):
            self._require_device(name, value, expected_device)

        if primary_hint is None:
            if hint_reliability is not None:
                raise ValueError(
                    "hint_reliability requires a corresponding primary_hint"
                )
            hint = None
            reliability = None
        else:
            hint = self._detached_floating_tensor(
                "primary_hint", primary_hint, ndim=4
            )
            if hint.shape[0] != batch_size or hint.shape[1] != 1:
                raise ValueError(
                    "primary_hint must have shape [B, 1, H_hint, W_hint]"
                )
            self._require_device("primary_hint", hint, expected_device)
            if hint_reliability is None:
                reliability = torch.ones(
                    batch_size,
                    device=expected_device,
                    dtype=torch.float32,
                )
            else:
                reliability = self._detached_floating_tensor(
                    "hint_reliability", hint_reliability, ndim=1
                )
                if tuple(reliability.shape) != (batch_size,):
                    raise ValueError(
                        f"hint_reliability must have shape [{batch_size}]"
                    )
                self._require_device(
                    "hint_reliability", reliability, expected_device
                )
        should_validate_values = (
            self.validate_input_values
            if validate_input_values is None
            else validate_input_values
        )
        if should_validate_values:
            named_values = [
                ("source_latent_grid", source),
                ("current_latent", current),
                ("delta_velocity", delta),
                ("pooled_instruction", instruction),
                ("progress", normalized_progress),
            ]
            if hint is not None:
                named_values.append(("primary_hint", hint))
            if reliability is not None:
                named_values.append(("hint_reliability", reliability))
            self._values_are_valid(
                tuple(named_values),
                normalized_progress,
                hint,
                reliability,
            )

        return (
            source,
            current,
            delta,
            instruction,
            normalized_progress,
            hint,
            reliability,
        )

    def _modulate(
        self,
        features: torch.Tensor,
        instruction: torch.Tensor,
        scale_index: int,
    ) -> torch.Tensor:
        gamma, beta = self.film_layers[scale_index](instruction).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1.0 + gamma) * features + beta

    def _forward_runtime(
        self,
        source_latent_grid: torch.Tensor,
        current_latent: torch.Tensor,
        delta_velocity: torch.Tensor,
        pooled_instruction: torch.Tensor,
        progress: torch.Tensor,
        primary_hint: torch.Tensor | None = None,
        hint_reliability: torch.Tensor | None = None,
        *,
        _validate_values: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw tensors using optional instance-level runtime hardening.

        The default follows ``validate_input_values``.  Normal pipeline use keeps
        its default ``False`` to avoid repeated step-level scans, while callers
        can opt in at construction time.  Public :meth:`forward` always forces
        strict validation independently of this setting.
        """

        validate_runtime_values = (
            self.validate_input_values
            if _validate_values is None
            else _validate_values
        )

        (
            source,
            current,
            delta,
            instruction,
            normalized_progress,
            hint,
            reliability,
        ) = self._validate_inputs(
            source_latent_grid,
            current_latent,
            delta_velocity,
            pooled_instruction,
            progress,
            primary_hint,
            hint_reliability,
            validate_input_values=validate_runtime_values,
        )

        target_size = current.shape[-2:]
        source_fp32 = source.to(dtype=torch.float32)
        if source.shape[-2:] != target_size:
            source_fp32 = F.interpolate(
                source_fp32,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        if hint is None:
            hint_fp32 = torch.zeros(
                current.shape[0],
                1,
                *target_size,
                device=current.device,
                dtype=torch.float32,
            )
            reliability_fp32 = torch.zeros(
                current.shape[0],
                1,
                1,
                1,
                device=current.device,
                dtype=torch.float32,
            )
        else:
            hint_fp32 = hint.to(dtype=torch.float32)
            if hint.shape[-2:] != target_size:
                hint_fp32 = F.interpolate(
                    hint_fp32,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            assert reliability is not None
            reliability_fp32 = reliability.to(dtype=torch.float32).view(-1, 1, 1, 1)

        delta_fp32 = delta.to(dtype=torch.float32)
        energy_fp32 = delta_fp32.abs().mean(dim=1, keepdim=True)
        progress_fp32 = normalized_progress.to(dtype=torch.float32).view(-1, 1, 1, 1)
        progress_map = progress_fp32.expand(-1, 1, *target_size)

        computation_dtype = self.stem.weight.dtype
        spatial_input = torch.cat(
            (
                source_fp32,
                current.to(dtype=torch.float32),
                delta_fp32,
                energy_fp32,
                progress_map,
                hint_fp32,
            ),
            dim=1,
        ).to(dtype=computation_dtype)
        projected_instruction = self.instruction_projection(
            instruction.to(dtype=computation_dtype)
        )

        features_0 = self.feature_blocks[0](self.stem(spatial_input))
        features_1 = self.feature_blocks[1](self.downsample(features_0))
        features_2 = self.feature_blocks[2](self.downsample(features_1))
        modulated = (
            self._modulate(features_0, projected_instruction, 0),
            self._modulate(features_1, projected_instruction, 1),
            self._modulate(features_2, projected_instruction, 2),
        )

        pyramid_2 = modulated[2]
        pyramid_1 = modulated[1] + F.interpolate(
            self.lateral_convs[1](pyramid_2),
            size=features_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pyramid_0 = modulated[0] + F.interpolate(
            self.lateral_convs[0](pyramid_1),
            size=features_0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        logits = self.output_head(pyramid_0).to(dtype=torch.float32)
        primary_logits, response_logits = logits.chunk(2, dim=1)
        predicted_primary = torch.sigmoid(primary_logits)
        primary = (
            reliability_fp32 * hint_fp32
            + (1.0 - reliability_fp32) * predicted_primary
        )
        response_given_not_primary = torch.sigmoid(response_logits)
        not_primary = 1.0 - primary
        probabilities = torch.cat(
            (
                primary,
                not_primary * response_given_not_primary,
                not_primary * (1.0 - response_given_not_primary),
            ),
            dim=1,
        )
        return probabilities, primary_logits, response_logits

    def forward(
        self,
        source_latent_grid: torch.Tensor,
        current_latent: torch.Tensor,
        delta_velocity: torch.Tensor,
        pooled_instruction: torch.Tensor,
        progress: torch.Tensor,
        primary_hint: torch.Tensor | None = None,
        hint_reliability: torch.Tensor | None = None,
    ) -> PRIRegionPrediction:
        """Return hierarchical logits and explicit FP32 P/R/I probabilities."""

        probabilities, primary_logits, response_logits = self._forward_runtime(
            source_latent_grid=source_latent_grid,
            current_latent=current_latent,
            delta_velocity=delta_velocity,
            pooled_instruction=pooled_instruction,
            progress=progress,
            primary_hint=primary_hint,
            hint_reliability=hint_reliability,
            _validate_values=True,
        )
        regions = PRIRegionProbabilities.from_probabilities(probabilities)
        return PRIRegionPrediction(
            regions=regions,
            primary_logits=primary_logits,
            response_logits=response_logits,
        )


__all__ = [
    "PRIRegionPrediction",
    "SemanticPRIPredictor",
    "masked_qwen_mean",
]
