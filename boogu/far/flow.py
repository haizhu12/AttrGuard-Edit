"""Pure flow-composition primitives for PRI regional editing."""

from itertools import pairwise
from math import isfinite
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn.functional as F

from .config import (
    MAX_RESPONSE_KERNEL_SIZE,
    ResponseTransportConfig,
    _normalize_finite_real,
    _normalize_invariant_strength,
    _normalize_kernel_sizes,
)
from .regions import PRIRegionProbabilities

_SUPPORTED_FLOW_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_FLOW_DESCRIPTION = (
    "dense strided float16, bfloat16, float32, and float64 tensors"
)
_WEIGHT_SUM_TOLERANCE = 1e-4


def _validate_flow(name: str, flow: Any) -> None:
    if not isinstance(flow, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if flow.layout != torch.strided or flow.dtype not in _SUPPORTED_FLOW_DTYPES:
        raise TypeError(f"{name} only supports {_SUPPORTED_FLOW_DESCRIPTION}")
    if flow.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W]")
    if any(dimension <= 0 for dimension in flow.shape):
        raise ValueError(f"{name} must have non-empty dimensions")


def _normalize_kernel_size(kernel_size: Any) -> int:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, Integral):
        raise TypeError("kernel_size must be an integer")
    normalized = int(kernel_size)
    if normalized <= 0 or normalized % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if normalized > MAX_RESPONSE_KERNEL_SIZE:
        raise ValueError(f"kernel_size must be at most {MAX_RESPONSE_KERNEL_SIZE}")
    return normalized


def _normalize_response_strength(response_strength: Any) -> float:
    if isinstance(response_strength, bool) or not isinstance(response_strength, Real):
        raise TypeError("response_strength must be a real number")
    try:
        normalized = float(response_strength)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("response_strength must be finite") from error
    if not isfinite(normalized):
        raise ValueError("response_strength must be finite")
    if not 0 <= normalized <= 1:
        raise ValueError("response_strength must be between 0 and 1")
    return normalized


def _normalize_support_epsilon(support_epsilon: Any) -> float:
    normalized = _normalize_finite_real("support_epsilon", support_epsilon)
    if not 0 < normalized < 1:
        raise ValueError("support_epsilon must be between 0 and 1")
    return normalized


def _validate_compatible_flows(
    reference_name: str,
    reference: torch.Tensor,
    other_name: str,
    other: torch.Tensor,
) -> None:
    if other.shape != reference.shape:
        raise ValueError(f"{other_name} shape must match {reference_name}")
    if other.dtype != reference.dtype:
        raise ValueError(f"{other_name} dtype must match {reference_name}")
    if other.device != reference.device:
        raise ValueError(f"{other_name} device must match {reference_name}")


def low_pass_flow(flow: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply a replicated-edge box filter with bounded, linear padding memory."""

    _validate_flow("flow", flow)
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size == 1:
        return flow
    working_flow = _to_accumulation_dtype(flow)
    filtered = _separable_box_mean(working_flow, kernel_size)
    return filtered.to(dtype=flow.dtype) if filtered.dtype != flow.dtype else filtered


def _to_accumulation_dtype(flow: torch.Tensor) -> torch.Tensor:
    if flow.dtype in _LOW_PRECISION_DTYPES:
        return flow.to(dtype=torch.float32)
    return flow


def _separable_box_mean(flow: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Compute a box mean without a quadratic two-dimensional padded tensor."""

    padding = kernel_size // 2
    horizontally_padded = F.pad(
        flow / kernel_size,
        (padding, padding, 0, 0),
        mode="replicate",
    )
    horizontally_filtered = F.avg_pool2d(
        horizontally_padded,
        kernel_size=(1, kernel_size),
        stride=1,
        divisor_override=1,
    )
    vertically_padded = F.pad(
        horizontally_filtered / kernel_size,
        (0, 0, padding, padding),
        mode="replicate",
    )
    return F.avg_pool2d(
        vertically_padded,
        kernel_size=(kernel_size, 1),
        stride=1,
        divisor_override=1,
    )


def build_response_flow(
    edit_flow: torch.Tensor,
    preserve_flow: torch.Tensor,
    response_strength: float,
    kernel_size: int,
) -> torch.Tensor:
    """Build the low-frequency contextual response around a primary edit."""

    _validate_flow("edit_flow", edit_flow)
    _validate_flow("preserve_flow", preserve_flow)
    _validate_compatible_flows(
        "edit_flow",
        edit_flow,
        "preserve_flow",
        preserve_flow,
    )
    response_strength = _normalize_response_strength(response_strength)
    kernel_size = _normalize_kernel_size(kernel_size)
    if response_strength == 0:
        return preserve_flow
    if kernel_size == 1 and response_strength == 1:
        return edit_flow

    edit_working = _to_accumulation_dtype(edit_flow)
    preserve_working = _to_accumulation_dtype(preserve_flow)
    if kernel_size == 1:
        response_working = (
            1 - response_strength
        ) * preserve_working + response_strength * edit_working
    else:
        # Scale before subtracting so opposite finite FP16/BF16-range endpoints do
        # not overflow. This is linearly equivalent to
        # preserve + s * LP(edit - preserve), but uses only one box filter.
        stable_scale = 0.25
        scaled_strength = response_strength * stable_scale
        scaled_delta = (
            edit_working * scaled_strength - preserve_working * scaled_strength
        )
        filtered_scaled_delta = _separable_box_mean(scaled_delta, kernel_size)
        if edit_flow.dtype in _LOW_PRECISION_DTYPES:
            # Match the precision of the returned flow before undoing the safety
            # scale. Besides avoiding false precision, this removes backend-level
            # accumulation residue that can survive the final FP16 cast.
            filtered_scaled_delta = filtered_scaled_delta.to(dtype=edit_flow.dtype).to(
                dtype=edit_working.dtype
            )
        response_working = preserve_working * stable_scale + filtered_scaled_delta
        response_working = response_working / stable_scale
    if response_working.dtype != edit_flow.dtype:
        return response_working.to(dtype=edit_flow.dtype)
    return response_working


def normalized_denoising_progress(step_index: int, num_steps: int) -> float:
    """Return scheduler-independent progress in the closed unit interval."""

    if isinstance(step_index, bool) or not isinstance(step_index, Integral):
        raise TypeError("step_index must be an integer")
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
        raise TypeError("num_steps must be an integer")
    step_index = int(step_index)
    num_steps = int(num_steps)
    if num_steps <= 0 or not 0 <= step_index < num_steps:
        raise ValueError("step_index must identify a step in a positive schedule")
    return 0.0 if num_steps == 1 else step_index / (num_steps - 1)


def _validate_progress(progress: Any) -> None:
    if not isinstance(progress, torch.Tensor):
        raise TypeError("progress must be a torch.Tensor")
    if progress.layout != torch.strided or progress.dtype not in _SUPPORTED_FLOW_DTYPES:
        raise TypeError(f"progress only supports {_SUPPORTED_FLOW_DESCRIPTION}")
    if progress.ndim != 1 or progress.shape[0] <= 0:
        raise ValueError("progress must have shape [B] with a positive batch size")
    if not bool(torch.isfinite(progress).all()):
        raise ValueError("progress must be finite")
    if bool(((progress < 0) | (progress > 1)).any()):
        raise ValueError("progress must be between 0 and 1")


def analytic_scale_weights(
    progress: torch.Tensor,
    config: ResponseTransportConfig,
) -> torch.Tensor:
    """Compute differentiable coarse-to-fine Gaussian scale weights."""

    if not isinstance(config, ResponseTransportConfig):
        raise TypeError("config must be a ResponseTransportConfig")
    _validate_progress(progress)
    return _analytic_scale_weights_runtime(progress, config)


def _analytic_scale_weights_runtime(
    progress: torch.Tensor,
    config: ResponseTransportConfig,
) -> torch.Tensor:
    """Trusted tensor-only scale routing used inside the denoising hot path."""

    # FP64 keeps valid finite temperatures whose square is below FP32 range
    # from collapsing every Gaussian logit to ``-inf``.
    working_progress = progress.to(dtype=torch.float64)
    centers = working_progress.new_tensor(config.schedule_centers)
    squared_distance = (working_progress[:, None] - centers[None, :]) ** 2
    relative_squared_distance = squared_distance - squared_distance.amin(
        dim=1,
        keepdim=True,
    )
    logits = -relative_squared_distance / (2 * config.schedule_temperature**2)
    weights = torch.softmax(logits, dim=1)
    return (
        weights.to(dtype=progress.dtype) if weights.dtype != progress.dtype else weights
    )


def _validate_support(support: Any, reference: torch.Tensor) -> None:
    _validate_flow("support", support)
    if support.shape[1] != 1:
        raise ValueError("support must have shape [B, 1, H, W]")
    if (
        support.shape[0] != reference.shape[0]
        or support.shape[2:] != reference.shape[2:]
    ):
        raise ValueError("support batch and spatial dimensions must match flows")
    if support.dtype != reference.dtype:
        raise ValueError("support dtype must match flows")
    if support.device != reference.device:
        raise ValueError("support device must match flows")
    if not bool(torch.isfinite(support).all()):
        raise ValueError("support must be finite")
    if bool((support < 0).any()):
        raise ValueError("support must be non-negative")


def _validate_scale_weights(
    scale_weights: Any,
    reference: torch.Tensor,
    num_scales: int,
) -> None:
    if not isinstance(scale_weights, torch.Tensor):
        raise TypeError("scale_weights must be a torch.Tensor")
    if (
        scale_weights.layout != torch.strided
        or scale_weights.dtype not in _SUPPORTED_FLOW_DTYPES
    ):
        raise TypeError(f"scale_weights only supports {_SUPPORTED_FLOW_DESCRIPTION}")
    if scale_weights.ndim not in (2, 4):
        raise ValueError("scale_weights must have shape [B, S] or [B, S, H, W]")
    if scale_weights.shape[0] != reference.shape[0]:
        raise ValueError("scale_weights batch dimension must match flows")
    if scale_weights.shape[1] != num_scales:
        raise ValueError("scale_weights scale dimension must match kernel_sizes")
    if scale_weights.ndim == 4 and scale_weights.shape[2:] != reference.shape[2:]:
        raise ValueError("spatial scale_weights dimensions must match flows")
    if scale_weights.dtype != reference.dtype:
        raise ValueError("scale_weights dtype must match flows")
    if scale_weights.device != reference.device:
        raise ValueError("scale_weights device must match flows")
    if not bool(torch.isfinite(scale_weights).all()):
        raise ValueError("scale_weights must be finite")
    if bool((scale_weights < 0).any()):
        raise ValueError("scale_weights must be non-negative")

    accumulation_dtype = (
        torch.float32
        if scale_weights.dtype in _LOW_PRECISION_DTYPES
        else scale_weights.dtype
    )
    scale_sums = scale_weights.sum(dim=1, dtype=accumulation_dtype)
    tolerance = max(_WEIGHT_SUM_TOLERANCE, torch.finfo(scale_weights.dtype).eps)
    if not torch.allclose(
        scale_sums,
        torch.ones_like(scale_sums),
        rtol=0,
        atol=tolerance,
    ):
        raise ValueError("scale_weights must sum to 1 along the scale dimension")


def _is_exact_single_scale_fallback(
    edit_flow: torch.Tensor,
    preserve_flow: torch.Tensor,
    support: torch.Tensor,
    scale_weights: torch.Tensor,
    kernel_sizes: tuple[int, ...],
) -> bool:
    if len(kernel_sizes) != 1:
        return False
    if torch.is_grad_enabled():
        return False
    if any(
        value.requires_grad
        for value in (edit_flow, preserve_flow, support, scale_weights)
    ):
        return False
    return torch.equal(support, torch.ones_like(support)) and torch.equal(
        scale_weights,
        torch.ones_like(scale_weights),
    )


def build_multiscale_response_flow(
    edit_flow: torch.Tensor,
    preserve_flow: torch.Tensor,
    *,
    support: torch.Tensor,
    scale_weights: torch.Tensor,
    response_strength: float,
    kernel_sizes: tuple[int, ...],
    support_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Build support-normalized, timestep-routed contextual response flow."""

    _validate_flow("edit_flow", edit_flow)
    _validate_flow("preserve_flow", preserve_flow)
    _validate_compatible_flows(
        "edit_flow",
        edit_flow,
        "preserve_flow",
        preserve_flow,
    )
    kernel_sizes = _normalize_kernel_sizes("kernel_sizes", kernel_sizes)
    if any(left <= right for left, right in pairwise(kernel_sizes)):
        raise ValueError("kernel_sizes must be strictly decreasing")
    response_strength = _normalize_response_strength(response_strength)
    support_epsilon = _normalize_support_epsilon(support_epsilon)
    _validate_support(support, edit_flow)
    _validate_scale_weights(scale_weights, edit_flow, len(kernel_sizes))

    if response_strength == 0:
        return preserve_flow
    if _is_exact_single_scale_fallback(
        edit_flow,
        preserve_flow,
        support,
        scale_weights,
        kernel_sizes,
    ):
        return build_response_flow(
            edit_flow,
            preserve_flow,
            response_strength,
            kernel_sizes[0],
        )

    return _build_multiscale_response_flow_runtime(
        edit_flow,
        preserve_flow,
        support=support,
        scale_weights=scale_weights,
        response_strength=response_strength,
        kernel_sizes=kernel_sizes,
        support_epsilon=support_epsilon,
    )


def _low_pass_flow_runtime(flow: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return flow if kernel_size == 1 else _separable_box_mean(flow, kernel_size)


def _build_multiscale_response_flow_runtime(
    edit_flow: torch.Tensor,
    preserve_flow: torch.Tensor,
    *,
    support: torch.Tensor,
    scale_weights: torch.Tensor,
    response_strength: float,
    kernel_sizes: tuple[int, ...],
    support_epsilon: float,
) -> torch.Tensor:
    """Trusted tensor-only response transport for validated runtime tensors."""

    if response_strength == 0:
        return preserve_flow

    edit_working = _to_accumulation_dtype(edit_flow)
    preserve_working = _to_accumulation_dtype(preserve_flow)
    support_working = _to_accumulation_dtype(support)
    weights_working = _to_accumulation_dtype(scale_weights)

    # Form the potentially large residual after a power-of-two safety scale.
    # The corresponding output is unscaled only after cancellation with the
    # preserve path, keeping opposite finite FP16/BF16 endpoints finite.
    stable_scale = 0.25
    scaled_residual = edit_working * stable_scale - preserve_working * stable_scale
    transported_scales = []
    for kernel_size in kernel_sizes:
        support_mass = _low_pass_flow_runtime(support_working, kernel_size)
        transported = _low_pass_flow_runtime(
            support_working * scaled_residual,
            kernel_size,
        )
        active_support = support_mass > support_epsilon
        safe_support_mass = torch.where(
            active_support,
            support_mass,
            torch.ones_like(support_mass),
        )
        transported = transported / safe_support_mass
        transported_scales.append(
            torch.where(
                active_support,
                transported,
                torch.zeros_like(transported),
            )
        )

    stacked = torch.stack(transported_scales, dim=1)
    if weights_working.ndim == 2:
        broadcast_weights = weights_working[:, :, None, None, None]
    else:
        broadcast_weights = weights_working[:, :, None]
    mixed_scaled_residual = (stacked * broadcast_weights).sum(dim=1)
    response_working = (
        preserve_working * stable_scale + response_strength * mixed_scaled_residual
    ) / stable_scale
    if response_working.dtype != edit_flow.dtype:
        return response_working.to(dtype=edit_flow.dtype)
    return response_working


def compose_pri_flow(
    regions: PRIRegionProbabilities,
    edit_flow: torch.Tensor,
    preserve_flow: torch.Tensor,
    response_flow: torch.Tensor,
    *,
    invariant_strength: float = 0.5,
) -> torch.Tensor:
    """Compose PRI flows, blending edit and preserve within the invariant role."""

    invariant_strength = _normalize_invariant_strength(invariant_strength)
    if not isinstance(regions, PRIRegionProbabilities):
        raise TypeError("regions must be a PRIRegionProbabilities instance")
    _validate_flow("edit_flow", edit_flow)
    _validate_flow("preserve_flow", preserve_flow)
    _validate_flow("response_flow", response_flow)
    _validate_compatible_flows(
        "edit_flow",
        edit_flow,
        "preserve_flow",
        preserve_flow,
    )
    _validate_compatible_flows(
        "edit_flow",
        edit_flow,
        "response_flow",
        response_flow,
    )

    region_values = regions.values
    if region_values.ndim != 4 or region_values.shape[1] != 3:
        raise ValueError("regions must have shape [B, 3, H, W]")
    if (
        region_values.shape[0] != edit_flow.shape[0]
        or region_values.shape[2:] != edit_flow.shape[2:]
    ):
        raise ValueError("regions batch and spatial dimensions must match flows")
    if region_values.dtype != edit_flow.dtype:
        raise ValueError("regions dtype must match flows")
    if region_values.device != edit_flow.device:
        raise ValueError("regions device must match flows")

    return _compose_pri_flow_runtime(
        regions,
        edit_flow,
        preserve_flow,
        response_flow,
        invariant_strength=invariant_strength,
    )


def _compose_pri_flow_runtime(
    regions: PRIRegionProbabilities,
    edit_flow: torch.Tensor,
    preserve_flow: torch.Tensor,
    response_flow: torch.Tensor,
    *,
    invariant_strength: float = 0.5,
) -> torch.Tensor:
    """Compose already-validated runtime tensors without scalar reductions."""

    if invariant_strength == 1:
        # Preserve the original hard-I arithmetic path for exact reproduction.
        return (
            regions.primary * edit_flow
            + regions.response * response_flow
            + regions.invariant * preserve_flow
        )

    edit_working = _to_accumulation_dtype(edit_flow)
    preserve_working = _to_accumulation_dtype(preserve_flow)
    response_working = _to_accumulation_dtype(response_flow)
    # A convex blend avoids forming an overflowing edit/preserve residual.
    invariant_working = (
        edit_working
        if invariant_strength == 0
        else (1 - invariant_strength) * edit_working
        + invariant_strength * preserve_working
    )
    controlled_working = (
        _to_accumulation_dtype(regions.primary) * edit_working
        + _to_accumulation_dtype(regions.response) * response_working
        + _to_accumulation_dtype(regions.invariant) * invariant_working
    )
    return controlled_working.to(dtype=edit_flow.dtype)
