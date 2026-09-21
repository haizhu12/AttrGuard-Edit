"""Runtime-only orchestration for flow-aware regional editing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn.functional as F

from .config import FARConfig, ResponseTransportConfig, _normalize_invariant_strength
from .flow import (
    _analytic_scale_weights_runtime,
    _build_multiscale_response_flow_runtime,
    _compose_pri_flow_runtime,
    build_multiscale_response_flow,
)
from .predictor import masked_qwen_mean
from .predictor_types import PRIRegionPrediction
from .regions import PRIRegionProbabilities

_SUPPORTED_RUNTIME_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_SUPPORTED_RUNTIME_DESCRIPTION = (
    "dense strided float16, bfloat16, float32, and float64 tensors"
)
_INTEGRAL_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@dataclass(frozen=True)
class FARStepResult:
    """Tensors produced by one region-aware denoising-control step."""

    controlled_flow: torch.Tensor
    response_flow: torch.Tensor
    scale_weights: torch.Tensor


@dataclass(frozen=True)
class FARRuntimeContext:
    """Prepared, request-local state shared by all FAR denoising steps."""

    far_config: FARConfig
    transport_config: ResponseTransportConfig
    static_regions: PRIRegionProbabilities | None = None
    predictor: Callable[..., Any] | None = None
    pooled_instruction: torch.Tensor | None = None
    primary_hint: torch.Tensor | None = None
    hint_reliability: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.far_config, FARConfig):
            raise TypeError("far_config must be a FARConfig")
        if not self.far_config.enabled:
            raise ValueError("FAR runtime requires far_config.enabled=True")
        if not isinstance(self.transport_config, ResponseTransportConfig):
            raise TypeError(
                "transport_config must be a ResponseTransportConfig"
            )
        if (self.static_regions is None) == (self.predictor is None):
            raise ValueError("FAR runtime requires exactly one PRI source")
        if self.static_regions is not None:
            _validate_runtime_regions_metadata(self.static_regions)
            if any(
                value is not None
                for value in (
                    self.pooled_instruction,
                    self.primary_hint,
                    self.hint_reliability,
                )
            ):
                raise ValueError(
                    "static FAR runtime forbids predictor-only pooled, hint, and "
                    "reliability fields"
                )
        else:
            if not callable(self.predictor):
                raise TypeError("predictor must be callable")
            if self.pooled_instruction is None:
                raise ValueError("predictor runtime requires pooled_instruction")
            _validate_predictor_context_tensors(
                self.pooled_instruction,
                self.primary_hint,
                self.hint_reliability,
            )

    def regions_for_step(
        self,
        *,
        source_latent: torch.Tensor,
        current_latent: torch.Tensor,
        delta_velocity: torch.Tensor,
        progress: float,
    ) -> PRIRegionProbabilities:
        """Resolve static or predicted regions for one denoising step."""

        if self.static_regions is not None:
            return self.static_regions

        assert self.predictor is not None
        assert self.pooled_instruction is not None
        progress_tensor = torch.full(
            (current_latent.shape[0],),
            float(progress),
            device=current_latent.device,
            dtype=torch.float32,
        )
        predictor_kwargs = {
            "source_latent_grid": source_latent,
            "current_latent": current_latent,
            "delta_velocity": delta_velocity.to(dtype=torch.float32),
            "pooled_instruction": self.pooled_instruction,
            "progress": progress_tensor,
            "primary_hint": self.primary_hint,
            "hint_reliability": self.hint_reliability,
        }
        runtime_forward = getattr(self.predictor, "_forward_runtime", None)
        if callable(runtime_forward):
            probabilities, primary_logits, response_logits = runtime_forward(
                **predictor_kwargs
            )
        else:
            prediction = self.predictor(**predictor_kwargs)
            if not isinstance(prediction, PRIRegionPrediction):
                raise TypeError("predictor must return a PRIRegionPrediction")
            probabilities = prediction.probabilities
            primary_logits = prediction.primary_logits
            response_logits = prediction.response_logits
        return _project_predictor_probabilities(
            probabilities,
            current_latent,
            primary_logits=primary_logits,
            response_logits=response_logits,
            epsilon=self.far_config.probability_tolerance,
            tolerance=self.far_config.probability_tolerance,
        )


def _positive_finite_epsilon(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("epsilon must be a real number")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0:
        raise ValueError("epsilon must be finite and positive")
    return normalized


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_integral(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _validate_dense_supported_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or value.dtype not in _SUPPORTED_RUNTIME_DTYPES:
        raise TypeError(f"{name} only supports {_SUPPORTED_RUNTIME_DESCRIPTION}")
    return value


def _validate_runtime_flow(name: str, value: Any) -> torch.Tensor:
    value = _validate_dense_supported_tensor(name, value)
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W]")
    if any(dimension <= 0 for dimension in value.shape):
        raise ValueError(f"{name} must have positive dimensions")
    return value


def _validate_runtime_regions_metadata(
    regions: Any,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if not isinstance(regions, PRIRegionProbabilities):
        raise TypeError("regions must be a PRIRegionProbabilities instance")
    values = regions.values
    if not isinstance(values, torch.Tensor):
        raise TypeError("runtime region values must be a torch.Tensor")
    if values.layout != torch.strided:
        raise TypeError("runtime region values must be a dense strided tensor")
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("runtime regions must have shape [B, 3, H, W]")
    if values.shape[0] <= 0 or values.shape[2] <= 0 or values.shape[3] <= 0:
        raise ValueError("runtime regions must have positive dimensions")
    if values.dtype != torch.float32:
        raise ValueError("runtime regions must use FP32 probabilities")
    if reference is not None:
        if values.device != reference.device:
            raise ValueError("runtime regions and flows must be on the same device")
        if values.shape[0] != reference.shape[0]:
            raise ValueError("runtime region batch must match flow batch")
        if values.shape[2:] != reference.shape[2:]:
            raise ValueError("runtime region spatial dimensions must match flows")
    return values


def _validate_predictor_context_tensors(
    pooled_instruction: Any,
    primary_hint: Any,
    hint_reliability: Any,
) -> None:
    """Validate request-local predictor tensors once, outside the step loop."""

    pooled = _validate_dense_supported_tensor(
        "pooled_instruction",
        pooled_instruction,
    )
    if pooled.device.type == "meta":
        raise ValueError("pooled_instruction must not use the meta device")
    if pooled.ndim != 2:
        raise ValueError("pooled_instruction must have shape [B, D]")
    if any(dimension <= 0 for dimension in pooled.shape):
        raise ValueError("pooled_instruction must have positive dimensions")

    valid_values = torch.isfinite(pooled).all()
    if primary_hint is None:
        if hint_reliability is not None:
            raise ValueError("hint_reliability requires primary_hint")
    else:
        hint = _validate_dense_supported_tensor("primary_hint", primary_hint)
        if hint.ndim != 4 or hint.shape[1] != 1:
            raise ValueError("primary_hint must have shape [B, 1, H, W]")
        if any(dimension <= 0 for dimension in hint.shape):
            raise ValueError("primary_hint must have positive dimensions")
        if hint.shape[0] != pooled.shape[0]:
            raise ValueError("primary_hint batch must match pooled_instruction batch")
        if hint.device != pooled.device:
            raise ValueError(
                "primary_hint device must match pooled_instruction device"
            )
        valid_values = (
            valid_values
            & torch.isfinite(hint).all()
            & ((hint >= 0) & (hint <= 1)).all()
        )

    if hint_reliability is not None:
        reliability = _validate_dense_supported_tensor(
            "hint_reliability",
            hint_reliability,
        )
        if reliability.ndim != 1:
            raise ValueError("hint_reliability must have shape [B]")
        if reliability.shape[0] <= 0:
            raise ValueError("hint_reliability must have a positive batch dimension")
        if reliability.shape[0] != pooled.shape[0]:
            raise ValueError(
                "hint_reliability batch must match pooled_instruction batch"
            )
        if reliability.device != pooled.device:
            raise ValueError(
                "hint_reliability device must match pooled_instruction device"
            )
        valid_values = (
            valid_values
            & torch.isfinite(reliability).all()
            & ((reliability >= 0) & (reliability <= 1)).all()
        )

    if not valid_values.item():
        raise ValueError(
            "predictor runtime context tensors must be finite; primary_hint and "
            "hint_reliability values must lie in [0, 1]"
        )


def _validate_projection_metadata(
    values: Any,
    reference: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = _validate_dense_supported_tensor("values", values)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("values must have shape [B, 3, H, W]")
    if values.shape[0] <= 0 or values.shape[2] <= 0 or values.shape[3] <= 0:
        raise ValueError("values must have positive batch and spatial dimensions")
    reference = _validate_dense_supported_tensor("reference", reference)
    if reference.ndim != 4 or any(dimension <= 0 for dimension in reference.shape):
        raise ValueError("reference must have shape [B, C, H, W]")
    if values.shape[0] != reference.shape[0]:
        raise ValueError("values batch size must match reference batch size")
    return values, reference


def _resize_projection(
    values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if values.shape[-2:] == reference.shape[-2:]:
        return values
    return F.interpolate(
        values,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def validate_far_scheduler(scheduler: Any) -> None:
    """Require the exact project scheduler configuration used by FAR."""

    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler,
    )

    if type(scheduler) is not FlowMatchEulerDiscreteScheduler:
        raise TypeError(
            "FAR requires the exact in-project FlowMatchEulerDiscreteScheduler"
        )
    required = (
        ("do_shift", True),
        ("dynamic_time_shift", False),
        ("time_shift_version", "v1"),
        ("seq_len", 4096),
    )
    for name, expected in required:
        actual = getattr(scheduler.config, name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(
                f"FAR scheduler {name} must equal {expected!r}; got {actual!r}"
            )


def project_pri_probabilities(
    values: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float,
    tolerance: float = 1e-4,
) -> PRIRegionProbabilities:
    """Project a P/R/I tensor onto a latent-grid FP32 probability simplex."""

    epsilon = _positive_finite_epsilon(epsilon)
    values, reference = _validate_projection_metadata(values, reference)
    if not torch.isfinite(values).all().item():
        raise ValueError("external PRI values must be finite")
    if ((values < 0) | (values > 1)).any().item():
        raise ValueError("external PRI values must lie in [0, 1]")
    projected = values.to(device=reference.device, dtype=torch.float32)
    projected = _resize_projection(projected, reference)
    projected = projected.clip(0, 1)
    denominator = projected.sum(dim=1, keepdim=True, dtype=torch.float32)
    if not (
        torch.isfinite(denominator).all() & (denominator > epsilon).all()
    ).item():
        raise ValueError(
            "probability normalization denominator must be finite and above epsilon"
        )
    projected = projected / denominator
    return PRIRegionProbabilities.from_probabilities(projected, tolerance=tolerance)


def _project_predictor_probabilities(
    values: torch.Tensor,
    reference: torch.Tensor,
    *,
    primary_logits: torch.Tensor,
    response_logits: torch.Tensor,
    epsilon: float,
    tolerance: float,
) -> PRIRegionProbabilities:
    """Project one predictor output with exactly one aggregate host check."""

    epsilon = _positive_finite_epsilon(epsilon)
    values, reference = _validate_projection_metadata(values, reference)
    if values.device.type == "meta" or values.device != reference.device:
        raise ValueError(
            "predictor probability device must match the runtime reference device"
        )
    expected_logit_shape = (values.shape[0], 1, values.shape[2], values.shape[3])
    logits = []
    for name, value in (
        ("primary_logits", primary_logits),
        ("response_logits", response_logits),
    ):
        value = _validate_dense_supported_tensor(name, value)
        if value.shape != expected_logit_shape:
            raise ValueError(f"{name} must have shape {expected_logit_shape}")
        if value.device != values.device:
            raise ValueError(f"{name} must match predictor probability device")
        logits.append(value)

    valid_source = torch.isfinite(values).all() & (
        (values >= 0) & (values <= 1)
    ).all()
    projected = values.to(device=reference.device, dtype=torch.float32)
    finite_logits = torch.isfinite(logits[0]).all() & torch.isfinite(logits[1]).all()
    projected = _resize_projection(projected, reference)
    projected = projected.clip(0, 1)
    denominator = projected.sum(dim=1, keepdim=True, dtype=torch.float32)
    valid_denominator = torch.isfinite(denominator).all() & (
        denominator > epsilon
    ).all()
    safe_denominator = torch.where(
        denominator > epsilon,
        denominator,
        torch.ones_like(denominator),
    )
    normalized = projected / safe_denominator
    normalized_sums = normalized.sum(dim=1, dtype=torch.float32)
    valid_normalized = (
        torch.isfinite(normalized).all()
        & ((normalized >= 0) & (normalized <= 1)).all()
        & ((normalized_sums - 1).abs() <= tolerance).all()
    )
    valid = valid_source & finite_logits & valid_denominator & valid_normalized
    if not valid.item():
        raise ValueError(
            "predictor probabilities/logits must be finite and project to a valid "
            "PRI probability simplex"
        )
    return PRIRegionProbabilities._from_runtime_projection(normalized, tolerance)


def prepare_external_pri_regions(
    values: (
        torch.Tensor
        | PRIRegionProbabilities
        | list[torch.Tensor | PRIRegionProbabilities]
        | tuple[torch.Tensor | PRIRegionProbabilities, ...]
    ),
    reference: torch.Tensor,
    *,
    instruction_batch_size: int,
    num_images_per_instruction: int,
    epsilon: float,
    tolerance: float,
) -> PRIRegionProbabilities:
    """Expand an external PRI tensor in sample-major order and project it once."""

    instruction_batch_size = _positive_integral(
        "instruction_batch_size", instruction_batch_size
    )
    num_images_per_instruction = _positive_integral(
        "num_images_per_instruction", num_images_per_instruction
    )
    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference must be a torch.Tensor")
    if reference.ndim != 4 or any(dimension <= 0 for dimension in reference.shape):
        raise ValueError("reference must have shape [B, C, H, W]")
    expanded_batch_size = instruction_batch_size * num_images_per_instruction
    if reference.shape[0] != expanded_batch_size:
        raise ValueError(
            "reference batch must equal the expanded instruction batch"
        )

    if isinstance(values, (list, tuple)):
        if len(values) not in (instruction_batch_size, expanded_batch_size):
            raise ValueError(
                "external PRI list length must match the instruction or expanded sample batch"
            )
        projected_items = []
        for index, item in enumerate(values):
            raw_item = item.values if isinstance(item, PRIRegionProbabilities) else item
            if not isinstance(raw_item, torch.Tensor):
                raise TypeError("external PRI list entries must be tensors")
            if raw_item.ndim == 3:
                raw_item = raw_item.unsqueeze(0)
            if raw_item.ndim != 4 or raw_item.shape[0] != 1:
                raise ValueError("external PRI list entries must have shape [3, H, W]")
            reference_index = (
                index * num_images_per_instruction
                if len(values) == instruction_batch_size
                else index
            )
            projected_items.append(
                project_pri_probabilities(
                    raw_item,
                    reference[reference_index : reference_index + 1],
                    epsilon=epsilon,
                    tolerance=tolerance,
                ).values
            )
        projected = torch.cat(projected_items, dim=0)
        if len(values) == instruction_batch_size:
            projected = projected.repeat_interleave(
                num_images_per_instruction,
                dim=0,
            )
        return PRIRegionProbabilities.from_probabilities(
            projected,
            tolerance=tolerance,
        )

    raw = values.values if isinstance(values, PRIRegionProbabilities) else values
    if not isinstance(raw, torch.Tensor):
        raise TypeError("external PRI values must be a tensor or PRI probabilities")
    if raw.ndim != 4 or raw.shape[1] != 3:
        raise ValueError("external PRI values must have shape [B, 3, H, W]")
    if raw.shape[0] <= 0 or raw.shape[2] <= 0 or raw.shape[3] <= 0:
        raise ValueError("external PRI values must have positive dimensions")
    if raw.shape[0] == instruction_batch_size:
        raw = raw.repeat_interleave(num_images_per_instruction, dim=0)
    elif raw.shape[0] != expanded_batch_size:
        raise ValueError(
            "external PRI batch must match the instruction or expanded sample batch"
        )
    return project_pri_probabilities(
        raw,
        reference,
        epsilon=epsilon,
        tolerance=tolerance,
    )


def _negative_instruction_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, list):
        return bool(value) and all(
            isinstance(item, str) and item == "" for item in value
        )
    return False


def validate_far_request(
    *,
    far_config: FARConfig | None,
    response_transport_config: ResponseTransportConfig | None,
    far_region_probabilities: Any,
    pri_region_predictor: Any,
    primary_hint: Any,
    hint_reliability: Any,
    text_guidance_scale: float,
    image_guidance_scale: float,
    empty_instruction_guidance_scale: float,
    cfg_range: Any,
    negative_instruction: Any,
    negative_condition_overridden: bool,
    negative_branch_uses_reference: bool,
    rewriting_enabled: bool,
    boosted_guidance_enabled: bool,
    prompt_tuning_enabled: bool,
    lora_enabled: bool,
    acceleration_enabled: bool,
    flow_match_euler_scheduler: bool,
    standard_processing: bool,
    trajectory_capture_enabled: bool,
) -> tuple[FARConfig, ResponseTransportConfig]:
    """Validate the deliberately narrow Phase-2 inference protocol."""

    config = FARConfig() if far_config is None else far_config
    if not isinstance(config, FARConfig):
        raise TypeError("far_config must be a FARConfig")

    auxiliary_values = (
        response_transport_config,
        far_region_probabilities,
        pri_region_predictor,
        primary_hint,
        hint_reliability,
    )
    if not config.enabled:
        if any(value is not None for value in auxiliary_values):
            raise ValueError("FAR is disabled; auxiliary inputs require enabled FAR")
        return config, ResponseTransportConfig()

    if trajectory_capture_enabled:
        raise ValueError("FAR cannot be combined with trajectory capture")
    if (far_region_probabilities is None) == (pri_region_predictor is None):
        raise ValueError("enabled FAR requires exactly one PRI source")
    if response_transport_config is None:
        transport = ResponseTransportConfig(
            mode="fixed",
            kernel_sizes=(config.effect_low_pass_kernel,),
            schedule_centers=(0.5,),
            response_strength=config.effect_strength,
        )
    else:
        if not isinstance(response_transport_config, ResponseTransportConfig):
            raise TypeError(
                "response_transport_config must be a ResponseTransportConfig"
            )
        if config.effect_strength != 0.5 or config.effect_low_pass_kernel != 5:
            raise ValueError(
                "non-default legacy FAR effect controls conflict with explicit "
                "response transport"
            )
        transport = response_transport_config
    if primary_hint is not None and pri_region_predictor is None:
        raise ValueError("primary_hint is supported only with a PRI predictor")
    if hint_reliability is not None and primary_hint is None:
        raise ValueError("hint_reliability requires primary_hint")
    text_guidance_scale = _finite_real(
        "text_guidance_scale",
        text_guidance_scale,
    )
    image_guidance_scale = _finite_real(
        "image_guidance_scale",
        image_guidance_scale,
    )
    empty_instruction_guidance_scale = _finite_real(
        "empty_instruction_guidance_scale",
        empty_instruction_guidance_scale,
    )
    if text_guidance_scale <= 1:
        raise ValueError("FAR requires text_guidance_scale greater than 1")
    if image_guidance_scale != 1:
        raise ValueError("FAR requires image_guidance_scale equal to 1")
    if empty_instruction_guidance_scale != 0:
        raise ValueError("FAR requires empty_instruction_guidance_scale equal to 0")
    if not isinstance(cfg_range, (list, tuple)) or tuple(cfg_range) != (0.0, 1.0):
        raise ValueError("FAR requires cfg_range=(0, 1)")
    if not _negative_instruction_is_empty(negative_instruction):
        raise ValueError("FAR requires an empty negative_instruction")
    if negative_condition_overridden:
        raise ValueError("FAR requires the default empty negative condition")
    if negative_branch_uses_reference:
        raise ValueError("FAR negative branch must not use a reference image")
    if rewriting_enabled:
        raise ValueError("FAR requires instruction rewriting to be disabled")
    if boosted_guidance_enabled:
        raise ValueError("FAR requires boosted guidance to be disabled")
    if prompt_tuning_enabled:
        raise ValueError("FAR requires prompt tuning to be disabled")
    if lora_enabled:
        raise ValueError("FAR requires LoRA to be disabled")
    if acceleration_enabled:
        raise ValueError("FAR requires cache acceleration to be disabled")
    if not flow_match_euler_scheduler:
        raise ValueError("FAR requires the project FlowMatch Euler scheduler")
    if not standard_processing:
        raise ValueError("FAR requires standard Boogu Edit processing")
    return config, transport


def _expand_optional_predictor_input(
    value: torch.Tensor | None,
    *,
    name: str,
    instruction_batch_size: int,
    num_images_per_instruction: int,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expanded_batch_size = instruction_batch_size * num_images_per_instruction
    if value.shape[0] == instruction_batch_size:
        value = value.repeat_interleave(num_images_per_instruction, dim=0)
    elif value.shape[0] != expanded_batch_size:
        raise ValueError(
            f"{name} batch must match the instruction or expanded sample batch"
        )
    return value.to(device=device, dtype=torch.float32)


def _validate_unit_interval_tensor(name: str, value: torch.Tensor) -> None:
    _validate_dense_supported_tensor(name, value)
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    if ((value < 0) | (value > 1)).any().item():
        raise ValueError(f"{name} values must lie in [0, 1]")


def _validate_attention_mask(
    mask: Any,
    hidden: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("instruction_attention_mask must be a torch.Tensor")
    if mask.layout != torch.strided:
        raise TypeError("instruction_attention_mask must be a dense strided tensor")
    if mask.dtype != torch.bool and mask.dtype not in _INTEGRAL_DTYPES:
        raise TypeError(
            "instruction_attention_mask must use bool or an integral dtype"
        )
    if mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError(
            "instruction_attention_mask shape must match instruction embeddings [B, L]"
        )
    if mask.device != hidden.device:
        raise ValueError(
            "instruction_attention_mask must match the instruction embedding device"
        )
    if mask.dtype != torch.bool and ((mask != 0) & (mask != 1)).any().item():
        raise ValueError("instruction_attention_mask must contain only 0 and 1")
    return mask.to(dtype=torch.bool)


def _validate_predictor_module_state(
    predictor: torch.nn.Module,
    *,
    runtime_device: torch.device,
) -> None:
    parameters = tuple(predictor.parameters())
    computation_dtype = None
    for parameter in parameters:
        if parameter.device.type == "meta" or parameter.device != runtime_device:
            raise ValueError(
                "predictor parameters and buffers must match the runtime device"
            )
        if parameter.layout != torch.strided:
            raise TypeError("predictor parameters must be dense strided tensors")
        if parameter.dtype not in _SUPPORTED_RUNTIME_DTYPES:
            raise TypeError(
                "predictor parameters must use a supported floating dtype"
            )
        if computation_dtype is None:
            computation_dtype = parameter.dtype
        elif parameter.dtype != computation_dtype:
            raise ValueError("all predictor parameters must use the same dtype")
        if not torch.isfinite(parameter.detach()).all().item():
            raise ValueError("predictor parameters must be finite")

    for buffer in predictor.buffers():
        if buffer.device.type == "meta" or buffer.device != runtime_device:
            raise ValueError(
                "predictor parameters and buffers must match the runtime device"
            )
        if buffer.layout != torch.strided:
            raise TypeError("predictor buffers must be dense strided tensors")
        if buffer.dtype in _SUPPORTED_RUNTIME_DTYPES:
            if computation_dtype is None:
                computation_dtype = buffer.dtype
            elif buffer.dtype != computation_dtype:
                raise ValueError(
                    "predictor floating buffers must match the computation dtype"
                )
            if not torch.isfinite(buffer.detach()).all().item():
                raise ValueError("predictor buffers must be finite")
        elif buffer.dtype not in _INTEGRAL_DTYPES:
            raise TypeError(
                "predictor buffers must use the computation dtype or an integral dtype"
            )


def prepare_predictor_runtime_context(
    *,
    far_config: FARConfig,
    transport_config: ResponseTransportConfig,
    predictor: Callable[..., Any],
    instruction_embeds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    instruction_attention_mask: torch.Tensor,
    instruction_batch_size: int,
    num_images_per_instruction: int,
    latent_reference: torch.Tensor,
    primary_hint: torch.Tensor | None = None,
    hint_reliability: torch.Tensor | None = None,
) -> FARRuntimeContext:
    """Pool expanded Qwen tokens once and prepare predictor-side request state."""

    if not callable(predictor):
        raise TypeError("predictor must be callable")
    instruction_batch_size = _positive_integral(
        "instruction_batch_size", instruction_batch_size
    )
    num_images_per_instruction = _positive_integral(
        "num_images_per_instruction", num_images_per_instruction
    )
    expanded_batch_size = instruction_batch_size * num_images_per_instruction
    latent_reference = _validate_runtime_flow("latent_reference", latent_reference)
    if latent_reference.shape[0] != expanded_batch_size:
        raise ValueError(
            "latent reference batch must equal the expanded instruction batch"
        )

    if isinstance(instruction_embeds, (list, tuple)):
        if not instruction_embeds:
            raise ValueError("instruction_embeds sequence must not be empty")
        hidden = instruction_embeds[-1]
    else:
        hidden = instruction_embeds
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("instruction_embeds must contain tensors")
    if hidden.ndim != 3:
        raise ValueError("instruction embeddings must have shape [B, L, D]")
    if hidden.shape[0] != expanded_batch_size:
        raise ValueError(
            "instruction embedding batch must equal the expanded instruction batch"
        )
    if hidden.device != latent_reference.device:
        raise ValueError("instruction embeddings must match the runtime device")

    instruction_dim = getattr(predictor, "instruction_dim", None)
    if instruction_dim is not None and instruction_dim != hidden.shape[-1]:
        raise ValueError("predictor instruction_dim must match the pooled hidden width")
    latent_channels = getattr(predictor, "latent_channels", None)
    if latent_channels is not None and latent_channels != latent_reference.shape[1]:
        raise ValueError("predictor latent_channels must match the runtime latent channels")

    if isinstance(predictor, torch.nn.Module):
        _validate_predictor_module_state(
            predictor,
            runtime_device=latent_reference.device,
        )
    prepared_attention_mask = _validate_attention_mask(
        instruction_attention_mask,
        hidden,
    )
    pooled_instruction = masked_qwen_mean(
        hidden,
        prepared_attention_mask,
    )
    if primary_hint is not None:
        if not isinstance(primary_hint, torch.Tensor):
            raise TypeError("primary_hint must be a torch.Tensor")
        if (
            primary_hint.ndim != 4
            or primary_hint.shape[1] != 1
            or any(dimension <= 0 for dimension in primary_hint.shape)
        ):
            raise ValueError("primary_hint must have shape [B, 1, H, W]")
        _validate_unit_interval_tensor("primary_hint", primary_hint)
    prepared_hint = _expand_optional_predictor_input(
        primary_hint,
        name="primary_hint",
        instruction_batch_size=instruction_batch_size,
        num_images_per_instruction=num_images_per_instruction,
        device=latent_reference.device,
    )
    if prepared_hint is not None and prepared_hint.shape[-2:] != latent_reference.shape[-2:]:
        prepared_hint = F.interpolate(
            prepared_hint,
            size=latent_reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if hint_reliability is not None:
        if not isinstance(hint_reliability, torch.Tensor):
            raise TypeError("hint_reliability must be a torch.Tensor")
        if hint_reliability.ndim != 1 or hint_reliability.shape[0] <= 0:
            raise ValueError("hint_reliability must have shape [B]")
        _validate_unit_interval_tensor("hint_reliability", hint_reliability)
    prepared_reliability = _expand_optional_predictor_input(
        hint_reliability,
        name="hint_reliability",
        instruction_batch_size=instruction_batch_size,
        num_images_per_instruction=num_images_per_instruction,
        device=latent_reference.device,
    )
    if prepared_reliability is not None and prepared_hint is None:
        raise ValueError("hint_reliability requires primary_hint")
    return FARRuntimeContext(
        far_config=far_config,
        transport_config=transport_config,
        predictor=predictor,
        pooled_instruction=pooled_instruction,
        primary_hint=prepared_hint,
        hint_reliability=prepared_reliability,
    )


def _apply_far_step_runtime(
    regions: PRIRegionProbabilities,
    edit_flow: torch.Tensor,
    preserve_flow: torch.Tensor,
    *,
    progress: float,
    config: ResponseTransportConfig,
    invariant_strength: float = 0.5,
) -> FARStepResult:
    """Apply FAR using tensors already validated at the request/model boundary."""

    edit_fp32 = edit_flow.to(dtype=torch.float32)
    preserve_fp32 = preserve_flow.to(dtype=torch.float32)

    batch_size = edit_fp32.shape[0]
    if config.mode == "fixed":
        kernel_sizes = config.kernel_sizes
        support = regions.primary + regions.response
        scale_weights = torch.ones(
            batch_size,
            1,
            device=edit_fp32.device,
            dtype=torch.float32,
        )
        use_legacy_fallback = (
            not torch.is_grad_enabled()
            and not any(
                value.requires_grad
                for value in (edit_flow, preserve_flow, support, scale_weights)
            )
            and torch.equal(support, torch.ones_like(support))
        )
        if use_legacy_fallback:
            response_flow = build_multiscale_response_flow(
                edit_flow,
                preserve_flow,
                support=support.to(dtype=edit_flow.dtype),
                scale_weights=scale_weights.to(dtype=edit_flow.dtype),
                response_strength=config.response_strength,
                kernel_sizes=kernel_sizes,
                support_epsilon=config.support_epsilon,
            ).to(dtype=torch.float32)
        else:
            response_flow = _build_multiscale_response_flow_runtime(
                edit_fp32,
                preserve_fp32,
                support=support,
                scale_weights=scale_weights,
                response_strength=config.response_strength,
                kernel_sizes=kernel_sizes,
                support_epsilon=config.support_epsilon,
            )
    else:
        kernel_sizes = config.kernel_sizes
        progress_tensor = torch.full(
            (batch_size,),
            float(progress),
            device=edit_fp32.device,
            dtype=torch.float32,
        )
        scale_weights = _analytic_scale_weights_runtime(progress_tensor, config)
        response_flow = _build_multiscale_response_flow_runtime(
            edit_fp32,
            preserve_fp32,
            support=regions.primary + regions.response,
            scale_weights=scale_weights,
            response_strength=config.response_strength,
            kernel_sizes=kernel_sizes,
            support_epsilon=config.support_epsilon,
        )
    controlled_flow = _compose_pri_flow_runtime(
        regions,
        edit_fp32,
        preserve_fp32,
        response_flow,
        invariant_strength=invariant_strength,
    )
    return FARStepResult(
        controlled_flow=controlled_flow.to(dtype=edit_flow.dtype),
        response_flow=response_flow.to(dtype=edit_flow.dtype),
        scale_weights=scale_weights,
    )


def apply_far_step(
    regions: PRIRegionProbabilities,
    edit_flow: torch.Tensor,
    preserve_flow: torch.Tensor,
    *,
    progress: float,
    config: ResponseTransportConfig,
    invariant_strength: float = 0.5,
) -> FARStepResult:
    """Validate public inputs, then apply regional flow control."""

    if not isinstance(config, ResponseTransportConfig):
        raise TypeError("config must be a ResponseTransportConfig")
    invariant_strength = _normalize_invariant_strength(invariant_strength)
    if isinstance(progress, bool) or not isinstance(progress, Real):
        raise TypeError("progress must be a real number")
    normalized_progress = float(progress)
    if not isfinite(normalized_progress) or not 0 <= normalized_progress <= 1:
        raise ValueError("progress must be finite and between 0 and 1")

    edit_flow = _validate_runtime_flow("edit_flow", edit_flow)
    preserve_flow = _validate_runtime_flow("preserve_flow", preserve_flow)
    if preserve_flow.shape != edit_flow.shape:
        raise ValueError("preserve_flow shape must match edit_flow")
    if preserve_flow.dtype != edit_flow.dtype:
        raise ValueError("preserve_flow dtype must match edit_flow")
    if preserve_flow.device != edit_flow.device:
        raise ValueError("preserve_flow device must match edit_flow")
    _validate_runtime_regions_metadata(regions, edit_flow)

    if not torch.isfinite(edit_flow).all().item():
        raise ValueError("edit_flow must be finite")
    if not torch.isfinite(preserve_flow).all().item():
        raise ValueError("preserve_flow must be finite")
    PRIRegionProbabilities.from_probabilities(
        regions.values,
        tolerance=regions.tolerance,
    )
    return _apply_far_step_runtime(
        regions,
        edit_flow,
        preserve_flow,
        progress=normalized_progress,
        config=config,
        invariant_strength=invariant_strength,
    )
