"""Read-only capture primitives for offline teacher-forced Boogu trajectories.

This module deliberately contains no PRI flow-control logic.  It only records
states already computed by the frozen baseline edit path so they can be stored
as training examples for the semantic PRI predictor.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

import torch


def _detached_clone(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    return value.detach().clone()


@dataclass(frozen=True, slots=True)
class _EncodedConditionsPayload:
    pipeline: Any
    instruction: tuple[str, ...]
    source_identities: tuple[int, ...]
    source_fingerprints: tuple[str, ...]
    hbar: torch.Tensor
    positive: torch.Tensor
    positive_mask: torch.Tensor
    negative: torch.Tensor
    negative_mask: torch.Tensor


class EncodedTrajectoryConditions:
    """Opaque Qwen conditions issued only by the trusted pipeline encoder.

    Tensor payloads live in a private weak registry rather than on the public
    object.  The only public export operation returns detached deep copies, so
    cached features cannot alias the conditions later consumed by inference.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls, *args: Any, **kwargs: Any):
        del args, kwargs
        raise TypeError(
            "EncodedTrajectoryConditions can only be created by "
            "BooguImagePipeline.encode_trajectory_conditions()."
        )

    def export_for_cache(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(hbar, positive, pos_mask, negative, neg_mask)`` clones."""

        with _SEALED_PAYLOAD_LOCK:
            payload = _ENCODED_CONDITIONS_PAYLOADS.get(self)
        if payload is None:
            raise ValueError("the encoded trajectory conditions are invalid or expired.")
        return tuple(
            _detached_clone(name, value)
            for name, value in (
                ("hbar", payload.hbar),
                ("positive", payload.positive),
                ("positive_mask", payload.positive_mask),
                ("negative", payload.negative),
                ("negative_mask", payload.negative_mask),
            )
        )


@dataclass(frozen=True, slots=True)
class _CaptureContextPayload:
    pipeline: Any
    scheduler: Any
    observer: TrajectoryObserver


class TrajectoryCaptureContext:
    """Opaque authority connecting validated ``__call__`` to ``processing``."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *args: Any, **kwargs: Any):
        del args, kwargs
        raise TypeError("TrajectoryCaptureContext is issued internally by the pipeline.")


_SEALED_PAYLOAD_LOCK = RLock()
_ENCODED_CONDITIONS_PAYLOADS: WeakKeyDictionary[
    EncodedTrajectoryConditions, _EncodedConditionsPayload
] = WeakKeyDictionary()
_CAPTURE_CONTEXT_PAYLOADS: WeakKeyDictionary[
    TrajectoryCaptureContext, _CaptureContextPayload
] = WeakKeyDictionary()


def _normalize_instruction_identity(
    instruction: str | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(instruction, str):
        return (instruction,)
    if isinstance(instruction, Sequence) and not isinstance(
        instruction, (str, bytes)
    ):
        normalized = tuple(instruction)
        if normalized and all(isinstance(value, str) for value in normalized):
            return normalized
    raise TypeError("instruction must be a string or a non-empty sequence of strings.")


def _source_content_fingerprint(source: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(source, torch.Tensor):
        tensor = source.detach().to(device="cpu").contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser().resolve(strict=True)
        digest.update(b"path\0")
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # Avoid importing PIL at module import time.  Boogu's supported image type
    # has all three attributes below; hashing mode, size and raw pixels detects
    # both in-place edits and resizing between encoding and trajectory replay.
    if all(hasattr(source, name) for name in ("mode", "size", "tobytes")):
        digest.update(b"pil\0")
        digest.update(str(source.mode).encode("utf-8"))
        digest.update(repr(tuple(source.size)).encode("ascii"))
        digest.update(source.tobytes())
        palette = getattr(source, "palette", None)
        if palette is not None and hasattr(palette, "tobytes"):
            digest.update(palette.tobytes())
        return digest.hexdigest()

    raise TypeError(
        "trajectory condition sources must be PIL images, tensors, or image paths."
    )


def _reference_binding(
    reference_groups: Sequence[Sequence[Any] | None],
    *,
    batch_size: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    validate_single_reference_groups(reference_groups, batch_size=batch_size)
    sources = tuple(group[0] for group in reference_groups if group is not None)
    return (
        tuple(id(source) for source in sources),
        tuple(_source_content_fingerprint(source) for source in sources),
    )


def _validate_condition_tensor_bundle(
    *,
    instruction: tuple[str, ...],
    hbar: torch.Tensor,
    positive: torch.Tensor,
    positive_mask: torch.Tensor,
    negative: torch.Tensor,
    negative_mask: torch.Tensor,
) -> None:
    values = {
        "hbar": hbar,
        "positive": positive,
        "positive_mask": positive_mask,
        "negative": negative,
        "negative_mask": negative_mask,
    }
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
    batch_size = len(instruction)
    if positive.ndim != 3 or negative.ndim != 3:
        raise ValueError("positive and negative conditions must have shape [B, L, D].")
    if positive_mask.ndim != 2 or negative_mask.ndim != 2:
        raise ValueError("condition masks must have shape [B, L].")
    if hbar.ndim != 2:
        raise ValueError("hbar must have shape [B, D].")
    if any(value.shape[0] != batch_size for value in values.values()):
        raise ValueError("encoded condition batch must match the instruction batch.")
    if positive.shape[:2] != positive_mask.shape:
        raise ValueError("positive condition and attention-mask shapes do not match.")
    if negative.shape[:2] != negative_mask.shape:
        raise ValueError("negative condition and attention-mask shapes do not match.")
    if positive.shape[-1] != negative.shape[-1] or hbar.shape[-1] != positive.shape[-1]:
        raise ValueError("encoded condition hidden dimensions do not match.")
    if hbar.dtype != torch.float32:
        raise ValueError("hbar must be computed and stored in float32.")


def _create_encoded_trajectory_conditions(
    *,
    pipeline: Any,
    instruction: str | Sequence[str],
    reference_groups: Sequence[Sequence[Any] | None],
    hbar: torch.Tensor,
    positive: torch.Tensor,
    positive_mask: torch.Tensor,
    negative: torch.Tensor,
    negative_mask: torch.Tensor,
) -> EncodedTrajectoryConditions:
    normalized_instruction = _normalize_instruction_identity(instruction)
    source_identities, source_fingerprints = _reference_binding(
        reference_groups, batch_size=len(normalized_instruction)
    )
    _validate_condition_tensor_bundle(
        instruction=normalized_instruction,
        hbar=hbar,
        positive=positive,
        positive_mask=positive_mask,
        negative=negative,
        negative_mask=negative_mask,
    )
    bundle = object.__new__(EncodedTrajectoryConditions)
    payload = _EncodedConditionsPayload(
        pipeline=pipeline,
        instruction=normalized_instruction,
        source_identities=source_identities,
        source_fingerprints=source_fingerprints,
        hbar=_detached_clone("hbar", hbar),
        positive=_detached_clone("positive", positive),
        positive_mask=_detached_clone("positive_mask", positive_mask),
        negative=_detached_clone("negative", negative),
        negative_mask=_detached_clone("negative_mask", negative_mask),
    )
    with _SEALED_PAYLOAD_LOCK:
        _ENCODED_CONDITIONS_PAYLOADS[bundle] = payload
    return bundle


def resolve_encoded_trajectory_conditions(
    conditions: EncodedTrajectoryConditions,
    *,
    pipeline: Any,
    instruction: str | Sequence[str],
    reference_groups: Sequence[Sequence[Any] | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve a trusted bundle after checking its exact inference binding."""

    if not isinstance(conditions, EncodedTrajectoryConditions):
        raise TypeError(
            "trajectory_conditions must be issued by encode_trajectory_conditions()."
        )
    with _SEALED_PAYLOAD_LOCK:
        payload = _ENCODED_CONDITIONS_PAYLOADS.get(conditions)
    if payload is None:
        raise ValueError("the encoded trajectory conditions are invalid or expired.")
    if payload.pipeline is not pipeline:
        raise ValueError("trajectory conditions are bound to a different pipeline.")
    normalized_instruction = _normalize_instruction_identity(instruction)
    if normalized_instruction != payload.instruction:
        raise ValueError("trajectory conditions are bound to a different instruction.")
    source_identities, source_fingerprints = _reference_binding(
        reference_groups, batch_size=len(normalized_instruction)
    )
    if source_identities != payload.source_identities:
        raise ValueError("trajectory conditions are bound to a different source identity.")
    if source_fingerprints != payload.source_fingerprints:
        raise ValueError("trajectory conditions source content or size has changed.")
    return tuple(
        _detached_clone(name, value)
        for name, value in (
            ("positive", payload.positive),
            ("positive_mask", payload.positive_mask),
            ("negative", payload.negative),
            ("negative_mask", payload.negative_mask),
        )
    )


def _issue_trajectory_capture_context(
    *, pipeline: Any, scheduler: Any, observer: TrajectoryObserver
) -> TrajectoryCaptureContext:
    if not isinstance(observer, TrajectoryObserver):
        raise TypeError("observer must be a TrajectoryObserver instance.")
    context = object.__new__(TrajectoryCaptureContext)
    with _SEALED_PAYLOAD_LOCK:
        _CAPTURE_CONTEXT_PAYLOADS[context] = _CaptureContextPayload(
            pipeline=pipeline,
            scheduler=scheduler,
            observer=observer,
        )
    return context


def _resolve_trajectory_capture_context(
    context: TrajectoryCaptureContext,
    *,
    pipeline: Any,
    scheduler: Any,
) -> TrajectoryObserver:
    if not isinstance(context, TrajectoryCaptureContext):
        raise TypeError("trajectory capture context must be issued internally.")
    with _SEALED_PAYLOAD_LOCK:
        payload = _CAPTURE_CONTEXT_PAYLOADS.get(context)
    if payload is None:
        raise ValueError("trajectory capture context is invalid or expired.")
    if payload.pipeline is not pipeline or payload.scheduler is not scheduler:
        raise ValueError("trajectory capture context is not bound to this pipeline/scheduler.")
    return payload.observer


@dataclass(frozen=True, slots=True)
class TrajectorySnapshot:
    """One pre-scheduler state from an unmodified, teacher-forced trajectory."""

    step_index: int
    num_steps: int
    rho: float
    timestep: torch.Tensor
    z_s: torch.Tensor
    z_k: torch.Tensor
    v_e_k: torch.Tensor
    v_p_k: torch.Tensor
    delta_v_k: torch.Tensor

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TypeError("step_index must be an integer.")
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int):
            raise TypeError("num_steps must be an integer.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")
        if not 0 <= self.step_index < self.num_steps:
            raise ValueError(
                f"step_index={self.step_index} is outside a {self.num_steps}-step schedule."
            )
        if not math.isfinite(float(self.rho)) or not 0.0 <= float(self.rho) <= 1.0:
            raise ValueError("rho must be finite and lie in [0, 1].")

        cloned = {
            name: _detached_clone(name, getattr(self, name))
            for name in (
                "timestep",
                "z_s",
                "z_k",
                "v_e_k",
                "v_p_k",
                "delta_v_k",
            )
        }
        if cloned["timestep"].numel() != 1:
            raise ValueError("timestep must contain exactly one value.")
        state_shape = cloned["z_k"].shape
        if cloned["z_s"].ndim != cloned["z_k"].ndim or cloned["z_s"].shape[:2] != state_shape[:2]:
            raise ValueError(
                "z_s batch/channel dimensions must equal z_k; spatial dimensions "
                "may differ and are resampled by the PRI predictor."
            )
        for name in ("v_e_k", "v_p_k", "delta_v_k"):
            if cloned[name].shape != state_shape:
                raise ValueError(
                    f"{name} shape {tuple(cloned[name].shape)} must equal "
                    f"z_k shape {tuple(state_shape)}."
                )
        for name, value in cloned.items():
            object.__setattr__(self, name, value)

    def clone(self) -> TrajectorySnapshot:
        """Return a deep tensor copy suitable for an untrusted consumer."""

        return TrajectorySnapshot(
            step_index=self.step_index,
            num_steps=self.num_steps,
            rho=self.rho,
            timestep=self.timestep,
            z_s=self.z_s,
            z_k=self.z_k,
            v_e_k=self.v_e_k,
            v_p_k=self.v_p_k,
            delta_v_k=self.delta_v_k,
        )


class TrajectoryObserver:
    """Collect a preregistered subset of baseline trajectory states.

    The optional callback receives its own deep copy.  Mutating that copy can
    neither change the denoising tensors nor corrupt the observer's stored
    snapshots.
    """

    def __init__(
        self,
        selected_steps: Sequence[int],
        *,
        on_snapshot: Callable[[TrajectorySnapshot], None] | None = None,
    ) -> None:
        if isinstance(selected_steps, (str, bytes)):
            raise TypeError("selected_steps must be a non-empty integer sequence.")
        try:
            normalized = tuple(selected_steps)
        except TypeError as error:
            raise TypeError(
                "selected_steps must be a non-empty integer sequence."
            ) from error
        if not normalized:
            raise ValueError("selected_steps must be non-empty.")
        if any(isinstance(step, bool) or not isinstance(step, int) for step in normalized):
            raise TypeError("every selected step must be an integer (not bool).")
        if any(step < 0 for step in normalized):
            raise ValueError("selected steps must be non-negative.")
        if any(left >= right for left, right in pairwise(normalized)):
            raise ValueError("selected_steps must be strictly increasing and unique.")
        if on_snapshot is not None and not callable(on_snapshot):
            raise TypeError("on_snapshot must be callable or None.")

        self._selected_steps = normalized
        self._selected_step_set = frozenset(normalized)
        self._on_snapshot = on_snapshot
        self._snapshots: list[TrajectorySnapshot] = []
        self._captured_steps: set[int] = set()
        self._num_steps: int | None = None

    @property
    def selected_steps(self) -> tuple[int, ...]:
        return self._selected_steps

    @property
    def snapshots(self) -> tuple[TrajectorySnapshot, ...]:
        return tuple(snapshot.clone() for snapshot in self._snapshots)

    def validate_num_steps(self, num_steps: int) -> None:
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")
        if self._selected_steps[-1] >= num_steps:
            raise ValueError(
                f"selected step {self._selected_steps[-1]} is outside a "
                f"{num_steps}-step schedule (maximum index {num_steps - 1})."
            )
        if self._num_steps is not None and self._num_steps != num_steps:
            raise ValueError(
                "a TrajectoryObserver cannot be reused with a different schedule; "
                f"expected {self._num_steps} steps, got {num_steps}."
            )
        self._num_steps = num_steps

    def wants_step(self, step_index: int) -> bool:
        return step_index in self._selected_step_set

    def capture(
        self,
        *,
        step_index: int,
        num_steps: int,
        timestep: torch.Tensor | float,
        z_s: torch.Tensor,
        z_k: torch.Tensor,
        v_e_k: torch.Tensor,
        v_p_k: torch.Tensor,
    ) -> TrajectorySnapshot | None:
        self.validate_num_steps(num_steps)
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise TypeError("step_index must be an integer.")
        if not 0 <= step_index < num_steps:
            raise ValueError(
                f"step_index={step_index} is outside a {num_steps}-step schedule."
            )
        if not self.wants_step(step_index):
            return None
        if step_index in self._captured_steps:
            raise ValueError(f"selected step {step_index} was captured more than once.")

        timestep_tensor = (
            timestep if isinstance(timestep, torch.Tensor) else torch.as_tensor(timestep)
        )
        rho = 0.0 if num_steps == 1 else step_index / (num_steps - 1)
        snapshot = TrajectorySnapshot(
            step_index=step_index,
            num_steps=num_steps,
            rho=rho,
            timestep=timestep_tensor,
            z_s=z_s,
            z_k=z_k,
            v_e_k=v_e_k,
            v_p_k=v_p_k,
            delta_v_k=v_e_k - v_p_k,
        )
        self._snapshots.append(snapshot)
        self._captured_steps.add(step_index)
        if self._on_snapshot is not None:
            self._on_snapshot(snapshot.clone())
        return snapshot.clone()


def _generator_tuple(
    value: torch.Generator | list[torch.Generator] | None,
    *,
    name: str,
) -> tuple[torch.Generator, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Generator):
        return (value,)
    if isinstance(value, list):
        values = tuple(value)
        if not values:
            raise ValueError(f"{name} must not be an empty sequence.")
        if not all(isinstance(item, torch.Generator) for item in values):
            raise TypeError(f"every item in {name} must be a torch.Generator.")
        return values
    raise TypeError(
        f"{name} must be a torch.Generator, a list of generators, or None."
    )


def _validate_generator_batch_size(
    name: str,
    value: torch.Generator | list[torch.Generator],
    *,
    batch_size: int,
) -> None:
    count = len(_generator_tuple(value, name=name))
    if count not in (1, batch_size):
        raise ValueError(
            f"{name} must contain one or batch_size={batch_size} generators; "
            f"one generator denotes a single deterministic stream; got {count}."
        )


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number (not bool).")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def validate_independent_generators(
    generator: torch.Generator | list[torch.Generator] | None,
    reference_generator: torch.Generator | list[torch.Generator] | None,
) -> None:
    """Reject shared RNG objects while preserving the legacy all-None path."""

    noise_generators = _generator_tuple(generator, name="generator")
    reference_generators = _generator_tuple(
        reference_generator, name="reference_generator"
    )
    if {id(item) for item in noise_generators} & {
        id(item) for item in reference_generators
    }:
        raise ValueError(
            "generator and reference_generator must use independent RNG objects."
        )


def pipeline_has_cache_acceleration(pipeline: Any) -> bool:
    """Detect the acceleration flags used by Boogu and Cache-DiT."""

    transformer = getattr(pipeline, "transformer", None)
    checks = (
        getattr(pipeline, "_is_cached", False),
        getattr(type(pipeline), "_is_cached", False),
        getattr(pipeline, "enable_taylorseer", False),
        getattr(transformer, "_is_cached", False),
        getattr(transformer, "enable_taylorseer", False),
        getattr(transformer, "enable_taylorseer_for_all_layers", False),
        getattr(transformer, "enable_teacache", False),
        getattr(transformer, "enable_teacache_for_all_layers", False),
    )
    return any(bool(value) for value in checks)


def pipeline_has_loaded_lora(pipeline: Any) -> bool:
    """Conservatively detect loaded PEFT/LoRA state on pipeline components."""

    components = (
        pipeline,
        getattr(pipeline, "transformer", None),
        getattr(pipeline, "mllm", None),
        getattr(pipeline, "prompt_embedding", None),
    )
    for component in components:
        if component is None:
            continue
        if bool(getattr(component, "_hf_peft_config_loaded", False)):
            return True
        peft_config = getattr(component, "peft_config", None)
        if peft_config is not None and bool(peft_config):
            return True
    return False


def _validate_explicit_empty_negative_instruction(
    negative_instruction: str | Sequence[str] | None,
    *,
    batch_size: int,
) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(negative_instruction, str):
        if negative_instruction != "":
            raise ValueError(
                "negative_instruction must be the explicit empty string ''."
            )
        return
    if isinstance(negative_instruction, Sequence) and not isinstance(
        negative_instruction, (str, bytes)
    ):
        values = tuple(negative_instruction)
        if len(values) != batch_size or any(value != "" for value in values):
            raise ValueError(
                "negative_instruction batch must match batch_size and contain "
                "only explicit empty strings."
            )
        return
    raise ValueError(
        "negative_instruction must be explicitly supplied as '' or a matching "
        "batch of empty strings."
    )


def validate_trajectory_capture_configuration(
    *,
    observer: TrajectoryObserver,
    generator: torch.Generator | list[torch.Generator] | None,
    reference_generator: torch.Generator | list[torch.Generator] | None,
    batch_size: int,
    num_images_per_instruction: int,
    negative_instruction: str | Sequence[str] | None,
    instruction_embeds: torch.Tensor | None,
    instruction_attention_mask: torch.Tensor | None,
    negative_instruction_embeds: torch.Tensor | None,
    negative_instruction_attention_mask: torch.Tensor | None,
    empty_instruction_embeds: torch.Tensor | None,
    empty_instruction_attention_mask: torch.Tensor | None,
    latents: torch.Tensor | None,
    trajectory_conditions: EncodedTrajectoryConditions | None,
    text_guidance_scale: float,
    image_guidance_scale: float,
    empty_instruction_guidance_scale: float,
    cfg_range: Sequence[float],
    use_rewrite_text_instruction: bool,
    use_input_images_4_neg_instruct: bool,
    use_input_images_4_empty_instruct: bool,
    use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide: bool,
    use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide: bool,
    use_boosted_orthogonal_guidance: bool,
    attention_kwargs: Mapping[str, Any] | None,
    prompt_tuning_enabled: bool,
    lora_enabled: bool,
    acceleration_enabled: bool,
    scheduler: Any,
    timesteps: Sequence[int] | None,
) -> None:
    """Fail closed unless capture follows the preregistered baseline path."""

    if not isinstance(observer, TrajectoryObserver):
        raise TypeError("observer must be a TrajectoryObserver instance.")
    if generator is None:
        raise ValueError("generator must be explicitly supplied for trajectory capture.")
    if reference_generator is None:
        raise ValueError(
            "reference_generator must be explicitly supplied for trajectory capture."
        )
    if (
        isinstance(num_images_per_instruction, bool)
        or not isinstance(num_images_per_instruction, int)
    ):
        raise TypeError("num_images_per_instruction must be an integer (not bool).")
    if num_images_per_instruction != 1:
        raise ValueError("num_images_per_instruction must equal 1 during capture.")
    raw_conditions = (
        instruction_embeds,
        instruction_attention_mask,
        negative_instruction_embeds,
        negative_instruction_attention_mask,
        empty_instruction_embeds,
        empty_instruction_attention_mask,
    )
    if any(value is not None for value in raw_conditions):
        raise ValueError(
            "raw precomputed condition tensors are forbidden during trajectory "
            "capture; use trajectory_conditions from the trusted encoder."
        )
    if latents is not None:
        raise ValueError(
            "externally supplied latents are not accepted during trajectory capture."
        )
    if trajectory_conditions is not None and not isinstance(
        trajectory_conditions, EncodedTrajectoryConditions
    ):
        raise TypeError(
            "trajectory_conditions must be issued by encode_trajectory_conditions()."
        )
    validate_independent_generators(generator, reference_generator)
    _validate_generator_batch_size("generator", generator, batch_size=batch_size)
    _validate_generator_batch_size(
        "reference_generator", reference_generator, batch_size=batch_size
    )
    _validate_explicit_empty_negative_instruction(
        negative_instruction, batch_size=batch_size
    )

    for name, value in (
        ("use_rewrite_text_instruction", use_rewrite_text_instruction),
        ("use_input_images_4_neg_instruct", use_input_images_4_neg_instruct),
        ("use_input_images_4_empty_instruct", use_input_images_4_empty_instruct),
        (
            "use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide",
            use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide,
        ),
        (
            "use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide",
            use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide,
        ),
        ("use_boosted_orthogonal_guidance", use_boosted_orthogonal_guidance),
        ("prompt_tuning_enabled", prompt_tuning_enabled),
        ("lora_enabled", lora_enabled),
        ("acceleration_enabled", acceleration_enabled),
    ):
        _require_bool(name, value)
    if attention_kwargs is not None and not isinstance(attention_kwargs, Mapping):
        raise TypeError("attention_kwargs must be a mapping or None.")

    validate_trajectory_processing_configuration(
        observer=observer,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        empty_instruction_guidance_scale=empty_instruction_guidance_scale,
        cfg_range=cfg_range,
        use_input_images_4_empty_instruct=use_input_images_4_empty_instruct,
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=(
            use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide
        ),
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=(
            use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide
        ),
        use_boosted_orthogonal_guidance=use_boosted_orthogonal_guidance,
        acceleration_enabled=acceleration_enabled,
        scheduler=scheduler,
        timesteps=timesteps,
    )

    if use_rewrite_text_instruction:
        raise ValueError("instruction rewriting must be disabled during trajectory capture.")
    if use_input_images_4_neg_instruct:
        raise ValueError(
            "the negative CFG branch must not receive a reference image."
        )
    if attention_kwargs is not None and len(attention_kwargs) != 0:
        raise ValueError("attention_kwargs must be empty during trajectory capture.")
    if prompt_tuning_enabled:
        raise ValueError("Prompt Tuning must be disabled during trajectory capture.")
    if lora_enabled:
        raise ValueError("LoRA adapters must be disabled during trajectory capture.")


def validate_trajectory_processing_configuration(
    *,
    observer: TrajectoryObserver,
    text_guidance_scale: float,
    image_guidance_scale: float,
    empty_instruction_guidance_scale: float,
    cfg_range: Sequence[float],
    use_input_images_4_empty_instruct: bool,
    use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide: bool,
    use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide: bool,
    use_boosted_orthogonal_guidance: bool,
    acceleration_enabled: bool,
    scheduler: Any,
    timesteps: Sequence[int] | None,
) -> None:
    """Recheck controls available inside ``processing`` against bypasses."""

    if not isinstance(observer, TrajectoryObserver):
        raise TypeError("observer must be a TrajectoryObserver instance.")
    for name, value in (
        ("use_input_images_4_empty_instruct", use_input_images_4_empty_instruct),
        (
            "use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide",
            use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide,
        ),
        (
            "use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide",
            use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide,
        ),
        ("use_boosted_orthogonal_guidance", use_boosted_orthogonal_guidance),
        ("acceleration_enabled", acceleration_enabled),
    ):
        _require_bool(name, value)

    normalized_text_scale = _finite_real(
        "text_guidance_scale", text_guidance_scale
    )
    normalized_image_scale = _finite_real(
        "image_guidance_scale", image_guidance_scale
    )
    normalized_empty_scale = _finite_real(
        "empty_instruction_guidance_scale", empty_instruction_guidance_scale
    )
    if normalized_text_scale <= 1.0:
        raise ValueError(
            "text_guidance_scale must be finite and greater than 1 for standard text CFG."
        )
    if normalized_image_scale != 1.0:
        raise ValueError("image_guidance_scale must equal 1 for trajectory capture.")
    if normalized_empty_scale != 0.0:
        raise ValueError(
            "empty_instruction_guidance_scale must equal 0 for trajectory capture."
        )
    if not isinstance(cfg_range, Sequence) or isinstance(cfg_range, (str, bytes)):
        raise TypeError("cfg_range must contain two finite real numbers.")
    if len(cfg_range) != 2:
        raise ValueError("cfg_range must contain exactly two finite real numbers.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in cfg_range):
        raise TypeError("cfg_range must contain finite real numbers (not bool).")
    normalized_cfg_range = tuple(float(value) for value in cfg_range)
    if not all(math.isfinite(value) for value in normalized_cfg_range):
        raise ValueError("cfg_range values must be finite.")
    if normalized_cfg_range != (0.0, 1.0):
        raise ValueError("cfg_range must equal (0, 1) for full-trajectory CFG.")
    if (
        use_input_images_4_empty_instruct
        or use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide
        or use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide
    ):
        raise ValueError("double-guidance options must be disabled during capture.")
    if use_boosted_orthogonal_guidance:
        raise ValueError("BOG must be disabled during trajectory capture.")
    if acceleration_enabled:
        raise ValueError(
            "TeaCache, TaylorSeer, Cache-DiT, and other acceleration must be disabled."
        )
    validate_trajectory_scheduler(scheduler, timesteps=timesteps)


def validate_trajectory_scheduler(
    scheduler: Any,
    *,
    timesteps: Sequence[int] | None,
) -> None:
    """Require the exact preregistered Boogu FlowMatch Euler schedule."""

    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler,
    )

    if type(scheduler) is not FlowMatchEulerDiscreteScheduler:
        raise TypeError(
            "trajectory capture requires the in-project "
            "FlowMatchEulerDiscreteScheduler exactly."
        )
    if timesteps is not None:
        raise ValueError("custom timesteps are forbidden during trajectory capture.")
    config = scheduler.config
    required = (
        ("do_shift", True),
        ("dynamic_time_shift", False),
        ("time_shift_version", "v1"),
        ("seq_len", 4096),
    )
    for name, expected in required:
        actual = getattr(config, name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(
                f"scheduler {name} must equal {expected!r} during trajectory capture; "
                f"got {actual!r}."
            )


def validate_single_reference_groups(
    reference_groups: Sequence[Sequence[Any] | None],
    *,
    batch_size: int,
) -> None:
    """Require exactly one non-null source image in each normalized sample."""

    if not isinstance(reference_groups, Sequence) or isinstance(
        reference_groups, (str, bytes)
    ):
        raise TypeError("trajectory capture requires exactly one reference per sample.")
    if len(reference_groups) != batch_size:
        raise ValueError(
            "trajectory capture requires exactly one reference per sample and a "
            "reference batch matching batch_size."
        )
    for group in reference_groups:
        if (
            not isinstance(group, Sequence)
            or isinstance(group, (str, bytes))
            or len(group) != 1
            or group[0] is None
        ):
            raise ValueError(
                "trajectory capture requires exactly one reference per sample."
            )


def select_reference_generator(
    reference_generator: torch.Generator | list[torch.Generator] | None,
    *,
    sample_index: int,
    batch_size: int,
) -> torch.Generator | None:
    """Select the per-sample reference RNG without advancing any noise RNG."""

    generators = _generator_tuple(
        reference_generator, name="reference_generator"
    )
    if not generators:
        return None
    if len(generators) == 1:
        return generators[0]
    if len(generators) != batch_size:
        raise ValueError(
            "a reference_generator sequence must contain one generator per source sample."
        )
    return generators[sample_index]


__all__ = [
    "EncodedTrajectoryConditions",
    "TrajectoryCaptureContext",
    "TrajectoryObserver",
    "TrajectorySnapshot",
    "pipeline_has_cache_acceleration",
    "pipeline_has_loaded_lora",
    "resolve_encoded_trajectory_conditions",
    "select_reference_generator",
    "validate_independent_generators",
    "validate_single_reference_groups",
    "validate_trajectory_capture_configuration",
    "validate_trajectory_processing_configuration",
    "validate_trajectory_scheduler",
]
