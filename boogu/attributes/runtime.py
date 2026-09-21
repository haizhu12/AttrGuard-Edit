"""Opt-in Boogu adapter. Manual constraints are mechanism diagnostics only."""

from dataclasses import dataclass, field

import torch

from .checks import LocalizedAttributeObserver, TargetColorGoal
from .constraints import AttributeConstraints
from .control import BudgetController, ControlConfig
from .observers import AttributeObserver


@dataclass
class AttributeControlRequest:
    constraints: AttributeConstraints | None
    config: ControlConfig = field(default_factory=ControlConfig)
    operation: str = "recolor"
    observer_profile: str = "geometry"
    target_color: TargetColorGoal | None = None
    diagnostics: dict = field(default_factory=dict, init=False)


def _validate_observer_options(request):
    if request.observer_profile not in ("legacy", "localized", "geometry"):
        raise ValueError("observer_profile must be legacy, localized or geometry")
    if request.target_color is not None:
        if not isinstance(request.target_color, TargetColorGoal):
            raise TypeError("target_color must be an explicit TargetColorGoal")
        if request.operation != "recolor":
            raise ValueError("target color checks only support recolor operations")
        spec = request.constraints
        if not isinstance(spec, AttributeConstraints):
            raise TypeError("target color requires fixed source constraints")
        mask = request.target_color.mask.to(spec.edit)
        if mask.shape != spec.edit.shape or torch.any((mask > 0) & (spec.edit <= 0)):
            raise ValueError("target color mask must lie inside edit support")


def validate_attribute_request(
    pipeline, request, *, far=False, trajectory=False, acceleration=False, boosted=False
):
    if not isinstance(request, AttributeControlRequest):
        raise TypeError("attribute_request must be AttributeControlRequest")
    if not isinstance(request.config, ControlConfig):
        raise TypeError("attribute config must be ControlConfig")
    if request.config.beta == 0:
        request.diagnostics = {"mode": "zero_budget_bypass", "records": []}
        return False
    _validate_observer_options(request)
    if torch.is_inference_mode_enabled():
        raise ValueError(
            "active attribute control cannot run inside torch.inference_mode; use no_grad"
        )
    if far or trajectory or acceleration or boosted:
        raise ValueError(
            "attribute control cannot combine FAR, trajectory capture or acceleration/BOG"
        )
    for flag in ("enable_sequential_cpu_offload_flag", "enable_group_offload_flag"):
        if getattr(pipeline, flag, False):
            raise ValueError(
                "attribute control only supports model CPU offload or resident models"
            )
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler,
    )

    if not isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
        raise TypeError("attribute control requires the Boogu Euler scheduler")
    if request.operation not in ("recolor", "replace_text", "remove_text"):
        raise ValueError("unsupported attribute edit operation")
    if not isinstance(request.constraints, AttributeConstraints):
        raise TypeError("active attribute control requires fixed source constraints")
    return True


class _BooguController:
    def __init__(self, pipeline, controller, request, grad_flags):
        self.pipeline, self.controller, self.request = pipeline, controller, request
        self.grad_flags = grad_flags

    @property
    def config(self):
        return self.controller.config

    def step(self, *args, **kwargs):
        try:
            return self.controller.step(*args, **kwargs)
        finally:
            # All backward work has completed before moving VAE weights.
            if getattr(self.pipeline, "enable_model_cpu_offload_flag", False):
                self.pipeline.vae.to("cpu")

    def close(self):
        for parameter, flag in self.grad_flags:
            parameter.requires_grad_(flag)
        self.request.diagnostics.update(
            records=self.controller.records,
            spent=self.controller.spent,
            initial_budget=self.controller.initial_budget,
            observer_items=self.controller.observer.names,
            proxy_tolerances=self.controller.observer.tolerances.tolist(),
        )


def prepare_attribute_controller(pipeline, request, latents, input_images=None):
    if request.config.beta == 0:
        request.diagnostics = {"mode": "zero_budget_bypass", "records": []}
        return None
    _validate_observer_options(request)
    spec = request.constraints
    if not isinstance(spec, AttributeConstraints):
        raise TypeError("missing fixed attribute constraints")
    expected = (
        latents.shape[-2] * pipeline.vae_scale_factor,
        latents.shape[-1] * pipeline.vae_scale_factor,
    )
    if spec.source.shape[-2:] != expected or latents.shape[0] != 1:
        raise ValueError("source masks and generation must use one identical canvas")
    if input_images is not None:
        if len(input_images) != 1 or len(input_images[0]) != 1:
            raise ValueError("attribute control requires exactly one reference image")
        actual = (
            pipeline.image_processor.preprocess(
                input_images[0][0],
                height=expected[0],
                width=expected[1],
                max_pixels=expected[0] * expected[1],
                max_side_length=max(expected),
            ).float()
            / 2
            + 0.5
        )
        if actual.shape != spec.source.shape or not torch.allclose(
            actual.cpu(), spec.source.cpu(), atol=1e-5, rtol=0
        ):
            raise ValueError(
                "attribute source differs from the actual preprocessed reference"
            )
    device = latents.device
    prepared = AttributeConstraints(
        spec.source.to(device),
        spec.keep.to(device),
        spec.edit.to(device),
        spec.scales,
        spec.tolerances,
        spec.provenance,
        reference_valid=spec.reference_valid.to(device),
    )
    vae = pipeline.vae
    grad_flags = [(p, p.requires_grad) for p in vae.parameters()]
    for p, _ in grad_flags:
        p.requires_grad_(False)

    def decode(z):
        value = z.to(dtype=vae.dtype)
        if vae.config.scaling_factor is not None:
            value = value / vae.config.scaling_factor
        if vae.config.shift_factor is not None:
            value = value + vae.config.shift_factor
        rgb = vae.decode(value, return_dict=False)[0].float() / 2 + 0.5
        # Preserve invalid observations for the controller's numeric fallback.
        # A plain clamp would silently convert infinity to a valid black/white pixel.
        return torch.where(torch.isfinite(rgb), rgb.clamp(0, 1), rgb)

    try:
        with torch.no_grad():
            mode = vae.encode(
                (prepared.source * 2 - 1).to(dtype=vae.dtype)
            ).latent_dist.mode()
            if vae.config.shift_factor is not None:
                mode = mode - vae.config.shift_factor
            if vae.config.scaling_factor is not None:
                mode = mode * vae.config.scaling_factor
            roundtrip = decode(mode)
            observer_type = (
                LocalizedAttributeObserver
                if request.observer_profile == "localized"
                else AttributeObserver
            )
            options = (
                {"structure_mode": "geometry"}
                if request.observer_profile == "geometry"
                else {}
            )
            observer = observer_type(prepared, roundtrip=roundtrip, **options)
            if request.target_color is not None:
                # Snapshot again at generation start, like source constraints.
                # Mutating the caller's request cannot weaken an active goal.
                fixed_goal = TargetColorGoal(
                    request.target_color.mask,
                    request.target_color.target,
                )
                observer.goal_errors = fixed_goal.errors
        controller = BudgetController(
            observer,
            decode,
            source_latent=mode,
            initial_latent=latents,
            config=request.config,
        )
        request.diagnostics = {
            "mode": "external_fixed_attributes",
            "provenance": spec.provenance,
            "semantic_predictor_used": False,
            "source_canvas": list(expected),
            "observer_profile": request.observer_profile,
            "target_check": "rgb_direction_proxy"
            if request.target_color is not None
            else "unavailable",
            "target_guard_enforced": request.target_color is not None
            and request.config.feedback == "rollout",
            "unavailable_checks": getattr(observer, "unavailable_checks", []),
            "observer_coverage": observer.coverage,
        }
        if getattr(pipeline, "enable_model_cpu_offload_flag", False):
            vae.to("cpu")
        return _BooguController(pipeline, controller, request, grad_flags)
    except BaseException:
        for parameter, flag in grad_flags:
            parameter.requires_grad_(flag)
        raise


def make_pipeline_lookahead(
    pipeline,
    controller,
    *,
    dtype,
    instruction_embeds,
    negative_instruction_embeds,
    instruction_attention_mask,
    negative_instruction_attention_mask,
    freqs_cis,
    ref_latents,
):
    """Build an isolated no-cache text-CFG field; original sampling is unchanged."""
    from .rollout import EulerLookahead

    if (
        pipeline.image_guidance_scale > 1.0
        or pipeline.empty_instruction_guidance_scale != 0.0
    ):
        raise ValueError("attribute lookahead currently supports only text-only CFG")
    inner = getattr(controller, "controller", controller)
    count = len(pipeline.scheduler.timesteps)

    @torch.no_grad()
    def predict(state, index):
        # Called after local proposal autograd has completed. Release VAE before
        # loading the DiT under model-offload; no graph crosses this boundary.
        if getattr(pipeline, "enable_model_cpu_offload_flag", False):
            pipeline.vae.to("cpu")
        arguments = {
            "t": pipeline.scheduler.timesteps[index],
            "latents": state,
            "freqs_cis": freqs_cis,
            "ref_image_hidden_states": ref_latents,
        }
        positive = pipeline.predict(
            **arguments,
            instruction_embeds=instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
        )
        scale = (
            pipeline.text_guidance_scale
            if pipeline.cfg_range[0] <= index / count <= pipeline.cfg_range[1]
            else 1.0
        )
        if scale > 1.0:
            negative = pipeline.predict(
                **arguments,
                instruction_embeds=negative_instruction_embeds,
                instruction_attention_mask=negative_instruction_attention_mask,
            )
            # Keep the main pipeline's operation order and low-precision math.
            return positive + (scale - 1) * (positive - negative)
        return positive

    return EulerLookahead(
        pipeline.scheduler,
        predict,
        inner.decode,
        dtype,
        horizon=inner.config.lookahead_steps,
    )
