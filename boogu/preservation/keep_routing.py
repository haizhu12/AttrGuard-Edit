"""Inference-only routing of fully known pixel-keep native latent patches.

The source path is analytic noise-to-image interpolation, not inversion. Source
memory substitutes block inputs before attention; it is not a direct KV cache.
No weights, native transformer code, or reference sampling are changed.
"""

import math
from numbers import Real

import torch

from .adapters import SourceConditionedTransformer
from .conditioning import SourceCondition, _finite_float

KEEP_MODES = ("off", "project", "source-memory")
TIME_FORMULA = "source_at(t) = (1 - t) * initial_noise + t * source_latent; t=0 noise, t=1 image"
MEMORY_STRATEGY = "unadapted base with empty instruction; replace eligible generated-tail inputs at every double/single-stream block before attention"


def validate_keep_mode(mode, *, guidance=1.0, negative_instruction=""):
    """Validate the public opt-in before model loading or denoising."""
    if mode not in KEEP_MODES:
        raise ValueError(f"preservation_keep_mode must be one of {KEEP_MODES}")
    if mode == "source-memory" and (
        isinstance(guidance, bool) or guidance != 1.0 or negative_instruction != ""
    ):
        raise ValueError("source-memory requires text_guidance_scale=1 and empty negative_instruction")


class KeepRouting:
    """Private source/noise snapshots and an exact, unresized native patch mask.

    A patch is eligible only when pixel K=1, M=U=0 and every attribute M=0.
    Attribute-only keeps do not imply pixel identity. CPU conditions may be
    explicitly transferred to the initial noise device, as in the adapter path.
    """

    def __init__(self, condition, initial_noise):
        if not isinstance(condition, SourceCondition):
            raise TypeError("condition must be a SourceCondition")
        shape, p = condition.latent_shape, condition.patch_size
        if (
            not isinstance(shape, tuple) or len(shape) != 4
            or any(type(d) is not int or d <= 0 for d in shape) or shape[0] != 1
        ):
            raise ValueError("latent_shape must describe [1,C,H,W]")
        if type(p) is not int or p <= 0 or shape[-2] % p or shape[-1] % p:
            raise ValueError("patch_size must form an integral native latent grid")
        _, channels, height, width = shape
        count = (height // p) * (width // p)
        for name, value in (("source_tokens", condition.source_tokens),
                            ("roles", condition.roles), ("initial_noise", initial_noise)):
            _finite_float(value, name)
        if condition.source_tokens.shape != (1, count, p * p * channels):
            raise ValueError("source_tokens do not match the native latent patch grid")
        if condition.roles.shape != (1, count, 15):
            raise ValueError("roles must have shape [1,N,15]")
        if tuple(initial_noise.shape) != shape:
            raise ValueError("initial_noise must match latent_shape")
        roles = condition.roles.detach().to(dtype=torch.float64).reshape(1, count, 5, 3)
        if ((roles < 0) | (roles > 1)).any().item() or not torch.allclose(
            roles.sum(-1), torch.ones_like(roles[..., 0]), atol=1e-5, rtol=0
        ):
            raise ValueError("roles must contain K/M/U probabilities summing to one")
        # Evaluate strict eligibility in the original precision, before any cast.
        pixel = roles[:, :, 4]
        eligible = (pixel[..., 0] == 1) & (pixel[..., 1] == 0) & (pixel[..., 2] == 0)
        eligible &= (roles[..., 1] == 0).all(-1)
        dtype = torch.promote_types(torch.float32, torch.promote_types(
            condition.source_tokens.dtype, initial_noise.dtype
        ))
        source = condition.source_tokens.detach().to(device=initial_noise.device, dtype=dtype)
        self._source = source.reshape(1, height // p, width // p, p, p, channels).permute(
            0, 5, 1, 3, 2, 4
        ).reshape(shape).clone()
        self._noise = initial_noise.detach().to(dtype=dtype).clone()
        self._token_mask = eligible.to(device=initial_noise.device).clone()
        self._latent_mask = self._token_mask.reshape(1, 1, height // p, width // p).repeat_interleave(
            p, dim=2
        ).repeat_interleave(p, dim=3).clone()
        self.latent_shape = shape
        self.patch_size = p
        self.token_count = count
        self.eligible_token_count = int(eligible.sum().item())

    @property
    def token_mask(self):
        return self._token_mask.clone()

    def source_at(self, time):
        if isinstance(time, torch.Tensor):
            _finite_float(time, "time")
            if time.numel() != 1:
                raise ValueError("time must contain one value")
            time = time.detach().item()
        if isinstance(time, bool) or not isinstance(time, Real):
            raise TypeError("time must be a finite scalar in [0,1]")
        if not math.isfinite(time) or not 0 <= time <= 1:
            raise ValueError("time must be a finite scalar in [0,1]")
        return (1 - time) * self._noise + time * self._source

    def project(self, current, time):
        _finite_float(current, "current")
        if tuple(current.shape) != self.latent_shape or current.device != self._source.device:
            raise ValueError("current must match the latent shape and initial noise device")
        source = self.source_at(time).to(dtype=current.dtype)
        return torch.where(self._latent_mask, source, current)

    def metadata(self):
        return {
            "eligible_token_count": self.eligible_token_count,
            "token_count": self.token_count,
            "eligible_token_fraction": self.eligible_token_count / self.token_count,
            "time_formula": TIME_FORMULA,
            "eligibility": "pixel K=1, M=U=0; all attribute M=0",
            "warning": "Inference-only; analytic source path is not exact inversion, block-input routing is not direct KV caching, and preservation quality is unestablished.",
        }


class SourceMemoryRouting:
    """Owned reversible pre-hooks with a fresh source cache for each edit call.

    Call capture with the unadapted source path and compact empty-instruction
    features, then edit once. Only protected generated tokens are cached. The
    wrapper's own output hooks remain installed and inactive during capture.
    """

    def __init__(self, model, routing):
        self.model = model
        self.routing = routing
        self._handles = []
        self._cache = {}
        self._phase = None
        self._visits = 0
        self._text_count = None
        self._layers = []
        self._step_signature = None
        self._indices = routing._token_mask[0].nonzero().flatten()

    def _check_model(self):
        if torch.is_grad_enabled():
            raise RuntimeError("source-memory is inference-only and requires no_grad")
        if not isinstance(self.model, SourceConditionedTransformer) or self.model._closed:
            raise ValueError("source-memory requires an open SourceConditionedTransformer")
        base = self.model.base
        if getattr(base, "_source_memory_owner", self) is not self:
            raise RuntimeError("source-memory routing is already installed on this base")
        SourceConditionedTransformer._check_supported(base)
        if any(getattr(module, flag, False) for module in base.modules() for flag in (
            "enable_cache_dit_caching", "enable_cache_dit_caching_for_all_layers",
        )):
            raise RuntimeError("source-memory does not support cache-dit caching")
        if self.model.training or any(module.training for module in base.modules()):
            raise RuntimeError("source-memory requires an eval model")
        if self.model._context.get() is not None:
            raise RuntimeError("source-memory cannot run inside an active adapter invocation")
        if (base.config.in_channels != self.routing.latent_shape[1]
                or base.config.patch_size != self.routing.patch_size):
            raise ValueError("source-memory grid must match the native architecture")
        layers = []
        for kind in ("double", "single"):
            sequence = getattr(base, f"{kind}_stream_layers", None)
            if sequence is None:
                raise ValueError("source-memory requires native double and single stream layers")
            layers.extend((kind, layer) for layer in sequence)
        if not layers or len({id(layer) for _, layer in layers}) != len(layers):
            raise ValueError("source-memory requires distinct native layers")
        for name, expected in (("num_layers", len(layers)),
                               ("num_double_stream_layers", len(base.double_stream_layers))):
            if getattr(base.config, name, expected) != expected:
                raise ValueError("source-memory native layer count does not match config")
        if self._layers and layers != self._layers:
            raise RuntimeError("source-memory native layers changed during routing")
        return layers

    def __enter__(self):
        if self._handles:
            raise RuntimeError("source-memory context is already active")
        self._layers = self._check_model()
        try:
            self.model.base._source_memory_owner = self
            for index, (kind, layer) in enumerate(self._layers):
                self._handles.append(layer.register_forward_pre_hook(
                    self._hook(index, kind), with_kwargs=True
                ))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cache.clear()
        self._step_signature = None
        self._phase = None
        self._visits = 0
        if getattr(self.model.base, "_source_memory_owner", None) is self:
            delattr(self.model.base, "_source_memory_owner")

    def _hook(self, index, kind):
        def route(module, args, kwargs):
            if self._phase not in ("source", "edit") or index != self._visits:
                raise RuntimeError("source-memory native layer visit order changed")
            key = "img_hidden_states" if kind == "double" else "hidden_states"
            named = key in kwargs
            if named and args or not named and not args:
                raise ValueError("source-memory requires one native block input")
            hidden = kwargs[key] if named else args[0]
            count = self.routing.token_count
            length = 2 * count + (self._text_count if kind == "single" else 0)
            if (not isinstance(hidden, torch.Tensor) or hidden.shape != (
                    1, length, self.model.base.config.hidden_size)
                    or hidden.device != self.routing._source.device
                    or not hidden.is_floating_point()):
                raise ValueError("source-memory native token layout must be compact text + reference + generated")
            self._visits += 1
            indices = self._indices + length - count
            if self._phase == "source":
                self._cache[index] = hidden[:, indices].detach().clone()
                return None
            cached = self._cache.get(index)
            if cached is None or cached.dtype != hidden.dtype or cached.device != hidden.device:
                raise RuntimeError("source-memory cache is missing or incompatible")
            hidden = hidden.clone()
            hidden[:, indices] = cached
            if named:
                return args, {**kwargs, key: hidden}
            return (hidden, *args[1:]), kwargs
        return route

    def _forward(self, phase, kwargs):
        self._check_model()
        if not self._handles or self._phase is not None:
            raise RuntimeError("source-memory forward requires an idle active context")
        # Reuse the native layout validator, without activating adapter hooks.
        self.model._snapshot(kwargs.get("preservation", self._condition), kwargs)
        self._text_count = kwargs["instruction_attention_mask"].shape[1]
        self._phase, self._visits = phase, 0
        try:
            result = (self.model.base if phase == "source" else self.model)(**kwargs)
            if self._visits != len(self._layers):
                raise RuntimeError("source-memory native layer visits are incomplete")
            return result
        finally:
            self._phase = None

    @property
    def _condition(self):
        # Layout validation only; strict permissions were snapshotted by routing.
        count, p = self.routing.token_count, self.routing.patch_size
        source = self.routing._source
        _, channels, height, width = source.shape
        tokens = source.reshape(1, channels, height // p, p, width // p, p).permute(
            0, 2, 4, 3, 5, 1
        ).reshape(1, count, p * p * channels)
        roles = source.new_zeros(1, count, 15)
        roles[..., 2::3] = 1
        return SourceCondition(tokens, roles, self.routing.latent_shape, p)

    def capture(self, **native_kwargs):
        self._cache.clear()
        self._step_signature = None
        try:
            if "preservation" in native_kwargs:
                raise ValueError("source-memory source pass must bypass adapters")
            self._forward("source", native_kwargs)
            self._step_signature = (
                native_kwargs["timestep"].detach().clone(),
                native_kwargs["ref_image_hidden_states"][0][0].detach().clone(),
            )
        except BaseException:
            self._cache.clear()
            raise

    def edit(self, **kwargs):
        try:
            if len(self._cache) != len(self._layers):
                raise RuntimeError("source-memory requires a fresh complete source pass")
            time, reference = self._step_signature
            if not torch.equal(time, kwargs.get("timestep")):
                raise ValueError("source-memory source/edit timestep must match")
            refs = kwargs.get("ref_image_hidden_states")
            if (not isinstance(refs, (list, tuple)) or len(refs) != 1
                    or not isinstance(refs[0], (list, tuple)) or len(refs[0]) != 1
                    or not isinstance(refs[0][0], torch.Tensor)
                    or not torch.equal(reference, refs[0][0])):
                raise ValueError("source-memory source/edit reference must match")
            return self._forward("edit", kwargs)
        finally:
            self._cache.clear()
            self._step_signature = None
