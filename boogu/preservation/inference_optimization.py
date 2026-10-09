"""Bounded, source-only latent interventions through a frozen FP32 decoder.

This is a stop-gradient surrogate endpoint objective, not differentiation through
the denoising trajectory or a guarantee of semantic preservation in the output.
"""

import copy
import math
from dataclasses import asdict, dataclass

import torch

from .objectives import preservation_loss
from .edit_anchoring import edit_anchor_loss


LIMITATIONS = (
    "Approximate stop-gradient clean-endpoint intervention; no DiT backward or trajectory differentiation.",
    "Keep features are RGB/chromaticity/derivative proxies, not identity, OCR or semantic metrics.",
    "Edit anchoring preserves the current prediction in M; it does not guarantee the requested edit.",
    "Surrogate objective acceptance does not establish improvement of the final decoded image.",
)


def validate_optimization(mode, *, keep_mode="off", guidance=1.0, method="source-conditioned", edit_anchor="rgb"):
    """Reject unsupported combinations before model loading or sampler work."""
    if mode not in ("off", "latent"):
        raise ValueError("preservation_optimization must be off or latent")
    if edit_anchor not in ("rgb", "attribute"):
        raise ValueError("preservation_optimization edit_anchor must be rgb or attribute")
    if mode == "off" and edit_anchor != "rgb":
        raise ValueError("preservation_optimization latent is required for attribute edit_anchor")
    if mode == "latent" and (
        keep_mode != "off" or isinstance(guidance, bool) or guidance != 1.0
        or method != "source-conditioned"
    ):
        raise ValueError(
            "latent preservation_optimization requires source-conditioned, CFG=1 and preservation_keep_mode=off"
        )


@dataclass(frozen=True)
class InferenceOptimizationConfig:
    """Fixed prototype defaults; mode and edit-anchor representation are exposed."""

    mode: str = "latent"
    edit_anchor: str = "rgb"
    start_time: float = 0.60
    end_time: float = 0.95
    interval: int = 5
    inner_steps: int = 4
    step_rms: float = 0.03
    max_delta_rms: float = 0.08
    edit_anchor_weight: float = 1.0
    trust_weight: float = 0.1
    backtracking_steps: int = 4
    gradient_epsilon: float = 1e-12

    def __post_init__(self):
        validate_optimization(self.mode, edit_anchor=self.edit_anchor)
        for name in ("interval", "inner_steps", "backtracking_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("start_time", "end_time", "step_rms", "max_delta_rms",
                     "edit_anchor_weight", "trust_weight", "gradient_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
        if not 0 <= self.start_time <= self.end_time <= 1:
            raise ValueError("optimization time window must satisfy 0 <= start_time <= end_time <= 1")
        if min(self.step_rms, self.max_delta_rms, self.gradient_epsilon) <= 0:
            raise ValueError("step_rms, max_delta_rms and gradient_epsilon must be positive")
        if min(self.edit_anchor_weight, self.trust_weight) < 0:
            raise ValueError("objective weights must be nonnegative")


def _finite_float(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def _time(value, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not value.is_floating_point():
            raise ValueError(f"{name} must be a scalar time")
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _rms(value):
    return value.square().mean().sqrt()


def optimization_metadata(config=None, trace=()):
    """JSON provenance also used by the off path without creating an optimizer."""
    config = config or InferenceOptimizationConfig(mode="off")
    return {
        "mode": config.mode,
        "config": asdict(config),
        "training_steps": 0,
        "weight_updates": 0,
        "target_conditioning": False,
        "rgb_replacement": False,
        "reference": "detached decode(VAE posterior mode source latent)",
        "edit_reference": "detached live predicted endpoint, not target supervision",
        "edit_anchor_representation": config.edit_anchor,
        "acceptance": "actual dtype-representable applied correction; strict objective decrease and nonincreasing keep proxy",
        "schedule": "start_time <= native_t <= end_time and (index + 1) % interval == 0",
        "limitations": list(LIMITATIONS),
        "trace": copy.deepcopy(list(trace)),
    }


class InferenceLatentOptimizer:
    """Optimize private FP32 latents; source/permissions/weights are never optimized.

    ``source_latent`` is the scaled VAE posterior mode. K/M maps are original RGB
    permissions [objects,5,H,W], merged with precisely the conditioning semantics.
    The caller owns the frozen, eval-mode FP32 VAE and its device residency.
    """

    def __init__(self, vae, source_latent, keep, modify, reference_valid=None, config=None):
        self.config = InferenceOptimizationConfig() if config is None else config
        if not isinstance(self.config, InferenceOptimizationConfig):
            raise TypeError("config must be InferenceOptimizationConfig")
        _finite_float(source_latent, "source_latent")
        if source_latent.ndim != 4 or source_latent.shape[0] != 1 or min(source_latent.shape) <= 0:
            raise ValueError("source_latent must have shape [1,C,H,W]")
        for name, value in (("keep", keep), ("modify", modify)):
            _finite_float(value, name)
            if value.ndim != 4 or value.shape[1] != 5 or min(value.shape) < 1:
                raise ValueError(f"{name} must have shape [objects,5,Hrgb,Wrgb]")
            if ((value < 0) | (value > 1)).any().item():
                raise ValueError(f"{name} must be in [0,1]")
        if keep.shape != modify.shape or keep.device != modify.device:
            raise ValueError("keep and modify must have identical shapes and devices")
        height, width = source_latent.shape[-2:]
        rgb_height, rgb_width = keep.shape[-2:]
        if rgb_height % height or rgb_width % width or rgb_height // height != rgb_width // width:
            raise ValueError("RGB and latent grids must have a common integral isotropic scale")
        if reference_valid is None:
            reference_valid = torch.ones_like(keep, dtype=torch.bool)
        if not isinstance(reference_valid, torch.Tensor) or reference_valid.dtype != torch.bool:
            raise TypeError("reference_valid must be boolean")
        if reference_valid.shape != keep.shape or reference_valid.device != keep.device:
            raise ValueError("reference_valid must match the permission shape and device")
        with torch.no_grad():
            merged_keep = keep.detach().float().amax(0, keepdim=True)
            merged_modify = modify.detach().float().amax(0, keepdim=True)
            if ((merged_keep > 0) & (merged_modify > 0)).any().item():
                raise ValueError("overlapping keep/modify permissions conflict across objects")
            self.keep = (keep.detach().float() * reference_valid).amax(0, keepdim=True).to(source_latent.device).clone()
            self.edit_mask = merged_modify.amax(1, keepdim=True).to(source_latent.device).clone()
            self.modify = modify.detach().float().to(source_latent.device).clone()
        if not isinstance(vae, torch.nn.Module) or vae.training:
            raise ValueError("VAE must be an eval-mode frozen FP32 module")
        for value in vae.parameters():
            if value.requires_grad:
                raise ValueError("VAE parameters must be frozen")
        for value in (*vae.parameters(), *vae.buffers()):
            if value.is_floating_point() and value.dtype != torch.float32:
                raise ValueError("VAE must use FP32 weights and buffers")
            if value.device != source_latent.device:
                raise ValueError("VAE and source_latent must share a device")
        self.vae = vae
        self.scale = _time(vae.config.scaling_factor, "VAE scaling_factor")
        self.shift = _time(vae.config.shift_factor, "VAE shift_factor")
        if self.scale <= 0:
            raise ValueError("VAE scaling_factor must be positive")
        self.latent_shape = tuple(source_latent.shape)
        self.device = source_latent.device
        self.has_keep = bool(self.keep.any().item())
        self.edit_mass = self.edit_mask.sum() * 3
        self.has_edit = bool(self.edit_mass > 0)
        self.trace = []
        self.source_codec = None
        if self.config.mode == "latent" and self.has_keep:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                self.source_codec = self._decode(source_latent.detach().float()).detach().clone()

    def _decode(self, latent):
        decoded = self.vae.decode(latent / self.scale + self.shift, return_dict=False)[0]
        _finite_float(decoded, "decoded RGB")
        if decoded.dtype != torch.float32 or tuple(decoded.shape) != (1, 3, *self.keep.shape[-2:]):
            raise ValueError("VAE must decode FP32 RGB at the original permission resolution")
        return (decoded / 2 + 0.5).clamp(0, 1)

    def _objective(self, x, anchor, edit_reference):
        prediction = self._decode(x)
        keep_loss, _ = preservation_loss(prediction, self.source_codec, self.keep)
        if self.config.edit_anchor == "rgb":
            # Keep the historical numerical operation order exactly for replay.
            edit = ((prediction - edit_reference).abs() * self.edit_mask).sum() / self.edit_mass if self.has_edit else prediction.sum() * 0
        else:
            edit, _ = edit_anchor_loss(prediction, edit_reference, self.modify, mode="attribute")
        trust = (x - anchor).square().mean()
        total = keep_loss + self.config.edit_anchor_weight * edit + self.config.trust_weight * trust
        return total, {"objective": float(total.detach()), "keep": float(keep_loss.detach()),
                       "edit_anchor": float(edit.detach()), "trust": float(trust.detach())}

    @torch.no_grad()
    def step(self, next_latent, *, before_latent, velocity, t, next_t, index):
        """Intervene after native Euler; return the exact input object on rejection.

        Every candidate is rounded at the *sampler addition*, then its effective
        clean correction is checked. Rounding the complete clean anchor would
        introduce an unrelated correction and is deliberately avoided.
        """
        if isinstance(next_latent, torch.Tensor) and next_latent.requires_grad:
            next_latent = next_latent.detach()
        t, next_t = _time(t, "t"), _time(next_t, "next_t")
        if not 0 <= t <= next_t <= 1 or type(index) is not int or index < 0:
            raise ValueError("optimization requires 0 <= t <= next_t <= 1 and a nonnegative index")
        entry = {"index": index, "t": t, "next_t": next_t, "selected": False,
                 "accepted_steps": 0, "backtracking_trials": 0, "correction_rms": 0.0,
                 "applied_correction_rms": 0.0, "before": None, "after": None, "inner": []}
        self.trace.append(entry)
        cfg = self.config
        if cfg.mode == "off" or not self.has_keep:
            entry["reason"] = "off" if cfg.mode == "off" else "no_valid_keep"
            return next_latent
        if not (cfg.start_time <= t <= cfg.end_time and (index + 1) % cfg.interval == 0):
            entry["reason"] = "outside_schedule"
            return next_latent
        entry["selected"] = True
        for name, value in (("next_latent", next_latent), ("before_latent", before_latent), ("velocity", velocity)):
            _finite_float(value, name)
            if tuple(value.shape) != self.latent_shape or value.device != self.device:
                raise ValueError(f"{name} must match source latent shape and device")
        if next_latent.dtype not in (torch.float32, torch.bfloat16) or before_latent.dtype != next_latent.dtype or velocity.dtype != next_latent.dtype:
            raise ValueError("sampler tensors must share FP32 or BF16 dtype")
        if next_t == 0:
            entry["reason"] = "zero_next_time"
            return next_latent
        with torch.autocast(device_type=self.device.type, enabled=False):
            anchor = before_latent.detach().float() + (1 - t) * velocity.detach().float()
            _finite_float(anchor, "anchor")
            edit_reference = self._decode(anchor).detach()
            if cfg.edit_anchor == "attribute" and bool(self.modify[:, :4].any()):
                _, entry["edit_anchor_support"] = edit_anchor_loss(
                    edit_reference, edit_reference, self.modify, mode="attribute")
            x = anchor.clone()
            _, current = self._objective(x, anchor, edit_reference)
            entry["before"] = current.copy()
            result = next_latent
            for inner_index in range(cfg.inner_steps):
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    total, _ = self._objective(x, anchor, edit_reference)
                    gradient = torch.autograd.grad(total, x, only_inputs=True)[0].detach()
                _finite_float(gradient, "latent gradient")
                gradient_rms = float(_rms(gradient))
                inner = {"index": inner_index, "gradient_rms": gradient_rms, "trials": [], "accepted": False}
                entry["inner"].append(inner)
                if gradient_rms == 0:
                    entry["reason"] = "zero_gradient"
                    break
                direction = gradient / max(gradient_rms, cfg.gradient_epsilon)
                for attempt in range(cfg.backtracking_steps):
                    entry["backtracking_trials"] += 1
                    proposal = x.detach() - (cfg.step_rms / (2 ** attempt)) * direction
                    delta = proposal - anchor
                    delta = delta * (cfg.max_delta_rms / _rms(delta).clamp_min(cfg.max_delta_rms))
                    candidate_result = (next_latent.detach().float() + next_t * delta).to(next_latent.dtype)
                    effective_delta = (candidate_result.float() - next_latent.detach().float()) / next_t
                    correction_rms = float(_rms(effective_delta))
                    trial = {"attempt": attempt, "step_rms": cfg.step_rms / (2 ** attempt), "correction_rms": correction_rms}
                    inner["trials"].append(trial)
                    if not math.isfinite(correction_rms) or correction_rms > cfg.max_delta_rms:
                        trial["reason"] = "representable_correction_exceeds_bound"
                        continue
                    if torch.equal(candidate_result, result):
                        trial["reason"] = "unchanged_after_rounding"
                        continue
                    candidate_x = anchor + effective_delta
                    _, candidate = self._objective(candidate_x, anchor, edit_reference)
                    trial["objective"] = candidate
                    if candidate["objective"] < current["objective"] and candidate["keep"] <= current["keep"]:
                        x, result, current = candidate_x.detach(), candidate_result.detach(), candidate
                        entry["accepted_steps"] += 1
                        entry["correction_rms"] = correction_rms
                        entry["applied_correction_rms"] = float(_rms(result.float() - next_latent.float()))
                        inner["accepted"] = True
                        trial["reason"] = "accepted"
                        break
                    trial["reason"] = "objective_or_keep_rejected"
                if not inner["accepted"]:
                    break
            entry["after"] = current.copy()
            entry["reason"] = "accepted" if entry["accepted_steps"] else entry.get("reason", "no_accepted_candidate")
            return result

    def metadata(self):
        return optimization_metadata(self.config, self.trace)
