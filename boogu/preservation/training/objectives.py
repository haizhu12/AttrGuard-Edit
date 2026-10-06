"""Paired Full objective: native flow plus detached target-codec supervision."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from ..objectives import preservation_loss


def _matching_fields(*fields):
    first = fields[0]
    for value in fields:
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.ndim != 4
            or min(value.shape) < 1
            or value.shape != first.shape
            or value.device != first.device
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("matching finite floating BCHW tensors required")


def _mask_tensor(mask, image):
    if isinstance(mask, torch.Tensor):
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("support must be boolean HxW")
        return mask.detach().to(device=image.device)[None, None]
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("support must be boolean HxW")
    return torch.from_numpy(array.copy()).to(image.device)[None, None]


def flow_sample(noise, target, t):
    """Native z(t)=(1-t)*noise+t*target, velocity=target-noise, in FP32."""
    _matching_fields(noise, target)
    time = torch.as_tensor(t, device=noise.device, dtype=torch.float32)
    if not torch.isfinite(time).all().item() or ((time < 0) | (time > 1)).any().item():
        raise ValueError("flow time must be normalized in [0,1]")
    if time.ndim == 1 and time.shape[0] == noise.shape[0]:
        time = time[:, None, None, None]
    if time.ndim != 0 and tuple(time.shape) != (noise.shape[0], 1, 1, 1):
        raise ValueError("flow time must be scalar or one value per batch item")
    return (
        1 - time
    ) * noise.float() + time * target.float(), target.float() - noise.float()


def endpoint_fp32(z_fp32, prediction, t):
    _matching_fields(z_fp32, prediction)
    if (
        z_fp32.dtype != torch.float32
        or not math.isfinite(float(t))
        or not 0 <= float(t) <= 1
    ):
        raise ValueError("endpoint requires original FP32 z and normalized time")
    return z_fp32 + (1 - float(t)) * prediction.float()


def balanced_flow_loss(prediction, velocity, allowed_edit):
    """Average separately normalized edit/context MSE, omitting empty regions."""
    _matching_fields(prediction, velocity)
    edit = F.adaptive_avg_pool2d(
        _mask_tensor(allowed_edit, prediction).float(), prediction.shape[-2:]
    )
    error = (prediction.float() - velocity.float()).square()
    terms = []
    for mask in (edit, 1 - edit):
        mass = mask.sum() * error.shape[1] * error.shape[0]
        if mass.item() > 0:
            terms.append((error * mask).sum() / mass)
    return torch.stack(terms).mean()


def decode(vae, latent):
    """Differentiable FP32 VAE decoding with native latent normalization."""
    _matching_fields(latent)
    if vae.dtype != torch.float32:
        raise ValueError("training requires an FP32 VAE")
    scale, shift = vae.config.scaling_factor, vae.config.shift_factor
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(shift):
        raise ValueError("invalid VAE latent normalization")
    image = vae.decode(latent.float() / scale + shift, return_dict=False)[0]
    if not torch.isfinite(image).all().item():
        raise ValueError("nonfinite VAE output")
    return (image.float() / 2 + 0.5).clamp(0, 1)


def _masked_rgb_mae(prediction, reference, support):
    if support.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("RGB support must be canvas aligned")
    count = int(torch.count_nonzero(support))
    if not count:
        return None, 0
    weight = support.detach().float()
    return ((prediction.float() - reference.detach().float()).abs() * weight).sum() / (
        weight.sum() * prediction.shape[1]
    ), count


def training_objective(
    prediction,
    velocity,
    rgb,
    target_codec_reference,
    keep,
    allowed_edit,
    protected_roi_masks,
    edit_core,
):
    """Flow + .5 target-K + .5 protected ROI macro + .5 edit-core RGB.

    Only this paired-training wrapper gives edited-target semantics to the
    existing K feature algebra. Targets never enter source conditioning.
    """
    reference = target_codec_reference.detach()
    flow = balanced_flow_loss(prediction.float(), velocity.detach(), allowed_edit)
    keep_loss, keep_details = preservation_loss(rgb, reference, keep)
    roi_terms, roi_details = [], {}
    pixel_k = keep[:, 4:5].detach() > 0
    for name, mask in sorted(protected_roi_masks.items()):
        support = _mask_tensor(mask, rgb)
        if support.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("protected ROI must be canvas aligned")
        term, count = _masked_rgb_mae(rgb, reference, support & pixel_k)
        roi_details[name] = {
            "loss": None if term is None else float(term.detach()),
            "pixels": count,
        }
        if term is not None:
            roi_terms.append(term)
    roi_macro = torch.stack(roi_terms).mean() if roi_terms else None
    edit, edit_count = _masked_rgb_mae(rgb, reference, _mask_tensor(edit_core, rgb))
    total = flow + 0.5 * keep_loss
    for value in (roi_macro, edit):
        if value is not None:
            total = total + 0.5 * value
    detail = {
        name: None if value is None else float(value.detach())
        for name, value in {
            "flow_balanced": flow,
            "target_keep_loss": keep_loss,
            "protected_roi_macro_rgb": roi_macro,
            "edit_core_rgb": edit,
            "total": total,
        }.items()
    }
    detail.update(
        target_keep_details=keep_details,
        protected_roi_details=roi_details,
        edit_core_pixels=edit_count,
        target_reference="detached_target_posterior_mode_codec_rgb",
    )
    return total, detail
