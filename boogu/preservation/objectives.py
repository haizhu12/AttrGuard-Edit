"""Continuous, source-aligned keep supervision for the experimental generator.

Use a detached source VAE round-trip as reference when supervising decoded
latents. There is deliberately no region-mean tolerance hinge: a small local
error cannot disappear inside a global dead zone. These RGB/chromaticity/value
derivative proxies are not face identity, OCR or semantic preservation metrics.
Unknown and modify roles never become keep labels here.
"""

import torch
from torch.nn import functional as F


def _features(rgb):
    rgb = rgb.float()
    value = rgb.amax(dim=1, keepdim=True)
    dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    kernels = (
        rgb.new_tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
        )[:, None]
        / 8
    )
    sobel = F.conv2d(F.pad(value, (1, 1, 1, 1), mode="replicate"), kernels)
    highpass = value - F.avg_pool2d(
        F.pad(value, (2, 2, 2, 2), mode="replicate"), 5, stride=1
    )
    chroma = rgb / rgb.sum(dim=1, keepdim=True).clamp_min(1e-4)
    return {
        "color": chroma,
        "structure_finite": torch.cat((dx, dy), 1),
        "structure_sobel": sobel,
        "texture": highpass,
        "glyph": rgb,
        "pixel": rgb,
    }


def _erode(weight, radius):
    padded = F.pad(weight, (radius,) * 4, value=0)
    return -F.max_pool2d(-padded, 2 * radius + 1, stride=1)


def preservation_loss(prediction, reference, keep):
    """Return differentiable loss and per-term diagnostics.

    Args:
        prediction: finite floating BCHW RGB, decoded in [0,1].
        reference: matching detached source/codec RGB, NOT an edited target.
        keep: floating [B,5,H,W], source-only validated keep weights in [0,1].

    Derivative supports are eroded so no stencil borrows unknown/edit pixels.
    Active terms are normalized by their own support mass, then averaged.
    Source/label tensors never receive gradients.
    """
    for name, value in (("prediction", prediction), ("reference", reference)):
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.ndim != 4
            or value.shape[1] != 3
            or min(value.shape) < 1
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{name} must be finite floating BCHW RGB")
    if prediction.shape != reference.shape or prediction.device != reference.device:
        raise ValueError("reference must match prediction shape and device")
    expected = (prediction.shape[0], 5, *prediction.shape[-2:])
    if (
        not isinstance(keep, torch.Tensor)
        or not keep.is_floating_point()
        or tuple(keep.shape) != expected
        or not bool(torch.isfinite(keep).all())
        or not bool(((keep >= 0) & (keep <= 1)).all())
    ):
        raise ValueError("keep must be finite floating [B,5,H,W] weights in [0,1]")
    weights = keep.detach().to(device=prediction.device, dtype=torch.float32)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        out = _features(prediction)
        ref = _features(reference.detach())
        masks = {
            "color": weights[:, 0:1],
            "structure_finite": _erode(weights[:, 1:2], 1),
            "structure_sobel": _erode(weights[:, 1:2], 1),
            "texture": _erode(weights[:, 2:3], 2),
            "glyph": weights[:, 3:4],
            "pixel": weights[:, 4:5],
        }
        terms, details = [], {}
        for name, mask in masks.items():
            mass = mask.sum()
            if not bool(mass > 0):
                continue
            term = ((out[name] - ref[name]).abs() * mask).sum() / (
                mass * out[name].shape[1]
            )
            terms.append(term)
            details[name] = {"loss": float(term.detach()), "weight_mass": float(mass)}
        loss = torch.stack(terms).mean() if terms else prediction.float().sum() * 0
    return loss, details
