"""Construct source-only conditioning; targets are deliberately not accepted."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

ATTRIBUTE_NAMES = ("color", "structure", "texture", "glyph", "pixel")


@dataclass(frozen=True)
class SourceCondition:
    """Detached source patches and five ordered K/M/U triples per patch.

    Tensor contents remain mutable in PyTorch. The transformer therefore takes
    a fresh private snapshot on every forward, including checkpoint recomputes.
    """

    source_tokens: torch.Tensor
    roles: torch.Tensor
    latent_shape: tuple[int, int, int, int]
    patch_size: int = 2


def _finite_float(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def prepare_condition(
    source_latent, keep, modify, reference_valid=None, *, patch_size=2
):
    """Pool RGB permission maps into aligned native latent patches.

    RGB maps must use the same aspect ratio and an integral, isotropic scale
    relative to the latent. There is no resizing, inferred crop, or target data.
    Object permissions merge by maximum; overlapping K and M are ambiguous and
    rejected before unavailable reference pixels turn K into U.
    """
    _finite_float(source_latent, "source_latent")
    if (
        source_latent.ndim != 4
        or source_latent.shape[0] != 1
        or min(source_latent.shape) <= 0
    ):
        raise ValueError("source_latent must have shape [1,C,H,W]")
    if (
        not isinstance(patch_size, int)
        or isinstance(patch_size, bool)
        or patch_size <= 0
    ):
        raise ValueError("patch_size must be a positive integer")
    _, channels, height, width = source_latent.shape
    if height % patch_size or width % patch_size:
        raise ValueError("latent grid must be divisible by patch_size")
    for name, value in (("keep", keep), ("modify", modify)):
        _finite_float(value, name)
        if (
            value.ndim != 4
            or value.shape[0] < 1
            or value.shape[1] != 5
            or min(value.shape) <= 0
        ):
            raise ValueError(f"{name} must have shape [O,5,Hrgb,Wrgb]")
        if ((value < 0) | (value > 1)).any().item():
            raise ValueError(f"{name} must be in [0,1]")
    if keep.shape != modify.shape or keep.device != modify.device:
        raise ValueError("keep and modify must have identical shapes and devices")
    rgb_height, rgb_width = keep.shape[-2:]
    if (
        rgb_height % height
        or rgb_width % width
        or rgb_height // height != rgb_width // width
    ):
        raise ValueError(
            "RGB and latent grids must have a common integral isotropic scale"
        )
    if reference_valid is None:
        reference_valid = torch.ones_like(keep, dtype=torch.bool)
    if (
        not isinstance(reference_valid, torch.Tensor)
        or reference_valid.dtype != torch.bool
    ):
        raise TypeError("reference_valid must be a boolean tensor")
    if reference_valid.shape != keep.shape or reference_valid.device != keep.device:
        raise ValueError("reference_valid must match the permission shape and device")
    with torch.no_grad():
        merged_keep = keep.detach().float().amax(dim=0, keepdim=True)
        merged_modify = modify.detach().float().amax(dim=0, keepdim=True)
        if ((merged_keep > 0) & (merged_modify > 0)).any().item():
            raise ValueError(
                "overlapping keep/modify permissions conflict across objects"
            )
        merged_keep = (keep.detach().float() * reference_valid).amax(
            dim=0, keepdim=True
        )
        cell = patch_size * (rgb_height // height)
        pooled_keep = F.avg_pool2d(merged_keep, kernel_size=cell, stride=cell)
        pooled_modify = F.avg_pool2d(merged_modify, kernel_size=cell, stride=cell)
        pooled_unknown = (1.0 - pooled_keep - pooled_modify).clamp_min(0)
        roles = torch.stack((pooled_keep, pooled_modify, pooled_unknown), dim=-1)
        roles = roles.permute(0, 2, 3, 1, 4).reshape(1, -1, 15).contiguous().clone()
        p = patch_size
        tokens = source_latent.detach().reshape(
            1, channels, height // p, p, width // p, p
        )
        tokens = (
            tokens.permute(0, 2, 4, 3, 5, 1)
            .reshape(1, -1, p * p * channels)
            .contiguous()
            .clone()
        )
    return SourceCondition(tokens, roles, tuple(source_latent.shape), patch_size)
