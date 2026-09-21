"""Opt-in fixed-support diagnostics, not object segmentation or OCR metrics."""

import torch
import torch.nn.functional as F

from .observers import AttributeObserver, erode_support, gaussian, gradient


def _pool(x, size=64):
    h, w = x.shape[-2:]
    padded = F.pad(x, (0, (-w) % size, 0, (-h) % size))
    # Two smaller reductions avoid avg_pool2d's long sequential FP32 sum on CPU.
    tiled = padded.reshape(
        *x.shape[:2], padded.shape[-2] // size, size, padded.shape[-1] // size, size
    )
    return tiled.mean(-1).mean(-2)


def _luma(x):
    return (x * x.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(
        1, keepdim=True
    )


def _edge_energy(x):
    # Positive-affine brightness and uniform two-color recoloring invariance,
    # after normalization, not invariance to arbitrary recoloring/shading.
    smooth = gaussian(x, 2)
    g = gradient(smooth.reshape(-1, 1, *x.shape[-2:]))
    return g.square().sum((0, 1), keepdim=True)


class LocalizedAttributeObserver:
    """Legacy proxies plus worst fixed RGB/luma patch and spatial edge mass.

    Spatial edge mass is a source-aligned histogram, NOT semantic shape identity.
    References and supports are fixed before sampling. Extra pixel checks exist
    only for explicit RGB K, never for color K or legally edited pixels.
    """

    def __init__(self, constraints, roundtrip=None):
        self.base = AttributeObserver(constraints, roundtrip=roundtrip)
        self.source, self.edit = self.base.source, self.base.edit
        self.names = list(self.base.names)
        self.items = list(self.base.items)
        self.unavailable_checks = list(self.base.unavailable_checks)
        self.coverage = self.base.coverage
        self.extra = []
        tolerances = []
        keep = constraints.keep.to(self.source) * constraints.reference_valid.to(
            self.source
        )
        keep[:, 4] *= self.edit[0, 0] <= 0
        owners = keep.argmax(dim=0)
        keep *= (
            torch.arange(keep.shape[0], device=keep.device)[:, None, None, None]
            == owners
        )
        for o in range(keep.shape[0]):
            pixel = keep[o : o + 1, 4:5]
            mass = _pool(pixel)
            valid = mass >= 4 / 64**2
            if valid.any():
                for name in ("local_rgb", "local_luma"):
                    self.extra.append(
                        (name, pixel, mass, valid, None, constraints.scales[4])
                    )
                    self.names.append(f"object{o}:{name}")
                    tolerances.append(constraints.tolerances[4] / constraints.scales[4])
            mask = keep[o : o + 1, 1:2]
            if mask.any():
                name = f"object{o}:structure_spatial"
                if min(self.source.shape[-2:]) <= 14:
                    self.unavailable_checks.append(name)
                    continue
                mask = erode_support(mask, 7)
                energy = _edge_energy(self.source) * mask
                if float(mask.sum()) < 4 or float(energy.sum()) < 1e-8:
                    self.unavailable_checks.append(name)
                    continue
                ref = _pool(energy, 16)
                ref = (ref / ref.sum()).detach()
                self.extra.append(
                    ("structure_spatial", mask, None, None, ref, constraints.scales[1])
                )
                self.names.append(name)
                tolerances.append(constraints.tolerances[1] / constraints.scales[1])
        # Item contents are opaque to the controller, which checks availability.
        self.items.extend(self.extra)
        self.weights = torch.cat(
            (self.base.weights, self.source.new_ones(len(self.extra)))
        )
        extra_tolerance = self.source.new_tensor(tolerances)
        if roundtrip is not None and self.extra:
            extra_tolerance += self._extra_errors(roundtrip.to(self.source)).detach()
        self.tolerances = torch.cat((self.base.tolerances, extra_tolerance))

    def _extra_errors(self, image):
        values = []
        for name, mask, mass, valid, ref, scale in self.extra:
            if name == "structure_spatial":
                hist = _pool(_edge_energy(image) * mask, 16)
                # If edges disappear, TV remains positive; availability is
                # source-derived and cannot be changed by the candidate.
                hist = hist / hist.sum().clamp_min(1e-12)
                error = 0.5 * (hist - ref).abs().sum()
            else:
                diff = image - self.source
                diff = (
                    _luma(diff).abs()
                    if name == "local_luma"
                    else diff.abs().mean(1, keepdim=True)
                )
                error = (_pool(diff * mask) / mass.clamp_min(1e-12))[valid].max()
            values.append(error / scale)
        return torch.stack(values) if values else image.new_empty(0)

    def errors(self, image):
        basic = self.base.errors(image)  # validates exact canvas and finiteness
        return torch.cat((basic, self._extra_errors(image)))


class TargetColorGoal:
    """Fixed-mask RGB-direction goal, not a full semantic recoloring metric.

    Nonzero brightness scaling is intentionally tolerated; black is not treated
    as a match. A target must be an explicit nonblack RGB direction in [0,1].
    """

    def __init__(self, mask, target_rgb):
        if (
            not isinstance(mask, torch.Tensor)
            or mask.ndim != 4
            or mask.shape[:2] != (1, 1)
            or not torch.is_floating_point(mask)
            or not torch.isfinite(mask).all()
            or mask.min() < 0
            or mask.max() > 1
            or float(mask.sum()) < 4
        ):
            raise ValueError(
                "target color requires a finite nonempty fixed [1,1,H,W] mask"
            )
        target = torch.as_tensor(target_rgb, dtype=torch.float32)
        if (
            target.shape != (3,)
            or not torch.isfinite(target).all()
            or target.min() < 0
            or target.max() > 1
            or float(target.norm()) == 0
        ):
            raise ValueError("target_rgb must be three finite [0,1] values, not black")
        self.mask = mask.detach().float().clone()
        self.target = (target / target.norm()).detach().clone()

    def errors(self, image):
        if (
            image.shape != (1, 3, *self.mask.shape[-2:])
            or not torch.isfinite(image).all()
        ):
            raise ValueError("invalid target-color observation canvas or values")
        image = image.float()  # Keep the zero-color norm floor representable in FP16.
        mask = self.mask.to(image)
        unit = image / image.square().sum(1, keepdim=True).clamp_min(1e-12).sqrt()
        cosine = (unit * self.target.to(image)[None, :, None, None]).sum(
            1, keepdim=True
        )
        return (((1 - cosine.clamp(-1, 1)) * mask).sum() / mask.sum()).reshape(1)
