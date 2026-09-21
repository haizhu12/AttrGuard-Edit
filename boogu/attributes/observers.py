"""Differentiable, fixed-coordinate attribute proxies (not semantic metrics)."""

import torch
import torch.nn.functional as F

from .constraints import ATTRIBUTES, AttributeConstraints


def gaussian(x, sigma):
    radius = 3 * sigma
    line = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-line.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    x = F.conv2d(
        F.pad(x, (radius, radius, 0, 0), mode="reflect"),
        kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
        groups=channels,
    )
    return F.conv2d(
        F.pad(x, (0, 0, radius, radius), mode="reflect"),
        kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
        groups=channels,
    )


def gradient(y):
    kernel = y.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    return F.conv2d(
        F.pad(y, (1, 1, 1, 1), mode="reflect"),
        torch.stack((kernel, kernel.t())).unsqueeze(1),
    )


def direction(y):
    g = gradient(y)
    return g / torch.sqrt(g.square().sum(1, keepdim=True) + 1e-6)


def local_pattern(y, sigma):
    high = y - gaussian(y, sigma)
    return high / torch.sqrt(gaussian(high.square(), sigma) + 1e-6)


def structure_tensor(image):
    """Polarity-invariant RGB edge orientation/occupancy in fixed coordinates.

    Channel permutation and contrast inversion preserve this proxy. Arbitrary
    local recoloring, shading and very weak contrast need not preserve it. The
    fixed energy floor makes flat/new/disappearing edges observable and keeps
    gradients finite; it is not semantic shape correspondence.
    """
    smooth = gaussian(image, 2)
    b, c, h, w = smooth.shape
    g = gradient(smooth.reshape(b * c, 1, h, w)).reshape(b, c, 2, h, w)
    xx, yy = g[:, :, 0].square().sum(1), g[:, :, 1].square().sum(1)
    xy = (g[:, :, 0] * g[:, :, 1]).sum(1)
    return torch.stack((xx, yy, 2**0.5 * xy), 1) / (xx + yy + 1e-6)[:, None]


def feature(image, attribute, structure_mode="legacy"):
    if attribute == 4:
        return image
    if attribute == 0:
        return torch.cat(
            (image[:, 0:1] - image[:, 1:2], image[:, 2:3] - image[:, 1:2]), 1
        )
    if attribute == 1 and structure_mode == "geometry":
        return structure_tensor(image)
    y = (image * image.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(
        1, keepdim=True
    )
    if attribute == 1:
        return direction(gaussian(y, 2))
    if attribute == 2:
        return torch.cat((local_pattern(y, 2), local_pattern(y, 4)), 1)
    return torch.cat((local_pattern(y, 2), direction(y)), 1)


def erode_support(mask, radius):
    if not radius:
        return mask
    invalid = (mask <= 0).float()
    return mask * (F.max_pool2d(invalid, 2 * radius + 1, stride=1, padding=radius) == 0)


class AttributeObserver:
    """Masks and feature references are snapshotted, never optimized with q."""

    def __init__(
        self,
        constraints: AttributeConstraints,
        roundtrip=None,
        *,
        structure_mode="legacy",
    ):
        if structure_mode not in ("legacy", "geometry"):
            raise ValueError("structure_mode must be legacy or geometry")
        self.structure_mode = structure_mode
        self.source = constraints.source.detach().clone()
        self.edit = constraints.edit.to(self.source).detach().clone()
        requested = constraints.keep.to(self.source).detach().clone()
        valid_keep = requested * constraints.reference_valid.to(self.source)
        keep = valid_keep.clone()
        # RGB identity cannot coexist with any actual requested edit.
        keep[:, 4] *= self.edit[0, 0] <= 0
        owners = keep.argmax(dim=0)
        keep *= (
            torch.arange(keep.shape[0], device=keep.device)[:, None, None, None]
            == owners
        )
        self.items = []
        self.names = []
        self._refs = {}
        self._scales = constraints.scales
        self.decode_count = 0
        self.coverage = []
        self.unavailable_checks = []
        for a in range(5):
            radius = (0, 7, 24, 12, 0)[a]
            small = a in (1, 2, 3) and min(self.source.shape[-2:]) <= 2 * radius
            ref = None
            if not small and keep[:, a].any():
                ref = feature(self.source, a, self.structure_mode).detach()
                self._refs[a] = ref
            for o in range(keep.shape[0]):
                name = f"object{o}:{ATTRIBUTES[a]}"
                row = {
                    "name": name,
                    "requested_mass": float(requested[o, a].sum()),
                    "reference_valid_mass": float(valid_keep[o, a].sum()),
                    "owned_mass": float(keep[o, a].sum()),
                    "eroded_mass": 0.0,
                    "active_mass": 0.0,
                    "coverage_fraction": 0.0,
                    "active": False,
                    "reason": "not_requested",
                }
                self.coverage.append(row)
                if row["requested_mass"] == 0:
                    continue
                if ref is None or row["owned_mass"] == 0:
                    row["reason"] = (
                        "reference_invalid"
                        if row["reference_valid_mass"] == 0
                        else "canvas_too_small"
                        if small
                        else "excluded_by_edit_or_ownership"
                    )
                    self.unavailable_checks.append(name)
                    continue
                mask = erode_support(keep[o : o + 1, a : a + 1], radius)
                row["eroded_mass"] = float(mask.sum())
                # Only spatial-pattern proxies require nonzero source energy.
                # Geometry retains flat coordinates to detect newly created edges.
                if a in (1, 2, 3) and not (a == 1 and structure_mode == "geometry"):
                    mask = mask * (ref.square().mean(1, keepdim=True) > 1e-6)
                mass = float(mask.sum())
                row["active_mass"] = mass
                row["coverage_fraction"] = mass / row["requested_mass"]
                if mass < 4:
                    row["reason"] = (
                        "insufficient_support_after_erosion"
                        if row["eroded_mass"] < 4
                        else "insufficient_source_feature_energy"
                    )
                    self.unavailable_checks.append(name)
                    continue
                row.update(active=True, reason="active")
                self.items.append((a, mask.detach().clone(), mass))
                self.names.append(name)
        totals = {
            a: sum(mass for aa, _, mass in self.items if aa == a) for a in range(5)
        }
        self.weights = self.source.new_tensor(
            [mass / totals[a] for a, _, mass in self.items]
        )
        self.tolerances = self.source.new_tensor(
            [
                constraints.tolerances[a] / constraints.scales[a]
                for a, _, _ in self.items
            ]
        )
        if roundtrip is not None:
            self.tolerances = (
                self.tolerances + self.errors(roundtrip.to(self.source)).detach()
            )

    @staticmethod
    def _distance(f, ref, mask, attribute):
        if attribute != 0:
            return ((f - ref).abs().mean(1, keepdim=True) * mask).sum() / mask.sum()
        # Pool weighted source-aligned 16px blocks; exclude insufficient blocks.
        size = 16
        h, w = f.shape[-2:]
        pad = (0, (-w) % size, 0, (-h) % size)
        maskpad = F.pad(mask, pad)
        mass = F.avg_pool2d(maskpad, size, stride=size)
        means = F.avg_pool2d(F.pad(f * mask, pad), size, stride=size) / mass.clamp_min(
            1e-8
        )
        refs = F.avg_pool2d(F.pad(ref * mask, pad), size, stride=size) / mass.clamp_min(
            1e-8
        )
        weight = mass * (mass >= 4 / (size * size))
        block = ((means - refs).abs().mean(1, keepdim=True) * weight).sum()
        whole = (
            (
                (f * mask).sum((2, 3)) / mask.sum()
                - (ref * mask).sum((2, 3)) / mask.sum()
            )
            .abs()
            .mean()
        )
        return (block + whole) / (weight.sum() + 1)

    def errors(self, image):
        if image.shape != self.source.shape:
            raise ValueError("decoded image does not match source coordinates")
        if not torch.isfinite(image).all():
            raise ValueError("nonfinite decoded image")
        cache = {a: feature(image, a, self.structure_mode) for a in self._refs}
        values = [
            self._distance(cache[a], self._refs[a], mask, a) / self._scales[a]
            for a, mask, _ in self.items
        ]
        return torch.stack(values) if values else image.new_empty(0)
