"""Source-only, fixed constraints. Attribute order is a public contract."""

import math
from dataclasses import dataclass

import torch

ATTRIBUTES = ("color", "structure", "texture", "glyph", "pixel")


@dataclass(frozen=True)
class AttributeConstraints:
    source: torch.Tensor
    keep: torch.Tensor
    edit: torch.Tensor
    scales: tuple[float, ...]
    tolerances: tuple[float, ...]
    provenance: str
    reference_valid: torch.Tensor | None = None

    def __post_init__(self):
        source, keep, edit = self.source, self.keep, self.edit
        if source.ndim != 4 or source.shape[:2] != (1, 3):
            raise ValueError("source must be [1,3,H,W]; batch control is not supported")
        h, w = source.shape[-2:]
        if keep.ndim != 4 or keep.shape[0] < 1 or keep.shape[1:] != (5, h, w):
            raise ValueError("keep must be [objects,5,H,W] on source coordinates")
        if edit.shape != (1, 1, h, w):
            raise ValueError("edit must be [1,1,H,W] on source coordinates")
        valid = self.reference_valid
        if valid is None:
            valid = torch.ones_like(keep, dtype=torch.bool)
        if (
            not isinstance(valid, torch.Tensor)
            or valid.shape != keep.shape
            or not torch.isfinite(valid).all()
            or not ((valid == 0) | (valid == 1)).all()
        ):
            raise ValueError("reference_valid must be binary [objects,5,H,W]")
        object.__setattr__(self, "reference_valid", valid.detach().bool().clone())
        for name, value in [("source", source), ("keep", keep), ("edit", edit)]:
            if not torch.is_floating_point(value) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating point values")
            if value.min() < 0 or value.max() > 1:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value.detach().float().clone())
        for name, values, strict in [
            ("scales", self.scales, True),
            ("tolerances", self.tolerances, False),
        ]:
            if len(values) != 5 or any(
                isinstance(v, bool)
                or not math.isfinite(v)
                or (v <= 0 if strict else v < 0)
                for v in values
            ):
                raise ValueError(f"{name} must contain five valid numbers")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("constraint provenance must be explicit")


def predicted_weights(
    probabilities, support, applicability, *, threshold=0.8, unknown_max=0.2
):
    """Calibrated probabilities [O,5,3,H,W] -> K/M weights; no implicit targets."""
    if probabilities.ndim != 5 or probabilities.shape[1:3] != (5, 3):
        raise ValueError("probabilities must be [O,5,3,H,W] in K/M/U order")
    o, _, _, h, w = probabilities.shape
    if support.shape != (o, 1, h, w) or applicability.shape != (o, 5, h, w):
        raise ValueError("support/applicability shape mismatch")
    if not 0 <= unknown_max <= 1 or not 0.5 < threshold <= 1:
        raise ValueError("invalid calibrated thresholds")
    for value in (probabilities, support, applicability):
        if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError("invalid probability/support values")
    if not torch.allclose(
        probabilities.sum(2), torch.ones_like(probabilities[:, :, 0]), atol=1e-5, rtol=0
    ):
        raise ValueError("roles must sum to one per attribute")
    # Gate K as well as M BEFORE spatial normalization. A diffuse near-zero
    # support must not become a unit-strength object through mask normalization.
    common = (
        support * (support >= threshold) * applicability * (applicability >= threshold)
    )
    common = common * (probabilities[:, :, 2] <= unknown_max)
    weights = [
        common * probabilities[:, :, r] * (probabilities[:, :, r] >= threshold)
        for r in (0, 1)
    ]
    return tuple(value.detach().clone() for value in weights)
