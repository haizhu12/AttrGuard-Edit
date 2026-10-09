"""Inference-only relaxation of an anchor decoded from the current prediction.

The reference is the detached current prediction, which may already contain an
incorrect edit. It is not a desired edited target. Attribute mode weakens its
spatial color constraint to per-object RGB moments; it does not establish edit
correctness or semantic preservation. Glyph and pixel terms are RGB proxies,
with no OCR or semantic guarantee. No training objective or weights live here.
"""

import torch

from .objectives import _erode, _features


_ATTRIBUTES = ("color", "structure", "texture", "glyph", "pixel")


def _validate(prediction, reference, modify, mode):
    if mode not in ("rgb", "attribute"):
        raise ValueError("edit anchor mode must be rgb or attribute")
    for name, value in (("prediction", prediction), ("reference", reference)):
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.ndim != 4
            or tuple(value.shape[:2]) != (1, 3)
            or min(value.shape) < 1
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{name} must be finite floating [1,3,H,W] RGB")
    if prediction.shape != reference.shape or prediction.device != reference.device:
        raise ValueError("reference must match prediction shape and device")
    if (
        not isinstance(modify, torch.Tensor)
        or not modify.is_floating_point()
        or modify.ndim != 4
        or modify.shape[0] < 1
        or modify.shape[1] != 5
        or modify.shape[-2:] != prediction.shape[-2:]
        or not bool(torch.isfinite(modify).all())
        or not bool(((modify >= 0) & (modify <= 1)).all())
    ):
        raise ValueError("modify must be finite floating [objects,5,H,W] weights in [0,1]")
    if modify.device != prediction.device:
        raise ValueError("modify must share the prediction device")


def _rgb_anchor(prediction, reference, weights, details):
    # Keep the legacy reduction, multiplication and division order unchanged.
    mask = weights.amax(0, keepdim=True).amax(1, keepdim=True)
    mass = mask.sum() * 3
    active = bool(mass > 0)
    loss = (
        ((prediction - reference).abs() * mask).sum() / mass
        if active
        else prediction.sum() * 0
    )
    details.update(
        effective_representation="rgb",
        active_objects=int(weights.flatten(1).any(1).sum()),
        active_terms=[{
            "attribute": "union",
            "representation": "rgb",
            "support_mass": float(mask.sum()),
            "loss": float(loss.detach()),
        }] if active else [],
        objects=[],
        loss=float(loss.detach()),
    )
    return loss, details


def _masked_mae(prediction, reference, mask, mass):
    return ((prediction - reference).abs() * mask).sum() / (
        mass * prediction.shape[1]
    )


def _color_moments(rgb, mask, mass):
    mean = (rgb * mask).sum(dim=(-2, -1), keepdim=True) / mass
    # Center first: E[x^2] - E[x]^2 can become negative near uniform colors.
    variance = ((rgb - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / mass
    return mean, (variance + 1e-8).sqrt()


def edit_anchor_loss(prediction, reference, modify, mode="rgb"):
    """Return a differentiable anchor relaxation and JSON-safe diagnostics.

    ``prediction`` and ``reference`` are matching finite floating [1,3,H,W]
    tensors. ``modify`` is finite floating [objects,5,H,W] in [0,1], ordered as
    color, structure, texture, glyph, pixel. All tensors share a device. Only
    prediction receives gradients; reference and permissions are detached.

    RGB mode exactly retains the legacy per-pixel MAE on the union of every
    object and attribute. Attribute mode also uses that exact path whenever
    the first four permission channels are empty, including pixel-only maps.
    Otherwise each active attribute has one vote within its object, and each
    active object has one vote in the result. Structure's finite and Sobel MAEs
    share one vote. Derivative masks are eroded before applying their stencils.

    Color uses the original, uneroded mask and compares RGB means plus standard
    deviations. This permits spatial rearrangements sharing those moments, and
    changes the objective's scale in general; a uniform color shift retains its
    full mean penalty. Existing optimization weights are not calibrated here.
    """
    _validate(prediction, reference, modify, mode)
    weights = modify.detach().float()
    reference = reference.detach()
    details = {"mode": mode, "kind": "prediction_anchor_relaxation"}
    if mode == "rgb" or not bool(weights[:, :4].any()):
        return _rgb_anchor(prediction, reference, weights, details)

    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction_rgb, reference_rgb = prediction.float(), reference.float()
        out, ref = _features(prediction_rgb), _features(reference_rgb)
        object_losses, objects = [], []
        for object_index in range(weights.shape[0]):
            attribute_losses, active_terms, skipped_terms = [], [], []
            for attribute_index, attribute in enumerate(_ATTRIBUTES):
                original_mask = weights[object_index:object_index + 1, attribute_index:attribute_index + 1]
                original_mass = original_mask.sum()
                if not bool(original_mass > 0):
                    continue
                radius = 1 if attribute == "structure" else 2 if attribute == "texture" else 0
                mask = _erode(original_mask, radius) if radius else original_mask
                mass = mask.sum()
                if not bool(mass > 0):
                    skipped_terms.append({
                        "attribute": attribute,
                        "support_mass": float(mass),
                        "original_support_mass": float(original_mass),
                        "reason": "empty_eroded_support",
                    })
                    continue
                term_details = {"attribute": attribute, "support_mass": float(mass)}
                if attribute == "color":
                    mean, std = _color_moments(prediction_rgb, mask, mass)
                    reference_mean, reference_std = _color_moments(reference_rgb, mask, mass)
                    term = (mean - reference_mean).abs().mean() + (std - reference_std).abs().mean()
                    term_details["representation"] = "masked_rgb_mean_std"
                elif attribute == "structure":
                    components = {
                        name: _masked_mae(out[name], ref[name], mask, mass)
                        for name in ("structure_finite", "structure_sobel")
                    }
                    term = torch.stack(list(components.values())).mean()
                    term_details["representation"] = "finite_sobel"
                    term_details["components"] = {
                        name: {"loss": float(value.detach()), "support_mass": float(mass)}
                        for name, value in components.items()
                    }
                else:
                    term = _masked_mae(out[attribute], ref[attribute], mask, mass)
                    term_details["representation"] = "highpass" if attribute == "texture" else "rgb_proxy"
                term_details["loss"] = float(term.detach())
                active_terms.append(term_details)
                attribute_losses.append(term)
            object_loss = torch.stack(attribute_losses).mean() if attribute_losses else None
            if object_loss is not None:
                object_losses.append(object_loss)
            objects.append({
                "object_index": object_index,
                "active_terms": active_terms,
                "skipped_terms": skipped_terms,
                "loss": float(object_loss.detach()) if object_loss is not None else 0.0,
            })
        loss = torch.stack(object_losses).mean() if object_losses else prediction.float().sum() * 0
    details.update(
        effective_representation="attribute",
        active_objects=len(object_losses),
        objects=objects,
        loss=float(loss.detach()),
    )
    return loss, details
