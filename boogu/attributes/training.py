"""Explicit attribute supervision and versioned, weights-only checkpoints.

Labels must be independently annotated attribute roles/support/applicability, not
derived from image differences. Synthetic fitting is a diagnostic only. Provenance
is a recorded claim, not cryptographic verification of data or semantic efficacy.
"""

import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.nn import functional as F

from .predictor import (
    ARCHITECTURE,
    ATTRIBUTE_ORDER,
    ROLE_ORDER,
    AttributeRolePrediction,
    AttributeRolePredictor,
)


def _mask(target, mask, shape, valid, name):
    if target.shape != shape or mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError(
            f"{name} target/mask shape must be {tuple(shape)}; mask must be boolean"
        )
    if target.device != valid.device or mask.device != valid.device:
        raise ValueError(f"{name} targets and masks must share the prediction device")
    return mask & valid


def _binary_loss(logits, target, mask, valid, name):
    mask = _mask(target, mask, logits.shape, valid, name)
    observed = target[mask]
    if (
        not torch.isfinite(observed).all()
        or (observed < 0).any()
        or (observed > 1).any()
    ):
        raise ValueError(f"observed {name} targets must lie in [0,1]")
    if not mask.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[mask].float(), observed.float())


def attribute_role_loss(
    prediction: AttributeRolePrediction,
    role_targets,
    role_valid_mask,
    support_targets,
    support_valid_mask,
    applicability_targets,
    applicability_valid_mask,
    *,
    role_quality_weights=None,
    object_weight=1.0,
    applicability_weight=1.0,
    paired_prediction=None,
    pair_valid_mask=None,
    margin=1.0,
    pair_weight=0.0,
):
    """Return total/roles/support/applicability/paired scalar losses.

    Roles are integer B,5,H,W (K=0,M=1,U=2); role masks have that same shape.
    Support targets/masks are B,1,H,W, applicability targets/masks B,5,H,W. Masks
    express available annotations, additionally restricted to valid image pixels.
    Role CE uses fixed, nonnegative quality weights of shape B,5,H,W (default one)
    and is normalized within each attribute before averaging attributes with
    positive supervision weight. Masked quality values are ignored. Object and
    applicability coefficients default to one for compatibility.
    Optional paired supervision explicitly orders the first instruction as keep
    and paired instruction as modify for selected B,5,H,W locations; it penalizes
    failure to increase (M-K) by margin. It does not infer such pairs from images.
    """
    for name, value in (
        ("margin", margin),
        ("pair_weight", pair_weight),
        ("object_weight", object_weight),
        ("applicability_weight", applicability_weight),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    logits, valid = prediction.role_logits, prediction.valid_mask
    role_mask = _mask(
        role_targets,
        role_valid_mask,
        logits.shape[:2] + logits.shape[3:],
        valid,
        "role",
    )
    if role_targets.dtype != torch.long:
        raise ValueError("role targets must be int64 class indices")
    observed = role_targets[role_mask]
    if (observed < 0).any() or (observed > 2).any():
        raise ValueError("observed role targets must be K=0, M=1, or U=2")
    if role_quality_weights is None:
        quality = torch.ones_like(role_targets, dtype=torch.float32)
    else:
        if (
            role_quality_weights.shape != role_targets.shape
            or role_quality_weights.device != logits.device
            or not role_quality_weights.is_floating_point()
        ):
            raise ValueError(
                "role quality weights must be floating B,5,H,W on the prediction device"
            )
        quality = role_quality_weights.detach().float()
        observed_quality = quality[role_mask]
        if not torch.isfinite(observed_quality).all() or (observed_quality < 0).any():
            raise ValueError(
                "observed role quality weights must be finite and nonnegative"
            )
    terms = []
    for attribute in range(5):
        mask = role_mask[:, attribute] & (quality[:, attribute] > 0)
        if mask.any():
            selected = logits[:, attribute].permute(0, 2, 3, 1)[mask]
            per_pixel = F.cross_entropy(
                selected.float(), role_targets[:, attribute][mask], reduction="none"
            )
            weights = quality[:, attribute][mask]
            # Rescaling preserves the weighted mean and prevents a large sum overflowing.
            weights = weights / weights.max()
            terms.append((per_pixel * weights).sum() / weights.sum())
    roles = torch.stack(terms).mean() if terms else logits.sum() * 0
    support = _binary_loss(
        prediction.support_logits, support_targets, support_valid_mask, valid, "support"
    )
    applicability = _binary_loss(
        prediction.applicability_logits,
        applicability_targets,
        applicability_valid_mask,
        valid,
        "applicability",
    )
    paired = logits.sum() * 0
    if paired_prediction is not None or pair_valid_mask is not None or pair_weight > 0:
        if paired_prediction is None or pair_valid_mask is None:
            raise ValueError(
                "paired loss requires both paired_prediction and pair_valid_mask"
            )
        if paired_prediction.role_logits.shape != logits.shape:
            raise ValueError("paired role prediction shapes must match")
        pair_mask = _mask(
            role_targets,
            pair_valid_mask,
            role_targets.shape,
            valid & paired_prediction.valid_mask,
            "paired",
        )
        if pair_mask.any():
            first = logits[:, :, 1] - logits[:, :, 0]
            second = (
                paired_prediction.role_logits[:, :, 1]
                - paired_prediction.role_logits[:, :, 0]
            )
            paired = F.relu(margin - (second - first))[pair_mask].float().mean()
    return {
        "total": (
            roles
            + object_weight * support
            + applicability_weight * applicability
            + pair_weight * paired
        ),
        "roles": roles,
        "support": support,
        "applicability": applicability,
        "paired": paired,
    }


def validate_source_scene_splits(
    training_scene_ids, calibration_scene_ids, evaluation_scene_ids=None
):
    """Reject source-scene overlap using caller-supplied canonical scene IDs.

    Repeated instructions within one scene are allowed within a split. This check
    cannot discover aliases or incorrectly assigned IDs; dataset names and this
    API alone are not evidence that actual images were independently partitioned.
    """
    partitions = [
        ("training", training_scene_ids),
        ("calibration", calibration_scene_ids),
    ]
    if evaluation_scene_ids is not None:
        partitions.append(("evaluation", evaluation_scene_ids))
    checked = []
    for name, values in partitions:
        if isinstance(values, (str, bytes, Mapping)):
            raise TypeError(
                f"{name} requires an explicit nonempty collection of source scene IDs"
            )
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ValueError(f"{name} requires source scene IDs") from exc
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{name} source scene IDs must be nonempty strings")
        ids = {value.strip() for value in values}
        for previous_name, previous_ids in checked:
            overlap = ids & previous_ids
            if overlap:
                raise ValueError(
                    f"source scene overlap between {previous_name} and {name}: {sorted(overlap)}"
                )
        checked.append((name, ids))


@torch.no_grad()
def fit_role_temperatures(
    role_logits,
    role_targets,
    role_valid_mask,
    *,
    minimum=0.05,
    maximum=100.0,
    iterations=64,
):
    """Fit five temperatures by held-out NLL; return detached CPU float32 values.

    Inputs are logits B,5,3,H,W and explicit labels/mask B,5,H,W; the mask must
    already exclude invalid canvas locations and missing annotations. Each
    attribute needs observed labels. Invalid masked labels/logits are ignored.
    A bounded scalar search in inverse temperature uses convexity of multiclass
    NLL and includes T=1 as a candidate. It changes no network weights or metadata.
    Callers must validate source-scene partitioning before collecting held-out
    predictions. Fitting synthetic inputs is only a numerical diagnostic, never
    semantic calibration evidence or automatic confidence-threshold selection.
    """
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or not 0 < minimum <= 1 <= maximum
        or minimum >= maximum
    ):
        raise ValueError("temperature bounds must be finite, positive and contain one")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("temperature iterations must be a positive integer")
    if (
        role_logits.ndim != 5
        or role_logits.shape[1:3] != (5, 3)
        or min(role_logits.shape) < 1
        or not role_logits.is_floating_point()
    ):
        raise ValueError("calibration logits must be floating B,5,3,H,W")
    expected = role_logits.shape[:2] + role_logits.shape[3:]
    if (
        role_targets.shape != expected
        or role_targets.dtype != torch.long
        or role_valid_mask.shape != expected
        or role_valid_mask.dtype != torch.bool
    ):
        raise ValueError("calibration labels/mask must be int64/boolean B,5,H,W")
    if (
        role_targets.device != role_logits.device
        or role_valid_mask.device != role_logits.device
    ):
        raise ValueError("calibration inputs must share a device")
    temperatures = []
    for attribute in range(5):
        mask = role_valid_mask[:, attribute]
        if not mask.any():
            raise ValueError(
                f"held-out calibration needs observed labels for {ATTRIBUTE_ORDER[attribute]}"
            )
        logits = (
            role_logits[:, attribute].permute(0, 2, 3, 1)[mask].detach().double().cpu()
        )
        labels = role_targets[:, attribute][mask].detach().cpu()
        if not torch.isfinite(logits).all() or (labels < 0).any() or (labels > 2).any():
            raise ValueError(
                "observed calibration logits/labels must be finite with K/M/U labels"
            )
        true_logits = logits.gather(1, labels[:, None]).squeeze(1)
        low, high = 1.0 / maximum, 1.0 / minimum
        for _ in range(iterations):
            inverse = (low + high) / 2
            derivative = ((logits * inverse).softmax(1) * logits).sum(1) - true_logits
            if derivative.mean() > 0:
                high = inverse
            else:
                low = inverse
        candidates = (1.0, 1.0 / maximum, 1.0 / minimum, (low + high) / 2)
        best_inverse = min(
            candidates, key=lambda value: float(F.cross_entropy(logits * value, labels))
        )
        temperatures.append(1.0 / best_inverse)
    return torch.tensor(temperatures, dtype=torch.float32)


def _provenance(value):
    if not isinstance(value, dict) or value.get("kind") not in (
        "diagnostic",
        "semantic",
    ):
        raise ValueError("checkpoint requires diagnostic or semantic provenance")
    if value["kind"] == "diagnostic":
        if value.get("trained") is not False or value.get("calibrated") is not False:
            raise ValueError(
                "diagnostic checkpoints cannot claim trained semantic calibration"
            )
    else:
        required = (
            "training_dataset",
            "calibration_dataset",
            "annotation_source",
            "calibration_method",
        )
        if any(
            not isinstance(value.get(key), str) or not value[key].strip()
            for key in required
        ):
            raise ValueError(
                "semantic provenance requires training, calibration and annotation evidence"
            )
        if value.get("trained") is not True or value.get("calibrated") is not True:
            raise ValueError("semantic checkpoints must be trained and calibrated")
        steps = value.get("training_steps")
        if type(steps) is not int or steps < 1:
            raise ValueError("semantic provenance requires positive training_steps")
        if value["annotation_source"] != "explicit_attribute_roles":
            raise ValueError("semantic annotations must be explicit_attribute_roles")
        if value["training_dataset"] == value["calibration_dataset"]:
            raise ValueError("calibration must identify a held-out dataset")
        # Dataset names are metadata, not source-scene deduplication or training evidence.
    if any(
        not isinstance(key, str) or type(item) not in (str, bool, int, float)
        for key, item in value.items()
    ):
        raise ValueError("provenance must contain only primitive named fields")
    return dict(value)


def _validate_state(state):
    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            for key, value in state.items()
        )
    ):
        raise ValueError("checkpoint state must contain finite named tensors")
    temperature = state.get("role_temperatures")
    if temperature is None or temperature.shape != (5,) or (temperature <= 0).any():
        raise ValueError("checkpoint requires five positive role temperatures")


def save_attribute_checkpoint(
    model: AttributeRolePredictor, path, *, provenance=None
) -> None:
    """Save a self-contained CPU checkpoint; the default is diagnostic, never trained.

    Semantic provenance requires kind='semantic', trained=True, calibrated=True,
    distinct training_dataset/calibration_dataset IDs, positive training_steps,
    annotation_source='explicit_attribute_roles', and a calibration_method string.
    Temperatures are explicit state and should be fitted on the held-out split.
    """
    provenance = _provenance(
        provenance
        if provenance is not None
        else {"kind": "diagnostic", "trained": False, "calibrated": False}
    )
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    _validate_state(state)
    payload = {
        "schema": "boogu.attribute_roles",
        "version": 1,
        "architecture": ARCHITECTURE,
        "attributes": list(ATTRIBUTE_ORDER),
        "roles": list(ROLE_ORDER),
        "provenance": provenance,
        "state_dict": state,
    }
    torch.save(payload, Path(path))


def load_attribute_checkpoint(path, *, for_inference=True, device="cpu"):
    """Load with weights_only=True, strict architecture/schema and provenance checks.

    Inference defaults to requiring a semantically trained, calibrated artifact.
    for_inference=False permits diagnostic training/debugging only. Returns model
    in eval mode and primitive metadata; callers explicitly select train() to fit.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "boogu.attribute_roles"
        or type(payload.get("version")) is not int
        or payload["version"] != 1
    ):
        raise ValueError("unsupported attribute checkpoint schema/version")
    if (
        payload.get("architecture") != ARCHITECTURE
        or payload.get("attributes") != list(ATTRIBUTE_ORDER)
        or payload.get("roles") != list(ROLE_ORDER)
    ):
        raise ValueError("incompatible attribute checkpoint architecture/order")
    provenance = _provenance(payload.get("provenance"))
    if for_inference and provenance["kind"] != "semantic":
        raise ValueError(
            "diagnostic checkpoint is not a trained, calibrated semantic predictor"
        )
    state = payload.get("state_dict")
    _validate_state(state)
    model = AttributeRolePredictor()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata
