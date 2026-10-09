"""Inference edit anchors relax the current prediction, never supervise a target."""

import importlib
import importlib.util
import json

import pytest
import torch

def anchor(*args, **kwargs):
    module = "boogu.preservation.edit_anchoring"
    assert importlib.util.find_spec(module) is not None, "edit anchoring objective missing"
    return importlib.import_module(module).edit_anchor_loss(*args, **kwargs)


def data(height=12, width=12, objects=1):
    reference = torch.rand(
        1, 3, height, width, generator=torch.Generator().manual_seed(19)
    )
    return reference, torch.zeros(objects, 5, height, width)


def legacy(prediction, reference, modify):
    mask = modify.detach().float().amax(0, keepdim=True).amax(1, keepdim=True)
    mass = mask.sum() * 3
    return (
        ((prediction - reference.detach()).abs() * mask).sum() / mass
        if bool(mass > 0)
        else prediction.sum() * 0
    )


def terms(details, object_index=0):
    return {term["attribute"]: term for term in details["objects"][object_index]["active_terms"]}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rgb_is_exact_legacy_union_with_soft_overlapping_unequal_objects(dtype):
    reference, modify = data(objects=3)
    reference = reference.to(dtype)
    prediction = (reference * 0.2 + 0.3).requires_grad_()
    modify[0, 0, 1:4, 1:5] = 0.4
    modify[1, 2, 2:10, 2:11] = 0.8
    modify[2, 4, 5:11, 1:9] = 0.2
    result, details = anchor(prediction, reference, modify)
    assert torch.equal(result, legacy(prediction, reference, modify))
    assert details["mode"] == details["effective_representation"] == "rgb"
    assert details["kind"] == "prediction_anchor_relaxation"
    json.dumps(details, allow_nan=False)


def test_pixel_only_attribute_mode_is_exact_legacy_union():
    reference, modify = data(objects=3)
    modify[0, 4, 1:4, 1:5] = 0.4
    modify[1, 4, 2:10, 2:11] = 0.8
    modify[2, 4, 5:11, 1:9] = 0.2
    prediction = (reference * 0.2).requires_grad_()
    result, details = anchor(prediction, reference, modify, mode="attribute")
    assert torch.equal(result, legacy(prediction, reference, modify))
    assert details["mode"] == "attribute"
    assert details["effective_representation"] == "rgb"


def test_color_statistics_allow_spatial_permutation_despite_rgb_difference():
    reference = torch.zeros(1, 3, 8, 8)
    reference[..., :4] = 0.25
    reference[..., 4:] = 0.75
    prediction = reference.roll(4, -1).requires_grad_()
    modify = torch.zeros(1, 5, 8, 8)
    modify[:, 0] = 1
    relaxed, details = anchor(prediction, reference, modify, mode="attribute")
    rgb, _ = anchor(prediction, reference, modify, mode="rgb")
    assert relaxed.item() == 0
    assert rgb.item() == 0.5
    assert terms(details)["color"]["representation"] == "masked_rgb_mean_std"


@pytest.mark.parametrize("reference_value,prediction_value", [(0.0, 1.0), (1.0, 0.0), (0.25, 0.75)])
def test_color_uniform_shift_detects_value_and_has_finite_zero_variance_gradients(
    reference_value, prediction_value
):
    reference = torch.full((1, 3, 8, 8), reference_value, requires_grad=True)
    prediction = torch.full((1, 3, 8, 8), prediction_value, requires_grad=True)
    modify = torch.zeros(1, 5, 8, 8)
    modify[:, 0] = 1
    modify.requires_grad_()
    loss, _ = anchor(prediction, reference, modify, mode="attribute")
    assert loss.item() == pytest.approx(abs(prediction_value - reference_value))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert reference.grad is None and modify.grad is None


def test_color_standard_deviation_detects_contrast_with_equal_mean():
    reference = torch.full((1, 3, 8, 8), 0.5)
    prediction = torch.zeros_like(reference)
    prediction[..., ::2] = 1
    prediction.requires_grad_()
    modify = torch.zeros(1, 5, 8, 8)
    modify[:, 0] = 1
    loss, _ = anchor(prediction, reference, modify, mode="attribute")
    assert loss.item() == pytest.approx((0.25 + 1e-8) ** 0.5 - 1e-4, abs=1e-7)
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


def test_color_is_computed_per_object_so_swapping_object_colors_is_penalized():
    reference = torch.zeros(1, 3, 8, 12)
    reference[:, 0, :, :6] = 1
    reference[:, 2, :, 6:] = 1
    prediction = reference.flip(-1).requires_grad_()
    modify = torch.zeros(2, 5, 8, 12)
    modify[0, 0, :, :6] = 1
    modify[1, 0, :, 6:] = 1
    loss, details = anchor(prediction, reference, modify, mode="attribute")
    assert loss.item() == pytest.approx(2 / 3)
    assert details["active_objects"] == 2
    assert all(terms(details, index)["color"]["support_mass"] == 48 for index in range(2))


def test_identical_whole_object_duplication_does_not_rescale_loss():
    reference, modify = data()
    modify[:, 0, 1:9, 1:7] = 1
    modify[:, 1, 2:11, 2:11] = 1
    modify[:, 4, 3:10, 3:9] = 0.4
    prediction = reference * 0.4
    single, _ = anchor(prediction, reference, modify, mode="attribute")
    duplicate, _ = anchor(prediction, reference, modify.repeat(3, 1, 1, 1), mode="attribute")
    assert torch.equal(single, duplicate)


def test_objects_have_equal_votes_despite_unequal_support_masses():
    reference = torch.zeros(1, 3, 8, 12)
    prediction = torch.zeros_like(reference)
    prediction[..., :2] = 1
    prediction[..., 2:] = 0.25
    modify = torch.zeros(3, 5, 8, 12)
    modify[0, 4, :, :2] = 1
    modify[1, 0, :, 2:] = 1
    loss, details = anchor(prediction, reference, modify, mode="attribute")
    assert loss.item() == pytest.approx((1 + 0.25) / 2)
    assert details["active_objects"] == 2


def test_active_attributes_have_equal_votes_and_structure_counts_once():
    reference, modify = data()
    prediction = reference.clone()
    prediction[..., 4:8, 4:8] *= 0.2
    modify[:, (0, 1, 4)] = 1
    combined, details = anchor(prediction, reference, modify, mode="attribute")
    separate = []
    for attribute in (0, 1, 4):
        isolated = torch.zeros_like(modify)
        isolated[:, attribute] = 1
        loss, _ = anchor(prediction, reference, isolated, mode="attribute")
        separate.append(loss)
    assert torch.allclose(combined, torch.stack(separate).mean(), rtol=0, atol=1e-8)
    active = terms(details)
    assert set(active) == {"color", "structure", "pixel"}
    structure = active["structure"]
    components = structure["components"]
    assert set(components) == {"structure_finite", "structure_sobel"}
    assert structure["loss"] == pytest.approx(sum(item["loss"] for item in components.values()) / 2)


@pytest.mark.parametrize("attribute,radius,name", [(1, 1, "structure"), (2, 2, "texture")])
def test_derivative_erosion_prevents_unknown_border_leakage(attribute, radius, name):
    reference, modify = data()
    modify[:, attribute, 3:9, 3:9] = 1
    prediction = reference.clone()
    outside = torch.ones(12, 12, dtype=torch.bool)
    outside[3:9, 3:9] = False
    prediction[:, :, outside] = 1 - prediction[:, :, outside]
    prediction.requires_grad_()
    loss, details = anchor(prediction, reference, modify, mode="attribute")
    assert loss.item() == 0
    assert terms(details)[name]["support_mass"] == (6 - 2 * radius) ** 2
    loss.backward()
    assert prediction.grad[:, :, outside].abs().sum() == 0


@pytest.mark.parametrize("attribute,name", [(1, "structure"), (2, "texture")])
def test_empty_eroded_support_is_skipped_with_diagnostics(attribute, name):
    reference, modify = data()
    prediction = (reference * 0.5).requires_grad_()
    modify[:, attribute, 4:6, 4:6] = 1
    loss, details = anchor(prediction, reference, modify, mode="attribute")
    assert loss.item() == 0 and details["active_objects"] == 0
    skipped = details["objects"][0]["skipped_terms"]
    assert skipped == [{"attribute": name, "support_mass": 0.0, "original_support_mass": 4.0,
                        "reason": "empty_eroded_support"}]
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("attribute,name", [(3, "glyph"), (4, "pixel")])
def test_glyph_and_pixel_rgb_proxies_use_their_own_uneroded_mask(attribute, name):
    reference, modify = data()
    prediction = reference.clone()
    prediction[..., 4, 4] += 0.2
    modify[:, 0, 8:10, 8:10] = 1  # Avoid the global pixel-only fallback.
    modify[:, attribute, 4, 4] = 0.5
    loss, details = anchor(prediction, reference, modify, mode="attribute")
    assert terms(details)[name]["support_mass"] == 0.5
    assert terms(details)[name]["loss"] == pytest.approx(0.2)
    assert loss.item() == pytest.approx(0.1)


@pytest.mark.parametrize("mode", ["rgb", "attribute"])
def test_empty_modify_is_differentiable_zero(mode):
    reference, modify = data(objects=2)
    prediction = (reference * 0.5).requires_grad_()
    loss, details = anchor(prediction, reference, modify, mode=mode)
    assert loss.item() == 0 and details["active_objects"] == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("mode", ["rgb", "attribute"])
def test_reference_and_masks_are_detached(mode):
    reference, modify = data(objects=2)
    modify[0, 0, 1:9, 1:9] = 1
    modify[1, 2, 2:11, 2:11] = 1
    reference.requires_grad_()
    modify.requires_grad_()
    prediction = (reference.detach() * 0.5).requires_grad_()
    loss, _ = anchor(prediction, reference, modify, mode=mode)
    loss.backward()
    assert reference.grad is None and modify.grad is None
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["wrong", "RGB", None, True])
def test_invalid_mode_is_rejected(mode):
    reference, modify = data()
    with pytest.raises(ValueError, match="mode"):
        anchor(reference, reference, modify, mode=mode)


@pytest.mark.parametrize("bad", ["prediction_batch", "prediction_channels", "prediction_integer",
                                "prediction_nan", "reference_shape", "reference_integer",
                                "reference_inf", "modify_channels", "modify_spatial",
                                "modify_objects", "modify_integer", "modify_nan",
                                "modify_negative", "modify_above_one"])
def test_invalid_inputs_are_rejected(bad):
    reference, modify = data()
    prediction = reference.clone()
    if bad == "prediction_batch":
        prediction = prediction.repeat(2, 1, 1, 1)
    elif bad == "prediction_channels":
        prediction = prediction[:, :2]
    elif bad == "prediction_integer":
        prediction = prediction.long()
    elif bad == "prediction_nan":
        prediction[0, 0, 0, 0] = float("nan")
    elif bad == "reference_shape":
        reference = reference[..., :4]
    elif bad == "reference_integer":
        reference = reference.long()
    elif bad == "reference_inf":
        reference[0, 0, 0, 0] = float("inf")
    elif bad == "modify_channels":
        modify = modify[:, :4]
    elif bad == "modify_spatial":
        modify = modify[..., :4]
    elif bad == "modify_objects":
        modify = modify[:0]
    elif bad == "modify_integer":
        modify = modify.long()
    elif bad == "modify_nan":
        modify[0, 0, 0, 0] = float("nan")
    elif bad == "modify_negative":
        modify[0, 0, 0, 0] = -0.1
    elif bad == "modify_above_one":
        modify[0, 0, 0, 0] = 1.1
    with pytest.raises((ValueError, TypeError)):
        anchor(prediction, reference, modify, mode="attribute")
