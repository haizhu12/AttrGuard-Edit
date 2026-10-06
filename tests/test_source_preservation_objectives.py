"""Continuous protection supervision must actually react to kept-detail changes."""

import importlib
import importlib.util

import pytest
import torch


def api():
    assert importlib.util.find_spec("boogu.preservation") is not None, (
        "preservation package missing"
    )
    assert importlib.util.find_spec("boogu.preservation.objectives") is not None, (
        "continuous objective missing"
    )
    return importlib.import_module("boogu.preservation.objectives")


def data():
    source = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(5))
    keep = torch.zeros(1, 5, 16, 16)
    return source, keep


def test_exact_reference_zero():
    source, keep = data()
    keep[:] = 1
    loss, details = api().preservation_loss(source, source, keep)
    assert loss.item() == 0
    assert all(term["loss"] == 0 for term in details.values())


@pytest.mark.parametrize("attribute", range(5))
def test_kept_attribute_perturbation_has_nonzero_gradient(attribute):
    source, keep = data()
    keep[:, attribute] = 1
    predicted = source.clone()
    predicted[:, :, 6:10, 6:10] *= 0.3
    if attribute == 0:
        predicted[:, 0, 6:10, 6:10] += 0.25
    predicted.requires_grad_()
    loss, details = api().preservation_loss(predicted, source, keep)
    assert loss.item() > 0 and details
    loss.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0


def test_unknown_or_modify_only_has_no_keep_signal():
    source, keep = data()
    predicted = (source * 0.3).requires_grad_()
    loss, details = api().preservation_loss(predicted, source, keep)
    loss.backward()
    assert loss.item() == 0 and details == {}
    assert predicted.grad.abs().sum() == 0


def test_unprotected_change_does_not_enter_pixel_or_stencil_loss():
    source, keep = data()
    keep[:, :, 5:11, 5:11] = 1
    predicted = source.clone()
    predicted[:, :, :5] = 0
    loss, _ = api().preservation_loss(predicted, source, keep)
    assert loss.item() == 0


def test_small_local_error_not_cancelled_by_region_mean_dead_zone():
    source, keep = data()
    keep[:, 4] = 1
    predicted = source.clone()
    predicted[:, :, 7, 7] += 1e-4
    predicted.requires_grad_()
    loss, _ = api().preservation_loss(predicted, source, keep)
    assert 0 < loss.item() < 1 / 255
    loss.backward()
    assert predicted.grad[:, :, 7, 7].abs().sum() > 0


@pytest.mark.parametrize("bad", ["nan", "shape", "negative", "integer"])
def test_bad_weights_rejected(bad):
    source, keep = data()
    if bad == "nan":
        keep[0, 0, 0, 0] = float("nan")
    if bad == "shape":
        keep = keep[:, :4]
    if bad == "negative":
        keep[0, 0, 0, 0] = -0.1
    if bad == "integer":
        keep = keep.long()
    with pytest.raises((ValueError, TypeError)):
        api().preservation_loss(source, source, keep)


def test_reference_and_labels_do_not_receive_gradients():
    source, keep = data()
    source.requires_grad_()
    keep[:, 4] = 1
    keep.requires_grad_()
    predicted = (source.detach() * 0.7).requires_grad_()
    loss, _ = api().preservation_loss(predicted, source, keep)
    loss.backward()
    assert source.grad is None and keep.grad is None
    assert predicted.grad.abs().sum() > 0
