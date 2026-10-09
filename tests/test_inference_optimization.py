"""CPU tests of frozen-weight, source-only inference latent optimization."""

import importlib
import importlib.util
import json
from types import SimpleNamespace

import pytest
import torch

from boogu.preservation.objectives import preservation_loss


@pytest.fixture
def api():
    class RequiredAPI:
        def __getattr__(self, member):
            name = "boogu.preservation.inference_optimization"
            assert importlib.util.find_spec(name) is not None, "inference optimizer is missing"
            return getattr(importlib.import_module(name), member)
    return RequiredAPI()


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=0.5, shift_factor=0.1)
        self.calls = 0
        self.eval()

    def decode(self, latent, return_dict=False):
        self.calls += 1
        return (torch.tanh(latent * self.gain),)


def inputs(dtype=torch.float32):
    source = torch.zeros(1, 3, 8, 8, requires_grad=True)
    keep = torch.zeros(2, 5, 8, 8)
    keep[0, 4, :, :4] = 1
    modify = torch.zeros_like(keep)
    modify[1, 4, :, 4:] = 1
    before = torch.full_like(source, 0.45, dtype=dtype)
    velocity = torch.full_like(before, 0.2)
    next_latent = (before.float() + 0.05 * velocity.float()).to(dtype)
    return source, keep, modify, before, velocity, next_latent


def run_step(optimizer, before, velocity, next_latent, **overrides):
    arguments = dict(before_latent=before, velocity=velocity, t=0.7, next_t=0.75, index=4)
    arguments.update(overrides)
    with torch.no_grad():
        return optimizer.step(next_latent, **arguments)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_decoder_decreases_keep_and_total_without_weight_or_source_gradients(api, dtype):
    source, keep, modify, before, velocity, next_latent = inputs(dtype)
    decoder = TinyDecoder()
    original_weight = decoder.gain.detach().clone()
    optimizer = api.InferenceLatentOptimizer(decoder, source, keep, modify)
    result = run_step(optimizer, before, velocity, next_latent)
    anchor = before.float() + 0.3 * velocity.float()
    effective_delta = (result.float() - next_latent.float()) / 0.75
    endpoint = anchor + effective_delta
    decode = lambda x: (decoder.decode(x / 0.5 + 0.1)[0] / 2 + 0.5).clamp(0, 1)
    merged_keep = keep.amax(0, keepdim=True)
    baseline = preservation_loss(decode(anchor), decode(source.detach()), merged_keep)[0]
    after = preservation_loss(decode(endpoint), decode(source.detach()), merged_keep)[0]
    assert after < baseline
    assert float(effective_delta.square().mean().sqrt()) <= 0.08
    assert torch.equal(result[..., 4:], next_latent[..., 4:])  # edit region is not pulled to source
    assert float((decode(endpoint)[..., 4:] - decode(source.detach())[..., 4:]).abs().mean()) > 0.2
    assert result.dtype == dtype and not result.requires_grad
    assert source.grad is None and decoder.gain.grad is None
    assert torch.equal(decoder.gain, original_weight)
    trace = optimizer.metadata()["trace"][-1]
    assert trace["after"]["objective"] < trace["before"]["objective"]
    assert trace["after"]["keep"] <= trace["before"]["keep"]
    assert trace["accepted_steps"] > 0
    assert trace["correction_rms"] <= 0.08
    assert optimizer.metadata()["training_steps"] == 0
    json.dumps(optimizer.metadata(), allow_nan=False)


@pytest.mark.parametrize("case", ["off", "unknown", "invalid_reference", "early", "late", "interval", "zero_gradient"])
def test_exact_noop_conditions(api, case):
    source, keep, modify, before, velocity, next_latent = inputs(torch.bfloat16)
    valid = None
    config = api.InferenceOptimizationConfig(mode="off" if case == "off" else "latent")
    if case == "unknown":
        keep.zero_()
    if case == "invalid_reference":
        valid = torch.zeros_like(keep, dtype=torch.bool)
    if case == "zero_gradient":
        before.zero_()
        velocity.zero_()
        next_latent.zero_()
    decoder = TinyDecoder()
    optimizer = api.InferenceLatentOptimizer(decoder, source, keep, modify, valid, config)
    decode_count = decoder.calls
    overrides = {"early": {"t": 0.5}, "late": {"t": 0.98, "next_t": 1.0}, "interval": {"index": 3}}.get(case, {})
    result = run_step(optimizer, before, velocity, next_latent, **overrides)
    assert result is next_latent and torch.equal(result, next_latent)
    if case != "zero_gradient":
        assert decoder.calls == decode_count


def test_sub_bf16_ulp_proposal_is_exact_noop_not_anchor_quantization(api):
    source, keep, modify, before, velocity, next_latent = inputs(torch.bfloat16)
    config = api.InferenceOptimizationConfig(step_rms=1e-8)
    optimizer = api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify, config=config)
    result = run_step(optimizer, before, velocity, next_latent)
    assert result is next_latent
    trace = optimizer.metadata()["trace"][-1]
    assert trace["accepted_steps"] == 0 and trace["backtracking_trials"] > 0


def test_rgb_masks_keep_original_resolution_and_validity_before_max(api):
    source, keep, modify, *_ = inputs()
    keep[0, 4, :, :4] = 0.3
    keep[1, 4, :, :4] = 0.8
    valid = torch.ones_like(keep, dtype=torch.bool)
    valid[1, 4, :, :4] = False
    optimizer = api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify, valid)
    assert optimizer.keep.shape == (1, 5, 8, 8)
    assert torch.all(optimizer.keep[:, 4, :, :4] == 0.3)
    keep.zero_()
    source.data.fill_(99)
    assert torch.all(optimizer.keep[:, 4, :, :4] == 0.3)


@pytest.mark.parametrize("field,value", [("start_time", -0.1), ("end_time", 1.1), ("interval", 0), ("inner_steps", True), ("step_rms", 0), ("max_delta_rms", float("nan")), ("trust_weight", -1), ("edit_anchor_weight", float("inf")), ("backtracking_steps", 0), ("gradient_epsilon", 0), ("mode", "typo")])
def test_config_rejects_invalid_values(api, field, value):
    with pytest.raises((TypeError, ValueError)):
        api.InferenceOptimizationConfig(**{field: value})


@pytest.mark.parametrize("case", ["nan_source", "nan_keep", "out_of_range", "shape", "conflict", "invalid_boolean", "trainable_decoder", "bf16_decoder"])
def test_constructor_rejects_malformed_inputs(api, case):
    source, keep, modify, *_ = inputs()
    source = source.detach()
    decoder = TinyDecoder()
    valid = None
    if case == "nan_source": source[0, 0, 0, 0] = float("nan")
    if case == "nan_keep": keep[0, 0, 0, 0] = float("nan")
    if case == "out_of_range": modify[0, 0, 0, 0] = 2
    if case == "shape": modify = modify[:1]
    if case == "conflict":
        modify[1, 4, :, :4] = 1
        valid = torch.zeros_like(keep, dtype=torch.bool)  # conflict precedes validity
    if case == "invalid_boolean": valid = torch.ones_like(keep)
    if case == "trainable_decoder": decoder.gain.requires_grad_(True)
    if case == "bf16_decoder": decoder.to(torch.bfloat16)
    with pytest.raises((TypeError, ValueError)):
        api.InferenceLatentOptimizer(decoder, source, keep, modify, valid)


@pytest.mark.parametrize("field,value", [("t", float("nan")), ("next_t", 0.6), ("index", -1)])
def test_step_rejects_invalid_schedule_inputs(api, field, value):
    source, keep, modify, before, velocity, next_latent = inputs()
    optimizer = api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify)
    with pytest.raises((TypeError, ValueError)):
        run_step(optimizer, before, velocity, next_latent, **{field: value})


def test_step_rejects_nonfinite_velocity(api):
    source, keep, modify, before, velocity, next_latent = inputs()
    optimizer = api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify)
    velocity[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run_step(optimizer, before, velocity, next_latent)


def test_epsilon_stabilizes_small_gradient_without_turning_it_into_zero(api):
    source, keep, modify, before, velocity, next_latent = inputs()
    config = api.InferenceOptimizationConfig(gradient_epsilon=1.0)
    optimizer = api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify, config=config)
    result = run_step(optimizer, before, velocity, next_latent)
    assert not torch.equal(result, next_latent)
    assert optimizer.metadata()["trace"][-1]["accepted_steps"] > 0


def test_noop_detaches_an_existing_sampler_graph(api):
    source, keep, modify, before, velocity, next_latent = inputs()
    optimizer = api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify)
    next_latent.requires_grad_(True)
    result = run_step(optimizer, before, velocity, next_latent, index=0)
    assert torch.equal(result, next_latent) and not result.requires_grad
    assert next_latent.grad is None


@pytest.mark.parametrize("config", [0, False, {}, "latent"])
def test_constructor_rejects_non_config_values(api, config):
    source, keep, modify, *_ = inputs()
    with pytest.raises(TypeError, match="config"):
        api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify, config=config)


@pytest.mark.parametrize("mode", ["rgb", "attribute"])
def test_edit_anchor_selector_and_permission_snapshot(api, mode):
    source, keep, modify, *_ = inputs()
    config = api.InferenceOptimizationConfig(edit_anchor=mode)
    optimizer = api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify, config=config)
    assert optimizer.metadata()["config"]["edit_anchor"] == mode
    assert torch.equal(optimizer.modify, modify)
    modify.zero_()
    assert optimizer.modify.sum() > 0


def test_edit_anchor_config_defaults_and_invalid_modes(api):
    assert api.InferenceOptimizationConfig().edit_anchor == "rgb"
    with pytest.raises(ValueError, match="edit_anchor"):
        api.InferenceOptimizationConfig(edit_anchor="semantic")
    with pytest.raises(ValueError, match="edit_anchor"):
        api.validate_optimization("off", edit_anchor="attribute")


def test_pixel_only_attribute_optimizer_replays_rgb_bitwise(api):
    source, keep, modify, before, velocity, next_latent = inputs(torch.bfloat16)
    results, traces = [], []
    for mode in ("rgb", "attribute"):
        optimizer = api.InferenceLatentOptimizer(TinyDecoder(), source, keep, modify,
            config=api.InferenceOptimizationConfig(edit_anchor=mode))
        results.append(run_step(optimizer, before, velocity, next_latent))
        traces.append(optimizer.metadata()["trace"])
    assert torch.equal(*results)
    assert traces[0] == traces[1]
