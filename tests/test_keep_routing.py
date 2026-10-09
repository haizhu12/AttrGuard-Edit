"""CPU tests of inference-only latent projection and source block-input routing."""

import importlib
import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from boogu.preservation.conditioning import prepare_condition


def test_keep_routing_module_exists():
    assert importlib.util.find_spec("boogu.preservation.keep_routing") is not None


@pytest.fixture
def routing_module():
    assert importlib.util.find_spec("boogu.preservation.keep_routing") is not None, (
        "Executable keep routing must be implemented"
    )
    return importlib.import_module("boogu.preservation.keep_routing")


def condition(*, all_keep=False):
    source = torch.arange(48, dtype=torch.float32).reshape(1, 2, 4, 6) / 7
    keep = torch.zeros(1, 5, 8, 12)
    keep[:, 4, :, :] = float(all_keep)
    keep[:, 4, :4, :4] = 1
    return source, prepare_condition(source, keep, torch.zeros_like(keep))


def test_eligibility_pixel_only_and_all_attribute_modify_veto(routing_module):
    source, cond = condition(all_keep=True)
    roles = cond.roles.reshape(1, 6, 5, 3)
    roles[0, 1, 4] = torch.tensor([0, 0, 1])  # Other keep is insufficient.
    roles[0, 1, 1] = torch.tensor([1, 0, 0])
    roles[0, 1, 0] = torch.tensor([0, 1, 0])
    roles[0, 2, 4] = torch.tensor([0.5, 0, 0.5])
    roles[0, 3, 4] = torch.tensor([0, 1, 0])
    roles[0, 4, 0] = torch.tensor([0, 0.1, 0.9])
    route = routing_module.KeepRouting(cond, torch.zeros_like(source))
    assert route.token_mask.tolist() == [[True, False, False, False, False, True]]
    assert route.metadata()["eligible_token_count"] == 2
    assert route.metadata()["eligible_token_fraction"] == pytest.approx(2 / 6)


def test_invalid_reference_and_mixed_pixel_cells_are_free(routing_module):
    source, _ = condition()
    keep = torch.ones(1, 5, 8, 12)
    valid = torch.ones_like(keep, dtype=torch.bool)
    valid[:, 4, :4, :4] = False
    valid[:, 4, 4, 4] = False
    cond = prepare_condition(source, keep, torch.zeros_like(keep), valid)
    assert routing_module.KeepRouting(cond, source).token_mask.tolist() == [
        [False, True, True, True, False, True]
    ]


def test_exact_nonsquare_unpatch_private_clones_and_time_direction(routing_module):
    source, cond = condition(all_keep=True)
    noise = torch.randn_like(source)
    expected_source, expected_noise = source.clone(), noise.clone()
    route = routing_module.KeepRouting(cond, noise)
    cond.source_tokens.zero_()
    cond.roles.zero_()
    noise.zero_()
    route.token_mask.zero_()
    assert route.token_mask.all()
    assert torch.equal(route.source_at(0), expected_noise)
    assert torch.equal(route.source_at(1), expected_source)
    assert torch.equal(route.source_at(0.25), 0.75 * expected_noise + 0.25 * expected_source)
    route.source_at(1).zero_()
    assert torch.equal(route.source_at(1), expected_source)


def test_projection_preserves_bf16_identity_and_free_cells(routing_module):
    source, cond = condition()
    current = torch.randn_like(source).bfloat16()
    route = routing_module.KeepRouting(cond, current)
    projected = route.project(current, 1)
    expected = current.clone()
    expected[:, :, :2, :2] = source[:, :, :2, :2].bfloat16()
    assert projected.dtype == torch.bfloat16
    assert torch.equal(projected, expected)
    assert not projected.requires_grad


@pytest.mark.parametrize("bad", ["type", "batch", "patch", "grid", "source_shape", "source_int", "source_nan", "roles_shape", "roles_range", "roles_sum", "noise_shape", "noise_int", "noise_nan"])
def test_malformed_route_contract_rejected(routing_module, bad):
    source, cond = condition()
    noise = source.clone()
    if bad == "type": cond = None
    if bad == "batch": cond = replace(cond, latent_shape=(2, 2, 4, 6))
    if bad == "patch": cond = replace(cond, patch_size=True)
    if bad == "grid": cond = replace(cond, latent_shape=(1, 2, 3, 6))
    if bad == "source_shape": cond = replace(cond, source_tokens=cond.source_tokens[:, :-1])
    if bad == "source_int": cond = replace(cond, source_tokens=cond.source_tokens.long())
    if bad == "source_nan": cond.source_tokens[0, 0, 0] = float("nan")
    if bad == "roles_shape": cond = replace(cond, roles=cond.roles[:, :, :-1])
    if bad == "roles_range": cond.roles[0, 0, 0] = -1
    if bad == "roles_sum": cond.roles[0, 0, 0] = 0.5
    if bad == "noise_shape": noise = noise[:, :, :, :-1]
    if bad == "noise_int": noise = noise.long()
    if bad == "noise_nan": noise[0, 0, 0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError)):
        routing_module.KeepRouting(cond, noise)


@pytest.mark.parametrize("time", [-0.01, 1.01, float("nan"), True, torch.ones(2)])
def test_bad_time_rejected(routing_module, time):
    source, cond = condition()
    with pytest.raises((TypeError, ValueError)):
        routing_module.KeepRouting(cond, source).source_at(time)


class RecordingBlock(nn.Module):
    def __init__(self, double=False):
        super().__init__()
        self.double = double
        self.seen = []

    def forward(self, hidden_states=None, *, img_hidden_states=None, text=None):
        hidden = img_hidden_states if self.double else hidden_states
        self.seen.append(hidden.detach().clone())
        # Cross-token mixing tests routing before attention and source isolation.
        hidden = hidden + hidden.mean(dim=1, keepdim=True)
        return (hidden, text) if self.double else hidden


class RecordingNative(nn.Module):
    def __init__(self, *, positional=False):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, in_channels=2, patch_size=2,
                                      num_layers=3, num_double_stream_layers=1)
        self.double_stream_layers = nn.ModuleList([RecordingBlock(True)])
        self.single_stream_layers = nn.ModuleList([RecordingBlock(), RecordingBlock()])
        self.positional = positional
        self.inputs = []
        self.failure = False
        self.skip = False

    def forward(self, *, hidden_states, instruction_hidden_states, **kwargs):
        self.inputs.append(hidden_states.detach().clone())
        def patch(value):
            return value.reshape(1, 2, 2, 2, 3, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 6, 8)
        ref = kwargs["ref_image_hidden_states"][0][0].unsqueeze(0)
        image = torch.cat((patch(ref), patch(hidden_states)), dim=1)
        text = instruction_hidden_states
        image, text = self.double_stream_layers[0](img_hidden_states=image, text=text)
        joint = torch.cat((text, image), dim=1)
        for index, block in enumerate(self.single_stream_layers):
            if self.skip and index == 1: continue
            joint = block(joint) if self.positional else block(hidden_states=joint)
        if self.failure: raise RuntimeError("deliberate model error")
        return joint[:, -6:].reshape(1, 2, 3, 2, 2, 2).permute(0, 5, 1, 3, 2, 4).reshape(1, 2, 4, 6)


def native_kwargs(latent, text_length):
    return dict(hidden_states=latent, timestep=torch.tensor([0.5]),
                instruction_hidden_states=torch.full((1, text_length, 8), float(text_length)),
                instruction_attention_mask=torch.ones(1, text_length),
                ref_image_hidden_states=[[torch.full_like(latent[0], -3)]],
                freqs_cis=None, return_dict=False)


@pytest.mark.parametrize("positional", [False, True])
def test_memory_replaces_only_selected_generated_inputs_and_cleans(routing_module, positional):
    from boogu.preservation import SourceConditionedTransformer
    source, cond = condition()
    base = RecordingNative(positional=positional)
    wrapper = SourceConditionedTransformer(base, block_indices=(0, 1), rank=2, checkpointing=False).eval()
    incoming, observers = {}, []
    for index, block in enumerate([*base.double_stream_layers, *base.single_stream_layers]):
        def observe(module, args, kwargs, index=index):
            hidden = args[0] if args else kwargs.get("img_hidden_states", kwargs.get("hidden_states"))
            incoming.setdefault(index, []).append(hidden.clone())
        observers.append(block.register_forward_pre_hook(observe, with_kwargs=True))
    with torch.no_grad():
        for bank in wrapper.banks.values(): bank.W_out.bias.fill_(11)
        route = routing_module.KeepRouting(cond, torch.zeros_like(source))
        with routing_module.SourceMemoryRouting(wrapper, route) as memory:
            memory.capture(**native_kwargs(route.source_at(0.5), 2))
            # First cached generated token must be cloned, not an alias.
            source_inputs = [b.seen[-1].clone() for b in [*base.double_stream_layers, *base.single_stream_layers]]
            memory.edit(preservation=cond, **native_kwargs(torch.ones_like(source) * 17, 3))
            for index, block in enumerate([*base.double_stream_layers, *base.single_stream_layers]):
                original, edited = source_inputs[index], block.seen[-1]
                assert torch.equal(edited[:, -6], original[:, -6])
                assert not torch.equal(edited[:, -5:], original[:, -5:])
                before = incoming[index][-1]
                assert torch.equal(edited[:, :-6], before[:, :-6])
                assert torch.equal(edited[:, -5:], before[:, -5:])
                if index == 0: assert torch.equal(edited[:, :6], original[:, :6])
                else: assert edited.shape[1] == 15 and original.shape[1] == 14
            assert not memory._cache
        assert all(len(b._forward_pre_hooks) == 1 for b in [*base.double_stream_layers, *base.single_stream_layers])
        for observer in observers: observer.remove()
        assert all(len(b._forward_hooks) == 1 for b in base.single_stream_layers)
        wrapper.close()


@pytest.mark.parametrize("failure", ["source", "edit", "skip", "layout"])
def test_memory_exception_removes_owned_hooks_and_cache(routing_module, failure):
    from boogu.preservation import SourceConditionedTransformer
    source, cond = condition()
    base = RecordingNative()
    wrapper = SourceConditionedTransformer(base, block_indices=(0,), rank=2, checkpointing=False).eval()
    route = routing_module.KeepRouting(cond, source)
    with torch.no_grad(), pytest.raises((ValueError, RuntimeError)):
        with routing_module.SourceMemoryRouting(wrapper, route) as memory:
            base.failure = failure == "source"
            base.skip = failure == "skip"
            kwargs = native_kwargs(source, 2)
            if failure == "layout": kwargs["instruction_attention_mask"] = torch.ones(1, 3)
            memory.capture(**kwargs)
            base.failure = failure == "edit"
            memory.edit(preservation=cond, **native_kwargs(source, 3))
    assert not memory._cache
    assert all(not b._forward_pre_hooks for b in [*base.double_stream_layers, *base.single_stream_layers])
    assert len(base.single_stream_layers[0]._forward_hooks) == 1
    wrapper.close()


@torch.no_grad()
def test_denoise_projection_uses_actual_shifted_timetable(routing_module):
    from boogu.preservation.runtime import _denoise
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
    source, cond = condition()
    noise = torch.randn_like(source).bfloat16()
    recorded = []
    def model(**kwargs):
        recorded.append(kwargs["hidden_states"].clone())
        return torch.ones_like(kwargs["hidden_states"])
    scheduler = FlowMatchEulerDiscreteScheduler(do_shift=True)
    result = _denoise(model, scheduler, noise, noise, cond,
                      (torch.ones(1, 2, 8), torch.ones(1, 2), None, None),
                      freqs_cis=None, steps=3, guidance=1.0, keep_mode="project")
    route = routing_module.KeepRouting(cond, noise)
    for value, time in zip(recorded, scheduler.timesteps):
        assert torch.equal(value[:, :, :2, :2], route.source_at(time).bfloat16()[:, :, :2, :2])
    assert torch.equal(result[:, :, :2, :2], source.bfloat16()[:, :, :2, :2])
    expected = _denoise(model, FlowMatchEulerDiscreteScheduler(do_shift=True), noise, noise, cond,
                        (torch.ones(1, 2, 8), torch.ones(1, 2), None, None),
                        freqs_cis=None, steps=3, guidance=1.0, keep_mode="off")
    assert torch.equal(result[:, :, :, 2:], expected[:, :, :, 2:])


@pytest.mark.parametrize("mode", ["project", "source-memory"])
def test_empty_mask_exact_off_equivalence_without_extra_source_forward(routing_module, mode):
    from boogu.preservation.runtime import _denoise
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
    source, cond = condition()
    roles = torch.zeros_like(cond.roles)
    roles[..., 2::3] = 1
    cond = replace(cond, roles=roles)
    calls = []
    def model(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"] * 0.3 + 1
    def run(mode):
        return _denoise(model, FlowMatchEulerDiscreteScheduler(do_shift=False), source, source, cond,
                        (torch.ones(1, 2, 8), torch.ones(1, 2), None, None),
                        freqs_cis=None, steps=3, guidance=1.0, keep_mode=mode)
    expected = run("off")
    with torch.no_grad(): result = run(mode)
    assert len(calls) == 6
    assert torch.equal(result, expected)


def test_denoise_memory_fresh_source_each_step_and_final_projection(routing_module):
    from boogu.preservation import SourceConditionedTransformer
    from boogu.preservation.runtime import _denoise
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
    source, cond = condition()
    base = RecordingNative()
    wrapper = SourceConditionedTransformer(base, block_indices=(0,), rank=2, checkpointing=False).eval()
    with torch.no_grad():
        wrapper.banks["0"].W_out.bias.fill_(9)
        noise = torch.randn_like(source)
        scheduler = FlowMatchEulerDiscreteScheduler(do_shift=True)
        encoded = (torch.full((1, 3, 8), 3.), torch.ones(1, 3), torch.full((1, 2, 8), 2.), torch.ones(1, 2))
        output = _denoise(wrapper, scheduler, noise, torch.zeros_like(noise), cond, encoded,
                          freqs_cis=None, steps=3, guidance=1, keep_mode="source-memory")
        assert len(base.inputs) == 6
        route = routing_module.KeepRouting(cond, noise)
        for index, time in enumerate(scheduler.timesteps):
            assert torch.equal(base.inputs[2 * index], route.source_at(time))
        assert torch.equal(output[:, :, :2, :2], source[:, :, :2, :2])
        assert all(not b._forward_pre_hooks for b in [*base.double_stream_layers, *base.single_stream_layers])
    wrapper.close()


def test_denoise_rejects_unknown_mode_and_memory_cfg_before_model_call():
    from boogu.preservation.runtime import _denoise
    source, cond = condition()
    for mode, guidance in (("typo", 1), ("source-memory", 4)):
        with pytest.raises(ValueError, match="mode|source-memory"):
            _denoise(None, None, source, source, cond, (None,) * 4,
                     freqs_cis=None, steps=1, guidance=guidance, keep_mode=mode)


def test_small_native_transformer_memory_wrapper_smoke(routing_module):
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from boogu.preservation import SourceConditionedTransformer
    from boogu.preservation.runtime import _denoise
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
    source, cond = condition()
    base = BooguImageTransformer2DModel(
        patch_size=2, in_channels=2, hidden_size=24, num_layers=4,
        num_double_stream_layers=1, num_refiner_layers=1, num_attention_heads=2,
        num_kv_heads=1, multiple_of=8, axes_dim_rope=(4, 4, 4), axes_lens=(32, 32, 32),
        instruction_feature_configs={"instruction_feat_dim": 8, "reduce_type": "mean", "num_instruction_feat_layers": 1},
    )
    wrapper = SourceConditionedTransformer(base, block_indices=(0, 2), rank=2, checkpointing=False).eval()
    with torch.no_grad():
        base.norm_out.linear_2.weight.normal_(std=0.1)
        wrapper.banks["0"].W_out.bias.fill_(1)
        output = _denoise(wrapper, FlowMatchEulerDiscreteScheduler(do_shift=False),
                          torch.zeros_like(source), source, cond,
                          (torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.long), torch.randn(1, 1, 8), torch.ones(1, 1, dtype=torch.long)),
                          freqs_cis=base.rope_embedder.get_freqs_cis((4, 4, 4), (32, 32, 32), 10000),
                          steps=2, guidance=1, keep_mode="source-memory")
        assert torch.isfinite(output).all()
        assert torch.equal(output[:, :, :2, :2], source[:, :, :2, :2])
    assert all(not b._forward_pre_hooks for b in [*base.double_stream_layers, *base.single_stream_layers])
    wrapper.close()


@pytest.mark.parametrize("bad", ["grad", "train", "cache", "cache_dit", "offload"])
def test_memory_rejects_unsafe_model_state(routing_module, bad):
    from boogu.preservation import SourceConditionedTransformer
    source, cond = condition()
    wrapper = SourceConditionedTransformer(RecordingNative(), block_indices=(0,), rank=2, checkpointing=False).eval()
    if bad == "train": wrapper.base.train()
    if bad == "cache": wrapper.base.enable_teacache = True
    if bad == "cache_dit": wrapper.base.enable_cache_dit_caching = True
    if bad == "offload": wrapper.base.single_stream_layers[0]._hf_hook = object()
    with torch.set_grad_enabled(bad == "grad"), pytest.raises((ValueError, RuntimeError)):
        with routing_module.SourceMemoryRouting(wrapper, routing_module.KeepRouting(cond, source)):
            pass
    wrapper.close()


@pytest.mark.parametrize("bad", ["time", "reference", "nested"])
def test_memory_rejects_incompatible_reuse(routing_module, bad):
    from boogu.preservation import SourceConditionedTransformer
    source, cond = condition()
    wrapper = SourceConditionedTransformer(RecordingNative(), block_indices=(0,), rank=2, checkpointing=False).eval()
    route = routing_module.KeepRouting(cond, source)
    with torch.no_grad():
        with routing_module.SourceMemoryRouting(wrapper, route) as memory:
            memory.capture(**native_kwargs(source, 2))
            kwargs = native_kwargs(source, 3)
            if bad == "time": kwargs["timestep"] = torch.tensor([0.8])
            if bad == "reference": kwargs["ref_image_hidden_states"][0][0].add_(1)
            with pytest.raises((RuntimeError, ValueError), match="time|reference|active|installed"):
                if bad == "nested":
                    with routing_module.SourceMemoryRouting(wrapper, route): pass
                else:
                    memory.edit(preservation=cond, **kwargs)
        assert not memory._cache
    wrapper.close()


@pytest.mark.parametrize("mode", ["project", "source-memory"])
def test_nonoff_denoise_rejects_grad_enabled(mode):
    from boogu.preservation.runtime import _denoise
    source, cond = condition()
    with pytest.raises(RuntimeError, match="no_grad|inference-only"):
        _denoise(None, None, source, source, cond, (None,) * 4,
                 freqs_cis=None, steps=1, guidance=1, keep_mode=mode)
