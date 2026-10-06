"""CPU contract tests for source-conditioned generation, independent of model weights."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _condition(source=None, *, keep_value=1.0):
    from boogu.preservation import prepare_condition

    source = (
        torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) / 16
        if source is None
        else source
    )
    return prepare_condition(
        source,
        torch.full((1, 5, 8, 8), keep_value, dtype=torch.float32),
        torch.zeros(1, 5, 8, 8),
    )


def test_prepare_uses_native_patch_order_and_detached_snapshots():
    from boogu.preservation import prepare_condition

    source = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4).requires_grad_()
    keep = torch.ones(1, 5, 8, 8, requires_grad=True)
    modify = torch.zeros_like(keep)
    condition = prepare_condition(source, keep, modify)
    assert condition.latent_shape == (1, 2, 4, 4)
    assert condition.patch_size == 2
    assert condition.source_tokens[0, 0].tolist() == [0, 16, 1, 17, 4, 20, 5, 21]
    assert condition.source_tokens[0, 1].tolist() == [2, 18, 3, 19, 6, 22, 7, 23]
    assert condition.roles.shape == (1, 4, 15)
    assert condition.roles[0, 0].tolist() == [1, 0, 0] * 5
    assert (
        not condition.source_tokens.requires_grad and not condition.roles.requires_grad
    )
    with torch.no_grad():
        source.zero_()
        keep.zero_()
    assert condition.source_tokens[0, 0, 1] == 16
    assert condition.roles[0, 0, 0] == 1


def test_prepare_invalid_reference_becomes_unknown_and_soft_pool_conserves_mass():
    from boogu.preservation import prepare_condition

    keep = torch.zeros(2, 5, 8, 8)
    modify = torch.zeros_like(keep)
    keep[0, :, :4, :4] = 0.5
    keep[1, :, :4, :4] = 0.25
    modify[0, 0, 4:, 4:] = 0.75
    valid = torch.ones_like(keep, dtype=torch.bool)
    valid[:, 1, :4, :4] = False
    result = prepare_condition(torch.ones(1, 2, 4, 4), keep, modify, valid)
    roles = result.roles.reshape(1, 4, 5, 3)
    assert roles[0, 0, 0].tolist() == [0.5, 0, 0.5]
    assert roles[0, 0, 1].tolist() == [0, 0, 1]
    assert roles[0, 3, 0].tolist() == [0, 0.75, 0.25]
    torch.testing.assert_close(roles.sum(-1), torch.ones(1, 4, 5))


def test_prepare_rejects_cross_object_conflict_before_invalid_reference_suppression():
    from boogu.preservation import prepare_condition

    keep = torch.zeros(2, 5, 8, 8)
    modify = torch.zeros_like(keep)
    keep[0, 2, 1, 1] = 0.2
    modify[1, 2, 1, 1] = 0.3
    with pytest.raises(ValueError, match="conflict|overlap"):
        prepare_condition(
            torch.zeros(1, 2, 4, 4),
            keep,
            modify,
            torch.zeros_like(keep, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "case",
    [
        "batch",
        "channels",
        "odd",
        "scale",
        "nan",
        "negative",
        "valid_dtype",
        "mismatch",
        "empty",
    ],
)
def test_prepare_rejects_bad_contract(case):
    from boogu.preservation import prepare_condition

    source = torch.zeros(1, 2, 4, 4)
    keep = torch.ones(1, 5, 8, 8)
    modify = torch.zeros_like(keep)
    valid = None
    if case == "batch":
        source = source.expand(2, -1, -1, -1)
    if case == "channels":
        keep = keep[:, :4]
    if case == "odd":
        source = source[:, :, :3]
    if case == "scale":
        keep = torch.ones(1, 5, 8, 12)
        modify = torch.zeros_like(keep)
    if case == "nan":
        source[0, 0, 0, 0] = float("nan")
    if case == "negative":
        keep[0, 0, 0, 0] = -1
    if case == "valid_dtype":
        valid = torch.ones_like(keep)
    if case == "mismatch":
        modify = modify[:, :, :4]
    if case == "empty":
        keep = keep[:0]
        modify = modify[:0]
    with pytest.raises((ValueError, TypeError)):
        prepare_condition(source, keep, modify, valid)


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, hidden, *args):
        return hidden + torch.tanh(self.linear(hidden))


class TinyNative(nn.Module):
    """Real torch layers mirroring native joint-token and checkpoint contracts."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, in_channels=2, patch_size=2)
        self.single_stream_layers = nn.ModuleList([TinyBlock(8) for _ in range(4)])
        self.layers = list(self.single_stream_layers)
        self.gradient_checkpointing = False
        self.enable_teacache = False
        self.enable_taylorseer = False

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        instruction_hidden_states,
        instruction_attention_mask,
        ref_image_hidden_states,
        freqs_cis=None,
        return_dict=False,
    ):
        if isinstance(instruction_hidden_states, (list, tuple)):
            instruction_hidden_states = torch.stack(instruction_hidden_states).mean(0)

        def tokens(value):
            return (
                value.reshape(1, 2, 2, 2, 2, 2)
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(1, 4, 8)
            )

        hidden = torch.cat(
            (
                instruction_hidden_states,
                tokens(ref_image_hidden_states[0][0]),
                tokens(hidden_states),
            ),
            1,
        )
        for layer in self.single_stream_layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden = self._gradient_checkpointing_func(layer, hidden)
            else:
                hidden = layer(hidden)
        return hidden


def _kwargs(source=None):
    source = (
        torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) / 16
        if source is None
        else source
    )
    return {
        "hidden_states": torch.randn(1, 2, 4, 4),
        "timestep": torch.tensor([0.4]),
        "instruction_hidden_states": torch.randn(1, 3, 8),
        "instruction_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "ref_image_hidden_states": [[source[0].clone()]],
        "freqs_cis": None,
        "return_dict": False,
    }


def _wrapper(*, checkpointing=True, **kwargs):
    from boogu.preservation import SourceConditionedTransformer

    torch.manual_seed(72)
    base = TinyNative()
    return SourceConditionedTransformer(
        base, block_indices=(1, 3), rank=4, checkpointing=checkpointing, **kwargs
    )


def _activate(wrapper):
    with torch.no_grad():
        for bank in wrapper.banks.values():
            bank.W_out.weight.normal_(std=0.2)
            bank.W_out.bias.fill_(0.1)


def test_zero_adapter_native_identity_inactive_bypass_and_base_frozen():
    wrapper = _wrapper()
    args = _kwargs()
    with torch.no_grad():
        expected = wrapper.base(**args)
        torch.testing.assert_close(
            wrapper(preservation=_condition(), **args), expected, rtol=0, atol=0
        )
        _activate(wrapper)
        torch.testing.assert_close(
            wrapper(preservation=None, **args), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(wrapper.base(**args), expected, rtol=0, atol=0)
    wrapper.train()
    assert not wrapper.base.training
    assert all(not p.requires_grad for p in wrapper.base.parameters())
    assert all(
        p.requires_grad and p.dtype == torch.float32
        for p in wrapper.adapter_parameters().values()
    )
    wrapper.close()


def test_only_generated_tokens_change_at_selected_block_output():
    wrapper = _wrapper(checkpointing=False)
    _activate(wrapper)
    args = _kwargs()
    observed = []
    handle = wrapper.base.single_stream_layers[1].register_forward_hook(
        lambda m, a, out: observed.append(out.detach().clone())
    )
    with torch.no_grad():
        wrapper.base(**args)
        wrapper(preservation=_condition(), **args)
    assert len(observed) == 2
    torch.testing.assert_close(observed[0][:, :7], observed[1][:, :7], rtol=0, atol=0)
    assert not torch.equal(observed[0][:, 7:], observed[1][:, 7:])
    handle.remove()
    wrapper.close()


@pytest.mark.parametrize("channel", ["source", "roles"])
def test_nonzero_adapter_uses_source_and_role_conditions_and_ablation(channel):
    kwargs = {"use_source": channel != "source", "use_roles": channel != "roles"}
    active = _wrapper(checkpointing=False)
    ablated = _wrapper(checkpointing=False, **kwargs)
    _activate(active)
    with torch.no_grad():
        for name, value in active.adapter_parameters().items():
            ablated.adapter_parameters()[name].copy_(value)
    args = _kwargs()
    first = _condition()
    second = (
        _condition(torch.zeros(1, 2, 4, 4))
        if channel == "source"
        else _condition(keep_value=0.0)
    )
    with torch.no_grad():
        assert not torch.equal(
            active(preservation=first, **args), active(preservation=second, **args)
        )
        torch.testing.assert_close(
            ablated(preservation=first, **args),
            ablated(preservation=second, **args),
            rtol=0,
            atol=0,
        )
    active.close()
    ablated.close()


def test_checkpoint_backward_after_scope_exit_matches_uncheckpointed_and_snapshots():
    plain = _wrapper(checkpointing=False)
    checked = _wrapper(checkpointing=True)
    _activate(plain)
    checked.load_state(plain.export_state())
    args = _kwargs()
    first = _condition()
    second = _condition()
    out_plain = plain(preservation=first, **args)
    out_checked = checked(preservation=second, **args)
    second.source_tokens.zero_()
    second.roles.fill_(7)
    out_plain.square().mean().backward()
    out_checked.square().mean().backward()
    for name, param in plain.adapter_parameters().items():
        other = checked.adapter_parameters()[name]
        assert param.grad is not None and other.grad is not None
        torch.testing.assert_close(param.grad, other.grad)
    assert checked.adapter_parameters()["banks.1.W_out.weight"].grad.abs().sum() > 0
    assert all(p.grad is None for p in checked.base.parameters())
    plain.close()
    checked.close()


def test_optimizer_updates_only_bank_and_fp32_survives_cast_and_autocast():
    wrapper = _wrapper()
    original = {n: p.detach().clone() for n, p in wrapper.base.named_parameters()}
    bank_original = {
        n: p.detach().clone() for n, p in wrapper.adapter_parameters().items()
    }
    wrapper.to(dtype=torch.bfloat16)
    for name, value in wrapper.adapter_parameters().items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, bank_original[name], rtol=0, atol=0)
    args = _kwargs()
    for name in ("hidden_states", "instruction_hidden_states"):
        args[name] = args[name].bfloat16()
    args["ref_image_hidden_states"][0][0] = args["ref_image_hidden_states"][0][
        0
    ].bfloat16()
    optim = torch.optim.SGD(wrapper.adapter_parameters().values(), lr=0.01)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = wrapper(preservation=_condition(), **args)
    result.float().square().mean().backward()
    optim.step()
    assert any(
        not torch.equal(bank_original[n], p)
        for n, p in wrapper.adapter_parameters().items()
    )
    for name, param in wrapper.base.named_parameters():
        torch.testing.assert_close(param, original[name].bfloat16(), rtol=0, atol=0)
        assert param.grad is None
    wrapper.close()


@pytest.mark.parametrize(
    "invalid",
    ["padding", "reference_count", "reference_size", "time", "shape", "roles"],
)
def test_forward_rejects_unsupported_conditioning_contract(invalid):
    wrapper = _wrapper()
    args = _kwargs()
    condition = _condition()
    if invalid == "padding":
        args["instruction_attention_mask"][0, -1] = False
    if invalid == "reference_count":
        args["ref_image_hidden_states"][0].append(args["ref_image_hidden_states"][0][0])
    if invalid == "reference_size":
        args["ref_image_hidden_states"][0][0] = torch.zeros(2, 4, 2)
    if invalid == "time":
        args["timestep"] = torch.tensor([400.0])
    if invalid == "shape":
        args["hidden_states"] = torch.zeros(2, 2, 4, 4)
    if invalid == "roles":
        condition.roles.fill_(float("nan"))
    with pytest.raises((ValueError, TypeError)):
        wrapper(preservation=condition, **args)
    wrapper.close()


def test_export_reload_strict_atomic_validation_and_metadata():
    wrapper = _wrapper(use_roles=False)
    _activate(wrapper)
    state = wrapper.export_state()
    assert set(state) == {"metadata", "adapters"}
    assert state["metadata"]["block_indices"] == [1, 3]
    assert state["metadata"]["use_roles"] is False
    assert all(
        t.dtype == torch.float32 and t.device.type == "cpu"
        for t in state["adapters"].values()
    )
    other = _wrapper(use_roles=False)
    other.load_state(state)
    args = _kwargs()
    with torch.no_grad():
        torch.testing.assert_close(
            wrapper(preservation=_condition(), **args),
            other(preservation=_condition(), **args),
            rtol=0,
            atol=0,
        )
    before = other.export_state()
    for corruption in (
        "metadata",
        "metadata_type",
        "metadata_bool",
        "nan",
        "dtype",
        "shape",
        "missing",
        "extra",
        "subclass",
    ):
        bad = copy.deepcopy(state)
        names = list(bad["adapters"])
        bad["adapters"][names[0]].add_(1)
        if corruption == "metadata":
            bad["metadata"]["rank"] = 999
        if corruption == "metadata_type":
            bad["metadata"]["rank"] = float(bad["metadata"]["rank"])
        if corruption == "metadata_bool":
            bad["metadata"]["use_roles"] = 0
        if corruption == "nan":
            bad["adapters"][names[-1]].fill_(float("nan"))
        if corruption == "dtype":
            bad["adapters"][names[-1]] = bad["adapters"][names[-1]].bfloat16()
        if corruption == "shape":
            bad["adapters"][names[-1]] = torch.zeros(999)
        if corruption == "missing":
            del bad["adapters"][names[-1]]
        if corruption == "extra":
            bad["adapters"]["base.weight"] = torch.ones(1)
        if corruption == "subclass":
            bad["adapters"][names[-1]] = nn.Parameter(bad["adapters"][names[-1]])
        with pytest.raises((ValueError, TypeError)):
            other.load_state(bad)
        for name, value in other.export_state()["adapters"].items():
            torch.testing.assert_close(value, before["adapters"][name], rtol=0, atol=0)
    wrapper.close()
    other.close()


def test_close_restores_original_checkpoint_function_and_keeps_foreign_hooks():
    from boogu.preservation import SourceConditionedTransformer

    base = TinyNative()
    original_function = lambda layer, *args: layer(*args)
    base.gradient_checkpointing = True
    base._gradient_checkpointing_func = original_function
    seen = []
    foreign = base.single_stream_layers[1].register_forward_hook(
        lambda m, a, o: seen.append(1)
    )
    layers = list(base.layers)
    wrapper = SourceConditionedTransformer(base, block_indices=(1,), rank=4)
    assert base.gradient_checkpointing
    assert base.layers == layers
    with pytest.raises(RuntimeError, match="already|installed"):
        SourceConditionedTransformer(base, block_indices=(1,), rank=4)
    wrapper.close()
    assert (
        base.gradient_checkpointing is True
        and base._gradient_checkpointing_func is original_function
    )
    assert all(not p.requires_grad for p in base.parameters())
    base(**_kwargs())
    assert seen == [1]
    foreign.remove()
    wrapper.close()


def test_close_guards_pending_backward_and_restores_absent_checkpoint_function():
    wrapper = _wrapper()
    result = wrapper(preservation=_condition(), **_kwargs())
    with pytest.raises(RuntimeError, match="backward"):
        wrapper.close()
    result.sum().backward()
    wrapper.close()
    assert wrapper.base.gradient_checkpointing is False
    assert not hasattr(wrapper.base, "_gradient_checkpointing_func")


def test_close_guards_graph_when_only_downstream_loss_is_kept():
    wrapper = _wrapper()
    loss = wrapper(preservation=_condition(), **_kwargs()).sum()
    with pytest.raises(RuntimeError, match="backward"):
        wrapper.close()
    loss.backward()
    wrapper.close()


@pytest.mark.parametrize(
    "unsupported", ["teacache", "taylorseer", "offload", "cache_hook"]
)
def test_unsupported_caches_offload_fail_closed_without_mutating_base(unsupported):
    from boogu.preservation import SourceConditionedTransformer

    base = TinyNative()
    if unsupported == "teacache":
        base.enable_teacache = True
    if unsupported == "taylorseer":
        base.single_stream_layers[0].enable_taylorseer = True
    if unsupported == "offload":
        base._hf_hook = object()
    if unsupported == "cache_hook":
        base._diffusers_hook = object()
    with pytest.raises((ValueError, RuntimeError), match="cache|offload|Taylor|hook"):
        SourceConditionedTransformer(base, block_indices=(1,), rank=4)
    assert base.gradient_checkpointing is False
    assert all(p.requires_grad for p in base.parameters())


def test_native_instruction_feature_list_is_supported_with_matching_actual_mask():
    wrapper = _wrapper(checkpointing=False)
    args = _kwargs()
    args["instruction_hidden_states"] = [
        args["instruction_hidden_states"],
        args["instruction_hidden_states"].clone(),
    ]
    with torch.no_grad():
        torch.testing.assert_close(
            wrapper(preservation=_condition(), **args),
            wrapper.base(**args),
            atol=0,
            rtol=0,
        )
    args["instruction_hidden_states"][1] = torch.zeros(1, 4, 8)
    with pytest.raises(ValueError, match="instruction|mask"):
        wrapper(preservation=_condition(), **args)
    wrapper.close()


def test_load_only_accepts_plain_fp32_tensors():
    wrapper = _wrapper()
    state = wrapper.export_state()
    name = next(iter(state["adapters"]))
    state["adapters"][name] = nn.Parameter(state["adapters"][name])
    with pytest.raises((TypeError, ValueError), match="FP32|tensor"):
        wrapper.load_state(state)
    wrapper.close()


def test_nested_bypass_and_exception_restore_condition_context():
    wrapper = _wrapper(checkpointing=False)
    _activate(wrapper)
    args = _kwargs()
    with torch.no_grad():
        expected_native = wrapper.base(**args)
        expected_conditioned = wrapper(preservation=_condition(), **args)
    nesting = [False]
    nested_outputs = []

    def nested(module, inputs, output):
        if not nesting[0]:
            nesting[0] = True
            nested_outputs.append(wrapper(preservation=None, **args))
            nesting[0] = False

    handle = wrapper.base.single_stream_layers[0].register_forward_hook(nested)
    with torch.no_grad():
        torch.testing.assert_close(
            wrapper(preservation=_condition(), **args),
            expected_conditioned,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(nested_outputs[0], expected_native, rtol=0, atol=0)
    handle.remove()

    def fail(module, inputs, output):
        raise RuntimeError("injected block failure")

    handle = wrapper.base.single_stream_layers[2].register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="injected block"):
        wrapper(preservation=_condition(), **args)
    handle.remove()
    with torch.no_grad():
        torch.testing.assert_close(
            wrapper.base(**args), expected_native, rtol=0, atol=0
        )
        torch.testing.assert_close(
            wrapper(preservation=_condition(), **args),
            expected_conditioned,
            rtol=0,
            atol=0,
        )
    wrapper.close()


def test_forward_condition_is_snapshotted_before_native_hooks_can_mutate_caller():
    wrapper = _wrapper(checkpointing=False)
    _activate(wrapper)
    args = _kwargs()
    condition = _condition()
    with torch.no_grad():
        expected = wrapper(preservation=condition, **args)

    def mutate(module, inputs):
        condition.source_tokens.zero_()
        condition.roles.zero_()

    handle = wrapper.base.single_stream_layers[0].register_forward_pre_hook(mutate)
    with torch.no_grad():
        actual = wrapper(preservation=condition, **args)
    handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    wrapper.close()


def test_two_outstanding_checkpointed_forwards_keep_separate_conditions():
    plain = _wrapper(checkpointing=False)
    checked = _wrapper(checkpointing=True)
    _activate(plain)
    checked.load_state(plain.export_state())
    args = _kwargs()
    conditions = [_condition(), _condition(torch.zeros(1, 2, 4, 4), keep_value=0)]
    expected = [plain(preservation=c, **args) for c in conditions]
    actual = [checked(preservation=c, **args) for c in conditions]
    sum(out.square().mean() for out in expected).backward()
    sum(out.square().mean() for out in actual).backward()
    for name, value in plain.adapter_parameters().items():
        torch.testing.assert_close(value.grad, checked.adapter_parameters()[name].grad)
    plain.close()
    checked.close()


def test_failed_installation_restores_flags_freeze_and_owned_hooks(monkeypatch):
    from boogu.preservation import SourceConditionedTransformer

    base = TinyNative()
    base.single_stream_layers[0].eval()
    before_training = [module.training for module in base.modules()]

    def fail(*args, **kwargs):
        raise RuntimeError("hook registration failed")

    monkeypatch.setattr(base.single_stream_layers[3], "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        SourceConditionedTransformer(base, block_indices=(1, 3), rank=4)
    assert base.gradient_checkpointing is False
    assert not hasattr(base, "_gradient_checkpointing_func")
    assert [module.training for module in base.modules()] == before_training
    assert all(p.requires_grad for p in base.parameters())
    assert len(base.single_stream_layers[1]._forward_hooks) == 0
    assert not hasattr(base, "_source_preservation_owner")


def test_small_real_native_model_zero_identity_and_checkpoint_gradient():
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from boogu.preservation import SourceConditionedTransformer

    torch.manual_seed(19)
    base = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=2,
        hidden_size=24,
        num_layers=4,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        multiple_of=8,
        axes_dim_rope=(4, 4, 4),
        axes_lens=(32, 32, 32),
        instruction_feature_configs={
            "instruction_feat_dim": 8,
            "reduce_type": "mean",
            "num_instruction_feat_layers": 1,
        },
    )
    # Native initialization zeros the final projection; randomize that fixture
    # so this integration detects whether adapter effects reach the output.
    with torch.no_grad():
        base.norm_out.linear_2.weight.normal_(std=0.1)
        for block in base.single_stream_layers:
            block.norm1.linear.weight.normal_(std=0.01)
    args = _kwargs()
    args["freqs_cis"] = base.rope_embedder.get_freqs_cis((4, 4, 4), (32, 32, 32), 10000)
    wrapper = SourceConditionedTransformer(base, block_indices=(0, 2), rank=4)
    with torch.no_grad():
        native = base(**args)
        torch.testing.assert_close(
            wrapper(preservation=_condition(), **args), native, rtol=0, atol=0
        )
    _activate(wrapper)
    output = wrapper(preservation=_condition(), **args)
    assert not torch.equal(output, native)
    output.square().mean().backward()
    for index in (0, 2):
        assert (
            wrapper.adapter_parameters()[f"banks.{index}.W_out.weight"].grad.abs().sum()
            > 0
        )
    assert all(parameter.grad is None for parameter in base.parameters())
    wrapper.close()
