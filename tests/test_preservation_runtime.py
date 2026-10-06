"""CPU contract checks for the staged source-conditioned production runtime."""

import hashlib
import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps


@pytest.fixture
def runtime():
    assert importlib.util.find_spec("boogu.preservation.runtime") is not None, (
        "The source-conditioned production runtime must exist"
    )
    return importlib.import_module("boogu.preservation.runtime")


def permission_payload(image):
    return {
        "keep": np.ones((1, 5, image.height, image.width), dtype=np.float32),
        "modify": np.zeros((1, 5, image.height, image.width), dtype=np.float32),
        "source_sha256": np.array(hashlib.sha256(image.tobytes()).hexdigest()),
        "schema": np.array("boogu_preservation_permissions_v1"),
        "attributes": np.array(["color", "structure", "texture", "glyph", "pixel"]),
    }


def test_canonical_hash_uses_exif_transposed_rgb(runtime):
    image = Image.new("RGBA", (16, 32), (10, 20, 30, 255))
    image.getexif()[274] = 6
    expected = hashlib.sha256(
        ImageOps.exif_transpose(image).convert("RGB").tobytes()
    ).hexdigest()
    assert runtime.source_rgb_sha256(image) == expected


def test_permission_round_trip(runtime, tmp_path):
    image = Image.new("RGB", (32, 16), "green")
    path = tmp_path / "permissions.npz"
    np.savez(path, **permission_payload(image))
    keep, modify, valid = runtime.load_permissions(path, image)
    assert tuple(keep.shape) == (1, 5, 16, 32)
    assert keep.dtype == torch.float32
    assert torch.all(keep == 1) and torch.all(modify == 0)
    assert valid.dtype == torch.bool and torch.all(valid)


@pytest.mark.parametrize(
    "bad",
    [
        "binding",
        "shape",
        "nan",
        "range",
        "conflict",
        "schema",
        "attributes",
        "valid_dtype",
        "object",
        "missing",
    ],
)
def test_permission_rejects_invalid_payload(runtime, tmp_path, bad):
    image = Image.new("RGB", (32, 16), "green")
    payload = permission_payload(image)
    if bad == "binding":
        payload["source_sha256"] = np.array("0" * 64)
    if bad == "shape":
        payload["keep"] = payload["keep"][:, :, :, :16]
    if bad == "nan":
        payload["keep"][0, 0, 0, 0] = np.nan
    if bad == "range":
        payload["keep"][0, 0, 0, 0] = 2
    if bad == "conflict":
        payload["modify"][0, 0, 0, 0] = 1
    if bad == "schema":
        payload["schema"] = np.array("another_schema")
    if bad == "attributes":
        payload["attributes"] = payload["attributes"][::-1]
    if bad == "valid_dtype":
        payload["reference_valid"] = np.ones_like(payload["keep"])
    if bad == "object":
        payload["keep"] = np.array([object()], dtype=object)
    if bad == "missing":
        del payload["source_sha256"]
    path = tmp_path / "bad.npz"
    np.savez(path, **payload)
    with pytest.raises((ValueError, TypeError)):
        runtime.load_permissions(path, image)


@pytest.mark.parametrize(
    "missing",
    [
        "preservation_adapter",
        "preservation_permissions",
        "input_image_paths",
        "instruction",
    ],
)
def test_required_inputs_fail_before_heavy_loading(runtime, missing, monkeypatch):
    args = SimpleNamespace(
        preservation_adapter="adapter.pt",
        preservation_permissions="permissions.npz",
        input_image_paths=["image.png"],
        instruction="Turn it blue",
        device="cuda",
        dtype="bf16",
    )
    setattr(args, missing, None)
    monkeypatch.setattr(
        runtime,
        "_load_encoding_components",
        lambda *a: pytest.fail("Loaded weights before preflight"),
    )
    with pytest.raises(ValueError, match=missing):
        runtime.run_source_conditioned(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_teacache", True),
        ("custom_transformer_lora_path", "weights.pt"),
        ("enable_group_offload_flag", True),
        ("use_batch_inference", True),
        ("scheduler", "dpmsolver++"),
        ("image_guidance_scale", 2),
        ("cfg_range_start", 0.2),
        ("batch_size", 2),
        ("num_images_per_instruction", 2),
        ("truncate_instruction_sequence", True),
        ("use_prompt_tuning", True),
        ("unknown_new_option", 1),
    ],
)
def test_unsupported_programmatic_options_fail_closed(runtime, field, value):
    with pytest.raises(ValueError, match=field):
        runtime._validate_options(SimpleNamespace(**{field: value}))


def test_even_explicit_inert_legacy_option_is_rejected(runtime):
    with pytest.raises(ValueError, match="teacache_rel_l1_thresh"):
        runtime._validate_options(
            SimpleNamespace(
                teacache_rel_l1_thresh=0.05,
                _explicit_options=("teacache_rel_l1_thresh",),
            )
        )


def adapter_state():
    config = {
        "hidden_size": 8,
        "in_channels": 16,
        "patch_size": 2,
        "num_layers": 4,
        "num_double_stream_layers": 1,
    }
    metadata = {
        "schema": "boogu_source_preservation_v1",
        "block_indices": [2],
        "rank": 2,
        "hidden_size": 8,
        "in_channels": 16,
        "patch_size": 2,
        "attributes": ["color", "structure", "texture", "glyph", "pixel"],
        "role_order": ["keep", "modify", "unknown"],
        "use_source": True,
        "use_roles": True,
    }
    tensors = {}
    for name, shape in {
        "W_h": (2, 8),
        "W_s": (2, 64),
        "W_r": (2, 15),
        "W_t": (2, 1),
        "W_out": (8, 2),
    }.items():
        tensors[f"banks.2.{name}.weight"] = torch.zeros(shape)
        tensors[f"banks.2.{name}.bias"] = torch.zeros(shape[0])
    return {"metadata": metadata, "adapters": tensors}, config


def test_adapter_validates_shapes_without_allocating_transformer(runtime):
    state, config = adapter_state()
    assert runtime._validate_adapter(state, config) == state["metadata"]


@pytest.mark.parametrize(
    "bad",
    [
        "rank_bool",
        "out_of_range",
        "wrong_architecture",
        "disabled_source",
        "extra_metadata",
        "missing_tensor",
        "shape",
        "nan",
        "dtype",
    ],
)
def test_adapter_preflight_rejects_corrupt_state(runtime, bad):
    state, config = adapter_state()
    if bad == "rank_bool":
        state["metadata"]["rank"] = True
    if bad == "out_of_range":
        state["metadata"]["block_indices"] = [3]
    if bad == "wrong_architecture":
        state["metadata"]["hidden_size"] = 10
    if bad == "disabled_source":
        state["metadata"]["use_source"] = False
    if bad == "extra_metadata":
        state["metadata"]["extra"] = 1
    key = "banks.2.W_h.weight"
    if bad == "missing_tensor":
        del state["adapters"][key]
    if bad == "shape":
        state["adapters"][key] = torch.zeros(2, 9)
    if bad == "nan":
        state["adapters"][key][0, 0] = float("nan")
    if bad == "dtype":
        state["adapters"][key] = state["adapters"][key].bfloat16()
    with pytest.raises(ValueError, match="adapter"):
        runtime._validate_adapter(state, config)


def test_compaction_removes_masked_tokens_in_each_feature_layer(runtime):
    feature = torch.arange(24).reshape(1, 4, 6).float()
    mask = torch.tensor([[1, 0, 1, 0]])
    layers, compact_mask = runtime._compact_condition([feature, feature + 1], mask)
    assert torch.equal(layers[0], feature[:, [0, 2]])
    assert torch.equal(layers[1], (feature + 1)[:, [0, 2]])
    assert compact_mask.tolist() == [[1, 1]]


def test_native_cfg_conditions_both_branches_and_uses_latent_pixel_count(runtime):
    from boogu.preservation.conditioning import prepare_condition
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler,
    )

    latent = torch.zeros(1, 16, 2, 4)
    condition = prepare_condition(
        latent, torch.ones(1, 5, 16, 32), torch.zeros(1, 5, 16, 32)
    )
    calls = []

    class Model:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return torch.full_like(
                kwargs["hidden_states"],
                float(kwargs["instruction_hidden_states"].mean()),
            )

    class Scheduler(FlowMatchEulerDiscreteScheduler):
        def set_timesteps(self, *args, **kwargs):
            assert kwargs["num_tokens"] == 8
            super().set_timesteps(*args, **kwargs)

    encoded = (
        torch.full((1, 1, 6), 2.0),
        torch.ones(1, 1),
        torch.ones(1, 1, 6),
        torch.ones(1, 1),
    )
    result = runtime._denoise(
        Model(),
        Scheduler(do_shift=False),
        latent,
        latent,
        condition,
        encoded,
        freqs_cis=None,
        steps=2,
        guidance=4.0,
    )
    assert torch.all(result == 5.0)  # positive + (4 - 1) * (positive - negative)
    assert len(calls) == 4 and all(call["preservation"] is condition for call in calls)


def test_runtime_has_no_research_script_dependency(runtime):
    import inspect

    source = inspect.getsource(runtime)
    assert "import inference" not in source
    assert "outputs/" not in source and "outputs\\" not in source


def test_cfg1_uses_only_positive_branch(runtime):
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
    calls = []
    latent = torch.zeros(1, 16, 2, 4)
    positive = torch.ones(1, 1, 6)
    def model(**kwargs):
        assert kwargs['instruction_hidden_states'] is positive
        calls.append(kwargs)
        return torch.ones_like(kwargs['hidden_states'])
    result = runtime._denoise(model, FlowMatchEulerDiscreteScheduler(do_shift=False),
        latent, latent, None, (positive, torch.ones(1,1), None, None),
        freqs_cis=None, steps=2, guidance=1.0)
    assert len(calls) == 2
    assert torch.equal(result, torch.ones_like(result))


@pytest.fixture
def inputs(runtime, tmp_path):
    state, config = adapter_state()
    config.update(
        instruction_feature_configs={
            "instruction_feat_dim": 6,
            "num_instruction_feature_layers": 1,
        },
        axes_dim_rope=[2, 2, 2],
        axes_lens=[16, 16, 16],
    )
    base = tmp_path / "Boogu-Image-0.1-Edit"
    for component in ("transformer", "vae", "mllm", "scheduler", "processor"):
        (base / component).mkdir(parents=True)
    vae_config = {
        "latent_channels": 16,
        "block_out_channels": [128, 256, 512, 512],
        "shift_factor": 0.1159,
        "scaling_factor": 0.3611,
    }
    mllm_config = {
        "model_type": "qwen3_vl",
        "text_config": {"hidden_size": 6, "num_hidden_layers": 3},
    }
    for component, value in (
        ("transformer", config),
        ("vae", vae_config),
        ("mllm", mllm_config),
    ):
        (base / component / "config.json").write_text(json.dumps(value))
        (base / component / "weights.safetensors").touch()
    (base / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"})
    )
    image = Image.new("RGB", (32, 16), "green")
    image.save(tmp_path / "source.png")
    np.savez(tmp_path / "permissions.npz", **permission_payload(image))
    torch.save(state, tmp_path / "adapter.pt")
    args = SimpleNamespace(
        preservation_adapter=str(tmp_path / "adapter.pt"),
        preservation_permissions=str(tmp_path / "permissions.npz"),
        input_image_paths=[str(tmp_path / "source.png")],
        pretrained_pipeline_name_or_path=str(base),
        output_image_path=str(tmp_path / "output.png"),
        instruction="Make it blue",
        device="cuda",
        dtype="bf16",
        seed=42,
        num_inference_steps=2,
    )
    return args


def test_complete_preflight_stays_cpu_and_derives_independent_seed(runtime, inputs):
    checked = runtime._preflight(inputs)
    assert checked.seed == 42 and checked.reference_seed == 43
    assert checked.image.size == (32, 16)
    assert all(
        tensor.device.type == "cpu" for tensor in checked.state["adapters"].values()
    )


def test_initial_noise_matches_native_cpu_generator_without_changing_global_rng(
    runtime,
):
    before = torch.get_rng_state().clone()
    actual = runtime._initial_noise((1, 16, 2, 4), 42, torch.device("cpu"))
    expected = torch.randn(
        (1, 16, 2, 4),
        generator=torch.Generator(device="cpu").manual_seed(42),
        dtype=torch.bfloat16,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), before)


def test_preflight_hashes_detect_changed_input_before_metadata(runtime, inputs):
    checked = runtime._preflight(inputs)
    from pathlib import Path

    Path(inputs.preservation_adapter).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        runtime._assert_input_hashes(checked)


@pytest.mark.parametrize(
    "bad", ["qwen", "feature_dim", "axes", "shard", "quantization"]
)
def test_native_configuration_and_shard_failures_precede_loading(runtime, inputs, bad):
    from pathlib import Path

    base = Path(inputs.pretrained_pipeline_name_or_path)
    if bad in ("qwen", "feature_dim", "quantization"):
        path = base / "mllm" / "config.json"
        config = json.loads(path.read_text())
        if bad == "qwen":
            config["model_type"] = "other_model"
        if bad == "feature_dim":
            config["text_config"]["hidden_size"] = 9
        if bad == "quantization":
            config["quantization_config"] = {"quant_method": "fp8"}
    elif bad == "axes":
        path = base / "transformer" / "config.json"
        config = json.loads(path.read_text())
        config["axes_dim_rope"] = [2]
    else:
        path = base / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        config = {"weight_map": {"weight": "missing_shard.safetensors"}}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        runtime._preflight(inputs)


def test_invalid_requested_cuda_ordinal_fails_before_device_context(
    runtime, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda index: pytest.fail("entered invalid device context"),
    )
    with pytest.raises(ValueError, match="device"):
        runtime._check_cuda_device(torch.device("cuda:2"))


def test_bf16_is_checked_on_requested_cuda_device(runtime, monkeypatch):
    from contextlib import contextmanager

    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    @contextmanager
    def device_context(index):
        selected.append(index)
        yield

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: selected[-1] == 0)
    with pytest.raises(ValueError, match="BF16"):
        runtime._check_cuda_device(torch.device("cuda:1"))
    assert selected == [1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("height", 32),
        ("width", 64),
        ("max_input_image_pixels", 100),
        ("max_input_image_side_length", 16),
        ("num_inference_steps", 0),
        ("text_guidance_scale", float("nan")),
        ("seed", -1),
        ("reference_seed", 42),
        ("dtype", "fp16"),
        ("device", "cpu"),
    ],
)
def test_preflight_rejects_unsupported_numeric_or_resource_settings(
    runtime, inputs, field, value
):
    setattr(inputs, field, value)
    with pytest.raises(ValueError):
        runtime._preflight(inputs)


@pytest.mark.parametrize("bad", ["source", "model", "jpeg", "existing", "sidecar"])
def test_preflight_protects_output_paths(runtime, inputs, bad, tmp_path):
    if bad == "source":
        inputs.output_image_path = inputs.input_image_paths[0]
    if bad == "model":
        inputs.output_image_path = str(tmp_path / "Boogu-Image-0.1-Edit" / "new.png")
    if bad == "jpeg":
        inputs.output_image_path = str(tmp_path / "lossy.jpg")
    if bad == "existing":
        (tmp_path / "output.png").touch()
    if bad == "sidecar":
        (tmp_path / "output.png.json").touch()
    with pytest.raises(ValueError):
        runtime._preflight(inputs)


@pytest.mark.parametrize("fail_denoising", [False, True])
def test_staged_runtime_uses_mode_sample_fp32_decode_and_removes_hooks(
    runtime, inputs, monkeypatch, fail_denoising
):
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler,
    )

    checked = runtime._preflight(inputs)
    checked.device = torch.device(
        "cpu"
    )  # Small doubles exercise orchestration without GPU/model weights.
    monkeypatch.setattr(runtime, "_preflight", lambda args: checked)
    monkeypatch.setattr(runtime, "_check_cuda_device", lambda device: device)
    monkeypatch.setattr(runtime, "_free_cuda", lambda: None)
    calls = []

    class Posterior:
        def mode(self):
            calls.append("mode")
            return torch.zeros(1, 16, 2, 4)

        def sample(self, generator):
            calls.append(("sample", generator.initial_seed()))
            return torch.ones(1, 16, 2, 4)

    class VAE(torch.nn.Module):
        def encode(self, value):
            assert value.dtype == torch.float32
            return SimpleNamespace(latent_dist=Posterior())

        def decode(self, value, return_dict):
            assert value.dtype == torch.float32
            calls.append("decode")
            return (torch.zeros(1, 3, 16, 32),)

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(**checked.config)
            self.single_stream_layers = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(3)]
            )
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kwargs):
            assert hasattr(self, "_source_preservation_owner")
            if fail_denoising:
                raise RuntimeError("native denoising failure")
            return torch.zeros_like(kwargs["hidden_states"])

    base = Base()
    monkeypatch.setattr(
        BooguImageTransformer2DModel, "from_pretrained", lambda *args, **kwargs: base
    )

    def encode(*args, **kwargs):
        assert kwargs["truncate_instruction_sequence"] is False
        assert kwargs["do_classifier_free_guidance"] is True
        return (
            torch.ones(1, 2, 6),
            torch.tensor([[1, 0]]),
            torch.zeros(1, 1, 6),
            torch.ones(1, 1),
            None,
            None,
        )

    pipe = SimpleNamespace(encode_instruction=encode)
    monkeypatch.setattr(
        runtime,
        "_load_encoding_components",
        lambda inputs: (
            pipe,
            torch.nn.Identity(),
            VAE(),
            FlowMatchEulerDiscreteScheduler(do_shift=False),
        ),
    )
    if fail_denoising:
        with pytest.raises(RuntimeError, match="native denoising failure"):
            runtime.run_source_conditioned(inputs)
    else:
        result, metadata = runtime.run_source_conditioned(inputs)
        assert result.device.type == "cpu" and result.shape == (1, 3, 16, 32)
        assert torch.all(result == 0.5)
        assert metadata["seed"] == 42 and metadata["reference_seed"] == 43
        json.dumps(metadata, allow_nan=False)
        assert calls == ["mode", ("sample", 43), "decode"]
    assert not hasattr(base, "_source_preservation_owner")
    assert all(not layer._forward_hooks for layer in base.single_stream_layers)
