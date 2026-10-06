import argparse
import ast
import os
import sys
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Optional

import pytest
import torch

from boogu.far.config import FARConfig, ResponseTransportConfig

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = ROOT / "inference.py"
TURBO_PATH = ROOT / "inference_turbo.py"


class ListConfig(list):
    """Small OmegaConf ListConfig stand-in for isolated CLI tests."""


class _FakeOmegaConf:
    configs: ClassVar[dict] = {}

    @classmethod
    def load(cls, path):
        return cls.configs[path]


def _tree(path: Path = INFERENCE_PATH) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _argument_call(option: str) -> ast.Call:
    function = _function("parse_args")
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == option
    )


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name),
        None,
    )


def _literal_keyword(call: ast.Call, name: str):
    value = _keyword(call, name)
    assert value is not None
    return ast.literal_eval(value)


def _load_prepare_far_inputs(loader):
    node = _function("_prepare_far_inputs")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        "argparse": argparse,
        "FARConfig": FARConfig,
        "ResponseTransportConfig": ResponseTransportConfig,
        "load_pri_region_map_list": loader,
        "torch": torch,
        "Integral": Integral,
        "Real": Real,
        "isfinite": isfinite,
        "ListConfig": ListConfig,
        "OmegaConf": _FakeOmegaConf,
    }
    exec(compile(module, str(INFERENCE_PATH), "exec"), namespace)  # noqa: S102
    return namespace["_prepare_far_inputs"]


def _parse_cli_args(monkeypatch, *extra):
    nodes = [_function("to_bool"), _function("parse_args")]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {
        "argparse": argparse,
        "get_device_validator": lambda additional_types=None: str,
    }
    exec(compile(module, str(INFERENCE_PATH), "exec"), namespace)  # noqa: S102
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--method",
            "baseline",
            "--pretrained_pipeline_name_or_path",
            "models/Boogu-Image-0.1-Edit",
            "--device",
            "cpu",
            *extra,
        ],
    )
    return namespace["parse_args"]()


def _args(**overrides):
    values = {
        "method": "baseline",
        "seed": 11,
        "reference_seed": 13,
        "device": "cpu",
        "use_far": True,
        "far_region_map_paths": ["region.pt"],
        "pri_predictor_checkpoint": None,
        "far_invariant_strength": None,
        "far_effect_strength": None,
        "far_effect_low_pass_kernel": None,
        "far_response_transport_mode": None,
        "far_response_kernel_sizes": None,
        "negative_instruction": "",
        "scheduler": "euler",
        "text_guidance_scale": 4.0,
        "image_guidance_scale": 1.0,
        "empty_instruction_guidance_scale": 0.0,
        "cfg_range_start": 0.0,
        "cfg_range_end": 1.0,
        "num_images_per_instruction": 1,
        "use_batch_inference": False,
        "use_rewrite_text_instruction": False,
        "merge_original_and_rewritten_instructions": False,
        "use_dashscope_remote_rewriting": False,
        "save_rewritten_instruction": False,
        "enable_inner_devices_manager": False,
        "custom_local_instruction_rewriter_model": None,
        "use_boosted_orthogonal_guidance": False,
        "use_prompt_tuning": False,
        "custom_prompt_tuning_model_path": None,
        "custom_prompt_tuning_model_lora_weights_path": None,
        "custom_transformer_lora_path": None,
        "enable_teacache": False,
        "enable_taylorseer": False,
        "enable_teacache_for_all_layers": False,
        "enable_taylorseer_for_all_layers": False,
        "enable_cache_dit_caching": False,
        "enable_cache_dit_caching_for_all_layers": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _valid_inputs():
    return ["replace the cup"], [[object()]], [["reference.png"]]


@pytest.mark.parametrize(
    ("option", "expected_type", "nargs", "choices"),
    [
        ("--use_far", "to_bool", None, None),
        ("--far_region_map_paths", None, "*", None),
        ("--pri_predictor_checkpoint", "str", None, None),
        ("--far_invariant_strength", "float", None, None),
        ("--far_effect_strength", "float", None, None),
        ("--far_effect_low_pass_kernel", "int", None, None),
        (
            "--far_response_transport_mode",
            "str",
            None,
            ["fixed", "analytic"],
        ),
        ("--far_response_kernel_sizes", "int", "+", None),
    ],
)
def test_far_parser_contract(option, expected_type, nargs, choices):
    call = _argument_call(option)
    assert _literal_keyword(call, "default") is (
        False if option == "--use_far" else None
    )
    type_node = _keyword(call, "type")
    if expected_type is not None:
        assert isinstance(type_node, ast.Name)
        assert type_node.id == expected_type
    else:
        assert type_node is None
    if nargs is not None:
        assert _literal_keyword(call, "nargs") == nargs
    else:
        assert _keyword(call, "nargs") is None
    if choices is not None:
        assert _literal_keyword(call, "choices") == choices


def test_turbo_cli_does_not_expose_far_options():
    source = TURBO_PATH.read_text(encoding="utf-8")
    assert "--use_far" not in source
    assert "--far_region_map_paths" not in source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("far_region_map_paths", []),
        ("pri_predictor_checkpoint", "predictor-checkpoint"),
        ("far_invariant_strength", 0.25),
        ("far_effect_strength", 0.25),
        ("far_effect_low_pass_kernel", 5),
        ("far_response_transport_mode", "analytic"),
        ("far_response_kernel_sizes", [15, 7, 3]),
    ],
)
def test_disabled_far_rejects_every_auxiliary_option(field, value):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: None)
    args = _args(use_far=False, far_region_map_paths=None)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="disabled|use_far"):
        prepare(args, *_valid_inputs())


def test_disabled_far_without_auxiliary_options_is_a_noop():
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: None)
    args = _args(use_far=False, far_region_map_paths=None)
    assert prepare(args, *_valid_inputs()) == (None, None, None)


def test_reference_seed_parser_defaults_to_derived_pair_and_accepts_override(
    monkeypatch,
):
    default_args = _parse_cli_args(monkeypatch)
    explicit_args = _parse_cli_args(monkeypatch, "--reference_seed", "23")

    assert default_args.reference_seed is None
    assert explicit_args.reference_seed == 23


def test_inference_generator_pair_is_distinct_and_reproducible():
    node = next(
        (
            node
            for node in _tree().body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_create_inference_generators"
        ),
        None,
    )
    assert node is not None, "the independent inference RNG builder is missing"
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"argparse": argparse, "torch": torch}
    exec(compile(module, str(INFERENCE_PATH), "exec"), namespace)  # noqa: S102
    create_generators = namespace["_create_inference_generators"]

    noise_a, reference_a = create_generators(_args(seed=17, reference_seed=29))
    noise_b, reference_b = create_generators(_args(seed=17, reference_seed=29))

    assert noise_a is not reference_a
    assert noise_a.initial_seed() == 17
    assert reference_a.initial_seed() == 29
    assert torch.equal(torch.randn(8, generator=noise_a), torch.randn(8, generator=noise_b))
    assert torch.equal(
        torch.randn(8, generator=reference_a),
        torch.randn(8, generator=reference_b),
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"seed": -1},
        {"reference_seed": -1},
        {"seed": 2**63},
        {"reference_seed": 2**63},
        {"seed": 7, "reference_seed": 7},
        {"seed": True},
        {"reference_seed": False},
    ],
)
def test_inference_generator_pair_rejects_invalid_or_equal_seeds(updates):
    node = next(
        (
            node
            for node in _tree().body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_create_inference_generators"
        ),
        None,
    )
    assert node is not None, "the independent inference RNG builder is missing"
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"argparse": argparse, "torch": torch}
    exec(compile(module, str(INFERENCE_PATH), "exec"), namespace)  # noqa: S102
    with pytest.raises((TypeError, ValueError), match="seed|independent|distinct"):
        namespace["_create_inference_generators"](_args(**updates))


def test_inference_generator_pair_derives_a_distinct_reference_seed():
    node = next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_inference_generators"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"argparse": argparse, "torch": torch}
    exec(compile(module, str(INFERENCE_PATH), "exec"), namespace)  # noqa: S102
    create_generators = namespace["_create_inference_generators"]

    noise, reference = create_generators(_args(seed=41, reference_seed=None))
    max_noise, wrapped_reference = create_generators(
        _args(seed=2**63 - 1, reference_seed=None)
    )

    assert (noise.initial_seed(), reference.initial_seed()) == (41, 42)
    assert (max_noise.initial_seed(), wrapped_reference.initial_seed()) == (
        2**63 - 1,
        0,
    )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"negative_instruction": "default negative"}, "negative_instruction"),
        ({"negative_instruction": " "}, "negative_instruction"),
        ({"scheduler": "dpmsolver++"}, "euler"),
        ({"text_guidance_scale": 1.0}, "text_guidance_scale"),
        ({"image_guidance_scale": 0.9}, "image_guidance_scale"),
        (
            {"empty_instruction_guidance_scale": 0.1},
            "empty_instruction_guidance_scale",
        ),
        ({"cfg_range_start": 0.1}, "cfg_range"),
        ({"cfg_range_end": 0.9}, "cfg_range"),
        ({"num_images_per_instruction": 2}, "num_images_per_instruction"),
    ],
)
def test_enabled_far_rejects_incompatible_generation_settings(updates, match):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["map"])
    with pytest.raises((TypeError, ValueError), match=match):
        prepare(_args(**updates), *_valid_inputs())


@pytest.mark.parametrize(
    "field",
    [
        "use_rewrite_text_instruction",
        "use_dashscope_remote_rewriting",
        "save_rewritten_instruction",
        "enable_inner_devices_manager",
        "use_boosted_orthogonal_guidance",
        "use_prompt_tuning",
        "enable_teacache",
        "enable_taylorseer",
        "enable_teacache_for_all_layers",
        "enable_taylorseer_for_all_layers",
        "enable_cache_dit_caching",
        "enable_cache_dit_caching_for_all_layers",
    ],
)
def test_enabled_far_rejects_incompatible_boolean_features(field):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["map"])
    with pytest.raises(ValueError, match=field):
        prepare(_args(**{field: True}), *_valid_inputs())


@pytest.mark.parametrize(
    "field",
    [
        "custom_local_instruction_rewriter_model",
        "custom_prompt_tuning_model_path",
        "custom_prompt_tuning_model_lora_weights_path",
        "custom_transformer_lora_path",
    ],
)
def test_enabled_far_rejects_incompatible_model_paths(field):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["map"])
    with pytest.raises(ValueError, match=field):
        prepare(_args(**{field: "custom.safetensors"}), *_valid_inputs())


@pytest.mark.parametrize(
    ("instructions", "images", "paths", "map_paths"),
    [
        (["edit"], [[]], [["reference.png"]], ["region.pt"]),
        (["edit"], [[object(), object()]], [["a.png", "b.png"]], ["region.pt"]),
        (["edit"], [[object()]], [["reference.png"]], []),
        (["edit"], [[object()]], [["reference.png"]], ["a.pt", "b.pt"]),
        (["a", "b"], [[object()], [object()]], [["a.png"], ["b.png"]], ["a.pt", "b.pt"]),
    ],
)
def test_enabled_far_requires_exactly_one_sample_reference_and_map(
    instructions,
    images,
    paths,
    map_paths,
):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["map"])
    with pytest.raises((TypeError, ValueError), match="one|single|exactly"):
        prepare(
            _args(far_region_map_paths=map_paths),
            instructions,
            images,
            paths,
        )


def test_enabled_far_loads_one_map_safely_and_builds_default_analytic_transport():
    calls = []
    loaded = [object()]

    def loader(paths, **kwargs):
        calls.append((paths, kwargs))
        return loaded

    prepare = _load_prepare_far_inputs(loader)
    far_config, transport, regions = prepare(_args(), *_valid_inputs())

    assert far_config == FARConfig(enabled=True)
    assert transport.mode == "analytic_multiscale"
    assert transport.kernel_sizes == (15, 7, 3)
    assert transport.schedule_centers == (0.0, 0.5, 1.0)
    assert regions is loaded
    assert calls == [(["region.pt"], {"dtype": torch.float32})]


def test_enabled_far_accepts_predictor_checkpoint_without_loading_static_maps():
    loader_calls = []
    prepare = _load_prepare_far_inputs(
        lambda *args, **kwargs: loader_calls.append((args, kwargs))
    )

    far_config, transport, regions = prepare(
        _args(
            far_region_map_paths=None,
            pri_predictor_checkpoint="predictor-checkpoint",
        ),
        *_valid_inputs(),
    )

    assert far_config.enabled is True
    assert transport.mode == "analytic_multiscale"
    assert regions is None
    assert loader_calls == []


@pytest.mark.parametrize(
    ("map_paths", "checkpoint"),
    [
        (None, None),
        (["region.pt"], "predictor-checkpoint"),
    ],
)
def test_enabled_far_requires_exactly_one_static_or_learned_pri_source(
    map_paths,
    checkpoint,
):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["map"])

    with pytest.raises(ValueError, match="exactly one|PRI source|mutually exclusive"):
        prepare(
            _args(
                far_region_map_paths=map_paths,
                pri_predictor_checkpoint=checkpoint,
            ),
            *_valid_inputs(),
        )


@pytest.mark.parametrize(
    ("args", "mode", "kernels", "strength"),
    [
        (
            _args(
                far_response_transport_mode="fixed",
                far_effect_low_pass_kernel=9,
                far_effect_strength=0.25,
            ),
            "fixed",
            (9,),
            0.25,
        ),
        (
            _args(
                far_response_transport_mode="fixed",
                far_response_kernel_sizes=[7],
            ),
            "fixed",
            (7,),
            0.5,
        ),
        (
            _args(
                far_response_transport_mode="analytic",
                far_response_kernel_sizes=[17, 9, 3],
                far_effect_strength=0.75,
            ),
            "analytic_multiscale",
            (17, 9, 3),
            0.75,
        ),
    ],
)
def test_enabled_far_builds_requested_transport(args, mode, kernels, strength):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["map"])
    _, transport, _ = prepare(args, *_valid_inputs())
    assert transport.mode == mode
    assert transport.kernel_sizes == kernels
    assert transport.response_strength == strength


@pytest.mark.parametrize(
    "args",
    [
        _args(
            far_response_transport_mode="fixed",
            far_response_kernel_sizes=[9, 3],
        ),
        _args(
            far_response_transport_mode="fixed",
            far_effect_low_pass_kernel=5,
            far_response_kernel_sizes=[5],
        ),
        _args(
            far_response_transport_mode="analytic",
            far_response_kernel_sizes=[15, 7],
        ),
        _args(
            far_response_transport_mode="analytic",
            far_effect_low_pass_kernel=5,
        ),
    ],
)
def test_enabled_far_rejects_ambiguous_or_wrong_transport_shapes(args):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["map"])
    with pytest.raises((TypeError, ValueError), match="kernel|low.pass|three|one"):
        prepare(args, *_valid_inputs())


def test_prepare_far_inputs_does_not_depend_on_weight_loading(monkeypatch):
    loader_calls = []
    prepare = _load_prepare_far_inputs(
        lambda paths, **kwargs: loader_calls.append(paths) or ["map"]
    )
    monkeypatch.setitem(sys.modules, "transformers", None)
    far_config, _, regions = prepare(_args(), *_valid_inputs())
    assert far_config.enabled is True
    assert regions == ["map"]
    assert loader_calls == [["region.pt"]]


def test_legacy_far_requires_explicit_guidance_above_current_default(monkeypatch):
    args = _parse_cli_args(
        monkeypatch, "--use_far", "true", "--far_region_map_paths", "region.pt",
        "--negative_instruction", "",
    )
    assert args.text_guidance_scale == 1
    prepare = _load_prepare_far_inputs(lambda paths, **kwargs: ["map"])
    with pytest.raises(ValueError, match="text_guidance_scale must be greater than 1"):
        prepare(args, *_valid_inputs())


def test_parser_with_explicit_legacy_far_guidance_passes_preflight(monkeypatch):
    args = _parse_cli_args(
        monkeypatch,
        "--use_far",
        "true",
        "--far_region_map_paths",
        "region.pt",
        "--negative_instruction",
        "",
        "--text_guidance_scale",
        "6",
    )
    prepare = _load_prepare_far_inputs(lambda paths, **kwargs: ["map"])

    far_config, transport, regions = prepare(args, *_valid_inputs())

    assert args.merge_original_and_rewritten_instructions is True
    assert far_config.enabled is True
    assert transport.mode == "analytic_multiscale"
    assert regions == ["map"]


def test_disabled_batch_does_not_read_pri_config_or_call_loader():
    class FailIfRead:
        def __getattr__(self, name):
            raise AssertionError(f"disabled FAR read batch field {name}")

    loader_calls = []
    prepare = _load_prepare_far_inputs(
        lambda *args, **kwargs: loader_calls.append((args, kwargs))
    )
    args = _args(
        use_far=False,
        use_batch_inference=True,
        far_region_map_paths=None,
        batch_data_config_path="unused.yml",
        _resolved_batch_data=FailIfRead(),
    )

    assert prepare(args, *_valid_inputs()) == (None, None, None)
    assert loader_calls == []


def _batch_args(data, **overrides):
    return _args(
        use_batch_inference=True,
        far_region_map_paths=None,
        batch_data_config_path="batch.yml",
        _resolved_batch_data=data,
        **overrides,
    )


def _three_sample_inputs():
    return (
        ["edit zero", "edit one", "edit two"],
        [["image-0"], ["image-1"], ["image-2"]],
        [["ref-0.png"], ["ref-1.png"], ["ref-2.png"]],
    )


def test_batch_far_loads_flat_config_maps_once_in_sample_order():
    data = SimpleNamespace(
        far_region_map_paths=ListConfig(
            ["region-0.pt", "region-1.pt", "region-2.pt"]
        )
    )
    calls = []
    loaded = ["region-0", "region-1", "region-2"]

    def loader(paths, **kwargs):
        calls.append((paths, kwargs))
        return loaded

    prepare = _load_prepare_far_inputs(loader)
    _, _, regions = prepare(_batch_args(data), *_three_sample_inputs())

    assert regions is loaded
    assert calls == [
        (
            ["region-0.pt", "region-1.pt", "region-2.pt"],
            {"dtype": torch.float32},
        )
    ]


@pytest.mark.parametrize(
    "maps",
    [None, "region.pt", [], ["region-0.pt"], ["region-0.pt", " "], [1, 2]],
)
def test_batch_far_rejects_missing_or_misaligned_config_maps(maps):
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["unused"])
    data = SimpleNamespace(far_region_map_paths=maps)
    with pytest.raises((TypeError, ValueError), match="map|paths|length|sample"):
        prepare(_batch_args(data), *_three_sample_inputs())


def test_batch_far_rejects_cli_and_config_maps_together():
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["unused"])
    data = SimpleNamespace(
        far_region_map_paths=["region-0.pt", "region-1.pt", "region-2.pt"]
    )
    args = _batch_args(data)
    args.far_region_map_paths = ["cli-region.pt"]
    with pytest.raises(ValueError, match="both|command|CLI|config"):
        prepare(args, *_three_sample_inputs())


def test_batch_far_rejects_cli_maps_with_predictor_checkpoint():
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["unused"])
    data = SimpleNamespace(far_region_map_paths=None)
    args = _batch_args(data, pri_predictor_checkpoint="predictor-checkpoint")
    args.far_region_map_paths = ["cli-region.pt"]

    with pytest.raises(ValueError, match="exactly one|PRI source|mutually exclusive"):
        prepare(args, *_three_sample_inputs())


def test_batch_far_requires_one_reference_per_sample():
    data = SimpleNamespace(
        far_region_map_paths=["region-0.pt", "region-1.pt", "region-2.pt"]
    )
    instructions, images, paths = _three_sample_inputs()
    images[1] = ["image-a", "image-b"]
    paths[1] = ["a.png", "b.png"]
    prepare = _load_prepare_far_inputs(lambda *args, **kwargs: ["unused"] * 3)
    with pytest.raises(ValueError, match="exactly one reference"):
        prepare(_batch_args(data), instructions, images, paths)


def test_inference_predictor_loader_freezes_fp32_model_on_requested_device():
    node = next(
        (
            node
            for node in _tree().body
            if isinstance(node, ast.FunctionDef) and node.name == "_load_pri_predictor"
        ),
        None,
    )
    assert node is not None, "the inference predictor loader is missing"
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    source = torch.nn.Linear(2, 2)
    calls = []

    def factory(path):
        calls.append(path)
        return source, {"architecture": {}}

    namespace = {
        "argparse": argparse,
        "torch": torch,
        "load_predictor_from_checkpoint": factory,
    }
    exec(compile(module, str(INFERENCE_PATH), "exec"), namespace)  # noqa: S102

    predictor = namespace["_load_pri_predictor"](
        _args(pri_predictor_checkpoint="predictor-checkpoint")
    )

    assert predictor is source
    assert calls == ["predictor-checkpoint"]
    assert predictor.training is False
    assert all(not parameter.requires_grad for parameter in predictor.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in predictor.parameters())
    assert all(parameter.device.type == "cpu" for parameter in predictor.parameters())


def _load_run(generator_pair=None):
    node = _function("run")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        "argparse": argparse,
        "BooguImagePipeline": object,
        "Image": SimpleNamespace(Image=object),
        "List": list,
        "Optional": Optional,
        "FARConfig": FARConfig,
        "ResponseTransportConfig": ResponseTransportConfig,
        "torch": torch,
        "_create_inference_generators": (
            (lambda args: generator_pair)
            if generator_pair is not None
            else lambda args: (
                torch.Generator(device=args.device).manual_seed(args.seed),
                torch.Generator(device=args.device).manual_seed(
                    args.seed + 1
                    if args.reference_seed is None
                    else args.reference_seed
                ),
            )
        ),
    }
    exec(compile(module, str(INFERENCE_PATH), "exec"), namespace)  # noqa: S102
    return namespace["run"]


class _CapturePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[])


def test_run_disabled_preserves_pipeline_kwargs_without_any_far_key(monkeypatch):
    args = _parse_cli_args(monkeypatch)
    pipeline = _CapturePipeline()
    run = _load_run()

    run(args, pipeline, "edit", args.negative_instruction, [["image"]], [["ref"]])

    assert len(pipeline.calls) == 1
    assert not {key for key in pipeline.calls[0] if key.startswith("far_")}
    assert "response_transport_config" not in pipeline.calls[0]
    assert pipeline.calls[0]["generator"] is not pipeline.calls[0]["reference_generator"]


def test_run_enabled_injects_exactly_three_far_kwargs(monkeypatch):
    args = _parse_cli_args(monkeypatch)
    pipeline = _CapturePipeline()
    run = _load_run()
    far_config = FARConfig(enabled=True)
    transport = ResponseTransportConfig()
    regions = ["region"]

    run(
        args,
        pipeline,
        "edit",
        "",
        [["image"]],
        [["ref"]],
        far_config=far_config,
        response_transport_config=transport,
        far_region_probabilities=regions,
    )

    far_keys = {
        "far_config",
        "response_transport_config",
        "far_region_probabilities",
    }
    assert far_keys <= pipeline.calls[0].keys()
    assert pipeline.calls[0]["far_config"] is far_config
    assert pipeline.calls[0]["response_transport_config"] is transport
    assert pipeline.calls[0]["far_region_probabilities"] is regions


def test_run_routes_one_learned_predictor_instead_of_static_regions(monkeypatch):
    args = _parse_cli_args(monkeypatch, "--reference_seed", "13")
    pipeline = _CapturePipeline()
    run = _load_run()
    far_config = FARConfig(enabled=True)
    transport = ResponseTransportConfig()
    predictor = object()

    run(
        args,
        pipeline,
        "edit",
        "",
        [["image"]],
        [["ref"]],
        far_config=far_config,
        response_transport_config=transport,
        pri_region_predictor=predictor,
    )

    assert pipeline.calls[0]["pri_region_predictor"] is predictor
    assert "far_region_probabilities" not in pipeline.calls[0]


def test_run_rejects_two_pri_sources_before_calling_pipeline(monkeypatch):
    args = _parse_cli_args(monkeypatch, "--reference_seed", "13")
    pipeline = _CapturePipeline()
    run = _load_run()

    with pytest.raises(ValueError, match="exactly one|PRI source|mutually exclusive"):
        run(
            args,
            pipeline,
            "edit",
            "",
            [["image"]],
            [["ref"]],
            far_config=FARConfig(enabled=True),
            response_transport_config=ResponseTransportConfig(),
            far_region_probabilities=["region"],
            pri_region_predictor=object(),
        )

    assert pipeline.calls == []


def _load_main(
    *,
    prepare_edit,
    prepare_far,
    load_pipeline,
    run,
    load_pri_predictor=lambda args: None,
    create_generators=lambda args: (object(), object()),
):
    node = _function("main")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))

    class Output:
        def __init__(self, images):
            self.images = images

    namespace = {
        "argparse": argparse,
        "str": str,
        "require_edit_checkpoint": lambda checkpoint: None,
        "_prepare_edit_inputs": prepare_edit,
        "_prepare_far_inputs": prepare_far,
        "load_pipeline": load_pipeline,
        "_load_pri_predictor": load_pri_predictor,
        "_create_inference_generators": create_generators,
        "run": run,
        "torch": SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
        ),
        "EditOnlyInputError": ValueError,
        "FMPipelineOutput": Output,
        "_ensure_output_parent": lambda path: None,
        "to_tensor": lambda image: 1,
        "create_collage": lambda images: SimpleNamespace(save=lambda path: None),
        "os": os,
    }
    exec(compile(module, str(INFERENCE_PATH), "exec"), namespace)  # noqa: S102
    return namespace["main"]


def test_main_runs_far_preflight_before_loading_pipeline(monkeypatch):
    args = _parse_cli_args(monkeypatch)
    events = []

    def fail_preflight(*unused):
        events.append("far-preflight")
        raise ValueError("invalid FAR request")

    main = _load_main(
        prepare_edit=lambda args: _valid_inputs(),
        prepare_far=fail_preflight,
        load_pipeline=lambda *unused: events.append("load-pipeline"),
        run=lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="invalid FAR"):
        main(args, str(ROOT))
    assert events == ["far-preflight"]


def test_main_pri_loader_error_occurs_before_pipeline_loading(monkeypatch):
    args = _parse_cli_args(
        monkeypatch,
        "--use_far",
        "true",
        "--far_region_map_paths",
        "missing.pt",
        "--negative_instruction",
        "",
        "--text_guidance_scale",
        "6",
    )
    events = []

    def loader(*unused, **kwargs):
        events.append("load-pri")
        raise ValueError("unsafe or missing PRI")

    prepare_far = _load_prepare_far_inputs(loader)
    main = _load_main(
        prepare_edit=lambda args: _valid_inputs(),
        prepare_far=prepare_far,
        load_pipeline=lambda *unused: events.append("load-pipeline"),
        run=lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="PRI"):
        main(args, str(ROOT))
    assert events == ["load-pri"]


def test_main_slices_batch_pri_regions_with_instruction_and_reference_indices(
    monkeypatch,
):
    args = _parse_cli_args(monkeypatch)
    args.batch_size = 2
    calls = []
    regions = ["region-0", "region-1", "region-2"]
    far_config = FARConfig(enabled=True)
    transport = ResponseTransportConfig()

    def fake_run(
        args,
        pipeline,
        instruction,
        negative_instruction,
        input_images,
        input_image_paths,
        **kwargs,
    ):
        calls.append(
            (instruction, input_images, input_image_paths, kwargs)
        )
        return SimpleNamespace(
            images=[SimpleNamespace(save=lambda path: None) for _ in instruction]
        )

    main = _load_main(
        prepare_edit=lambda args: _three_sample_inputs(),
        prepare_far=lambda *unused: (far_config, transport, regions),
        load_pipeline=lambda *unused: "pipeline",
        run=fake_run,
    )
    main(args, str(ROOT))

    assert [call[0] for call in calls] == [
        ["edit zero", "edit one"],
        ["edit two"],
    ]
    assert [call[1] for call in calls] == [
        [["image-0"], ["image-1"]],
        [["image-2"]],
    ]
    assert [call[2] for call in calls] == [
        [["ref-0.png"], ["ref-1.png"]],
        [["ref-2.png"]],
    ]
    assert [call[3]["far_region_probabilities"] for call in calls] == [
        ["region-0", "region-1"],
        ["region-2"],
    ]
    assert all(call[3]["far_config"] is far_config for call in calls)
    assert all(
        call[3]["response_transport_config"] is transport for call in calls
    )
    assert calls[0][3]["generator"] is calls[1][3]["generator"]
    assert calls[0][3]["reference_generator"] is calls[1][3]["reference_generator"]
    assert calls[0][3]["generator"] is not calls[0][3]["reference_generator"]


def test_main_loads_and_reuses_one_predictor_across_batches(monkeypatch):
    args = _parse_cli_args(monkeypatch)
    args.batch_size = 2
    args.pri_predictor_checkpoint = "predictor-checkpoint"
    args.far_region_map_paths = None
    predictor = object()
    events = []
    calls = []

    def fake_run(*positional, **kwargs):
        calls.append(kwargs)
        batch = positional[2]
        return SimpleNamespace(
            images=[SimpleNamespace(save=lambda path: None) for _ in batch]
        )

    main = _load_main(
        prepare_edit=lambda args: _three_sample_inputs(),
        prepare_far=lambda *unused: (
            FARConfig(enabled=True),
            ResponseTransportConfig(),
            None,
        ),
        load_pipeline=lambda *unused: events.append("pipeline") or "pipeline",
        load_pri_predictor=lambda args: events.append("predictor") or predictor,
        run=fake_run,
    )

    main(args, str(ROOT))

    assert events == ["predictor", "pipeline"]
    assert len(calls) == 2
    assert all(call["pri_region_predictor"] is predictor for call in calls)
    assert all("far_region_probabilities" not in call for call in calls)


def test_main_reuses_continuous_random_streams_across_batches(monkeypatch):
    args = _parse_cli_args(monkeypatch)
    args.batch_size = 2
    noise = object()
    reference = object()
    calls = []

    def fake_run(*positional, **kwargs):
        calls.append(kwargs)
        batch = positional[2]
        return SimpleNamespace(
            images=[SimpleNamespace(save=lambda path: None) for _ in batch]
        )

    main = _load_main(
        prepare_edit=lambda args: _three_sample_inputs(),
        prepare_far=lambda *unused: (None, None, None),
        load_pipeline=lambda *unused: "pipeline",
        run=fake_run,
        create_generators=lambda args: (noise, reference),
    )

    main(args, str(ROOT))

    assert len(calls) == 2
    assert all(call["generator"] is noise for call in calls)
    assert all(call["reference_generator"] is reference for call in calls)


def test_main_validates_seed_pair_before_loading_main_pipeline(monkeypatch):
    args = _parse_cli_args(monkeypatch)
    events = []

    def reject_seeds(args):
        events.append("seeds")
        raise ValueError("seeds must be distinct")

    main = _load_main(
        prepare_edit=lambda args: _valid_inputs(),
        prepare_far=lambda *unused: (None, None, None),
        load_pipeline=lambda *unused: events.append("pipeline"),
        run=lambda *args, **kwargs: None,
        create_generators=reject_seeds,
    )

    with pytest.raises(ValueError, match="distinct"):
        main(args, str(ROOT))
    assert events == ["seeds"]
