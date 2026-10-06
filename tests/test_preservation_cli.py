"""Public entry migration tests, without loading multi-gigabyte model weights."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

ROOT = Path(__file__).resolve().parents[1]


def load_functions(*names, **extra):
    tree = ast.parse((ROOT / "inference.py").read_text(encoding="utf-8"))
    module = ast.Module(
        body=[
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ],
        type_ignores=[],
    )
    namespace = {
        "argparse": argparse,
        "get_device_validator": lambda additional_types=None: str,
        **extra,
    }
    exec(compile(ast.fix_missing_locations(module), "inference.py", "exec"), namespace)  # noqa: S102 - isolated real CLI functions
    return namespace


def parse(*extra):
    ns = load_functions("parse_args", "to_bool")
    return ns["parse_args"](
        [
            "--pretrained_pipeline_name_or_path",
            "models/Boogu-Image-0.1-Edit",
            *extra,
            "--device",
            "cuda:0",
        ]
    )


def test_new_architecture_is_default_and_requires_actual_edit_inputs():
    args = parse()
    assert args.method == "source-conditioned"
    assert args.preservation_adapter is None
    assert args.preservation_permissions is None
    assert args.height is None and args.width is None
    assert args.instruction is None
    assert args.negative_instruction == ""


def test_explicit_values_are_preserved_and_recorded():
    args = parse(
        "--height=512",
        "--instruction",
        "make it blue",
        "--negative_instruction",
        "blur",
    )
    assert args.height == 512
    assert args.instruction == "make it blue"
    assert args.negative_instruction == "blur"
    assert {"height", "instruction", "negative_instruction"} <= set(
        args._explicit_options
    )


def test_explicit_baseline_keeps_original_defaults():
    args = parse("--method", "baseline")
    assert args.method == "baseline"
    assert args.height == args.width == 1024
    assert args.negative_instruction.startswith("(((deformed)))")


def test_abbreviated_options_rejected_instead_of_evading_tracking():
    with pytest.raises(SystemExit):
        parse("--preservation_ad", "adapter.pt")


@pytest.mark.parametrize("method", [None, "source-conditioned"])
def test_main_dispatches_new_path_without_loading_legacy_pipeline(method):
    calls = []
    values = {"pretrained_pipeline_name_or_path": "models/Boogu-Image-0.1-Edit"}
    if method is not None:
        values["method"] = method
    args = SimpleNamespace(**values)
    ns = load_functions(
        "main",
        require_edit_checkpoint=lambda path: None,
        _run_source_conditioned=lambda a: calls.append(a),
        _prepare_edit_inputs=lambda a: pytest.fail("legacy pipeline reached"),
    )
    ns["main"](args, str(ROOT))
    assert calls == [args]


def test_unknown_programmatic_method_fails_closed():
    ns = load_functions("main", require_edit_checkpoint=lambda path: None)
    with pytest.raises(ValueError, match="method"):
        ns["main"](
            SimpleNamespace(pretrained_pipeline_name_or_path="Edit", method="typo"),
            str(ROOT),
        )


def test_baseline_rejects_ignored_preservation_inputs():
    ns = load_functions("main", require_edit_checkpoint=lambda path: None)
    with pytest.raises(ValueError, match="preservation"):
        ns["main"](
            SimpleNamespace(
                pretrained_pipeline_name_or_path="Edit",
                method="baseline",
                preservation_adapter="trained.pt",
            ),
            str(ROOT),
        )


def test_short_launcher_forwards_current_inputs_without_an_implicit_pilot():
    spec = importlib.util.spec_from_file_location(
        "simple_edit", ROOT / "inference_ti2i_simple.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extra = [
        "--preservation_adapter",
        "chosen.pt",
        "--preservation_permissions",
        "roles.npz",
        "--input_image_paths",
        "my.png",
        "--instruction",
        "make the shirt blue",
    ]
    command = module.build_command(extra)
    assert command[-len(extra) :] == extra
    assert "--method" not in command
    assert not any(
        "adapter_only.pt" in token or "persimmon" in token for token in command
    )


def test_source_entry_saves_actual_png_and_provenance(monkeypatch, tmp_path):
    from boogu.preservation import runtime

    # Only the expensive GPU boundary is substituted; serialization is real.
    monkeypatch.setattr(
        runtime,
        "run_source_conditioned",
        lambda args: (
            torch.full((1, 3, 16, 16), 0.5),
            {"method": "source-conditioned", "seed": 42},
        ),
    )
    ns = load_functions(
        "_run_source_conditioned",
        to_pil_image=to_pil_image,
        _ensure_output_parent=lambda path: None,
    )
    path = tmp_path / "generated.png"
    ns["_run_source_conditioned"](SimpleNamespace(output_image_path=str(path)))
    with Image.open(path) as image:
        assert image.size == (16, 16)
        assert image.getpixel((0, 0)) == (128, 128, 128)
    metadata = json.loads(path.with_suffix(".png.json").read_text(encoding="utf-8"))
    assert metadata["code_version"] == "0.2.1"
    assert metadata["seed"] == 42 and metadata["method"] == "source-conditioned"


@pytest.mark.parametrize("existing", ["generated.png", "generated.png.json"])
def test_source_entry_will_not_overwrite_outputs_before_model_load(
    monkeypatch, tmp_path, existing
):
    from boogu.preservation import runtime

    monkeypatch.setattr(
        runtime,
        "run_source_conditioned",
        lambda args: pytest.fail("loaded before output check"),
    )
    (tmp_path / existing).write_text("previous result", encoding="utf-8")
    ns = load_functions("_run_source_conditioned")
    with pytest.raises(FileExistsError):
        ns["_run_source_conditioned"](
            SimpleNamespace(output_image_path=str(tmp_path / "generated.png"))
        )
    assert (tmp_path / existing).read_text(encoding="utf-8") == "previous result"


def test_legacy_shell_examples_select_baseline_explicitly():
    scripts = list((ROOT / "demo_scripts").glob("*.sh")) + list(
        ROOT.glob("test_ti2i*.sh")
    )
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        if "python inference.py" in source:
            assert "--method baseline" in source, script
