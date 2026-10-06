"""Portable paired-training contracts; no model weights or CUDA required."""

import hashlib
import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


def test_standalone_training_package_exists():
    assert importlib.util.find_spec("boogu.preservation.training") is not None


@pytest.fixture
def training():
    assert importlib.util.find_spec("boogu.preservation.training") is not None, (
        "A standalone preservation training package is required"
    )
    return importlib.import_module("boogu.preservation.training")


@pytest.fixture
def manifest(tmp_path):
    source = Image.new("RGB", (32, 16), (30, 60, 90))
    source.save(tmp_path / "source.png")
    Image.new("RGB", (32, 16), (40, 60, 90)).save(tmp_path / "target.png")
    np.savez(
        tmp_path / "permissions.npz",
        keep=np.ones((1, 5, 16, 32), dtype=np.float32),
        modify=np.zeros((1, 5, 16, 32), dtype=np.float32),
        schema=np.array("boogu_preservation_permissions_v1"),
        attributes=np.array(["color", "structure", "texture", "glyph", "pixel"]),
        source_sha256=np.array(hashlib.sha256(source.tobytes()).hexdigest()),
    )
    edit = np.zeros((16, 32), dtype=bool)
    edit[:, :16] = True
    np.savez(
        tmp_path / "supervision.npz",
        allowed_edit=edit,
        edit_core=edit,
        protected_roi__face=~edit,
    )
    payload = {
        "schema": "boogu_preservation_pairs_v1",
        "pairs": [
            {
                "pair_id": "train_1",
                "source": "source.png",
                "target": "target.png",
                "instruction": "Change the requested color",
                "permissions": "permissions.npz",
                "supervision": "supervision.npz",
                "split": "train",
                "source_group": "person_1",
                "admitted": True,
                "same_frame": True,
            }
        ],
    }
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_manifest_resolves_paths_and_loads_bound_source_roles(training, manifest):
    path, _ = manifest
    rows = training.load_manifest(path)
    assert len(rows) == 1 and rows[0].source.is_absolute()
    item = training.load_pair(rows[0])
    assert item["source"].size == item["target"].size == (32, 16)
    assert item["keep"].shape == (1, 5, 16, 32)
    assert item["protected_roi_masks"]["face"].sum() == 256


@pytest.mark.parametrize(
    "field,value",
    [
        ("admitted", False),
        ("same_frame", False),
        ("split", "validation"),
        ("split", "test"),
        ("source_group", ""),
        ("instruction", " "),
    ],
)
def test_training_manifest_rejects_ineligible_rows(training, manifest, field, value):
    path, payload = manifest
    payload["pairs"][0][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        training.load_manifest(path)


@pytest.mark.parametrize(
    "change", ["pair_id", "pair_content", "group_split", "pixel_split"]
)
def test_manifest_detects_duplicates_and_split_leakage(training, manifest, change):
    path, payload = manifest
    second = dict(payload["pairs"][0])
    if change != "pair_id":
        second["pair_id"] = "second"
    if change in ("group_split", "pixel_split"):
        second["split"] = "test"
        second["instruction"] = "A different edit"
    if change == "pixel_split":
        second["source_group"] = "renamed_person"
    payload["pairs"].append(second)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|split|leakage"):
        training.load_manifest(path)


def test_manifest_rejects_case_insensitive_pair_id_collision(training, manifest):
    path, payload = manifest
    payload["pairs"][0]["pair_id"] = "photoA"
    payload["pairs"].append(
        {**payload["pairs"][0], "pair_id": "photoa", "instruction": "A different edit"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate pair_id"):
        training.load_manifest(path)


@pytest.mark.parametrize(
    "pair_id",
    [
        "CON",
        "con.foo",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM9.txt",
        "LPT1",
        "lpt9.more",
        "photo.",
    ],
)
def test_manifest_rejects_windows_unsafe_pair_ids(training, manifest, pair_id):
    path, payload = manifest
    payload["pairs"][0]["pair_id"] = pair_id
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safe filename"):
        training.load_manifest(path)


@pytest.mark.parametrize(
    "bad", ["canvas", "frame", "binding", "mask_dtype", "core_outside"]
)
def test_pair_validation_rejects_bad_geometry_permissions_and_supervision(
    training, manifest, bad
):
    path, _ = manifest
    if bad in ("canvas", "frame"):
        Image.new("RGB", (31 if bad == "canvas" else 16, 16)).save(
            path.parent / "target.png"
        )
    elif bad == "binding":
        Image.new("RGB", (32, 16), "red").save(path.parent / "source.png")
    else:
        allowed = np.zeros((16, 32), dtype=bool)
        core = np.ones((16, 32), dtype=bool)
        if bad == "mask_dtype":
            core = core.astype(np.float32)
        np.savez(path.parent / "supervision.npz", allowed_edit=allowed, edit_core=core)
    with pytest.raises(ValueError):
        training.load_manifest(path)


def test_flow_endpoint_keeps_unrounded_float32_and_correct_gradient(training):
    obj = importlib.import_module("boogu.preservation.training.objectives")
    noise = torch.full((1, 16, 2, 4), 0.1003)
    target = torch.full_like(noise, 0.6123)
    z, velocity = obj.flow_sample(noise, target, 0.37)
    prediction = velocity.bfloat16().requires_grad_()
    result = obj.endpoint_fp32(z, prediction, 0.37)
    assert result.dtype == torch.float32
    assert torch.equal(result, z + 0.63 * prediction.float())
    assert not torch.equal(result, z.bfloat16().float() + 0.63 * prediction.float())
    result.sum().backward()
    assert torch.all(prediction.grad == torch.tensor(0.63).bfloat16())


def test_balanced_flow_regions_normalize_independently(training):
    obj = importlib.import_module("boogu.preservation.training.objectives")
    prediction = torch.tensor([[[[2.0, 1.0, 1.0, 1.0]]]])
    mask = np.array([[True, False, False, False]])
    assert obj.balanced_flow_loss(prediction, torch.zeros_like(prediction), mask) == 2.5


def test_objective_is_exact_weighted_target_supervision_with_detached_labels(training):
    obj = importlib.import_module("boogu.preservation.training.objectives")
    rgb = torch.full((1, 3, 16, 32), 0.2, requires_grad=True)
    target = torch.full_like(rgb, 0.6, requires_grad=True)
    keep = torch.zeros((1, 5, 16, 32), requires_grad=True)
    with torch.no_grad():
        keep[:, 4] = 1
    mask = np.ones((16, 32), dtype=bool)
    pred = torch.ones((1, 16, 2, 4), requires_grad=True)
    loss, detail = obj.training_objective(
        pred,
        torch.zeros_like(pred),
        rgb,
        target,
        keep,
        mask,
        {"face": mask, "empty": ~mask},
        mask,
    )
    assert loss.item() == pytest.approx(1.6)
    assert detail["protected_roi_details"]["empty"]["pixels"] == 0
    loss.backward()
    assert target.grad is None and keep.grad is None
    assert rgb.grad.abs().sum() > 0 and pred.grad.abs().sum() > 0


def test_source_encoder_receives_no_target_or_supervision(training, manifest):
    from boogu.preservation.training.runner import encode_source_features

    row = training.load_manifest(manifest[0])[0]
    received = []

    class Pipe:
        def encode_instruction(self, instruction, **kwargs):
            received.append((instruction, kwargs))
            return (
                torch.ones(1, 3, 6),
                torch.tensor([[1, 0, 1]]),
                None,
                None,
                None,
                None,
            )

    features = encode_source_features(Pipe(), row, torch.device("cpu"), 1280)
    assert features["positive"].shape == (1, 2, 6)
    args = received[0][1]
    assert args["do_classifier_free_guidance"] is False
    assert np.asarray(args["input_images"][0][0])[0, 0].tolist() == [30, 60, 90]
    assert "target" not in args and "supervision" not in args


def test_draw_schedule_is_stratified_seeded_and_source_balanced(training):
    from boogu.preservation.training.runner import make_draws

    rows = [SimpleNamespace(pair_id="b"), SimpleNamespace(pair_id="a")]
    draws = make_draws(rows, 8, 42)
    assert draws == make_draws(rows, 8, 42)
    assert draws != make_draws(rows, 8, 43)
    assert [draw["pair_id"] for draw in draws] == ["a", "b"] * 4
    assert all(0.01 <= draw["t"] < 0.99 for draw in draws)
    for key in ("a", "b"):
        bins = sorted(
            int((d["t"] - 0.01) / 0.98 * 4) for d in draws if d["pair_id"] == key
        )
        assert bins == [0, 1, 2, 3]


def test_training_configures_reproducible_native_math(training):
    from boogu.preservation.training import runner

    previous = torch.are_deterministic_algorithms_enabled()
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_benchmark = torch.backends.cudnn.benchmark
    try:
        runner.configure_determinism(42)
        value = torch.randn(3)
        runner.configure_determinism(42)
        assert torch.equal(value, torch.randn(3))
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cudnn.benchmark = old_benchmark


def test_cache_binds_manifest_files_and_detects_changes(training, manifest):
    from boogu.preservation.training.data import pair_binding

    row = training.load_manifest(manifest[0])[0]
    before = pair_binding(row)
    Image.new("RGB", (32, 16), "blue").save(row.target)
    assert pair_binding(row) != before


def test_training_cli_is_importable_without_model_loading(training):
    from boogu.preservation.training.runner import build_parser

    args = build_parser().parse_args(
        [
            "train",
            "--manifest",
            "pairs.json",
            "--base",
            "base",
            "--cache",
            "cache",
            "--output",
            "adapter.pt",
            "--steps",
            "5",
            "--lr",
            ".0003",
        ]
    )
    assert args.steps == 5 and args.lr == 0.0003 and args.seed == 42


def test_training_package_has_no_historical_dependencies(training):
    from pathlib import Path

    for path in Path(training.__file__).parent.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "outputs/" not in text and "outputs\\" not in text
        assert "import inference" not in text and "importlib.util" not in text


class SmallVAE(torch.nn.Module):
    """Differentiable factor-eight codec for CPU orchestration tests."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)

    @property
    def dtype(self):
        return self.weight.dtype

    def encode(self, rgb):
        value = torch.nn.functional.avg_pool2d(rgb.mean(1, keepdim=True), 8).repeat(
            1, 16, 1, 1
        )

        class Posterior:
            def mode(self):
                return value

            def sample(self, generator):
                return value + 0.01 * torch.randn(value.shape, generator=generator)

        return SimpleNamespace(latent_dist=Posterior())

    def decode(self, latent, return_dict):
        return (
            torch.nn.functional.interpolate(
                latent[:, :3] * self.weight, scale_factor=8
            ),
        )


def test_codec_source_features_do_not_depend_on_target(training, manifest):
    from boogu.preservation.training.runner import encode_pair_latents

    pair = training.load_pair(training.load_manifest(manifest[0])[0])
    vae = SmallVAE().requires_grad_(False)
    with torch.no_grad():
        before = encode_pair_latents(vae, pair, torch.device("cpu"), 43)
        pair["target"] = Image.new("RGB", (32, 16), "red")
        after = encode_pair_latents(vae, pair, torch.device("cpu"), 43)
    assert torch.equal(before["source_mode"], after["source_mode"])
    assert torch.equal(before["reference"], after["reference"])
    assert not torch.equal(before["target_mode"], after["target_mode"])
    assert not torch.equal(before["target_codec"], after["target_codec"])


def test_cached_pair_rejects_wrong_native_channel_count(training, manifest):
    from boogu.preservation.training.data import file_sha256
    from boogu.preservation.training.runner import _load_cached_pair

    row = training.load_manifest(manifest[0])[0]
    value = {
        "positive": torch.ones(1, 1, 6),
        "positive_mask": torch.ones(1, 1),
        "source_mode": torch.zeros(1, 15, 2, 4),
        "target_mode": torch.zeros(1, 15, 2, 4),
        "reference": torch.zeros(1, 15, 2, 4).bfloat16(),
        "target_codec": torch.zeros(1, 3, 16, 32),
    }
    path = manifest[0].parent / f"{row.pair_id}.pt"
    torch.save(value, path)
    with pytest.raises(ValueError, match="channel|canvas|shape"):
        _load_cached_pair(path.parent, row, file_sha256(path))


def test_staged_cache_and_single_update_export_formal_adapter(
    training, manifest, monkeypatch
):
    from diffusers import AutoencoderKL

    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from boogu.preservation.runtime import _validate_adapter
    from boogu.preservation.training import runner

    # A valid ID must not collide with another sample's intermediate filename.
    manifest[1]["pairs"].append(
        {
            **manifest[1]["pairs"][0],
            "pair_id": "train_1.features",
            "instruction": "A second instruction",
        }
    )
    manifest[0].write_text(json.dumps(manifest[1]), encoding="utf-8")
    base_dir = manifest[0].parent / "base"
    base_dir.mkdir()
    inputs = SimpleNamespace(
        base=base_dir,
        config={
            "hidden_size": 24,
            "in_channels": 16,
            "patch_size": 2,
            "num_layers": 33,
            "num_double_stream_layers": 1,
            "axes_dim_rope": [4, 4, 4],
            "axes_lens": [32, 32, 32],
        },
    )
    monkeypatch.setattr(runner, "_base_inputs", lambda args: inputs)
    monkeypatch.setattr(runner, "_check_cuda_device", lambda device: device)

    class Pipe:
        def encode_instruction(self, *args, **kwargs):
            return (
                torch.ones(1, 2, 8).bfloat16(),
                torch.ones(1, 2),
                None,
                None,
                None,
                None,
            )

    monkeypatch.setattr(
        runner,
        "_load_encoding_components",
        lambda inputs: (Pipe(), torch.nn.Identity(), SmallVAE(), None),
    )
    args = runner.build_parser().parse_args(
        [
            "prepare",
            "--manifest",
            str(manifest[0]),
            "--base",
            str(base_dir),
            "--cache",
            str(base_dir.parent / "cache"),
            "--device",
            "cpu",
        ]
    )
    cache = runner.prepare_cache(args)
    assert (cache / "index.json").is_file()
    with pytest.raises(ValueError, match="fresh"):
        runner.prepare_cache(args)

    base = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        hidden_size=24,
        num_layers=33,
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
    ).bfloat16()
    with torch.no_grad():
        base.norm_out.linear_2.weight.normal_(std=0.1)
    before = {name: p.detach().clone() for name, p in base.named_parameters()}
    monkeypatch.setattr(
        BooguImageTransformer2DModel, "from_pretrained", lambda *a, **kw: base
    )
    monkeypatch.setattr(AutoencoderKL, "from_pretrained", lambda *a, **kw: SmallVAE())
    args.command, args.output, args.steps, args.lr, args.seed = (
        "train",
        str(base_dir.parent / "adapter.pt"),
        1,
        0.0003,
        42,
    )
    output = runner.train(args)
    state = torch.load(output, weights_only=True)
    assert (
        _validate_adapter(state, inputs.config)["schema"]
        == "boogu_source_preservation_v1"
    )
    assert state["metadata"]["block_indices"] == [23, 27, 31]
    assert any(
        torch.count_nonzero(value)
        for name, value in state["adapters"].items()
        if ".W_out." in name
    )
    assert all(
        torch.equal(before[name], p) and p.grad is None
        for name, p in base.named_parameters()
    )
    assert not hasattr(base, "_source_preservation_owner")
    assert json.loads(output.with_suffix(".pt.json").read_text())["steps"] == 1
