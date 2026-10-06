"""Single-device, batch-one preparation and fresh Full-adapter training.

Feature preparation is a separate command so frozen Qwen is released before
DiT training. Each sample is read from disk on demand. No pretrained adapter,
dataset location, online download, evaluation or resume is implicit.
"""

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from boogu.edit_only import require_edit_checkpoint

from ..adapters import SourceConditionedTransformer
from ..conditioning import prepare_condition
from ..runtime import (
    _check_cuda_device,
    _compact_condition,
    _free_cuda,
    _load_encoding_components,
    _local_fp8_kernels,
    _move,
)
from .data import file_sha256, load_manifest, load_pair, pair_binding, read_rgb
from .objectives import decode, endpoint_fp32, flow_sample, training_objective

CACHE_SCHEMA = "boogu_preservation_training_cache_v1"


def configure_determinism(seed):
    """Match native deterministic math; unsupported nondeterministic ops fail."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _save_tensor(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)


def _base_inputs(args):
    require_edit_checkpoint(args.base)
    base = Path(args.base).resolve()
    if not base.is_dir() or "turbo" in base.name.lower():
        raise ValueError("base must be a local standard Edit pipeline")
    if args.use_fp8_weights != ("fp8" in base.name.lower()):
        raise ValueError("use_fp8_weights must match the base checkpoint name")
    configs = {}
    for component in ("transformer", "vae", "mllm"):
        configs[component] = json.loads(
            (base / component / "config.json").read_text(encoding="utf-8")
        )
    config, vae_config = configs["transformer"], configs["vae"]
    if (
        config.get("in_channels") != 16
        or config.get("patch_size") != 2
        or config.get("num_layers", 0) - config.get("num_double_stream_layers", 0) <= 31
    ):
        raise ValueError(
            "Full requires the native 16-channel, patch-2 architecture and sites 23/27/31"
        )
    if (
        vae_config.get("latent_channels") != 16
        or len(vae_config.get("block_out_channels", [])) != 4
        or not math.isfinite(vae_config.get("scaling_factor", float("nan")))
        or vae_config["scaling_factor"] <= 0
        or not math.isfinite(vae_config.get("shift_factor", float("nan")))
    ):
        raise ValueError(
            "native factor-8 VAE with finite latent normalization required"
        )
    if configs["mllm"].get("model_type") != "qwen3_vl":
        raise ValueError("native Qwen3-VL instruction encoder required")
    if (
        not (base / "processor").is_dir()
        or not (base / "scheduler" / "scheduler_config.json").is_file()
    ):
        raise ValueError("base must contain the local processor and scheduler")
    return SimpleNamespace(base=base, config=config, vae_config=vae_config)


def _base_binding(base):
    """Stream hashes of local model/config files without loading weights to RAM."""
    files = {}
    for component in ("transformer", "vae", "mllm", "processor", "scheduler"):
        folder = base / component
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                files[path.relative_to(base).as_posix()] = file_sha256(path)
    return {"path": str(base), "sha256": files}


def _encoding_options(args):
    if type(args.reference_seed) is not int or not 0 <= args.reference_seed < 2**63:
        raise ValueError("reference_seed must be in [0, 2**63)")
    if type(args.max_sequence_length) is not int or args.max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    return {
        "reference_seed": args.reference_seed,
        "max_sequence_length": args.max_sequence_length,
        "use_fp8_weights": args.use_fp8_weights,
        "max_vlm_input_pil_pixels": 384 * 384,
        "max_vlm_input_pil_side_length": 768,
        "vae_dtype": "float32",
        "training_branch": "positive",
        "source_latent": "posterior_mode",
        "target_latent": "posterior_mode",
        "reference_latent": "posterior_sample",
    }


def _cache_contract(rows, inputs, args):
    return {
        "schema": CACHE_SCHEMA,
        "base": _base_binding(inputs.base),
        "encoding": _encoding_options(args),
        "pairs": {row.pair_id: pair_binding(row) for row in rows},
    }


def encode_source_features(pipe, row, device, max_sequence_length):
    """Accept only a source image and instruction at the Qwen call boundary."""
    encoded = pipe.encode_instruction(
        row.instruction,
        do_classifier_free_guidance=False,
        negative_instruction="",
        input_images=[[read_rgb(row.source)]],
        use_input_images_4_neg_instruct=False,
        use_input_images_4_empty_instruct=False,
        max_vlm_input_pil_pixels=384 * 384,
        max_vlm_input_pil_side_length=768,
        num_images_per_instruction=1,
        device=device,
        max_sequence_length=max_sequence_length,
        truncate_instruction_sequence=False,
        use_rewrite_text_instruction=False,
        use_dashscope_remote_rewriting=False,
    )
    if len(encoded) != 6 or encoded[4] is not None or encoded[5] is not None:
        raise ValueError("native six-value instruction encoding contract changed")
    positive, mask = _compact_condition(encoded[0], encoded[1])
    return {
        "positive": _move(positive, "cpu"),
        "positive_mask": mask.to(device="cpu", dtype=torch.int64),
    }


def _rgb_tensor(image, device):
    return (
        torch.from_numpy(np.array(image, dtype=np.float32, copy=True))
        .permute(2, 0, 1)[None]
        .to(device)
        / 255
    )


def encode_pair_latents(vae, item, device, reference_seed):
    """Source mode/reference are isolated from target mode/codec supervision."""
    scale, shift = vae.config.scaling_factor, vae.config.shift_factor
    posterior = vae.encode(_rgb_tensor(item["source"], device) * 2 - 1).latent_dist
    source_mode = (posterior.mode() - shift) * scale
    reference = (
        (
            posterior.sample(
                generator=torch.Generator(device=device).manual_seed(reference_seed)
            )
            - shift
        )
        * scale
    ).to(torch.bfloat16)
    del posterior
    target_posterior = vae.encode(
        _rgb_tensor(item["target"], device) * 2 - 1
    ).latent_dist
    target_mode = (target_posterior.mode() - shift) * scale
    expected = (1, 16, item["source"].height // 8, item["source"].width // 8)
    if tuple(source_mode.shape) != expected or tuple(target_mode.shape) != expected:
        raise ValueError(
            "observed latent grid differs from aligned source/target canvas"
        )
    if source_mode.dtype != torch.float32 or target_mode.dtype != torch.float32:
        raise ValueError("source/target posterior modes must remain FP32")
    return {
        "source_mode": source_mode.cpu(),
        "reference": reference.cpu(),
        "target_mode": target_mode.cpu(),
        "target_codec": decode(vae, target_mode).cpu(),
    }


def prepare_cache(args):
    rows = load_manifest(args.manifest)
    inputs = _base_inputs(args)
    contract = _cache_contract(rows, inputs, args)
    cache = Path(args.cache).resolve()
    if cache == inputs.base or inputs.base in cache.parents:
        raise ValueError("cache must be outside base model files")
    if cache.exists():
        raise ValueError(
            "prepare requires a fresh cache directory; partial caches are not resumed"
        )
    configure_determinism(args.reference_seed)
    device = _check_cuda_device(torch.device(args.device))
    cache.mkdir(parents=True)
    (cache / "features").mkdir()
    _save_json(cache / "preparing.json", contract)
    pipe = mllm = vae = None
    try:
        with _local_fp8_kernels(args.use_fp8_weights):
            pipe, mllm, vae, _ = _load_encoding_components(inputs)
            mllm.requires_grad_(False).eval().to(device=device)
            with torch.no_grad():
                for row in rows:
                    print(f"[prepare] source instruction: {row.pair_id}", flush=True)
                    feature = encode_source_features(
                        pipe, row, device, args.max_sequence_length
                    )
                    _save_tensor(cache / "features" / f"{row.pair_id}.pt", feature)
                    del feature
            mllm.to(device="cpu")
            pipe.mllm = None
            pipe.text_instruction_rewriter = None
            pipe = mllm = None
            _free_cuda()
        vae.requires_grad_(False).eval().to(device=device)
        if vae.dtype != torch.float32:
            raise ValueError("VAE must remain FP32")
        files = {}
        with torch.no_grad():
            for row in rows:
                print(f"[prepare] source/target codec: {row.pair_id}", flush=True)
                item = load_pair(row)
                features = torch.load(
                    cache / "features" / f"{row.pair_id}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                features.update(
                    encode_pair_latents(vae, item, device, args.reference_seed)
                )
                path = cache / f"{row.pair_id}.pt"
                _save_tensor(path, features)
                files[row.pair_id] = file_sha256(path)
                del features, item
        # Final index is written only after a second binding check succeeds.
        if _cache_contract(rows, inputs, args) != contract:
            raise ValueError("training inputs or base files changed during preparation")
        _save_json(cache / "index.json", {"contract": contract, "files": files})
        return cache
    finally:
        pipe = mllm = vae = None
        _free_cuda()


def make_draws(rows, steps, seed):
    """Seeded round-robin pairs with per-pair stratified times in [.01,.99)."""
    if (
        type(steps) is not int
        or steps < 1
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("steps must be positive and seed must be in [0, 2**63)")
    keys = sorted(row.pair_id for row in rows)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("draw schedule requires distinct training pairs")
    rng = random.Random(seed)
    times = {}
    for index, key in enumerate(keys):
        count = (steps + len(keys) - 1 - index) // len(keys)
        bins = list(range(count))
        rng.shuffle(bins)
        times[key] = [
            0.01 + 0.98 * (bin_index + rng.random()) / count for bin_index in bins
        ]
    return [
        {
            "step": step + 1,
            "pair_id": keys[step % len(keys)],
            "t": times[keys[step % len(keys)]][step // len(keys)],
            "noise_seed": rng.randrange(1, 2**63),
        }
        for step in range(steps)
    ]


def _load_cached_pair(cache, row, expected_hash):
    path = cache / f"{row.pair_id}.pt"
    if file_sha256(path) != expected_hash:
        raise ValueError(f"cache file changed: {row.pair_id}")
    item = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "positive",
        "positive_mask",
        "source_mode",
        "reference",
        "target_mode",
        "target_codec",
    }
    if type(item) is not dict or set(item) != required:
        raise ValueError("invalid feature cache fields")
    for name in ("source_mode", "target_mode", "target_codec", "reference"):
        value = item[name]
        dtype = torch.bfloat16 if name == "reference" else torch.float32
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != dtype
            or value.ndim != 4
            or value.shape[0] != 1
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"invalid cached {name}")
    if (
        item["source_mode"].shape != item["target_mode"].shape
        or item["reference"].shape != item["source_mode"].shape
    ):
        raise ValueError("cached source/target/reference grids differ")
    image = read_rgb(row.source)
    if tuple(item["source_mode"].shape) != (
        1,
        16,
        image.height // 8,
        image.width // 8,
    ) or tuple(item["target_codec"].shape) != (1, 3, image.height, image.width):
        raise ValueError("cached channels or canvas shape differ from the source")
    item["positive"], item["positive_mask"] = _compact_condition(
        item["positive"], item["positive_mask"]
    )
    item["positive_mask"] = item["positive_mask"].to(dtype=torch.int64)
    return item


def training_step(wrapper, vae, features, pair, draw, freqs, device):
    """One positive-branch forward/backward loss; all supervision is detached."""
    condition = prepare_condition(
        features["source_mode"], pair["keep"], pair["modify"], pair["reference_valid"]
    )
    target = features["target_mode"].to(device)
    noise = torch.randn(
        target.shape,
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(draw["noise_seed"]),
    ).to(device)
    z_fp32, velocity = flow_sample(noise, target, draw["t"])
    prediction = wrapper(
        preservation=condition,
        hidden_states=z_fp32.to(torch.bfloat16),
        timestep=torch.tensor([draw["t"]], device=device, dtype=torch.bfloat16),
        instruction_hidden_states=_move(features["positive"], device),
        instruction_attention_mask=features["positive_mask"].to(device),
        ref_image_hidden_states=[[features["reference"].to(device)[0]]],
        freqs_cis=freqs,
        return_dict=False,
    )
    if prediction.dtype != torch.bfloat16 or prediction.shape != z_fp32.shape:
        raise ValueError("native prediction must be matching BF16 velocity")
    endpoint = endpoint_fp32(z_fp32, prediction, draw["t"])
    rgb = checkpoint(lambda value: decode(vae, value), endpoint, use_reentrant=False)
    keep = (pair["keep"] * pair["reference_valid"]).amax(dim=0, keepdim=True).to(device)
    loss, detail = training_objective(
        prediction,
        velocity,
        rgb,
        features["target_codec"].to(device),
        keep,
        pair["allowed_edit"],
        pair["protected_roi_masks"],
        pair["edit_core"],
    )
    if not torch.isfinite(loss).item():
        raise ValueError("nonfinite training loss")
    loss.backward()
    return detail


def train(args):
    rows = load_manifest(args.manifest)
    inputs = _base_inputs(args)
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    draws = make_draws(rows, args.steps, args.seed)
    output, cache = Path(args.output).resolve(), Path(args.cache).resolve()
    sidecar = Path(str(output) + ".json")
    if output.suffix.lower() != ".pt" or output.exists() or sidecar.exists():
        raise ValueError(
            "output must be a fresh .pt path with a fresh .pt.json sidecar"
        )
    if inputs.base in output.parents or cache in output.parents:
        raise ValueError("output must be outside base and cache directories")
    contract = _cache_contract(rows, inputs, args)
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    if (
        type(index) is not dict
        or set(index) != {"contract", "files"}
        or index["contract"] != contract
        or set(index["files"]) != {row.pair_id for row in rows}
    ):
        raise ValueError(
            "cache does not match current manifest, model files or encoding options"
        )
    for row in rows:
        if file_sha256(cache / f"{row.pair_id}.pt") != index["files"][row.pair_id]:
            raise ValueError(f"cache file changed: {row.pair_id}")
    configure_determinism(args.seed)
    device = _check_cuda_device(torch.device(args.device))
    from diffusers import AutoencoderKL

    from boogu.models.transformers.rope import BooguImageRotaryPosEmbed
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel

    vae = base = wrapper = None
    try:
        vae = (
            AutoencoderKL.from_pretrained(
                str(inputs.base / "vae"),
                torch_dtype=torch.float32,
                local_files_only=True,
                use_safetensors=True,
            )
            .requires_grad_(False)
            .eval()
            .to(device=device)
        )
        base = (
            BooguImageTransformer2DModel.from_pretrained(
                str(inputs.base / "transformer"),
                torch_dtype=torch.bfloat16,
                use_safetensors=not args.use_fp8_weights,
                local_files_only=True,
                device_map=None,
            )
            .requires_grad_(False)
            .eval()
            .to(device=device)
        )
        torch.manual_seed(args.seed)
        wrapper = SourceConditionedTransformer(
            base, block_indices=(23, 27, 31), rank=32, checkpointing=True
        ).train()
        parameters = wrapper.adapter_parameters()
        if any(
            p.dtype != torch.float32 or not p.requires_grad for p in parameters.values()
        ):
            raise ValueError("only FP32 adapter parameters may be trainable")
        frozen = [(p, p._version) for model in (base, vae) for p in model.parameters()]
        optimizer = torch.optim.AdamW(parameters.values(), lr=args.lr, weight_decay=0.0)
        freqs = BooguImageRotaryPosEmbed.get_freqs_cis(
            inputs.config["axes_dim_rope"], inputs.config["axes_lens"], theta=10000
        )
        by_id = {row.pair_id: row for row in rows}
        history = []
        for draw in draws:
            row = by_id[draw["pair_id"]]
            if pair_binding(row) != contract["pairs"][row.pair_id]:
                raise ValueError("training pair changed after cache validation")
            pair = load_pair(row)
            features = _load_cached_pair(cache, row, index["files"][row.pair_id])
            optimizer.zero_grad(set_to_none=True)
            detail = training_step(wrapper, vae, features, pair, draw, freqs, device)
            if any(
                p.grad is None or not torch.isfinite(p.grad).all().item()
                for p in parameters.values()
            ):
                raise ValueError("adapter gradient missing or nonfinite")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(parameters.values()), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if any(
                p.requires_grad or p.grad is not None or p._version != version
                for p, version in frozen
            ):
                raise ValueError("frozen base or VAE changed during training")
            history.append({**draw, "loss": detail, "gradient_norm": float(grad_norm)})
            print(
                f"[train] {draw['step']}/{args.steps} {row.pair_id} loss={detail['total']:.6f}",
                flush=True,
            )
            del features, pair
        if _cache_contract(rows, inputs, args) != contract:
            raise ValueError("training inputs or base files changed during training")
        state = wrapper.export_state()
        output.parent.mkdir(parents=True, exist_ok=True)
        _save_tensor(output, state)
        _save_json(
            sidecar,
            {
                "schema": "boogu_preservation_training_run_v1",
                "adapter_sha256": file_sha256(output),
                "cache_contract": contract,
                "steps": args.steps,
                "lr": args.lr,
                "seed": args.seed,
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "gradient_clip": 1.0,
                "rank": 32,
                "block_indices": [23, 27, 31],
                "fresh_initialization": True,
                "history": history,
            },
        )
        return output
    finally:
        gc.collect()
        if wrapper is not None:
            wrapper.close()
        wrapper = base = vae = None
        _free_cuda()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Validate manifest without loading model weights"
    )
    validate.add_argument("--manifest", required=True)
    for command in ("prepare", "train"):
        child = commands.add_parser(command)
        child.add_argument("--manifest", required=True)
        child.add_argument(
            "--base", required=True, help="Local standard Boogu Edit pipeline"
        )
        child.add_argument("--cache", required=True)
        child.add_argument("--device", default="cuda")
        child.add_argument("--use-fp8-weights", action="store_true")
        child.add_argument("--reference-seed", type=int, default=43)
        child.add_argument("--max-sequence-length", type=int, default=1280)
        if command == "train":
            child.add_argument("--output", required=True)
            child.add_argument("--steps", type=int, default=96)
            child.add_argument("--lr", type=float, default=0.0003)
            child.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        rows = load_manifest(args.manifest)
        print(f"Validated {len(rows)} admitted train pairs", flush=True)
    elif args.command == "prepare":
        print(f"Prepared cache: {prepare_cache(args)}", flush=True)
    else:
        print(f"Saved portable Full adapter: {train(args)}", flush=True)


if __name__ == "__main__":
    main()
