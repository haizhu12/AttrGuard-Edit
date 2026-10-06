"""Staged native inference with explicit source-bound preservation permissions.

Only one unchanged-size reference, CUDA BF16 denoising and FP32 VAE encoding /
decoding are supported. The adapter is required; its presence does not establish
that it preserves attributes or generalizes to a particular input domain.
"""

import gc
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps

from boogu.edit_only import require_edit_checkpoint

from .adapters import SourceConditionedTransformer, _same_metadata
from .conditioning import ATTRIBUTE_NAMES, prepare_condition

_SUPPORTED = {
    "method",
    "preservation_adapter",
    "preservation_permissions",
    "pretrained_pipeline_name_or_path",
    "use_fp8_weights",
    "device",
    "dtype",
    "input_image_paths",
    "instruction",
    "negative_instruction",
    "output_image_path",
    "height",
    "width",
    "seed",
    "reference_seed",
    "num_inference_steps",
    "text_guidance_scale",
    "image_guidance_scale",
    "batch_size",
    "num_images_per_instruction",
    "max_input_image_pixels",
    "max_input_image_side_length",
    "max_vlm_input_pil_pixels",
    "max_vlm_input_pil_side_length",
    "max_sequence_length",
    "scheduler",
}
# Inherited CLI defaults are tolerated only when not explicitly requested.
# A new CLI option must be reviewed here before source mode can accept it.
_LEGACY_DEFAULTS = {
    "empty_instruction": " ",
    "empty_instruction_guidance_scale": 0.0,
    "cfg_range_start": 0.0,
    "cfg_range_end": 1.0,
    "bog_range_start": 0.0,
    "bog_range_end": 1.0,
    "bog_interval": 2,
    "bog_mu": 0.1,
    "teacache_rel_l1_thresh": 0.05,
    "vision_token_ids": [],
    "merge_original_and_rewritten_instructions": True,
    "rewriter_max_new_tokens": 512,
    "resize_rewriter_ref_images": True,
    "rewriter_ref_images_max_pixels": 2048 * 2048,
    "rewriter_ref_images_max_side_length": 2560,
    "do_sample_for_local_rewriter": True,
    "dashscope_remote_rewriting_model": "qwen-vl-max-latest",
    "dashscope_base_http_api_url": "https://dashscope.aliyuncs.com/api/v1",
    "dashscope_api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
    "rewriter_system_prompt_type": "default",
    "custom_rewriter_system_prompts_list": [
        "Rewrite the user's reference-image editing request into a precise edit instruction while preserving all content they did not ask to change."
    ],
    "unload_rewriter_level": "destroy",
    "torch_compile_mode": "default",
    **{
        f"{kind}_momentum_rolling_sum_{term}_weight": value
        for kind in ("text", "image", "empty")
        for term, value in (("momentum", 0.05), ("current", 0.95))
    },
    **{
        name: None
        for name in (
            "custom_diffusion_transformer_path",
            "custom_pretrained_instruction_encoder_model_name_or_path",
            "custom_prompt_tuning_model_path",
            "custom_prompt_tuning_model_lora_weights_path",
            "custom_transformer_lora_path",
            "custom_local_instruction_rewriter_model",
            "far_region_map_paths",
            "pri_predictor_checkpoint",
            "far_invariant_strength",
            "far_effect_strength",
            "far_effect_low_pass_kernel",
            "far_response_transport_mode",
            "far_response_kernel_sizes",
            "batch_data_config_path",
            "save_rewritten_instruction_path",
            "rewriter_device",
        )
    },
    **{
        name: False
        for name in (
            "use_prompt_tuning",
            "use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide",
            "use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide",
            "use_boosted_orthogonal_guidance",
            "use_far",
            "mask_vision_tokens_feature",
            "truncate_instruction_sequence",
            "use_rewrite_text_instruction",
            "use_dashscope_remote_rewriting",
            "use_batch_inference",
            "save_rewritten_instruction",
            "enable_model_cpu_offload_flag",
            "enable_sequential_cpu_offload_flag",
            "enable_group_offload_flag",
            "enable_teacache",
            "enable_taylorseer",
            "enable_teacache_for_all_layers",
            "enable_taylorseer_for_all_layers",
            "enable_cache_dit_caching",
            "enable_cache_dit_caching_for_all_layers",
            "enable_inner_devices_manager",
            "enable_torch_compile",
        )
    },
}


def _validate_options(args):
    explicit = set(getattr(args, "_explicit_options", ()))
    for name in explicit - _SUPPORTED:
        raise ValueError(f"source-conditioned does not support explicit --{name}")
    for name, value in vars(args).items():
        if name in _SUPPORTED or name == "_explicit_options":
            continue
        if name not in _LEGACY_DEFAULTS or not _same_metadata(
            value, _LEGACY_DEFAULTS[name]
        ):
            raise ValueError(f"source-conditioned does not support {name}={value!r}")
    for name, choices in {
        "batch_size": (None, 1),
        "num_images_per_instruction": (1,),
        "image_guidance_scale": (1.0,),
        "scheduler": ("euler",),
    }.items():
        value = getattr(args, name, choices[0])
        if isinstance(value, bool) or value not in choices:
            raise ValueError(f"source-conditioned requires {name} in {choices}")


def source_rgb_sha256(image):
    """Hash EXIF-transposed RGB pixel bytes (permission shapes bind dimensions)."""
    return hashlib.sha256(
        ImageOps.exif_transpose(image).convert("RGB").tobytes()
    ).hexdigest()


def _scalar_string(value, name):
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode string")
    return value.item()


def load_permissions(path, image):
    """Read a source-bound NPZ without pickle; return validated CPU K/M/valid maps."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    with np.load(path, allow_pickle=False) as archive:
        required = {"keep", "modify", "source_sha256", "schema", "attributes"}
        if not required <= set(archive.files) or set(archive.files) - required - {
            "reference_valid"
        }:
            raise ValueError(
                "permissions require keep, modify, source_sha256, schema and attributes only"
            )
        if (
            _scalar_string(archive["schema"], "schema")
            != "boogu_preservation_permissions_v1"
        ):
            raise ValueError("unsupported permissions schema")
        attributes = archive["attributes"]
        if (
            attributes.dtype.kind != "U"
            or attributes.shape != (5,)
            or attributes.tolist() != list(ATTRIBUTE_NAMES)
        ):
            raise ValueError(
                "permissions attributes must be color, structure, texture, glyph, pixel in order"
            )
        if _scalar_string(
            archive["source_sha256"], "source_sha256"
        ) != source_rgb_sha256(image):
            raise ValueError(
                "permissions source_sha256 does not match the EXIF-transposed source RGB"
            )
        arrays = [archive[name] for name in ("keep", "modify")]
        for name, value in zip(("keep", "modify"), arrays):
            if (
                value.dtype.kind != "f"
                or value.ndim != 4
                or value.shape[0] < 1
                or value.shape[1:] != (5, image.height, image.width)
            ):
                raise ValueError(
                    f"{name} must be floating [objects,5,source_height,source_width]"
                )
            if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must contain finite values in [0,1]")
        keep, modify = [
            torch.from_numpy(value.astype(np.float32, copy=True)) for value in arrays
        ]
        valid = (
            archive["reference_valid"]
            if "reference_valid" in archive.files
            else np.ones(arrays[0].shape, dtype=bool)
        )
        if valid.dtype != np.bool_ or valid.shape != arrays[0].shape:
            raise ValueError(
                "reference_valid must be boolean and match the permission shape"
            )
        valid = torch.from_numpy(valid.copy())
    if image.width % 16 or image.height % 16:
        raise ValueError(
            "source canvas must be divisible by 16; resizing is unsupported"
        )
    # Reuse canonical cross-object conflicts, unknown roles and alignment checks.
    prepare_condition(
        torch.zeros(1, 16, image.height // 8, image.width // 8), keep, modify, valid
    )
    return keep, modify, valid


def _validate_adapter(state, config):
    """Validate portable adapter metadata and every tensor before large loads."""
    if (
        type(state) is not dict
        or set(state) != {"metadata", "adapters"}
        or type(state["metadata"]) is not dict
    ):
        raise ValueError(
            "adapter checkpoint must contain exactly metadata and adapters"
        )
    metadata = state["metadata"]
    indices, rank = metadata.get("block_indices"), metadata.get("rank")
    layer_count = config["num_layers"] - config["num_double_stream_layers"]
    if (
        type(indices) is not list
        or not indices
        or any(type(i) is not int or not 0 <= i < layer_count for i in indices)
        or len(set(indices)) != len(indices)
        or type(rank) is not int
        or rank <= 0
    ):
        raise ValueError(
            "adapter block_indices / rank are invalid for the native transformer"
        )
    expected = {
        "schema": "boogu_source_preservation_v1",
        "block_indices": indices,
        "rank": rank,
        "hidden_size": config["hidden_size"],
        "in_channels": config["in_channels"],
        "patch_size": config["patch_size"],
        "attributes": list(ATTRIBUTE_NAMES),
        "role_order": ["keep", "modify", "unknown"],
        "use_source": True,
        "use_roles": True,
    }
    if not _same_metadata(metadata, expected):
        raise ValueError(
            "adapter metadata does not exactly match the source-conditioned native architecture"
        )
    shapes = {}
    for index in indices:
        for name, shape in {
            "W_h": (rank, config["hidden_size"]),
            "W_s": (rank, config["patch_size"] ** 2 * config["in_channels"]),
            "W_r": (rank, 15),
            "W_t": (rank, 1),
            "W_out": (config["hidden_size"], rank),
        }.items():
            shapes[f"banks.{index}.{name}.weight"] = shape
            shapes[f"banks.{index}.{name}.bias"] = (shape[0],)
    tensors = state["adapters"]
    if type(tensors) is not dict or set(tensors) != set(shapes):
        raise ValueError("adapter tensor names do not exactly match metadata")
    for name, shape in shapes.items():
        value = tensors[name]
        if (
            type(value) is not torch.Tensor
            or value.layout != torch.strided
            or value.is_meta
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(
                f"adapter {name} requires the expected shape and finite FP32 tensor"
            )
    return metadata


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _single_limit(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} requires exactly one limit")
        value = value[0]
    return _positive_int(value, name)


@dataclass
class _Inputs:
    args: object
    image: Image.Image
    source_path: Path
    base: Path
    adapter_path: Path
    permissions_path: Path
    state: dict
    config: dict
    vae_config: dict
    permissions: tuple
    seed: int
    reference_seed: int
    device: torch.device
    input_hashes: dict


def _preflight(args):
    for name in (
        "preservation_adapter",
        "preservation_permissions",
        "input_image_paths",
        "instruction",
    ):
        value = getattr(args, name, None)
        if not value or isinstance(value, str) and not value.strip():
            raise ValueError(f"source-conditioned requires {name}")
    _validate_options(args)
    paths = args.input_image_paths
    if (
        not isinstance(paths, (list, tuple))
        or len(paths) != 1
        or not isinstance(paths[0], (str, Path))
    ):
        raise ValueError("input_image_paths must contain exactly one source image")
    if not isinstance(args.instruction, str) or not isinstance(
        getattr(args, "negative_instruction", ""), str
    ):
        raise TypeError("instruction and negative_instruction must be strings")
    if getattr(args, "method", "source-conditioned") != "source-conditioned":
        raise ValueError("runtime requires method=source-conditioned")
    source_path = Path(paths[0]).resolve()
    adapter_path = Path(args.preservation_adapter).resolve()
    permissions_path = Path(args.preservation_permissions).resolve()
    for name, path in (
        ("input_image_paths", source_path),
        ("preservation_adapter", adapter_path),
        ("preservation_permissions", permissions_path),
    ):
        if not path.is_file():
            raise ValueError(f"{name} must name an existing local file: {path}")
    input_hashes = {
        str(path): _sha256(path)
        for path in (source_path, adapter_path, permissions_path)
    }
    base_value = getattr(args, "pretrained_pipeline_name_or_path", None)
    if not isinstance(base_value, (str, Path)) or not str(base_value).strip():
        raise ValueError(
            "pretrained_pipeline_name_or_path must name a local Edit checkpoint"
        )
    require_edit_checkpoint(str(base_value))
    base = Path(base_value).resolve()
    if "turbo" in base.name.lower() or not base.is_dir():
        raise ValueError(
            "source-conditioned requires a local standard Edit checkpoint; Turbo is unsupported"
        )
    use_fp8 = getattr(args, "use_fp8_weights", False)
    if type(use_fp8) is not bool or use_fp8 != ("fp8" in base.name.lower()):
        raise ValueError("use_fp8_weights must match the local checkpoint format")
    output = Path(getattr(args, "output_image_path", "output.png")).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("output_image_path must be a lossless .png file")
    if (
        output in (source_path, adapter_path, permissions_path)
        or output == base
        or base in output.parents
    ):
        raise ValueError(
            "output_image_path must not overwrite inputs or checkpoint files"
        )
    if output.exists() or Path(str(output) + ".json").exists():
        raise ValueError("output_image_path and metadata sidecar must be fresh paths")
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if image.width % 16 or image.height % 16:
        raise ValueError(
            "source canvas must be divisible by 16; resizing is unsupported"
        )
    for name, actual in (("height", image.height), ("width", image.width)):
        value = getattr(args, name, None)
        if value is not None and (type(value) is not int or value != actual):
            raise ValueError(
                f"{name} must match the source canvas; resizing is unsupported"
            )
    for name, actual, default in (
        ("max_input_image_pixels", image.width * image.height, 4194304),
        ("max_input_image_side_length", max(image.size), 4096),
    ):
        if _single_limit(getattr(args, name, default), name) < actual:
            raise ValueError(f"{name} would resize the source; increase its limit")
    for name, default in (
        ("max_vlm_input_pil_pixels", 147456),
        ("max_vlm_input_pil_side_length", 768),
        ("max_sequence_length", 1024),
    ):
        _single_limit(getattr(args, name, default), name)
    _positive_int(getattr(args, "num_inference_steps", 50), "num_inference_steps")
    cfg = getattr(args, "text_guidance_scale", 1.0)
    if (
        isinstance(cfg, bool)
        or not isinstance(cfg, (int, float))
        or not math.isfinite(cfg)
        or cfg < 1
    ):
        raise ValueError("text_guidance_scale must be finite and >= 1")
    seed = getattr(args, "seed", 0)
    reference_seed = getattr(args, "reference_seed", None)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if reference_seed is None:
        reference_seed = (seed + 1) % 2**63
    if (
        type(reference_seed) is not int
        or not 0 <= reference_seed < 2**63
        or reference_seed == seed
    ):
        raise ValueError("reference_seed must be a distinct integer in [0, 2**63)")
    if getattr(args, "dtype", "bf16") != "bf16":
        raise ValueError("source-conditioned currently requires dtype=bf16")
    device = torch.device(getattr(args, "device", "cuda"))
    if device.type != "cuda":
        raise ValueError(
            "source-conditioned currently requires a CUDA device with BF16 support"
        )
    configs = {}
    for component, filename in (
        ("transformer", "config.json"),
        ("vae", "config.json"),
        ("mllm", "config.json"),
        ("scheduler", "scheduler_config.json"),
    ):
        path = base / component / filename
        if not path.is_file():
            raise ValueError(f"missing native {component} configuration: {path}")
        input_hashes[str(path)] = _sha256(path)
        configs[component] = json.loads(path.read_text(encoding="utf-8"))
        if type(configs[component]) is not dict:
            raise ValueError(f"invalid {component} configuration")
    config, vae_config = configs["transformer"], configs["vae"]
    for name in (
        "hidden_size",
        "in_channels",
        "patch_size",
        "num_layers",
        "num_double_stream_layers",
    ):
        _positive_int(config.get(name), f"transformer.{name}")
    if (
        config["in_channels"] != 16
        or config["patch_size"] != 2
        or config["num_layers"] <= config["num_double_stream_layers"]
    ):
        raise ValueError(
            "native transformer must use 16 latent channels and 2x2 patches"
        )
    if (
        vae_config.get("latent_channels") != 16
        or len(vae_config.get("block_out_channels", [])) != 4
        or vae_config.get("latents_mean") is not None
        or vae_config.get("latents_std") is not None
    ):
        raise ValueError(
            "native VAE must use 16 channels, 8x downsampling and scalar normalization"
        )
    for key in ("shift_factor", "scaling_factor"):
        value = vae_config.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (key == "scaling_factor" and value <= 0)
        ):
            raise ValueError(f"invalid VAE {key}")
    if configs["scheduler"].get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("native Euler scheduler configuration is required")
    if (
        not isinstance(config.get("instruction_feature_configs"), dict)
        or not config.get("axes_dim_rope")
        or not config.get("axes_lens")
    ):
        raise ValueError("native instruction and rotary configurations are required")
    for name in ("axes_dim_rope", "axes_lens"):
        axes = config[name]
        if (
            type(axes) is not list
            or len(axes) != 3
            or any(type(value) is not int or value <= 0 for value in axes)
        ):
            raise ValueError(f"native {name} must contain three positive integers")
    features = config["instruction_feature_configs"]
    mllm_config = configs["mllm"]
    text_config = mllm_config.get("text_config", {})
    if mllm_config.get("model_type") != "qwen3_vl" or type(text_config) is not dict:
        raise ValueError(
            "native Qwen3-VL instruction encoder configuration is required"
        )
    if features.get("instruction_feat_dim") != text_config.get("hidden_size"):
        raise ValueError(
            "Qwen hidden size must match the native instruction feature dimension"
        )
    layers = features.get("num_instruction_feature_layers", 1)
    if type(layers) is not int or not 1 <= layers <= text_config.get(
        "num_hidden_layers", 0
    ):
        raise ValueError(
            "native instruction feature layer count exceeds the Qwen encoder"
        )
    quantization = mllm_config.get("quantization_config")
    if (
        use_fp8
        and (
            type(quantization) is not dict
            or quantization.get("quant_method") != "fp8"
            or quantization.get("dequantize", False)
        )
    ) or (not use_fp8 and quantization is not None):
        raise ValueError(
            "use_fp8_weights must match native Qwen quantization without dequantizing"
        )
    if not (base / "processor").is_dir():
        raise ValueError("missing native processor directory")
    for component in ("transformer", "vae", "mllm"):
        if not any((base / component).glob("*.safetensors")) and not any(
            (base / component).glob("*.bin")
        ):
            raise ValueError(f"missing local {component} weights")
        for index_path in (base / component).glob("*.index.json"):
            input_hashes[str(index_path)] = _sha256(index_path)
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weights = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"invalid local {component} weight index")
            directory = (base / component).resolve()
            for shard in weights.values():
                if not isinstance(shard, str):
                    raise TypeError(f"invalid {component} shard filename")
                shard_path = (directory / shard).resolve()
                if directory not in shard_path.parents or not shard_path.is_file():
                    raise ValueError(
                        f"missing or escaped local {component} weight shard: {shard}"
                    )
    permissions = load_permissions(permissions_path, image)
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    _validate_adapter(state, config)
    inputs = _Inputs(
        args,
        image,
        source_path,
        base,
        adapter_path,
        permissions_path,
        state,
        config,
        vae_config,
        permissions,
        seed,
        reference_seed,
        device,
        input_hashes,
    )
    _assert_input_hashes(inputs)
    return inputs


def _assert_input_hashes(inputs):
    for path, expected in inputs.input_hashes.items():
        if not Path(path).is_file() or _sha256(path) != expected:
            raise ValueError(
                f"input changed after source-conditioned preflight: {path}"
            )


@contextmanager
def _local_fp8_kernels(enabled):
    """Use an installed cached kernel without a download or an inference import."""
    if not enabled:
        yield
        return
    import transformers.integrations.finegrained_fp8 as fp8
    from huggingface_hub.constants import HF_HUB_CACHE
    from kernels import get_local_kernel

    candidates = sorted(
        (
            Path(HF_HUB_CACHE)
            / "kernels--kernels-community--finegrained-fp8"
            / "snapshots"
        ).glob("*/build/torch-cuda"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise ValueError(
            "FP8 Qwen requires an installed local finegrained-fp8 kernel in the Hugging Face cache"
        )
    kernel = get_local_kernel(candidates[0])
    required = (
        "w8a8_fp8_matmul",
        "fp8_act_quant",
        "w8a8_fp8_matmul_batched",
        "w8a8_fp8_matmul_grouped",
    )
    if not all(callable(getattr(kernel, name, None)) for name in required):
        raise ValueError("local finegrained-fp8 kernel lacks the native entrypoints")

    def local_only(name, *args, **kwargs):
        if name != "finegrained-fp8":
            raise ValueError(f"unexpected kernel request: {name}")
        return kernel

    def disable_deepgemm(*args, **kwargs):
        raise ImportError(
            "DeepGEMM disabled; use local Triton finegrained-fp8 fallback"
        )

    replacements = {"lazy_load_kernel": local_only}
    for name in ("deepgemm_fp8_fp4_linear", "_load_deepgemm_kernel"):
        if hasattr(fp8, name):
            replacements[name] = disable_deepgemm
    previous = {name: getattr(fp8, name) for name in replacements}
    env_key = "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"
    old_env = os.environ.get(env_key)
    try:
        os.environ[env_key] = "1"
        for name, value in replacements.items():
            setattr(fp8, name, value)
        fp8._load_finegrained_fp8_kernel.cache_clear()
        yield
    finally:
        for name, value in previous.items():
            setattr(fp8, name, value)
        fp8._load_finegrained_fp8_kernel.cache_clear()
        if old_env is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_env


def _load_encoding_components(inputs):
    from diffusers import AutoencoderKL
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler,
    )

    mllm = Qwen3VLForConditionalGeneration.from_pretrained(
        str(inputs.base / "mllm"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map=None,
        use_safetensors=True,
    )
    processor = Qwen3VLProcessor.from_pretrained(
        str(inputs.base / "processor"), local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        str(inputs.base / "vae"),
        torch_dtype=torch.float32,
        local_files_only=True,
        use_safetensors=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(inputs.base / "scheduler"), local_files_only=True
    )
    mllm.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    pipe = BooguImagePipeline(
        transformer=None, vae=vae, scheduler=scheduler, mllm=mllm, processor=processor
    )
    metadata = torch.nn.Module()
    metadata.config = SimpleNamespace(**inputs.config)
    metadata.instruction_feature_configs = dict(
        inputs.config["instruction_feature_configs"]
    )
    pipe.transformer = metadata
    pipe.enable_inner_devices_manager = False
    return pipe, mllm, vae, scheduler


def _compact_condition(features, mask):
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 2
        or mask.shape[0] != 1
        or not ((mask == 0) | (mask == 1)).all().item()
    ):
        raise ValueError("native instruction mask must be binary [1,T]")
    valid = mask[0].bool()
    if not valid.any().item():
        raise ValueError("native instruction branch has no valid tokens")
    layers = features if isinstance(features, (list, tuple)) else [features]
    compact = []
    for feature in layers:
        if (
            not isinstance(feature, torch.Tensor)
            or feature.ndim != 3
            or feature.shape[:2] != mask.shape
            or not torch.isfinite(feature).all().item()
        ):
            raise ValueError(
                "native instruction features must be finite and match the mask"
            )
        compact.append(feature[:, valid.to(feature.device)].detach().clone())
    if not compact:
        raise ValueError("native instruction features must have at least one layer")
    return (
        compact if isinstance(features, (list, tuple)) else compact[0],
        torch.ones((1, int(valid.sum())), dtype=mask.dtype, device=mask.device),
    )


def _move(value, device):
    return (
        [_move(item, device) for item in value]
        if isinstance(value, (list, tuple))
        else value.to(device=device)
    )


def _free_cuda():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _initial_noise(shape, seed, device):
    # Match native diffusers randn_tensor with the public CLI's CPU generator.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        shape, generator=generator, device="cpu", dtype=torch.bfloat16
    ).to(device=device)


def _check_cuda_device(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "source-conditioned requires an available CUDA device with BF16 support"
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"requested CUDA device index is unavailable: {index}")
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise ValueError(f"requested CUDA device {index} does not support BF16")
    return torch.device("cuda", index)


def _denoise(
    model,
    scheduler,
    latent,
    reference,
    condition,
    encoded,
    *,
    freqs_cis,
    steps,
    guidance,
):
    positive, mask, negative, negative_mask = encoded
    scheduler.set_timesteps(
        steps, device=latent.device, num_tokens=latent.shape[-2] * latent.shape[-1]
    )
    for index, timestep in enumerate(scheduler.timesteps):
        print(f"[source-conditioned] Denoising {index + 1}/{steps}", flush=True)
        predictions = []
        branches = ((positive, mask),) if guidance == 1 else ((positive, mask), (negative, negative_mask))
        for features, attention in branches:
            result = model(
                preservation=condition,
                hidden_states=latent,
                timestep=timestep.expand(1).to(
                    device=latent.device, dtype=latent.dtype
                ),
                instruction_hidden_states=features,
                instruction_attention_mask=attention,
                ref_image_hidden_states=[[reference[0]]],
                freqs_cis=freqs_cis,
                return_dict=False,
            )
            if (
                not isinstance(result, torch.Tensor)
                or result.shape != latent.shape
                or not torch.isfinite(result).all().item()
            ):
                raise ValueError(
                    "native transformer must return a finite latent-shaped tensor"
                )
            predictions.append(result)
        velocity = predictions[0]
        if guidance != 1:
            velocity = velocity + (guidance - 1) * (velocity - predictions[1])
        latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0].to(
            latent.dtype
        )
    return latent


@torch.no_grad()
def run_source_conditioned(args):
    """Return CPU BCHW RGB [0,1] plus JSON-compatible provenance; save nothing."""
    print("[source-conditioned] Validating source, permissions and adapter", flush=True)
    inputs = _preflight(args)
    inputs.device = _check_cuda_device(inputs.device)
    pipe = mllm = vae = base = wrapper = None
    try:
        with _local_fp8_kernels(getattr(args, "use_fp8_weights", False)):
            print(
                "[source-conditioned] Loading native instruction encoder and FP32 VAE",
                flush=True,
            )
            pipe, mllm, vae, scheduler = _load_encoding_components(inputs)
            mllm.to(device=inputs.device)
            encoded = pipe.encode_instruction(
                args.instruction,
                do_classifier_free_guidance=True,
                negative_instruction=getattr(args, "negative_instruction", ""),
                input_images=[[inputs.image]],
                use_input_images_4_neg_instruct=False,
                use_input_images_4_empty_instruct=False,
                max_vlm_input_pil_pixels=_single_limit(
                    getattr(args, "max_vlm_input_pil_pixels", 147456),
                    "max_vlm_input_pil_pixels",
                ),
                max_vlm_input_pil_side_length=getattr(
                    args, "max_vlm_input_pil_side_length", 768
                ),
                num_images_per_instruction=1,
                device=inputs.device,
                max_sequence_length=getattr(args, "max_sequence_length", 1024),
                truncate_instruction_sequence=False,
                use_rewrite_text_instruction=False,
                use_dashscope_remote_rewriting=False,
            )
            if len(encoded) != 6 or encoded[4] is not None or encoded[5] is not None:
                raise ValueError(
                    "native six-value instruction encoding contract changed"
                )
            positive, mask = _compact_condition(encoded[0], encoded[1])
            negative, negative_mask = _compact_condition(encoded[2], encoded[3])
            encoded = tuple(
                _move(value, "cpu")
                for value in (positive, mask, negative, negative_mask)
            )
            del positive, mask, negative, negative_mask
            mllm.to(device="cpu")
            pipe.mllm = None
            pipe.text_instruction_rewriter = None
            mllm = None
            _free_cuda()
        print(
            "[source-conditioned] Encoding deterministic source and independently sampled reference",
            flush=True,
        )
        vae.to(device=inputs.device)
        rgb = (
            torch.from_numpy(np.array(inputs.image, dtype=np.float32, copy=True))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(inputs.device)
            / 255
        )
        posterior = vae.encode(rgb * 2 - 1).latent_dist
        shift, scale = (
            inputs.vae_config["shift_factor"],
            inputs.vae_config["scaling_factor"],
        )
        source = (posterior.mode() - shift) * scale
        reference_generator = torch.Generator(device=inputs.device).manual_seed(
            inputs.reference_seed
        )
        reference = (
            (posterior.sample(generator=reference_generator) - shift) * scale
        ).to(torch.bfloat16)
        condition = prepare_condition(source.cpu(), *inputs.permissions)
        reference = reference.cpu()
        del source, posterior, rgb
        vae.to(device="cpu")
        _free_cuda()
        print(
            "[source-conditioned] Loading native transformer and required trained adapter",
            flush=True,
        )
        from boogu.models.transformers.rope import BooguImageRotaryPosEmbed
        from boogu.models.transformers.transformer_boogu import (
            BooguImageTransformer2DModel,
        )

        base = BooguImageTransformer2DModel.from_pretrained(
            str(inputs.base / "transformer"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map=None,
            use_safetensors=not getattr(args, "use_fp8_weights", False),
        )
        base.requires_grad_(False).eval()
        # Device-only moves preserve native Float8Tensor quantization.
        base.to(device=inputs.device)
        metadata = inputs.state["metadata"]
        wrapper = SourceConditionedTransformer(
            base,
            block_indices=metadata["block_indices"],
            rank=metadata["rank"],
            checkpointing=False,
        )
        wrapper.load_state(inputs.state)
        wrapper.eval()
        encoded = tuple(_move(value, inputs.device) for value in encoded)
        reference = reference.to(device=inputs.device)
        latent = _initial_noise(reference.shape, inputs.seed, inputs.device)
        freqs = BooguImageRotaryPosEmbed.get_freqs_cis(
            inputs.config["axes_dim_rope"], inputs.config["axes_lens"], theta=10000
        )
        latent = _denoise(
            wrapper,
            scheduler,
            latent,
            reference,
            condition,
            encoded,
            freqs_cis=freqs,
            steps=getattr(args, "num_inference_steps", 50),
            guidance=getattr(args, "text_guidance_scale", 1.0),
        )
        latent = latent.float().cpu()
        wrapper.close()
        wrapper = None
        base.to(device="cpu")
        base = None
        del encoded, reference, freqs
        _free_cuda()
        print("[source-conditioned] Decoding with native FP32 VAE", flush=True)
        vae.to(device=inputs.device)
        decoded = vae.decode(
            latent.to(device=inputs.device) / scale + shift, return_dict=False
        )[0]
        if not torch.isfinite(decoded).all().item() or tuple(decoded.shape) != (
            1,
            3,
            inputs.image.height,
            inputs.image.width,
        ):
            raise ValueError(
                "native decoded image must be finite BCHW RGB with the original source canvas"
            )
        result = (decoded.float() / 2 + 0.5).clamp(0, 1).cpu()
        _assert_input_hashes(inputs)
        metadata = {
            "method": "source-conditioned",
            "source_path": str(inputs.source_path),
            "source_rgb_sha256": source_rgb_sha256(inputs.image),
            "source_file_sha256": inputs.input_hashes[str(inputs.source_path)],
            "adapter_path": str(inputs.adapter_path),
            "adapter_sha256": inputs.input_hashes[str(inputs.adapter_path)],
            "permissions_path": str(inputs.permissions_path),
            "permissions_sha256": inputs.input_hashes[str(inputs.permissions_path)],
            "input_file_sha256": inputs.input_hashes,
            "base_checkpoint": str(inputs.base),
            "seed": inputs.seed,
            "reference_seed": inputs.reference_seed,
            "instruction": args.instruction,
            "negative_instruction": getattr(args, "negative_instruction", ""),
            "max_vlm_input_pil_pixels": _single_limit(
                getattr(args, "max_vlm_input_pil_pixels", 147456),
                "max_vlm_input_pil_pixels",
            ),
            "max_vlm_input_pil_side_length": getattr(
                args, "max_vlm_input_pil_side_length", 768
            ),
            "max_sequence_length": getattr(args, "max_sequence_length", 1024),
            "initial_noise_device": "cpu",
            "num_inference_steps": getattr(args, "num_inference_steps", 50),
            "text_guidance_scale": getattr(args, "text_guidance_scale", 1.0),
            "negative_dit_branch": "skipped" if getattr(args, "text_guidance_scale", 1.0) == 1 else "conditioned",
            "image_guidance_scale": 1.0,
            "height": inputs.image.height,
            "width": inputs.image.width,
            "adapter_metadata": inputs.state["metadata"],
            "source_encoding": "VAE posterior mode",
            "reference_encoding": "independent VAE posterior sample",
            "vae_dtype": "float32",
            "denoiser_dtype": "bfloat16",
            "warning": "Adapter quality and generalization are not established; architecture availability is not evidence of preservation quality.",
        }
        return result, metadata
    finally:
        if wrapper is not None:
            wrapper.close()
        for module in (mllm, base, vae):
            if module is not None:
                module.to(device="cpu")
        pipe = mllm = base = vae = wrapper = None
        _free_cuda()
