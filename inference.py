"""
Boogu Image Editing Inference Demo

# Copyright (C) 2026 Boogu Team.
# This repository is a fork by Boogu Team; modifications have been made.
#
# Original work: Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.
#
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
import re
import warnings
from math import isfinite
from numbers import Integral, Real
from typing import List, Optional, Tuple

import cache_dit
import torch
from diffusers.hooks import apply_group_offloading
from omegaconf import ListConfig, OmegaConf
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from transformers import AutoProcessor

from boogu.edit_only import (
    EditOnlyInputError,
    require_edit_checkpoint,
    require_reference_batch,
)
from boogu.far.config import FARConfig, ResponseTransportConfig
from boogu.far.io import load_pri_region_map_list
from boogu.far.training import load_predictor_from_checkpoint
from boogu.models.transformers.transformer_boogu import (
    BooguImageTransformer2DModel,
    PromptEmbedding,
)
from boogu.pipelines.boogu.pipeline_boogu import (
    BooguImagePipeline,
    BooguImagePromptTuningPipeline,
    FMPipelineOutput,
)
from boogu.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from boogu.utils.validator_utils import (
    get_device_validator,
    validate_device_and_offload_strategy_compatibility,
)


def to_bool(s):
    if isinstance(s, bool):
        return s
    s = s.lower()
    if s in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "f", "0", "no", "n", "off", ""}:
        return False
    raise argparse.ArgumentTypeError("Expect bool value: true/false, 1/0, yes/no")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="AttrGuard-Edit 0.2.1 source-conditioned editing (legacy: --method baseline).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--method", choices=["source-conditioned", "baseline"],
        default="source-conditioned",
        help="Current source-conditioned architecture, or explicitly selected legacy baseline.",
    )
    parser.add_argument(
        "--preservation_adapter", default=None,
        help="Required source-conditioned adapter checkpoint (.pt); no implicit pilot weights.",
    )
    parser.add_argument(
        "--preservation_permissions", default=None,
        help="Required source-bound per-attribute K/M permission maps (.npz).",
    )
    parser.add_argument(
        "--custom_diffusion_transformer_path",
        type=str,
        default=None,
        help=(
            "Optional path to a custom diffusion transformer checkpoint. "
            "When set, it replaces the transformer stored in the pretrained pipeline."
        ),
    )

    parser.add_argument(
        "--pretrained_pipeline_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path or name of a Boogu-Image Edit pretrained pipeline.",
    )

    parser.add_argument(
        "--custom_pretrained_instruction_encoder_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Optional path or model name for a custom pretrained instruction encoder, usually a VLM/MLLM. "
            "When set, it replaces the MLLM stored in the pretrained pipeline and also updates the processor."
        ),
    )

    parser.add_argument(
        "--use_prompt_tuning",
        type=to_bool,
        default=False,
        help="Whether to use prompt tuning.",
    )

    parser.add_argument(
        "--custom_prompt_tuning_model_path",
        type=str,
        default=None,
        help=(
            "Optional path to a custom prompt-tuning embedding checkpoint. "
            "Leave unset or empty to keep prompt tuning disabled."
        ),
    )

    parser.add_argument(
        "--custom_prompt_tuning_model_lora_weights_path",
        type=str,
        default=None,
        help=(
            "Optional path to LoRA weights for the custom prompt-tuning embedding model. "
            "Only takes effect when `--use_prompt_tuning=True` and the path is non-empty."
        ),
    )

    parser.add_argument(
        "--custom_transformer_lora_path",
        type=str,
        default=None,
        help=(
            "Optional path to LoRA weights for the diffusion transformer. "
            "Leave unset or empty to skip loading transformer LoRA weights."
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver++"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for image editing."
    )
    parser.add_argument(
        "--reference_seed",
        type=int,
        default=None,
        help=(
            "Independent VAE reference-latent seed. When omitted, a distinct "
            "deterministic value is derived from --seed."
        ),
    )
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--max_input_image_pixels",
        type=int,
        nargs="+",
        default=4194304,
        help="Maximum number of pixels for each input image passed to the Diffusion Transformer.",
    )
    parser.add_argument(
        "--max_input_image_side_length",
        type=int,
        default=4096,
        help="Maximum side length of each input image passed to the Diffusion Transformer.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for model weights.",
    )
    parser.add_argument(
        "--text_guidance_scale", type=float, default=1.0,
        help="Text guidance scale (default: 1.0, conditional prediction without extra CFG amplification).",
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=1.0, help="Image guidance scale."
    )
    parser.add_argument(
        "--empty_instruction_guidance_scale",
        type=float,
        default=0.0,
        help="Auxiliary empty-instruction guidance scale for reference-image editing. It is used only with compatible double-guidance settings.",
    )
    parser.add_argument(
        "--use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide",
        type=to_bool,
        default=False,
        help="Whether to use empty negative instruction for image guidance in double guidance.",
    )
    parser.add_argument(
        "--use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide",
        type=to_bool,
        default=False,
        help="Whether to use empty negative instruction for text guidance in double guidance.",
    )
    parser.add_argument(
        "--use_boosted_orthogonal_guidance",
        type=to_bool,
        default=False,
        help="Whether to enable the Boosted Orthogonal Guidance (BOG).",
    )

    parser.add_argument(
        "--text_momentum_rolling_sum_momentum_weight",
        type=float,
        default=0.05,
        help="The weight for historical momentum state term in MomentumRollingSum for text guidance when BOG is enabled. Only takes effect when `--use_boosted_orthogonal_guidance=True`.",
    )

    parser.add_argument(
        "--text_momentum_rolling_sum_current_weight",
        type=float,
        default=0.95,
        help="The weight for current update in MomentumRollingSum for text guidance when BOG is enabled. Only takes effect when `--use_boosted_orthogonal_guidance=True`.",
    )

    parser.add_argument(
        "--image_momentum_rolling_sum_momentum_weight",
        type=float,
        default=0.05,
        help="The weight for historical momentum state term in MomentumRollingSum for reference image guidance when BOG is enabled. Only takes effect when `--use_boosted_orthogonal_guidance=True`.",
    )

    parser.add_argument(
        "--image_momentum_rolling_sum_current_weight",
        type=float,
        default=0.95,
        help="The weight for current update in MomentumRollingSum for reference image guidance when BOG is enabled. Only takes effect when `--use_boosted_orthogonal_guidance=True`.",
    )

    parser.add_argument(
        "--empty_momentum_rolling_sum_momentum_weight",
        type=float,
        default=0.05,
        help="The weight for historical momentum state term in MomentumRollingSum for empty instruction guidance when BOG is enabled. Only takes effect when `--use_boosted_orthogonal_guidance=True` and \
         `--use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=True` or `--use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=True`",
    )

    parser.add_argument(
        "--empty_momentum_rolling_sum_current_weight",
        type=float,
        default=0.95,
        help="The weight for current update in MomentumRollingSum for empty instruction guidance when BOG is enabled. Only takes effect when `--use_boosted_orthogonal_guidance=True` and \
         `--use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=True` or `--use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=True`",
    )

    parser.add_argument(
        "--bog_mu",
        type=float,
        default=0.1,
        help="The mu weight for BOG. Only takes effect when `--use_boosted_orthogonal_guidance=True`.",
    )

    parser.add_argument(
        "--cfg_range_start", type=float, default=0.0, help="Start of the CFG range."
    )
    parser.add_argument(
        "--cfg_range_end", type=float, default=1.0, help="End of the CFG range."
    )

    parser.add_argument(
        "--bog_range_start", type=float, default=0.0, help="Start of the BOG range."
    )
    parser.add_argument(
        "--bog_range_end", type=float, default=1.0, help="End of the BOG range."
    )

    parser.add_argument(
        "--bog_interval",
        type=int,
        default=2,
        help=(
            "Apply BOG once every N denoising steps during inference. "
            "BOG is used only when step %% bog_interval == 0. "
            "Set to 1 to apply BOG at every step. "
            "Only takes effect when `--use_boosted_orthogonal_guidance=True`."
        ),
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default="在画面右下角添加三个带叶子的柿子，并保持其余内容不变。",
        help="Instruction describing how to edit the required reference image.",
    )
    parser.add_argument(
        "--negative_instruction",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Optional negative conditioning for image editing.",
    )
    parser.add_argument(
        "--empty_instruction",
        type=str,
        default=" ",
        help="Auxiliary empty instruction used by compatible edit guidance modes.",
    )
    parser.add_argument(
        "--input_image_paths",
        type=str,
        nargs="*",
        default=None,
        help="Required reference image path(s) for editing.",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="output.png",
        help="Path to save output image.",
    )
    parser.add_argument(
        "--num_images_per_instruction",
        type=int,
        default=1,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--use_far",
        type=to_bool,
        default=False,
        help="Enable external PRI-guided flow-aware regional editing.",
    )
    parser.add_argument(
        "--far_region_map_paths",
        nargs="*",
        default=None,
        help="External PRI region-map path for each editing sample.",
    )
    parser.add_argument(
        "--pri_predictor_checkpoint",
        type=str,
        default=None,
        help=(
            "Safe PRI predictor checkpoint directory; mutually exclusive with "
            "--far_region_map_paths."
        ),
    )
    parser.add_argument(
        "--far_invariant_strength",
        type=float,
        default=None,
        help=(
            "Independent I-region proxy strength in [0, 1]; omitted uses the "
            "candidate default 0.5. 1 reproduces legacy hard I; 0 disables only "
            "I control, not the R response."
        ),
    )
    parser.add_argument(
        "--far_effect_strength",
        type=float,
        default=None,
        help="Optional FAR response strength in [0, 1].",
    )
    parser.add_argument(
        "--far_effect_low_pass_kernel",
        type=int,
        default=None,
        help="Legacy single-scale kernel used only by fixed FAR transport.",
    )
    parser.add_argument(
        "--far_response_transport_mode",
        type=str,
        choices=["fixed", "analytic"],
        default=None,
        help="FAR response transport; omitted selects analytic multiscale.",
    )
    parser.add_argument(
        "--far_response_kernel_sizes",
        type=int,
        nargs="+",
        default=None,
        help="One fixed kernel or three analytic multiscale kernels.",
    )
    # VLM vision-token masking options
    parser.add_argument(
        "--mask_vision_tokens_feature",
        type=to_bool,
        default=False,
        help="Remove vision-token features from VLM outputs during inference.",
    )
    parser.add_argument(
        "--vision_token_ids",
        type=int,
        nargs="*",
        default=[],
        help="A list of VISION token ids whose VLM features will be removed and not be passed to DiT (e.g., '151652 151655 151653' for qwen3-vl series.). Only takes effect when `mask_vision_tokens_feature=True`.",
    )
    # VLM image preprocess controls
    parser.add_argument(
        "--max_vlm_input_pil_pixels",
        type=int,
        nargs="+",
        default=147456,
        help="Maximum pixels for input images passed to the VLM (i.e., the instruction encoder). Type: int or list of ints. Default: 147456.",
    )
    parser.add_argument(
        "--max_vlm_input_pil_side_length",
        type=int,
        default=768,
        help="Maximum side length of input images passed to the VLM (i.e., the instruction encoder). Type: int. Default: 768.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=1024,
        help="Maximum sequence length of the tokenized input instruction (which refers to the combined input of both text and image tokens) passed to the VLM (i.e., the instruction encoder). \
                Type: int. Default: 1024. Only takes effect when `--truncate_instruction_sequence=True`.",
    )

    parser.add_argument(
        "--truncate_instruction_sequence",
        type=to_bool,
        default=False,
        help="Whether to set `truncation=True` when calling the `apply_chat_template` function of the image processor. When True, it will \
                try to truncate the tokenized input instruction (which refers to the combined input of both text and image tokens) to the length specified by \
                `--max_sequence_length`. However, for some versions of transformers, if the number of image tokens in the input instruction is larger than \
                `--max_sequence_length` and you set `--truncate_instruction_sequence=True`, it may cause some errors.",
    )

    parser.add_argument(
        "--use_rewrite_text_instruction",
        type=to_bool,
        default=False,
        help="Whether to use the text instruction rewriter model to rewrite the text instruction.",
    )

    parser.add_argument(
        "--merge_original_and_rewritten_instructions",
        type=to_bool,
        default=True,
        help="Whether to merge the original user instructions with the rewritten instructions.",
    )

    parser.add_argument(
        "--custom_local_instruction_rewriter_model",
        type=str,
        default=None,
        help="""The name of custom local instruction rewriter model. If not set, the built-in VLM instruction encoder is used as the text instruction rewriter by default.
                `custom_local_instruction_rewriter_model` must be a Hugging Face-format model, preferably a Qwen-VL series model (verified in our tests).
                This parameter only takes effect when both `use_rewrite_text_instruction=True` and `use_dashscope_remote_rewriting=False`.
        """,
    )

    parser.add_argument(
        "--rewriter_max_new_tokens",
        type=int,
        default=512,
        help="The maximum number of tokens generated by the local text instruction rewriter model. Only takes effect when `use_rewrite_text_instruction=True` and `use_dashscope_remote_rewriting=False`.",
    )

    parser.add_argument(
        "--resize_rewriter_ref_images",
        type=to_bool,
        default=True,
        help="Whether to resize the reference images passed to the instruction rewriter model. Only takes effect when `use_rewrite_text_instruction=True`.",
    )

    parser.add_argument(
        "--rewriter_ref_images_max_pixels",
        type=int,
        nargs="+",
        default=2048 * 2048,
        help="Maximum number of pixels for each reference input image passed to the instruction rewritter model. Only takes effect when `use_rewrite_text_instruction=True` and `resize_rewriter_ref_images=True`.",
    )

    parser.add_argument(
        "--rewriter_ref_images_max_side_length",
        type=int,
        default=2560,
        help="Maximum side length of each reference input image passed to the instruction rewritter model. Only takes effect when `use_rewrite_text_instruction=True` and `resize_rewriter_ref_images=True`.",
    )

    parser.add_argument(
        "--do_sample_for_local_rewriter",
        type=to_bool,
        default=True,
        help="Whether to use `do_sample` option for the local instruction rewriter to generate different polished instructions every time. Only takes effect when `use_rewrite_text_instruction` is True.",
    )

    parser.add_argument(
        "--use_dashscope_remote_rewriting",
        type=to_bool,
        default=False,
        help="Whether to use the remote rewriting model through dashscope. Only takes effect when `use_rewrite_text_instruction` is True.",
    )

    parser.add_argument(
        "--dashscope_remote_rewriting_model",
        type=str,
        default="qwen-vl-max-latest",
        help="The name of remote rewriting model to use. Only takes effect when `use_rewrite_text_instruction` is True and `use_dashscope_remote_rewriting` is True.",
    )

    parser.add_argument(
        "--dashscope_base_http_api_url",
        type=str,
        default="https://dashscope.aliyuncs.com/api/v1",
        help="The base http api url for dashscope. Only takes effect when `use_rewrite_text_instruction` is True and `use_dashscope_remote_rewriting` is True.",
    )

    parser.add_argument(
        "--dashscope_api_key",
        type=str,
        default="sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
        help="The api key for dashscope. Only takes effect when `use_rewrite_text_instruction` is True and `use_dashscope_remote_rewriting` is True.",
    )

    parser.add_argument(
        "--rewriter_system_prompt_type",
        type=str,
        default="default",
        choices=["default", "custom"],
        help="Image-editing rewriter preset. Available options: 'default' and 'custom'. Only takes effect when `use_rewrite_text_instruction` is True.",
    )

    parser.add_argument(
        "--custom_rewriter_system_prompts_list",
        type=str,
        nargs="+",
        default=[
            "Rewrite the user's reference-image editing request into a precise edit instruction while preserving all content they did not ask to change."
        ],
        help="The list of custom system prompts for the text instruction rewriter model. Only takes effect when `use_rewrite_text_instruction` is True and `rewriter_system_prompt_type` is 'custom'.",
    )

    parser.add_argument(
        "--use_batch_inference",
        type=to_bool,
        default=False,
        help="Whether to use batch inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="The batch size for batch inference. If not set, it will be inferred from the input data.",
    )

    parser.add_argument(
        "--batch_data_config_path",
        type=str,
        default=None,
        help="The path to the test data config (yml file) for batch inference.",
    )

    parser.add_argument(
        "--save_rewritten_instruction",
        type=to_bool,
        default=False,
        help="Whether to save the rewritten instruction.",
    )

    parser.add_argument(
        "--save_rewritten_instruction_path",
        type=str,
        default=None,
        help="The path to save the rewritten instruction.",
    )

    parser.add_argument(
        "--enable_model_cpu_offload_flag",
        type=to_bool,
        default=False,
        help="Enable model CPU offload.",
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload_flag",
        type=to_bool,
        default=False,
        help="Enable sequential CPU offload.",
    )
    parser.add_argument(
        "--enable_group_offload_flag",
        type=to_bool,
        default=False,
        help="Enable group offload.",
    )

    parser.add_argument(
        "--enable_teacache",
        type=to_bool,
        default=False,
        help="Enable TeaCache on single-stream layers to speed up inference.",
    )
    parser.add_argument(
        "--teacache_rel_l1_thresh",
        type=float,
        default=0.05,
        help="Relative L1 threshold for teacache.",
    )
    parser.add_argument(
        "--enable_taylorseer",
        type=to_bool,
        default=False,
        help="Enable TaylorSeer caching on single-stream layers.",
    )

    parser.add_argument(
        "--enable_teacache_for_all_layers",
        type=to_bool,
        default=False,
        help=(
            "Enable TeaCache for both double-stream and single-stream layers. "
            "This also enables `--enable_teacache` implicitly unless TaylorSeer is selected."
        ),
    )
    parser.add_argument(
        "--enable_taylorseer_for_all_layers",
        type=to_bool,
        default=False,
        help=(
            "Enable TaylorSeer for both double-stream and single-stream layers. "
            "This also enables `--enable_taylorseer` implicitly and takes priority over TeaCache."
        ),
    )

    parser.add_argument(
        "--enable_cache_dit_caching",
        type=to_bool,
        default=False,
        help="Enable Cache-DiT caching on single-stream layers.",
    )
    parser.add_argument(
        "--enable_cache_dit_caching_for_all_layers",
        type=to_bool,
        default=False,
        help=(
            "Enable Cache-DiT caching for both double-stream and single-stream layers. "
            "This also enables `--enable_cache_dit_caching` implicitly and takes priority over TaylorSeer and TeaCache."
        ),
    )

    parser.add_argument(
        "--enable_inner_devices_manager",
        type=to_bool,
        default=False,
        help=(
            "Enable pipeline inner devices manager for local-rewriter memory saving. "
            "This is mainly useful when local rewriting is enabled and the rewriter shares the same CUDA device "
            "as the pipeline (or uses `auto` on CUDA), but it can slow repeated pipeline reuse because weights may "
            "move back and forth."
        ),
    )

    parser.add_argument(
        "--device",
        type=get_device_validator(),
        required=True,
        help="The device for the main pipeline to use. "
        "Choose from ['cpu', 'cuda', 'cuda:x']. ",
    )

    parser.add_argument(
        "--rewriter_device",
        # Call the factory function and pass the extra supported types
        type=get_device_validator(additional_types=["auto"]),
        default=None,
        help="The device for the instruction reasoner (the rewriter) to use. "
        "Choose from ['cpu', 'cuda', 'cuda:x', 'auto']. "
        "Only takes effect when `use_rewrite_text_instruction` is True.",
    )

    parser.add_argument(
        "--unload_rewriter_level",
        type=str,
        default="destroy",
        choices=["keep", "cpu", "destroy"],
        help=(
            "Choose the unload level for the instruction reasoner (the rewriter) after it finishes rewriting. "
            "Options:\n"
            "  - 'keep': Do nothing. The rewriter remains in its current device (RAM/VRAM).\n"
            "  - 'cpu': Offload the rewriter to the CPU to save VRAM. Note: This option is "
            "invalid and will not work if `--rewriter_device` is set to 'auto' on a multi-GPU "
            "setup, because a model sharded across multiple GPUs cannot be directly offloaded to CPU.\n"
            "  - 'destroy': Completely destroy the rewriter. It will be removed from RAM and VRAM, "
            "set to None, and will no longer be usable.\n"
            "Only takes effect when `use_rewrite_text_instruction` is True.\n\n"
            "IMPORTANT NOTES:\n"
            "1. If `--custom_local_instruction_rewriter_model` is not set, the instruction reasoner (rewriter) defaults "
            "to the instruction encoder (the MLLM). In this case, NO unload operations will be performed on the rewriter, "
            "regardless of this setting.\n"
            "2. If `--use_dashscope_remote_rewriting` is also True, rewriting is performed remotely, so there is no "
            "local rewriter to unload."
        ),
    )

    parser.add_argument(
        "--enable_torch_compile",
        type=to_bool,
        default=False,
        help=(
            "Enable Torch Compile. Warning: this may occasionally produce all-black outputs on some GPUs/models; "
            "disable it first if you see black images."
        ),
    )

    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="The mode for Torch Compile. Only takes effect when `enable_torch_compile` is True.",
    )

    parser.add_argument(
        "--use_fp8_weights",
        type=to_bool,
        default=False,
        help="Whether to use fp8 quantized model weights for mllm and transformer. The fp8 weights of mllm are from Qwen's official repository.",
    )

    args = parser.parse_args(argv)
    args._explicit_options = tuple(sorted({
        parser._option_string_actions[token.split("=", 1)[0]].dest
        for token in argv
        if token.startswith("--") and token.split("=", 1)[0] in parser._option_string_actions
    }))
    if args.method == "source-conditioned":
        for name, value in {
            "height": None, "width": None, "instruction": None,
            "negative_instruction": "",
            "max_vlm_input_pil_pixels": 384 * 384,
            "max_vlm_input_pil_side_length": 768,
        }.items():
            if name not in args._explicit_options:
                setattr(args, name, value)
    return args


def check_pattern(s, pattern=r"\d+b_a\d+b"):
    if re.search(pattern, s):
        return True
    else:
        return False


def apply_cache_dit_caching(
    pipeline: BooguImagePipeline | BooguImagePromptTuningPipeline,
    *,
    all_layers: bool = False,
):
    """
    Apply Cache-DiT caching to the pipeline.
    Args:
        pipeline: The pipeline to apply Cache-DiT caching to.
        all_layers: Whether to apply Cache-DiT caching to all layers.
    Returns:
        The pipeline with Cache-DiT caching applied.
    """
    if all_layers:
        blocks = [
            pipeline.transformer.double_stream_layers,
            pipeline.transformer.single_stream_layers,
        ]
        forward_pattern = [
            cache_dit.ForwardPattern.Pattern_0,
            cache_dit.ForwardPattern.Pattern_3,
        ]
        params_modifiers = [
            cache_dit.ParamsModifier(
                cache_config=cache_dit.DBCacheConfig().reset(Fn_compute_blocks=2)
            ),
            cache_dit.ParamsModifier(
                cache_config=cache_dit.DBCacheConfig().reset(Fn_compute_blocks=4)
            ),
        ]
    else:
        blocks = [pipeline.transformer.single_stream_layers]
        forward_pattern = [cache_dit.ForwardPattern.Pattern_3]
        params_modifiers = [
            cache_dit.ParamsModifier(
                cache_config=cache_dit.DBCacheConfig().reset(Fn_compute_blocks=4)
            )
        ]

    adapter = cache_dit.BlockAdapter(
        pipe=pipeline,
        transformer=pipeline.transformer,
        blocks=blocks,
        forward_pattern=forward_pattern,
        params_modifiers=params_modifiers,
        check_forward_pattern=False,
        has_separate_cfg=True,
    )
    cache_config = cache_dit.DBCacheConfig(
        max_warmup_steps=4,
        max_continuous_cached_steps=6,
        residual_diff_threshold=0.12,
        enable_separate_cfg=True,
    )
    calibrator_config = cache_dit.TaylorSeerCalibratorConfig(taylorseer_order=1)

    cache_dit.enable_cache(
        adapter, cache_config=cache_config, calibrator_config=calibrator_config
    )

    return adapter.pipe


def load_llm_or_vlm(
    pretrained_model_name_or_path: str,
    weight_dtype,
    RETURN_GENERATION_MODEL: bool = True,
    device=None,
):
    if "vl" in pretrained_model_name_or_path.lower():
        if "intern" in pretrained_model_name_or_path.lower():
            print("[Custom Model Loader]: Loading InternVLForConditionalGeneration.")
            from transformers import InternVLForConditionalGeneration as CustomModel

        elif "qwen3" in pretrained_model_name_or_path.lower() and check_pattern(
            pretrained_model_name_or_path.lower()
        ):
            print("[Custom Model Loader]: Loading Qwen3VLMoeForConditionalGeneration.")
            from transformers import Qwen3VLMoeForConditionalGeneration as CustomModel

        elif "qwen3" in pretrained_model_name_or_path.lower():
            print("[Custom Model Loader]: Loading Qwen3VLForConditionalGeneration.")
            from transformers import Qwen3VLForConditionalGeneration as CustomModel

        elif (
            "qwen2" in pretrained_model_name_or_path.lower()
            and "5" in pretrained_model_name_or_path.lower()
        ):
            print("[Custom Model Loader]: Loading Qwen2_5_VLForConditionalGeneration.")
            from transformers import Qwen2_5_VLForConditionalGeneration as CustomModel

        else:
            warnings.warn(
                "[Custom Model Loader Warning]: No explicit VLM class matched this model path. "
                "`AutoModelForImageTextToText` will be used to infer the proper generation model class from config. "
                f"pretrained_model_name_or_path={pretrained_model_name_or_path!r}.",
                UserWarning,
            )
            print("[Custom Model Loader]: Loading AutoModelForImageTextToText.")
            from transformers import AutoModelForImageTextToText as CustomModel

    else:
        if "qwen3" in pretrained_model_name_or_path.lower():
            print("[Custom Model Loader]: Loading Qwen3ForCausalLM.")
            from transformers import Qwen3ForCausalLM as CustomModel

        elif "qwen2" in pretrained_model_name_or_path.lower():
            print("[Custom Model Loader]: Loading Qwen2ForCausalLM.")
            from transformers import Qwen2ForCausalLM as CustomModel

        elif (
            "gem" in pretrained_model_name_or_path.lower()
            or "dream " in pretrained_model_name_or_path.lower()
        ):
            print("[Custom Model Loader]: Loading Gemma4ForConditionalGeneration.")
            from transformers import Gemma4ForConditionalGeneration as CustomModel

        else:
            warnings.warn(
                "[Custom Model Loader Warning]: No explicit LLM class matched this model path. "
                "`AutoModelForCausalLM` will be used to infer the proper generation model class from config. "
                f"pretrained_model_name_or_path={pretrained_model_name_or_path!r}.",
                UserWarning,
            )
            print("[Custom Model Loader]: Loading AutoModelForCausalLM.")
            from transformers import AutoModelForCausalLM as CustomModel

    try:
        cus_mod = CustomModel.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=weight_dtype,
            device_map=device,
        )
    except Exception as exc:
        raise NotImplementedError(
            "[Custom Model Loader Error]: Failed to load the requested model with the selected Transformers class. "
            "This model may not be compatible with the current generic loading logic. Please manually extend "
            "`load_llm_or_vlm(...)` with the correct model class and loading arguments. "
            f"pretrained_model_name_or_path={pretrained_model_name_or_path!r}, "
            f"CustomModel={getattr(CustomModel, '__name__', str(CustomModel))!r}, "
            f"device={device!r}."
        ) from exc

    if not RETURN_GENERATION_MODEL:
        cus_mod = cus_mod.model

    return cus_mod


def _disable_deepgemm_for_fp8_vlm() -> None:
    # For transformers >= 5.11.0
    os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "1"

    try:
        import transformers.integrations.finegrained_fp8 as fg_fp8
    except ImportError:
        return

    def _raise_import_error(*args, **kwargs):
        raise ImportError("DeepGEMM disabled; forcing Triton finegrained-fp8 fallback.")

    if hasattr(fg_fp8, "deepgemm_fp8_fp4_linear"):
        # For 5.10.1 <= transformers < 5.11.0
        fg_fp8.deepgemm_fp8_fp4_linear = _raise_import_error
    elif hasattr(fg_fp8, "_load_deepgemm_kernel"):
        # For 5.5.0 <=transoformers < 5.10.1
        fg_fp8._load_deepgemm_kernel = _raise_import_error


def load_pipeline(
    args: argparse.Namespace, weight_dtype: torch.dtype
) -> BooguImagePipeline:
    """
    # Keep `device_map=None`; no need to set it to `args.device`.
    # The same applies below (Note: `rewriter` is an external module, so its device management differs).
    """
    require_edit_checkpoint(args.pretrained_pipeline_name_or_path)

    if args.pretrained_pipeline_name_or_path is not None:
        # Keep `device_map=None`; no need to set it to `args.device`.
        # The same applies below (Note: `rewriter` is an external module, so its device management differs).
        if args.use_prompt_tuning:
            PipelineClass = BooguImagePromptTuningPipeline
        else:
            PipelineClass = BooguImagePipeline
        if args.use_fp8_weights:
            assert "fp8" in args.pretrained_pipeline_name_or_path.lower(), (
                f"Invalid model path: {args.pretrained_pipeline_name_or_path}. The model must be a fp8 quantized model."
            )

            # Use Triton finegrained-fp8 kernel for better compatibility.
            _disable_deepgemm_for_fp8_vlm()

            # Need to load the fp8 transformer weights separately, because they are not in the safetensors format.
            print(
                "[Pipeline Loader]: Using fp8 quantized model weights for transformer."
            )
            fp8_transformer_path = os.path.join(
                args.pretrained_pipeline_name_or_path, "transformer"
            )
            fp8_transformer = BooguImageTransformer2DModel.from_pretrained(
                fp8_transformer_path,
                torch_dtype=weight_dtype,
                use_safetensors=False,
            )
            pipeline = PipelineClass.from_pretrained(
                args.pretrained_pipeline_name_or_path,
                torch_dtype=weight_dtype,
                trust_remote_code=True,
                transformer=fp8_transformer,
            )
        else:
            pipeline = PipelineClass.from_pretrained(
                args.pretrained_pipeline_name_or_path,
                torch_dtype=weight_dtype,
                trust_remote_code=True,
            )
    else:
        raise ValueError(
            f"args.pretrained_pipeline_name_or_path=({args.pretrained_pipeline_name_or_path}) must be properly provided."
        )

    if args.custom_pretrained_instruction_encoder_model_name_or_path:
        ## !! No need to set `device=args.device`, just keep `device=None`
        mllm = load_llm_or_vlm(
            args.custom_pretrained_instruction_encoder_model_name_or_path,
            weight_dtype,
            RETURN_GENERATION_MODEL=True,
        )
        pipeline.set_mllm(mllm)
        processor = AutoProcessor.from_pretrained(
            args.custom_pretrained_instruction_encoder_model_name_or_path
        )
        pipeline.set_processor(processor)

    if (
        args.use_prompt_tuning
        and args.custom_prompt_tuning_model_path is not None
        and args.custom_prompt_tuning_model_path.strip()
    ):
        print(
            f"[Pipeline Loader]: Custom Prompt Tuning Embedding weights loaded from {args.custom_prompt_tuning_model_path}"
        )
        _prompt_embedding = PromptEmbedding.from_pretrained(
            args.custom_prompt_tuning_model_path,
            weight_dtype=weight_dtype,
        )
        pipeline.set_prompt_embedding(_prompt_embedding)

    if args.custom_local_instruction_rewriter_model:
        print(
            f"[Pipeline Loader]: Using `custom_local_instruction_rewriter_model` as the instruction reasoner (the rewriter) model: {args.custom_local_instruction_rewriter_model}"
        )

        ## Use `args.rewriter_device`, not `args.device`
        _instruct_rewriter_model = load_llm_or_vlm(
            args.custom_local_instruction_rewriter_model,
            weight_dtype,
            RETURN_GENERATION_MODEL=True,
            device=args.rewriter_device,
        )
        pipeline.set_custom_local_instruction_rewriter_model(_instruct_rewriter_model)
        del _instruct_rewriter_model

        _instruction_rewriter_processor = AutoProcessor.from_pretrained(
            args.custom_local_instruction_rewriter_model
        )
        pipeline.set_instruction_rewriter_processor(_instruction_rewriter_processor)

    if args.custom_diffusion_transformer_path:
        print(
            f"[Pipeline Loader]: Custom Diffusion Transformer model weights loaded from {args.custom_diffusion_transformer_path}"
        )
        # Keep `device_map=None`; no need to set it to `args.device`.
        # The same applies below (Note: `rewriter` is an external module, so its device management differs).
        _transformer = BooguImageTransformer2DModel.from_pretrained(
            args.custom_diffusion_transformer_path,
            torch_dtype=weight_dtype,
        )
        pipeline.set_transformer(_transformer)

    if args.custom_transformer_lora_path:
        pipeline.load_lora_weights(args.custom_transformer_lora_path)
        print(
            f"[Pipeline Loader]: LoRA weights for Diffusion Transformer model loaded from {args.custom_transformer_lora_path}"
        )

    if (
        args.use_prompt_tuning
        and args.custom_prompt_tuning_model_lora_weights_path is not None
        and args.custom_prompt_tuning_model_lora_weights_path.strip()
    ):
        pipeline.load_lora_prompt_embedding_weights(
            args.custom_prompt_tuning_model_lora_weights_path
        )
        print(
            f"[Pipeline Loader]: LoRA weights for Prompt Tuning Model loaded from {args.custom_prompt_tuning_model_lora_weights_path}"
        )

    enable_taylorseer = args.enable_taylorseer or args.enable_taylorseer_for_all_layers
    enable_teacache = args.enable_teacache or args.enable_teacache_for_all_layers
    enable_cache_dit_caching = (
        args.enable_cache_dit_caching or args.enable_cache_dit_caching_for_all_layers
    )

    caching_strategies_count = sum(
        [enable_taylorseer, enable_teacache, enable_cache_dit_caching]
    )
    if caching_strategies_count > 1:
        warnings.warn(
            "[Cache Strategy Warning]: Caching strategies are mutually exclusive. "
            "Would enable in the following priority: Cache-DiT > TaylorSeer > TeaCache.",
            UserWarning,
        )
        if enable_cache_dit_caching:
            enable_taylorseer = False
            args.enable_taylorseer = False
            args.enable_taylorseer_for_all_layers = False
            enable_teacache = False
            args.enable_teacache = False
            args.enable_teacache_for_all_layers = False
        elif enable_taylorseer:
            enable_teacache = False
            args.enable_teacache = False
            args.enable_teacache_for_all_layers = False

    # The all-layer flags extend the selected cache method to double-stream layers.
    pipeline.transformer.enable_teacache_for_all_layers = bool(
        args.enable_teacache_for_all_layers and enable_teacache
    )
    pipeline.transformer.enable_taylorseer_for_all_layers = bool(
        args.enable_taylorseer_for_all_layers and enable_taylorseer
    )

    if enable_taylorseer:
        pipeline.enable_taylorseer = True
    elif enable_teacache:
        pipeline.transformer.enable_teacache = True
        pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    if enable_cache_dit_caching:
        pipeline = apply_cache_dit_caching(
            pipeline, all_layers=args.enable_cache_dit_caching_for_all_layers
        )
        print("[Pipeline Loader]: Cache-DiT caching applied to the pipeline.")

    # Configure VLM vision-token masking flags for inference to align with training
    try:
        pipeline.MASK_VISION_TOKENS_FEATURE = bool(args.mask_vision_tokens_feature)
        if args.vision_token_ids:
            pipeline.VISION_TOKEN_IDs = args.vision_token_ids
        else:
            pipeline.VISION_TOKEN_IDs = []
    except Exception as e:
        print(f"Warning: Failed to parse vision token settings: {e}. Using defaults.")

    if args.scheduler == "dpmsolver++":
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.set_scheduler(scheduler)

    if not validate_device_and_offload_strategy_compatibility(
        args.device,
        args.enable_sequential_cpu_offload_flag,
        args.enable_model_cpu_offload_flag,
        args.enable_group_offload_flag,
    ):
        raise ValueError(
            "[Device and Offload Strategy Compatibility Error]: The device and offload strategy are not compatible. "
            "Please make sure all three offload flags are valid booleans, at most one offload strategy is enabled, "
            "and `device` is a CUDA execution device when any CPU offload strategy is enabled. "
            f"device={args.device}, "
            f"enable_sequential_cpu_offload_flag={args.enable_sequential_cpu_offload_flag}, "
            f"enable_model_cpu_offload_flag={args.enable_model_cpu_offload_flag}, "
            f"enable_group_offload_flag={args.enable_group_offload_flag}."
        )

    if args.enable_inner_devices_manager:
        if not all(
            [
                args.use_rewrite_text_instruction,
                not args.use_dashscope_remote_rewriting,
                args.custom_local_instruction_rewriter_model is not None
                and args.custom_local_instruction_rewriter_model.strip(),
                args.rewriter_device is not None and args.rewriter_device != "cpu",
                (args.device == args.rewriter_device)
                or (args.rewriter_device == "auto" and args.device.startswith("cuda")),
                args.unload_rewriter_level != "keep",
            ]
        ):
            warnings.warn(
                "[Device Moving Warning]: `enable_inner_devices_manager` is meant for the local-rewriter memory-saving path "
                "and is usually useful only when all the following conditions are met: "
                "1. use_rewrite_text_instruction == True "
                "2. use_dashscope_remote_rewriting == False "
                "3. custom_local_instruction_rewriter_model is set and successfully loaded "
                "4. rewriter_device is not cpu "
                "5. device and rewriter_device are the same, or rewriter_device is auto and device is cuda or cuda:x "
                "6. unload_rewriter_level != 'keep'. "
                "Outside this case, it is usually unnecessary and can slow repeated reuse because weights are moved "
                "between CPU and GPU more often.",
                UserWarning,
            )
        warnings.warn(
            "[Device Moving Warning]: When `enable_inner_devices_manager` is True, do not run `pipeline.to(device)` manually. "
            "This switch is only meant to coordinate the pipeline with the local rewriter's device moves; mixing in "
            "manual pipeline moves can fight that behavior and trigger extra weight transfers.",
            UserWarning,
        )
    elif not any(
        [
            args.enable_sequential_cpu_offload_flag,
            args.enable_model_cpu_offload_flag,
            args.enable_group_offload_flag,
        ]
    ):
        pipeline.to(args.device)

    if args.enable_sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload_flag = True
        pipeline.enable_sequential_cpu_offload(device=args.device)
    elif args.enable_model_cpu_offload_flag:
        pipeline.enable_model_cpu_offload_flag = True
        pipeline.enable_model_cpu_offload(device=args.device)
    elif args.enable_group_offload_flag:
        pipeline.enable_group_offload_flag = True
        apply_group_offloading(
            pipeline.transformer,
            onload_device=args.device,
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
        )
        apply_group_offloading(
            pipeline.mllm,
            onload_device=args.device,
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
        )
        apply_group_offloading(
            pipeline.vae,
            onload_device=args.device,
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
        )
        if args.use_prompt_tuning:
            apply_group_offloading(
                pipeline.prompt_embedding,
                onload_device=args.device,
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=True,
            )

    if args.enable_torch_compile:
        warnings.warn(
            "[Torch Compile Warning]: Torch Compile may occasionally produce all-black images on some GPUs/models. "
            "If this happens, disable `--enable_torch_compile` first.",
            UserWarning,
        )
        if args.enable_sequential_cpu_offload_flag or args.enable_group_offload_flag:
            warnings.warn(
                "[Torch Compile Warning]: Torch Compile is not compatible with sequential CPU offload or group offload. "
                "Disabling Torch Compile.",
                UserWarning,
            )
            args.enable_torch_compile = False
        else:
            pipeline.transformer.compile_repeated_blocks(
                fullgraph=True, dynamic=True, mode=args.torch_compile_mode
            )

    return pipeline


def preprocess(input_image_paths: Optional[List[str]] = None) -> List[Image.Image]:
    """Load independent RGB copies of the reference images."""
    if isinstance(input_image_paths, str):
        input_image_paths = [input_image_paths]
    elif isinstance(input_image_paths, tuple):
        input_image_paths = list(input_image_paths)
    if not input_image_paths:
        return []

    image_paths = list(input_image_paths)
    if len(image_paths) == 1 and os.path.isdir(image_paths[0]):
        directory = image_paths[0]
        image_extensions = {
            ".bmp",
            ".gif",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        }
        image_paths = sorted(
            (
                os.path.join(directory, filename)
                for filename in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, filename))
                and os.path.splitext(filename)[1].lower() in image_extensions
            ),
            key=lambda path: os.path.basename(path).casefold(),
        )

    input_images = []
    for path in image_paths:
        with Image.open(path) as source_image:
            transposed_image = ImageOps.exif_transpose(source_image)
            input_images.append(transposed_image.convert("RGB"))
            if transposed_image is not source_image:
                transposed_image.close()
    return input_images


def _prepare_edit_inputs(
    args: argparse.Namespace,
) -> Tuple[List[str], List[List[Image.Image]], List[List[str]]]:
    """Load and validate one non-empty reference-image group per instruction."""
    if args.use_batch_inference:
        if not args.batch_data_config_path:
            raise EditOnlyInputError(
                "Image editing batch inference requires --batch_data_config_path."
            )
        batch_data_config = OmegaConf.load(args.batch_data_config_path)
        data = getattr(batch_data_config, "data", None)
        args._resolved_batch_data = data
        raw_instructions = getattr(data, "instructions", None)
        sequence_types = (list, tuple, ListConfig)
        if not isinstance(raw_instructions, sequence_types) or not raw_instructions:
            raise EditOnlyInputError(
                "Image editing batch instructions must be a non-empty list of strings."
            )
        instructions = list(raw_instructions)
        if any(
            not isinstance(instruction, str) or not instruction.strip()
            for instruction in instructions
        ):
            raise EditOnlyInputError(
                "Image editing batch instructions must be non-empty strings."
            )

        raw_input_image_paths = getattr(data, "input_image_paths", None)
        if not isinstance(raw_input_image_paths, sequence_types):
            raise EditOnlyInputError(
                "Image editing batch input_image_paths must be a list."
            )
        input_image_paths = []
        for paths in raw_input_image_paths:
            if isinstance(paths, str):
                normalized_paths = [paths]
            elif isinstance(paths, sequence_types):
                normalized_paths = list(paths)
            else:
                raise EditOnlyInputError(
                    "Each image editing reference group must be a string or list of strings."
                )
            if any(
                not isinstance(path, str) or not path.strip()
                for path in normalized_paths
            ):
                raise EditOnlyInputError(
                    "Each image editing reference path must be a non-empty string."
                )
            input_image_paths.append(normalized_paths)
    else:
        args._resolved_batch_data = None
        if not isinstance(args.instruction, str) or not args.instruction.strip():
            raise EditOnlyInputError(
                "Image editing requires a non-empty instruction."
            )
        instructions = [args.instruction]
        sample_paths = args.input_image_paths
        if isinstance(sample_paths, str):
            sample_paths = [sample_paths]
        elif isinstance(sample_paths, tuple):
            sample_paths = list(sample_paths)
        input_image_paths = [sample_paths] if sample_paths else None

    require_reference_batch(
        input_image_paths, expected_batch_size=len(instructions)
    )
    input_images = [preprocess(paths) for paths in input_image_paths]
    require_reference_batch(input_images, expected_batch_size=len(instructions))
    return instructions, input_images, input_image_paths


def _prepare_far_inputs(
    args: argparse.Namespace,
    instructions,
    input_images,
    input_image_paths,
):
    """Validate one learned or external PRI source before loading main weights."""
    auxiliary_names = (
        "far_region_map_paths",
        "pri_predictor_checkpoint",
        "far_invariant_strength",
        "far_effect_strength",
        "far_effect_low_pass_kernel",
        "far_response_transport_mode",
        "far_response_kernel_sizes",
    )
    use_far = getattr(args, "use_far", False)
    if not isinstance(use_far, bool):
        raise TypeError("use_far must be a bool")

    supplied_auxiliary_names = [
        name for name in auxiliary_names if getattr(args, name, None) is not None
    ]
    if not use_far:
        if supplied_auxiliary_names:
            supplied = ", ".join(supplied_auxiliary_names)
            raise ValueError(
                f"FAR is disabled; {supplied} requires --use_far true"
            )
        return None, None, None

    if not isinstance(instructions, (list, tuple)) or not instructions:
        raise ValueError("FAR CLI editing requires at least one instruction")
    if any(
        not isinstance(instruction, str) or not instruction.strip()
        for instruction in instructions
    ):
        raise ValueError("FAR CLI editing requires non-empty instructions")
    sample_count = len(instructions)
    is_batch = getattr(args, "use_batch_inference", False)
    if not is_batch and sample_count != 1:
        raise ValueError("FAR CLI editing requires exactly one instruction")

    for name, reference_groups in (
        ("input_images", input_images),
        ("input_image_paths", input_image_paths),
    ):
        if (
            not isinstance(reference_groups, (list, tuple))
            or len(reference_groups) != sample_count
            or any(
                not isinstance(group, (list, tuple)) or len(group) != 1
                for group in reference_groups
            )
        ):
            raise ValueError(
                f"FAR CLI editing requires exactly one reference per sample in {name}"
            )

    cli_region_map_paths = getattr(args, "far_region_map_paths", None)
    predictor_checkpoint = getattr(args, "pri_predictor_checkpoint", None)
    if predictor_checkpoint is not None and (
        not isinstance(predictor_checkpoint, str) or not predictor_checkpoint.strip()
    ):
        raise ValueError("pri_predictor_checkpoint must be a non-empty path")
    if is_batch:
        data = getattr(args, "_resolved_batch_data", None)
        if data is None:
            config_path = getattr(args, "batch_data_config_path", None)
            if not config_path:
                raise ValueError(
                    "Batch FAR editing requires a batch data config"
                )
            batch_data_config = OmegaConf.load(config_path)
            data = getattr(batch_data_config, "data", None)
        config_region_map_paths = getattr(data, "far_region_map_paths", None)
        if predictor_checkpoint is not None and (
            cli_region_map_paths is not None
            or config_region_map_paths is not None
        ):
            raise ValueError(
                "FAR CLI editing requires exactly one PRI source: external maps or "
                "a predictor checkpoint"
            )
        if cli_region_map_paths is not None and config_region_map_paths is not None:
            raise ValueError(
                "FAR map paths cannot be supplied by both CLI and batch config"
            )
        if cli_region_map_paths is not None and predictor_checkpoint is None:
            raise ValueError(
                "Batch FAR editing requires data.far_region_map_paths in the batch config"
            )
        region_map_paths = config_region_map_paths
    else:
        region_map_paths = cli_region_map_paths

    if (region_map_paths is None) == (predictor_checkpoint is None):
        raise ValueError(
            "FAR CLI editing requires exactly one PRI source: external maps or "
            "a predictor checkpoint"
        )
    if region_map_paths is not None:
        sequence_types = (list, tuple, ListConfig)
        if (
            not isinstance(region_map_paths, sequence_types)
            or len(region_map_paths) != sample_count
            or any(
                not isinstance(path, str) or not path.strip()
                for path in region_map_paths
            )
        ):
            raise ValueError(
                "FAR CLI editing requires one PRI map path per sample"
            )

    if getattr(args, "negative_instruction", None) != "":
        raise ValueError(
            "FAR CLI editing requires explicit --negative_instruction \"\""
        )
    if getattr(args, "scheduler", None) != "euler":
        raise ValueError("FAR CLI editing requires scheduler='euler'")

    numeric_requirements = (
        ("text_guidance_scale", lambda value: value > 1, "greater than 1"),
        ("image_guidance_scale", lambda value: value == 1, "equal to 1"),
        (
            "empty_instruction_guidance_scale",
            lambda value: value == 0,
            "equal to 0",
        ),
        ("cfg_range_start", lambda value: value == 0, "equal to 0"),
        ("cfg_range_end", lambda value: value == 1, "equal to 1"),
    )
    for name, predicate, requirement in numeric_requirements:
        value = getattr(args, name, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            or not predicate(value)
        ):
            raise ValueError(f"FAR CLI {name} must be {requirement}")

    num_images = getattr(args, "num_images_per_instruction", None)
    if (
        isinstance(num_images, bool)
        or not isinstance(num_images, Integral)
        or num_images != 1
    ):
        raise ValueError("FAR CLI num_images_per_instruction must equal 1")

    incompatible_boolean_names = (
        "use_rewrite_text_instruction",
        "use_dashscope_remote_rewriting",
        "save_rewritten_instruction",
        "enable_inner_devices_manager",
        "use_boosted_orthogonal_guidance",
        "use_prompt_tuning",
        "use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide",
        "use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide",
        "enable_teacache",
        "enable_taylorseer",
        "enable_teacache_for_all_layers",
        "enable_taylorseer_for_all_layers",
        "enable_cache_dit_caching",
        "enable_cache_dit_caching_for_all_layers",
    )
    for name in incompatible_boolean_names:
        if getattr(args, name, False):
            raise ValueError(f"FAR CLI editing requires {name}=False")

    incompatible_path_names = (
        "custom_local_instruction_rewriter_model",
        "custom_prompt_tuning_model_path",
        "custom_prompt_tuning_model_lora_weights_path",
        "custom_transformer_lora_path",
    )
    for name in incompatible_path_names:
        if getattr(args, name, None) not in (None, ""):
            raise ValueError(f"FAR CLI editing requires {name} to be unset")

    invariant_strength = getattr(args, "far_invariant_strength", None)
    far_config = (
        FARConfig(enabled=True)
        if invariant_strength is None
        else FARConfig(enabled=True, invariant_strength=invariant_strength)
    )
    default_transport = ResponseTransportConfig()
    response_strength = getattr(args, "far_effect_strength", None)
    if response_strength is None:
        response_strength = default_transport.response_strength

    requested_mode = getattr(args, "far_response_transport_mode", None)
    mode = "analytic" if requested_mode is None else requested_mode
    requested_kernels = getattr(args, "far_response_kernel_sizes", None)
    legacy_kernel = getattr(args, "far_effect_low_pass_kernel", None)

    if mode == "fixed":
        if legacy_kernel is not None and requested_kernels is not None:
            raise ValueError(
                "fixed FAR transport accepts one kernel source, not both "
                "far_effect_low_pass_kernel and far_response_kernel_sizes"
            )
        if requested_kernels is not None:
            if len(requested_kernels) != 1:
                raise ValueError("fixed FAR transport requires exactly one kernel")
            kernel_sizes = tuple(requested_kernels)
        else:
            kernel_sizes = (
                far_config.effect_low_pass_kernel
                if legacy_kernel is None
                else legacy_kernel,
            )
        response_transport_config = ResponseTransportConfig(
            mode="fixed",
            kernel_sizes=kernel_sizes,
            schedule_centers=(0.5,),
            response_strength=response_strength,
        )
    elif mode == "analytic":
        if legacy_kernel is not None:
            raise ValueError(
                "far_effect_low_pass_kernel is only valid for fixed FAR transport"
            )
        if requested_kernels is None:
            kernel_sizes = default_transport.kernel_sizes
        else:
            if len(requested_kernels) != 3:
                raise ValueError(
                    "analytic FAR transport requires exactly three kernels"
                )
            kernel_sizes = tuple(requested_kernels)
        response_transport_config = ResponseTransportConfig(
            mode="analytic_multiscale",
            kernel_sizes=kernel_sizes,
            schedule_centers=default_transport.schedule_centers,
            response_strength=response_strength,
        )
    else:
        raise ValueError("far_response_transport_mode must be fixed or analytic")

    regions = None
    if region_map_paths is not None:
        regions = load_pri_region_map_list(
            list(region_map_paths),
            dtype=torch.float32,
        )
        if not isinstance(regions, list) or len(regions) != sample_count:
            raise ValueError("FAR PRI loader must return one region map per sample")
    return far_config, response_transport_config, regions


def _create_inference_generators(
    args: argparse.Namespace,
) -> tuple[torch.Generator, torch.Generator]:
    """Create independent, reproducible noise and reference RNG streams."""

    maximum_seed = 2**63 - 1
    noise_seed = getattr(args, "seed", None)
    reference_seed = getattr(args, "reference_seed", None)
    for name, value in (("seed", noise_seed), ("reference_seed", reference_seed)):
        if value is None and name == "reference_seed":
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if not 0 <= value <= maximum_seed:
            raise ValueError(f"{name} must lie in [0, {maximum_seed}]")
    if reference_seed is None:
        reference_seed = 0 if noise_seed == maximum_seed else noise_seed + 1
    if reference_seed == noise_seed:
        raise ValueError("seed and reference_seed must be independent distinct seeds")

    noise_generator = torch.Generator(device=args.device).manual_seed(noise_seed)
    reference_generator = torch.Generator(device=args.device).manual_seed(reference_seed)
    return noise_generator, reference_generator


def _load_pri_predictor(args: argparse.Namespace):
    """Load one frozen FP32 PRI predictor onto the inference device."""

    checkpoint = getattr(args, "pri_predictor_checkpoint", None)
    if checkpoint is None:
        return None
    predictor, _ = load_predictor_from_checkpoint(checkpoint)
    predictor.requires_grad_(False)
    predictor.eval()
    return predictor.to(device=args.device, dtype=torch.float32)


def run(
    args: argparse.Namespace,
    pipeline: BooguImagePipeline,
    instruction: str,
    negative_instruction: str,
    input_images: Optional[List[List[Image.Image]]] = None,
    input_image_paths: Optional[List[List[str]]] = None,
    *,
    generator: torch.Generator | None = None,
    reference_generator: torch.Generator | None = None,
    far_config: FARConfig | None = None,
    response_transport_config: ResponseTransportConfig | None = None,
    far_region_probabilities=None,
    pri_region_predictor=None,
    attribute_request=None,
    source_anchor_request=None,
) -> Image.Image:
    """Run the image-editing pipeline with the given parameters."""
    if (generator is None) != (reference_generator is None):
        raise ValueError(
            "generator and reference_generator must be supplied together"
        )
    if generator is None:
        generator, reference_generator = _create_inference_generators(args)
    if far_config is not None and (
        (far_region_probabilities is None) == (pri_region_predictor is None)
    ):
        raise ValueError(
            "enabled FAR requires exactly one PRI source: external maps or a predictor"
        )

    pipeline_kwargs = {
        "instruction": instruction,
        "input_images": input_images,
        "width": args.width,
        "height": args.height,
        "max_input_image_pixels": args.max_input_image_pixels,
        "max_input_image_side_length": args.max_input_image_side_length,
        "num_inference_steps": args.num_inference_steps,
        "max_vlm_input_pil_pixels": args.max_vlm_input_pil_pixels,
        "max_vlm_input_pil_side_length": args.max_vlm_input_pil_side_length,
        "max_sequence_length": args.max_sequence_length,
        "truncate_instruction_sequence": args.truncate_instruction_sequence,
        "text_guidance_scale": args.text_guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "empty_instruction_guidance_scale": args.empty_instruction_guidance_scale,
        "cfg_range": (args.cfg_range_start, args.cfg_range_end),
        "negative_instruction": negative_instruction,
        "num_images_per_instruction": args.num_images_per_instruction,
        "generator": generator,
        "reference_generator": reference_generator,
        "output_type": "pil",
        "use_rewrite_text_instruction": args.use_rewrite_text_instruction,
        "rewriter_max_new_tokens": args.rewriter_max_new_tokens,
        "resize_rewriter_ref_images": args.resize_rewriter_ref_images,
        "rewriter_ref_images_max_pixels": args.rewriter_ref_images_max_pixels,
        "rewriter_ref_images_max_side_length": args.rewriter_ref_images_max_side_length,
        "rewriter_system_prompt_type": args.rewriter_system_prompt_type,
        "custom_rewriter_system_prompts_list": args.custom_rewriter_system_prompts_list,
        "merge_original_and_rewritten_instructions": args.merge_original_and_rewritten_instructions,
        "do_sample_for_local_rewriter": args.do_sample_for_local_rewriter,
        "empty_instruction": args.empty_instruction,
        "use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide": args.use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide,
        "use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide": args.use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide,
        "save_rewritten_instruction": args.save_rewritten_instruction,
        "save_rewritten_instruction_path": args.save_rewritten_instruction_path,
        "input_image_paths": input_image_paths,
        "use_dashscope_remote_rewriting": args.use_dashscope_remote_rewriting,
        "dashscope_remote_rewriting_model": args.dashscope_remote_rewriting_model,
        "dashscope_base_http_api_url": args.dashscope_base_http_api_url,
        "dashscope_api_key": args.dashscope_api_key,
        "use_boosted_orthogonal_guidance": args.use_boosted_orthogonal_guidance,
        "text_momentum_rolling_sum_momentum_weight": args.text_momentum_rolling_sum_momentum_weight,
        "text_momentum_rolling_sum_current_weight": args.text_momentum_rolling_sum_current_weight,
        "image_momentum_rolling_sum_momentum_weight": args.image_momentum_rolling_sum_momentum_weight,
        "image_momentum_rolling_sum_current_weight": args.image_momentum_rolling_sum_current_weight,
        "empty_momentum_rolling_sum_momentum_weight": args.empty_momentum_rolling_sum_momentum_weight,
        "empty_momentum_rolling_sum_current_weight": args.empty_momentum_rolling_sum_current_weight,
        "bog_mu": args.bog_mu,
        "bog_range": [args.bog_range_start, args.bog_range_end],
        "bog_interval": args.bog_interval,
        "device": args.device,
        "rewriter_device": args.rewriter_device,
        "unload_rewriter_level": args.unload_rewriter_level,
        "enable_inner_devices_manager": args.enable_inner_devices_manager,
    }
    if far_config is not None:
        pipeline_kwargs.update(
            far_config=far_config,
            response_transport_config=response_transport_config,
        )
        if pri_region_predictor is not None:
            pipeline_kwargs["pri_region_predictor"] = pri_region_predictor
        elif far_region_probabilities is not None:
            pipeline_kwargs["far_region_probabilities"] = far_region_probabilities

    if attribute_request is not None:
        pipeline_kwargs["attribute_request"] = attribute_request
    if source_anchor_request is not None:
        pipeline_kwargs["source_anchor_request"] = source_anchor_request
    results = pipeline(**pipeline_kwargs)

    return results


def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)

    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x : current_x + w] = img * 0.5 + 0.5
        current_x += w

    return to_pil_image(canvas)


def _ensure_output_parent(output_image_path: str) -> None:
    """Create a usable output parent for both bare and nested paths."""
    os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)


def _run_source_conditioned(args: argparse.Namespace) -> None:
    """Run the current method and save its image and provenance sidecar."""
    import json
    from pathlib import Path
    from PIL import Image

    from boogu.preservation.runtime import run_source_conditioned

    output_path = Path(args.output_image_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.suffix.lower() != ".png":
        raise ValueError("source-conditioned output_image_path must end in .png")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Output image or metadata already exists; choose a fresh output_image_path.")
    images, metadata = run_source_conditioned(args)
    _ensure_output_parent(str(output_path))
    # Exclusive creation prevents accidentally overwriting an input or an earlier run.
    with output_path.open("xb") as stream:
        pixels = images[0].detach().float().cpu().permute(1, 2, 0).numpy()
        Image.fromarray((pixels * 255).round().clip(0, 255).astype("uint8")).save(stream, format="PNG")
    metadata = {**metadata, "code_version": "0.2.1", "output_image_path": str(output_path.resolve()),
                "png_quantization": "round-to-nearest uint8"}
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Image saved to {output_path}; provenance saved to {metadata_path}")


def main(args: argparse.Namespace, root_dir: str) -> None:
    """Run reference-image editing from validated CLI inputs."""

    require_edit_checkpoint(args.pretrained_pipeline_name_or_path)
    method = getattr(args, "method", "source-conditioned")
    if method == "source-conditioned":
        return _run_source_conditioned(args)
    if method != "baseline":
        raise ValueError(f"Unknown editing method: {method!r}")
    if any(getattr(args, name, None) is not None for name in (
        "preservation_adapter", "preservation_permissions",
    )):
        raise ValueError("preservation inputs require --method source-conditioned, not baseline")
    instructions, input_images, input_image_paths = _prepare_edit_inputs(args)
    if args.batch_size is None:
        args.batch_size = len(instructions)
    elif (
        isinstance(args.batch_size, bool)
        or not isinstance(args.batch_size, int)
        or args.batch_size <= 0
    ):
        raise EditOnlyInputError(
            "Image editing batch_size must be a positive integer."
        )

    far_config, response_transport_config, far_regions = _prepare_far_inputs(
        args,
        instructions,
        input_images,
        input_image_paths,
    )

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == "fp16":
        weight_dtype = torch.float16
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16

    # Validate and create both random streams before loading any large weights.
    generator, reference_generator = _create_inference_generators(args)

    # Fail fast on a small predictor checkpoint before loading the main pipeline.
    pri_region_predictor = _load_pri_predictor(args)

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, weight_dtype)

    if (
        isinstance(args.max_input_image_pixels, (list, tuple))
        and len(args.max_input_image_pixels) == 1
    ):
        args.max_input_image_pixels = args.max_input_image_pixels[0]
    if (
        isinstance(args.max_vlm_input_pil_pixels, (list, tuple))
        and len(args.max_vlm_input_pil_pixels) == 1
    ):
        args.max_vlm_input_pil_pixels = args.max_vlm_input_pil_pixels[0]
    if (
        isinstance(args.rewriter_ref_images_max_pixels, (list, tuple))
        and len(args.rewriter_ref_images_max_pixels) == 1
    ):
        args.rewriter_ref_images_max_pixels = args.rewriter_ref_images_max_pixels[0]

    # Generate images
    images = []
    for i in range(0, len(instructions), args.batch_size):
        batch_instructions = instructions[i : i + args.batch_size]
        batch_input_images = (
            input_images[i : i + args.batch_size] if input_images else None
        )
        batch_input_image_paths = (
            input_image_paths[i : i + args.batch_size] if input_image_paths else None
        )
        far_run_kwargs = {}
        if far_config is not None:
            far_run_kwargs = {
                "far_config": far_config,
                "response_transport_config": response_transport_config,
            }
            if pri_region_predictor is not None:
                far_run_kwargs["pri_region_predictor"] = pri_region_predictor
            else:
                far_run_kwargs["far_region_probabilities"] = far_regions[
                    i : i + args.batch_size
                ]
        batch_results = run(
            args,
            pipeline,
            batch_instructions,
            args.negative_instruction,
            batch_input_images,
            batch_input_image_paths,
            generator=generator,
            reference_generator=reference_generator,
            **far_run_kwargs,
        )
        images.extend(batch_results.images)
    results = FMPipelineOutput(images=images)

    # Save output images
    _ensure_output_parent(args.output_image_path)
    if len(results.images) > 1:
        for i, image in enumerate(results.images):
            image_name, ext = os.path.splitext(args.output_image_path)
            image.save(f"{image_name}_{i}{ext}")

    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
    output_image = create_collage(vis_images)

    output_image.save(args.output_image_path)
    print(f"Image saved to {args.output_image_path}")


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)
