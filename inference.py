"""
Boogu Inference Demo

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
from typing import List, Optional, Tuple

import cache_dit
import torch
from diffusers.hooks import apply_group_offloading
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from transformers import AutoProcessor

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Boogu-Image generation script.")
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
        help="Path or name of the whole pretrained pipeline.",
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
        "--seed", type=int, default=0, help="Random seed for generation."
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
        "--text_guidance_scale", type=float, default=4.0, help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=1.0, help="Image guidance scale."
    )
    parser.add_argument(
        "--empty_instruction_guidance_scale",
        type=float,
        default=0.0,
        help="Empty instruction guidance scale. Can be >0.0, <0.0, or =0.0 (disable). Only takes effect for TI2I tasks when `text_guidance_scale>1.0` and `image_guidance_scale>1.0` and. `use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide != use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide`.",
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
        "--system_prompt_follows_task_type",
        type=to_bool,
        default=True,
        help="If True, the system prompt will be selected based on the task type. If False, the system prompt will be selected based on the input images and instruction.",
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
        default="一幅国风琉金风格的山水画作，展现了桂林山水在金光普照下的壮丽景象。远山层叠，江水如镜，山峰边缘勾勒着发光的金色线条。画面采用石青石绿岩彩与鎏金质感相结合，局部有厚涂油画笔触，空中飘浮着金色粒子，营造出梦幻朦胧而又磅礴大气的意境。",
        help="Text prompt for generation.",
    )
    parser.add_argument(
        "--negative_instruction",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation.",
    )
    parser.add_argument(
        "--empty_instruction",
        type=str,
        default=" ",
        help="Empty instruction for generation.",
    )
    parser.add_argument(
        "--input_image_paths",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to input image(s).",
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
        help="The type of system prompt for the text instruction rewriter model. Only takes effect when `use_rewrite_text_instruction` is True. Available options: 'default','ppt', 'custom'.",
    )

    parser.add_argument(
        "--custom_rewriter_system_prompts_list",
        type=str,
        nargs="+",
        default=[
            "You are a Prompt optimizer designed to rewrite user inputs into high-quality Prompts that are more complete and expressive while preserving the original meaning."
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

    return parser.parse_args()


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


def preprocess(input_image_paths: List[str] = []) -> List[Image.Image]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if isinstance(input_image_paths, (str, list, tuple)):
        input_image_paths = input_image_paths if len(input_image_paths) > 0 else None

    if input_image_paths:
        input_images = []
        if isinstance(input_image_paths, str):
            input_image_paths = [input_image_paths]

        if len(input_image_paths) == 1 and os.path.isdir(input_image_paths[0]):
            input_images = [
                Image.open(os.path.join(input_image_paths[0], f)).convert("RGB")
                for f in os.listdir(input_image_paths[0])
            ]
        else:
            input_images = [
                Image.open(path).convert("RGB") for path in input_image_paths
            ]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images


def run(
    args: argparse.Namespace,
    pipeline: BooguImagePipeline,
    instruction: str,
    negative_instruction: str,
    input_images: Optional[List[List[Image.Image]]] = None,
    input_image_paths: Optional[List[List[str]]] = None,
) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    results = pipeline(
        instruction=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        max_input_image_pixels=args.max_input_image_pixels,
        max_input_image_side_length=args.max_input_image_side_length,
        num_inference_steps=args.num_inference_steps,
        max_vlm_input_pil_pixels=args.max_vlm_input_pil_pixels,
        max_vlm_input_pil_side_length=args.max_vlm_input_pil_side_length,
        max_sequence_length=args.max_sequence_length,
        truncate_instruction_sequence=args.truncate_instruction_sequence,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        empty_instruction_guidance_scale=args.empty_instruction_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_instruction=negative_instruction,
        num_images_per_instruction=args.num_images_per_instruction,
        generator=generator,
        output_type="pil",
        use_rewrite_text_instruction=args.use_rewrite_text_instruction,
        rewriter_max_new_tokens=args.rewriter_max_new_tokens,
        resize_rewriter_ref_images=args.resize_rewriter_ref_images,
        rewriter_ref_images_max_pixels=args.rewriter_ref_images_max_pixels,
        rewriter_ref_images_max_side_length=args.rewriter_ref_images_max_side_length,
        rewriter_system_prompt_type=args.rewriter_system_prompt_type,
        custom_rewriter_system_prompts_list=args.custom_rewriter_system_prompts_list,
        merge_original_and_rewritten_instructions=args.merge_original_and_rewritten_instructions,
        do_sample_for_local_rewriter=args.do_sample_for_local_rewriter,
        empty_instruction=args.empty_instruction,
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=args.use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide,
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=args.use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide,
        save_rewritten_instruction=args.save_rewritten_instruction,
        save_rewritten_instruction_path=args.save_rewritten_instruction_path,
        input_image_paths=input_image_paths,
        use_dashscope_remote_rewriting=args.use_dashscope_remote_rewriting,
        dashscope_remote_rewriting_model=args.dashscope_remote_rewriting_model,
        dashscope_base_http_api_url=args.dashscope_base_http_api_url,
        dashscope_api_key=args.dashscope_api_key,
        system_prompt_follows_task_type=args.system_prompt_follows_task_type,
        use_boosted_orthogonal_guidance=args.use_boosted_orthogonal_guidance,
        text_momentum_rolling_sum_momentum_weight=args.text_momentum_rolling_sum_momentum_weight,
        text_momentum_rolling_sum_current_weight=args.text_momentum_rolling_sum_current_weight,
        image_momentum_rolling_sum_momentum_weight=args.image_momentum_rolling_sum_momentum_weight,
        image_momentum_rolling_sum_current_weight=args.image_momentum_rolling_sum_current_weight,
        empty_momentum_rolling_sum_momentum_weight=args.empty_momentum_rolling_sum_momentum_weight,
        empty_momentum_rolling_sum_current_weight=args.empty_momentum_rolling_sum_current_weight,
        bog_mu=args.bog_mu,
        bog_range=[args.bog_range_start, args.bog_range_end],
        bog_interval=args.bog_interval,
        device=args.device,
        rewriter_device=args.rewriter_device,
        unload_rewriter_level=args.unload_rewriter_level,
        enable_inner_devices_manager=args.enable_inner_devices_manager,
    )

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


def main(args: argparse.Namespace, root_dir: str) -> None:
    """Main function to run the image generation process."""

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == "fp16":
        weight_dtype = torch.float16
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, weight_dtype)

    if isinstance(args.input_image_paths, (str, list, tuple)):
        print(f">>>>> User Inputs: input_image_paths: {args.input_image_paths}")
        args.input_image_paths = (
            args.input_image_paths if len(args.input_image_paths) > 0 else None
        )
        print(f">>>>> Processed Inputs: input_image_paths: {args.input_image_paths}")

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

    if args.use_batch_inference:
        print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
        print(
            "⚠️ ⚠️ ⚠️  >>>>>>>>>>>>>>##############  Using Batch Inference !##############<<<<<<<<<<<<<<<<<<<<  ⚠️ ⚠️ ⚠️ "
        )
        print(
            "⚠️ ⚠️ ⚠️  >>>>>>>>>>>>>>##############  Using Batch Inference !##############<<<<<<<<<<<<<<<<<<<<  ⚠️ ⚠️ ⚠️ "
        )
        print(
            "⚠️ ⚠️ ⚠️  >>>>>>>>>>>>>>##############  Using Batch Inference !##############<<<<<<<<<<<<<<<<<<<<  ⚠️ ⚠️ ⚠️ "
        )
        print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")

        print("⚠️ ⚠️ ⚠️  The following paramters will NOT take effect:")
        print(f"⚠️ ⚠️ ⚠️  --input_image_paths: {args.input_image_paths}")
        print(f"⚠️ ⚠️ ⚠️  --instruction: {args.instruction}")
        print(
            f"⚠️ ⚠️ ⚠️  because these information should be contained in the `--batch_data_config_path` file: {args.batch_data_config_path}."
        )
        print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
        assert (
            args.batch_data_config_path is not None
            and len(args.batch_data_config_path) > 0
        ), (
            f"Please properly set `--batch_data_config_path`, got {args.batch_data_config_path}."
        )

        batch_data_config = OmegaConf.load(args.batch_data_config_path)
        instructions = [x for x in list(batch_data_config.data.instructions)]
        input_image_paths = (
            [list(x) for x in list(batch_data_config.data.input_image_paths)]
            if batch_data_config.data.input_image_paths
            else None
        )

        input_images = []
        if input_image_paths:
            for paths in input_image_paths:
                input_images.append(preprocess(paths))
        else:
            input_images = None
            input_image_paths = None

    else:
        assert args.instruction is not None and len(args.instruction) > 0, (
            f"Please properly set `--instruction`, got {args.instruction}."
        )
        instructions = [args.instruction]
        if args.input_image_paths:
            input_images = [preprocess(args.input_image_paths)]
            input_image_paths = [args.input_image_paths]
        else:
            input_images = None
            input_image_paths = None

    if args.batch_size is None:
        args.batch_size = len(instructions)

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
        batch_results = run(
            args,
            pipeline,
            batch_instructions,
            args.negative_instruction,
            batch_input_images,
            batch_input_image_paths,
        )
        images.extend(batch_results.images)
    results = FMPipelineOutput(images=images)

    # Save output images
    os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
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
