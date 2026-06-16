# Boogu Inference Guide

This document describes the current Boogu inference entry point for
text-to-image (T2I) generation and text/image-to-image (TI2I) editing:

```bash
inference.py
```

Many ready-to-run examples are available in:

```bash
demo_scripts/
```

These scripts cover common T2I, TI2I, batch inference, offload, prompt rewriting,
TeaCache, TaylorSeer, BOG, and prompt-tuning (**totally optional; coming soon**) combinations. For example,
`demo_t2i.sh` and `demo_ti2i.sh` are good starting points for basic generation
and editing, while scripts whose names contain `reasoning` show how to enable
the built-in instruction reasoner / rewriter. For most users, the fastest way to
start is to copy one of these scripts and adjust only the model path, prompt,
input image paths, device settings, and output name.

Most demo scripts also set:

```bash
export HF_MODULES_CACHE="$(pwd)/.hf_modules_cache"
```

This keeps dynamically loaded Hugging Face modules local to the repository
workspace.

## 1. Runtime Flow

The Python entry point follows this order:

1. Parse CLI arguments.
2. Load `BooguImagePipeline` or `BooguImagePromptTuningPipeline` (**coming soon**) from
   `--pretrained_pipeline_name_or_path`.
3. Optionally replace the MLLM instruction encoder with
   `--custom_pretrained_instruction_encoder_model_name_or_path`.
4. Optionally load prompt-tuning embeddings and LoRA weights.
5. Optionally attach a custom local instruction rewriter.
6. Optionally replace the diffusion transformer with
   `--custom_diffusion_transformer_path`.
7. Optionally load transformer LoRA weights.
8. Resolve TeaCache / TaylorSeer flags.
9. Validate device/offload compatibility.
10. Enable the selected offload strategy.
11. Prepare single-sample or batch inputs.
12. Run the pipeline and save the output image or collage.

## 2. Device Configuration

### 2.1 `export device` and `--device`

`device` must first be set as a shell environment variable, and the same value
should then be passed to the pipeline through `--device`:

```bash
export device="cuda:0"
...
--device "$device"
```

They should represent the same execution device for registered pipeline modules
such as the Boogu transformer, VAE, MLLM instruction encoder, scheduler tensors,
and optional prompt-tuning module.

The environment variable is required because some modules inspect
`os.getenv("device", "cpu")` during construction to decide whether CUDA/Triton
or Flash-Attention-oriented operators should be used. `inference.py` does not
copy `--device` back into `os.environ`, so set `export device=...` before
launching the script.

Supported `device` values:

| Value | Meaning |
| --- | --- |
| `cpu` | CPU execution. Do not enable CPU/group offload flags. |
| `cuda` | Default visible CUDA device, usually equivalent to `cuda:0`. |
| `cuda:x` | Specific visible CUDA device, e.g. `cuda:0`. |

The pipeline uses lazy placement in several paths. Models are often loaded on
CPU first, then moved to the requested execution device only when needed. If an
offload strategy is enabled, manual `.to(device)` moves are intentionally
ignored because the offload hook owns module movement.

### 2.2 `enable_inner_devices_manager`

`--enable_inner_devices_manager` is a narrow device-movement helper for saving
GPU memory during local instruction rewriting. It defaults to `False`.

It is mainly meant for this situation:

1. `use_rewrite_text_instruction == True`
2. `use_dashscope_remote_rewriting == False`
3. `custom_local_instruction_rewriter_model` is set and loaded successfully
4. `rewriter_device` is not `cpu`
5. `device` and `rewriter_device` are the same, or `rewriter_device == auto`
   while `device` is `cuda` / `cuda:x`
6. `unload_rewriter_level` is not `keep`

In that case, the pipeline tries to keep GPU memory lower by moving the local
rewriter and the pipeline modules as needed around the rewriting phase. This is
useful when VRAM is tight.

The tradeoff is that repeated reuse of the same loaded pipeline can become
slower, because weights may be moved back and forth between CPU and GPU.

Outside the local-rewriter case above, this switch is usually unnecessary.
If an offload strategy is enabled, the offload hook still owns module movement,
and the pipeline may ignore some user-requested moves.

Do not combine this mode with manual `pipeline.to(device)` calls unless you
really need to manage placement yourself.

### 2.3 `rewriter_device`

`rewriter_device` is a CLI argument passed through `--rewriter_device`. The
shell variable used in the demo scripts is only a convenience for keeping
commands readable. It controls the local instruction rewriter, also called the
instruction reasoner. It is separate from `device` because rewriting happens
before diffusion sampling and can use a different model.

Recommended shell pattern:

```bash
export device="cuda:0"
export rewriter_device="$device"
...
--device "$device" \
--rewriter_device "$rewriter_device"
```

Supported `rewriter_device` values:

| Value | Meaning |
| --- | --- |
| `cpu` | Run the local rewriter on CPU. Slow, but valid. |
| `cuda` | Run on the default visible CUDA device. |
| `cuda:x` | Run on a specific visible CUDA device. |
| `auto` | Use auto/meta placement when the model is loaded with auto device mapping. |

When all of the following are true, the pipeline reuses the instruction encoder
as the local instruction reasoner (the rewriter):

1. `--use_rewrite_text_instruction True`
2. `--use_dashscope_remote_rewriting False`
3. `--custom_local_instruction_rewriter_model` is empty or not provided

In this shared-model setup, keeping `device` and `rewriter_device` equal is
strongly recommended because the instruction encoder and the local rewriter are
the same underlying model. It becomes a hard compatibility requirement when
shared local rewriting is combined with offload.

This shared default is convenient, but for stable advanced usage it is strongly
recommended to attach a separate local rewriter model and its matching processor,
or use remote rewriting. When the rewriter and MLLM are shared, set:

```bash
rewriter_device="$device"
```

`cuda` and `cuda:0` are treated as equivalent for warning purposes. Splitting a
shared MLLM/rewriter across different devices can trigger expensive device moves
and is more likely to conflict with offload hooks.

In other cases, `device` and `rewriter_device` may be different. For example,
when `--custom_local_instruction_rewriter_model` points to a separate local
rewriter, it can run on another GPU such as `cuda:1`; when DashScope remote
rewriting is enabled, no local rewriter device is used. If one GPU has enough
VRAM, setting `rewriter_device="$device"` is always a valid and simple choice.

## 3. Model Loading

### 3.1 Base Pipeline

`--pretrained_pipeline_name_or_path` is required and should point to the full
pretrained Boogu pipeline directory.

If `--use_prompt_tuning True`, the loader uses `BooguImagePromptTuningPipeline`.
Otherwise it uses `BooguImagePipeline`.

### 3.2 MLLM Instruction Encoder

`--custom_pretrained_instruction_encoder_model_name_or_path` optionally replaces
the pipeline MLLM. The loader tries known classes first:

- `InternVLForConditionalGeneration`
- `Qwen3VLMoeForConditionalGeneration`
- `Qwen3VLForConditionalGeneration`
- `Qwen2_5_VLForConditionalGeneration`
- `Qwen3ForCausalLM`
- `Qwen2ForCausalLM`
- `Gemma4ForConditionalGeneration`

For unknown VLM/LLM names, it falls back to Transformers auto classes. If that
fails, extend `load_llm_or_vlm(...)` with the correct model class and loading
arguments.

### 3.3 Diffusion Transformer

`--custom_diffusion_transformer_path` optionally replaces the pipeline
transformer with a checkpoint loaded by
`BooguImageTransformer2DModel.from_pretrained`.

`--custom_transformer_lora_path` optionally loads LoRA weights into the
transformer via `pipeline.load_lora_weights(...)`.

### 3.4 Prompt Tuning (Optional Research Module; Coming Soon)

Prompt tuning is purely optional. It is included as reference material for the
open-source community and should not be treated as the main release path. The
default Boogu pipeline is the recommended version for normal use.

The prompt-tuning variant was trained on only a small subset of data, so it is
expected to be less robust than the default pipeline. Prompt-tuning weights may
be released later, but they are not required for standard inference.

Prompt tuning is controlled by:

| Flag | Meaning |
| --- | --- |
| `--use_prompt_tuning` | Select `BooguImagePromptTuningPipeline`. |
| `--custom_prompt_tuning_model_path` | Load a custom `PromptEmbedding` checkpoint. |
| `--custom_prompt_tuning_model_lora_weights_path` | Load LoRA weights into `prompt_embedding`. |

Prompt-tuning LoRA loading only runs when all conditions are true:

1. `--use_prompt_tuning True`
2. `--custom_prompt_tuning_model_lora_weights_path` is not `None`
3. `--custom_prompt_tuning_model_lora_weights_path.strip()` is non-empty

For prompt-tuning LoRA to be useful, load a base prompt-tuning model first with
`--custom_prompt_tuning_model_path`. Example prompt-tuning scripts under
`demo_scripts/` use the `_pt` suffix, but they are optional research examples
rather than the recommended default workflow. These scripts expect a
prompt-tuning-specific pipeline checkpoint path, not the main
`models/Boogu-Image-0.1-Base` release path.

## 4. Instruction Rewriting

### 4.1 Enabling Rewriting

Use:

```bash
--use_rewrite_text_instruction True
```

Main rewriting options:

| Flag | Meaning |
| --- | --- |
| `--merge_original_and_rewritten_instructions` | Merge original and rewritten instructions before encoding. |
| `--custom_local_instruction_rewriter_model` | Hugging Face-format local rewriter model. |
| `--rewriter_max_new_tokens` | Maximum tokens generated by the local rewriter. |
| `--do_sample_for_local_rewriter` | Whether local rewriting uses sampling. |
| `--resize_rewriter_ref_images` | Resize reference images before local rewriting. |
| `--rewriter_ref_images_max_pixels` | Pixel cap for rewriter reference images. |
| `--rewriter_ref_images_max_side_length` | Side-length cap for rewriter reference images. |
| `--unload_rewriter_level` | `keep`, `cpu`, or `destroy` after rewriting. |

If no custom local rewriter is set, the MLLM instruction encoder is reused as
the default local rewriter. In that case `unload_rewriter_level` will not unload
the shared model.

### 4.2 Remote Rewriting

DashScope remote rewriting is controlled by:

| Flag | Meaning |
| --- | --- |
| `--use_dashscope_remote_rewriting` | Use remote rewriting instead of local rewriting. |
| `--dashscope_remote_rewriting_model` | Remote model name, default `qwen-vl-max-latest`. |
| `--dashscope_base_http_api_url` | DashScope endpoint. |
| `--dashscope_api_key` | Must be a real key, not the placeholder value. |

When remote rewriting is enabled there is no local rewriter to unload.

### 4.3 Rewriter System Prompts

`--rewriter_system_prompt_type` selects the prompt preset used by the rewriter.
Internally, each preset is a list of rewrite steps. During rewriting, the
pipeline iterates over that list, updates the active system prompts for the
current step, and feeds the rewritten result into the next step.

| Value | Use case | Behavior |
| --- | --- | --- |
| `default` | General T2I/TI2I prompt polishing. | Uses the built-in image-generation and image-editing rewriter prompts from `InstructionReasonerStaticRewriteSkills`. This is the recommended baseline. |
| `ppt` | Slide, report, infographic, or presentation-style generation/editing. | Uses the PPT-specific prompt chain from `static_skills.py`. This preset can contain multiple rewrite steps. |
| `custom` | User-defined rewrite policy. | Uses `--custom_rewriter_system_prompts_list`; each string in the list becomes one rewrite step. |

Examples:

```bash
# General prompt polishing.
--use_rewrite_text_instruction True \
--rewriter_system_prompt_type default
```

```bash
# Presentation / infographic oriented rewriting.
--use_rewrite_text_instruction True \
--rewriter_system_prompt_type ppt
```

`--custom_rewriter_system_prompts_list` only takes effect when:

1. `--use_rewrite_text_instruction True`
2. `--rewriter_system_prompt_type custom`
3. `--custom_rewriter_system_prompts_list` is a non-empty `List[str]`

For `custom`, pass each rewrite step as one CLI string:

```bash
--rewriter_system_prompt_type custom \
--custom_rewriter_system_prompts_list \
  "You are a prompt optimizer. Rewrite the user request into a clear visual prompt while preserving its intent." \
  "Refine the result again with more concrete subject, layout, lighting, and style details."
```

In shell scripts, use array expansion:

```bash
custom_rewriter_system_prompts_list=(
  "You are a prompt optimizer. Rewrite the user request into a clear visual prompt while preserving its intent."
  "Refine the result again with more concrete subject, layout, lighting, and style details."
)

--custom_rewriter_system_prompts_list "${custom_rewriter_system_prompts_list[@]}"
```

Notes:

- `default` is usually best for normal image generation or image editing.
- `ppt` is specialized for slide-like visual design. Do not use it for ordinary
  photo or illustration prompts unless that layout-heavy behavior is desired.
- `custom` is useful for experiments and domain-specific rewrite policies. The
  current implementation registers each custom prompt for both Chinese and
  English, and for both image-generation and image-editing keys, so make the
  prompt text itself clear enough for your expected use case.
- A multi-step custom list is powerful but can over-edit the instruction. Start
  with one step, then add a second step only if the rewritten prompt still lacks
  structure or detail.

For frequently reused custom rewrite prompts, you can also register prompt
presets directly in code by following the existing logic in:

- `boogu/pipelines/boogu/instruct_reasoner_static_skills.py`
- `boogu/pipelines/boogu/static_skills.py`

This is convenient when you want a stable named prompt preset instead of passing
a long prompt list through the CLI.

### 4.4 Saving Rewritten Instructions

```bash
--save_rewritten_instruction True \
--save_rewritten_instruction_path outputs/.../rewritten_instruction.json
```

The saved file contains original and rewritten instructions for reproducibility.

## 5. Offload and Memory

The three offload flags are mutually exclusive. Exactly zero or one may be
enabled:

| Flag | Approx. VRAM without local rewriter | Notes |
| --- | ---: | --- |
| `enable_sequential_cpu_offload_flag=True` | `< 2 GB` | Lowest VRAM, slowest path. Uses fine-grained CPU/meta offload. |
| `enable_model_cpu_offload_flag=True` | `~21.9 GB` | Balanced Diffusers pipeline-level CPU offload. |
| `enable_group_offload_flag=True` | `~18-20.3 GB` | Group/block offload for transformer, MLLM, and VAE. |

The memory numbers above assume `use_rewrite_text_instruction=False`, so no
local rewriter model is active.

Validation rules enforced by the code:

- All three offload flags must be valid booleans.
- At most one offload flag may be `True`.
- If any offload flag is `True`, `device` must be `cuda` or `cuda:x`; `cpu` is
  rejected because CPU offload needs a non-CPU execution device.
- If prompt rewriting is enabled, offload is incompatible with a shared local
  MLLM/rewriter. Use a custom local rewriter or remote rewriting.

Recommended no-offload setup:

```bash
export device="cuda:0"
enable_sequential_cpu_offload_flag="False"
enable_model_cpu_offload_flag="False"
enable_group_offload_flag="False"
```

Low-memory setup:

```bash
export device="cuda:0"
enable_sequential_cpu_offload_flag="True"
enable_model_cpu_offload_flag="False"
enable_group_offload_flag="False"
```

Group-offload setup:

```bash
export device="cuda:0"
enable_sequential_cpu_offload_flag="False"
enable_model_cpu_offload_flag="False"
enable_group_offload_flag="True"
```

The offload variables above follow the `demo_scripts/` style: they are shell
helpers that should be passed to `inference.py` through the corresponding CLI
flags, for example `--enable_group_offload_flag $enable_group_offload_flag`.

Without offload and without a separate local rewriter, a T2I run typically needs
about `40 GB` VRAM. In that setup, the default shared MLLM rewriter does not add
another local model and stays around the same memory range.

If you attach a large custom local rewriter, memory requirements increase
substantially. A 32B local rewriter may require at least `60 GB` total VRAM for
local instruction reasoning plus Boogu inference.

## 6. TeaCache, TaylorSeer, and Cache-DiT

TeaCache, TaylorSeer, and Cache-DiT are optional transformer caching acceleration
strategies. They are independent from offload flags in control flow, but their
cache tensors can increase runtime memory.

When multiple strategies are requested, the active strategy follows this priority:
**Cache-DiT > TaylorSeer > TeaCache**.

### 6.1 Cache-DiT

| Flag | Effect |
| --- | --- |
| `--enable_cache_dit_caching True` | Enables Cache-DiT caching on single-stream layers. |
| `--enable_cache_dit_caching_for_all_layers True` | Extends Cache-DiT caching to all layers. |

Cache-DiT takes the highest priority over TaylorSeer and TeaCache. When it is
enabled, TaylorSeer and TeaCache flags are ignored.

### 6.2 TeaCache and TaylorSeer

| Flag | Effect |
| --- | --- |
| `--enable_teacache True` | Enables TeaCache on single-stream layers. |
| `--enable_taylorseer True` | Enables TaylorSeer on single-stream layers. |
| `--enable_teacache_for_all_layers True` | Extends TeaCache to double-stream and single-stream layers; implicitly enables TeaCache unless TaylorSeer is selected. |
| `--enable_taylorseer_for_all_layers True` | Extends TaylorSeer to double-stream and single-stream layers; implicitly enables TaylorSeer and takes priority over TeaCache. |
| `--teacache_rel_l1_thresh` | TeaCache relative L1 threshold, default `0.05`. |

TeaCache and TaylorSeer are mutually exclusive. If both are requested, TaylorSeer
is enabled and all TeaCache options are ignored with a warning.

Recommended order:

1. Start with all cache flags disabled.
2. Try `--enable_teacache True` for a lighter acceleration option.
3. Try `--enable_teacache_for_all_layers True` only after the single-stream path
   is stable.
4. Use TaylorSeer carefully; all-layer TaylorSeer stores more cache state and is
   more memory-intensive.
5. Use Cache-DiT when a higher-priority caching strategy is needed; it overrides
   all other cache flags.

## 7. Guidance Options

### 7.1 CFG and Task Modes

Core guidance flags:

| Flag | Meaning |
| --- | --- |
| `--text_guidance_scale` | Text CFG strength. `1.0` disables text CFG. |
| `--image_guidance_scale` | Reference-image CFG strength. `1.0` disables image CFG. |
| `--cfg_range_start`, `--cfg_range_end` | Step range where CFG is active. |
| `--negative_instruction` | Negative text instruction. |
| `--empty_instruction` | Empty instruction used by special TI2I branches. |

T2I uses no reference images. TI2I uses one or more reference images.

`--empty_instruction_guidance_scale` can be positive, negative, or zero. It only
takes effect for TI2I when:

- `text_guidance_scale > 1.0`
- `image_guidance_scale > 1.0`
- `use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide` and
  `use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide` are not
  equal

### 7.2 Empty-Instruction Branches

| Flag | Meaning |
| --- | --- |
| `--use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide` | Use `empty_instruction` for the image-guidance reference branch. |
| `--use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide` | Use `empty_instruction` for the text-guidance reference branch. |
| `--system_prompt_follows_task_type` | Select system prompt according to T2I/TI2I task type. |

### 7.3 Boosted Orthogonal Guidance

Enable BOG with:

```bash
--use_boosted_orthogonal_guidance True
```

BOG-related parameters:

| Flag | Meaning |
| --- | --- |
| `--bog_mu` | BOG strength. |
| `--bog_range_start`, `--bog_range_end` | Denoising range where BOG can take effect. |
| `--bog_interval` | Apply BOG every `N` denoising steps. `1` means every step. Default: `2`. |
| `--text_momentum_rolling_sum_momentum_weight`, `--text_momentum_rolling_sum_current_weight` | Momentum settings for text guidance. |
| `--image_momentum_rolling_sum_momentum_weight`, `--image_momentum_rolling_sum_current_weight` | Momentum settings for image guidance. |
| `--empty_momentum_rolling_sum_momentum_weight`, `--empty_momentum_rolling_sum_current_weight` | Momentum settings for empty-instruction guidance. |

Recommended starting point:

```bash
--use_boosted_orthogonal_guidance False
--bog_range_start 0.0
--bog_range_end 1.0
--bog_interval 2
--bog_mu 0.1
```

## 8. Image and VLM Preprocessing

Diffusion-transformer image limits:

| Flag | Meaning |
| --- | --- |
| `--max_input_image_pixels` | Max pixels for each input image passed to the diffusion transformer. For T2I, the internally created noise image is also treated as an input image. Accepts one or more integers. |
| `--max_input_image_side_length` | Max side length for each input image passed to the diffusion transformer. For T2I, this also constrains the internally created noise image. |

VLM instruction-encoder image/token limits:

| Flag | Meaning |
| --- | --- |
| `--max_vlm_input_pil_pixels` | Max pixels for images passed to the MLLM instruction encoder. Accepts one or more integers. |
| `--max_vlm_input_pil_side_length` | Max side length for images passed to the MLLM instruction encoder. |
| `--max_sequence_length` | Max tokenized instruction length when truncation is enabled. |
| `--truncate_instruction_sequence` | Whether to truncate the VLM input sequence. |

Be careful with `--truncate_instruction_sequence True`: some Transformers
versions can fail if image tokens alone exceed `--max_sequence_length`.

Vision-token masking:

| Flag | Meaning |
| --- | --- |
| `--mask_vision_tokens_feature` | Remove selected vision-token features from MLLM outputs. |
| `--vision_token_ids` | List of vision token ids to remove, e.g. Qwen3-VL ids. |

## 9. Single-Sample and Batch Inputs

### 9.1 Single-Sample T2I

For pure T2I, omit `--input_image_paths`:

```bash
--instruction "A cinematic photo of ..."
```

The script converts missing or empty `input_image_paths` to `None`, so the
pipeline detects the task as T2I.

### 9.2 Single-Sample TI2I

For image editing, provide one or more paths:

```bash
--instruction "Change the red car into a blue car." \
--input_image_paths ./examples/car.png
```

Multiple reference images are passed as multiple paths.

### 9.3 Batch Inference

Enable batch mode with:

```bash
--use_batch_inference True \
--batch_data_config_path ./batch_data_samples/ti2i_batch_data_sample.yml \
--batch_size 4
```

`--batch_size` controls how many samples are processed per execution segment.
The pipeline runs samples in batches of this size and aggregates the outputs at
the end. If omitted, all samples in the YAML are processed in one pass.

When batch mode is enabled:

- `--instruction` and `--input_image_paths` from the CLI are ignored.
- Instructions and image paths are read from the YAML file.
- `batch_data_config_path` must be non-empty.

Expected YAML structure:

```yaml
data:
  instructions:
    - "Prompt for sample 1"
    - "Prompt for sample 2"
  input_image_paths:
    - []
    - ["./examples/ref.png"]
```

For pure T2I batches, `input_image_paths` can be empty or omitted according to
the loader behavior. For TI2I samples, each sample should provide at least one
valid image path.

## 10. Output

Core output flags:

| Flag | Meaning |
| --- | --- |
| `--output_image_path` | Final image path. Use a path with a parent directory, e.g. `outputs/demo.png`. If multiple images are generated, numbered files are also saved. |
| `--num_images_per_instruction` | Number of images generated per instruction. |
| `--seed` | Random seed. |
| `--height`, `--width` | Output image size. |
| `--dtype` | Model dtype: `fp32`, `fp16`, or `bf16`. |
| `--scheduler` | `euler` or `dpmsolver++`. |
| `--num_inference_steps` | Number of denoising steps. |

When multiple images are produced, the script also creates a horizontal collage
and writes it to `--output_image_path`.

## 11. Minimal Recipes

### 11.1 Direct CUDA T2I

```bash
export CUDA_VISIBLE_DEVICES=0
export device="cuda:0"

python inference.py \
  --pretrained_pipeline_name_or_path models/Boogu-Image-0.1-Base \
  --instruction "A high-quality cinematic image of a mountain lake." \
  --output_image_path outputs/demo.png \
  --device "$device" \
  --rewriter_device "$device" \
  --enable_sequential_cpu_offload_flag False \
  --enable_model_cpu_offload_flag False \
  --enable_group_offload_flag False
```

### 11.2 Low-Memory T2I

```bash
export CUDA_VISIBLE_DEVICES=0
export device="cuda:0"

python inference.py \
  --pretrained_pipeline_name_or_path models/Boogu-Image-0.1-Base \
  --instruction "A high-quality cinematic image of a mountain lake." \
  --output_image_path outputs/demo_low_mem.png \
  --device "$device" \
  --rewriter_device "$device" \
  --enable_sequential_cpu_offload_flag True \
  --enable_model_cpu_offload_flag False \
  --enable_group_offload_flag False
```

Do not enable local rewriting with a shared default MLLM rewriter in this setup.
Use remote rewriting or a separate custom local rewriter if rewriting is needed.

### 11.3 Custom Local Rewriter

```bash
--use_rewrite_text_instruction True \
--use_dashscope_remote_rewriting False \
--custom_local_instruction_rewriter_model /path/to/qwen-vl-rewriter \
--rewriter_device cuda:1
```

Make sure the rewriter model and its processor match. Large local rewriters can
significantly increase VRAM requirements.

For TI2I editing, follow the TI2I demo scripts and use an edit-capable pipeline
checkpoint such as `models/Boogu-Image-0.1-Edit`.

## 12. Torch Compile and FP8

### 12.1 Torch Compile

`--enable_torch_compile` compiles the transformer's repeated blocks using
`torch.compile`, which can reduce per-step latency on supported GPUs.
Warning: on some GPUs/models this can occasionally produce all-black outputs.
If you see black images, disable `--enable_torch_compile` first.

| Flag | Meaning |
| --- | --- |
| `--enable_torch_compile` | Enable `torch.compile` on repeated transformer blocks. |
| `--torch_compile_mode` | Compilation mode passed to `torch.compile`, e.g. `reduce-overhead`. |

The first forward pass triggers compilation and is slower. Subsequent steps
use the compiled kernel. Compilation only applies to repeated blocks;
non-repeated layers run normally.

Note: `block_lumina2.py` switches `swiglu` to a pure PyTorch implementation
automatically when `torch.compiler.is_compiling()` is detected, to avoid
compilation incompatibilities.

### 12.2 FP8 Weights

`--use_fp8_weights` loads the transformer checkpoint as FP8-quantized weights
from the `transformer/` subdirectory inside `--pretrained_pipeline_name_or_path`,
reducing VRAM for weight storage.

| Flag | Meaning |
| --- | --- |
| `--use_fp8_weights` | Load transformer weights in FP8 format from `<pretrained_pipeline_name_or_path>/transformer`. |

The transformer must have been pre-quantized with `utils/quantize_transformer.py`
(uses `float8_weight_only` by default, excluding `embedder`, `embed`, and
`lm_head` layers).

To produce an FP8 checkpoint:

```bash
python utils/quantize_transformer.py \
  --model_path /path/to/transformer \
  --save_path /path/to/transformer_fp8
```

For the built-in FP8 loading path, place the quantized checkpoint under the
pipeline's `transformer/` directory and pass `--use_fp8_weights True`.

If you instead pass `--custom_diffusion_transformer_path`, that custom
transformer is loaded after the pipeline and replaces the transformer from the
pretrained pipeline. Do not rely on `--use_fp8_weights` to redirect loading to a
custom transformer path.

## 13. Compatibility Matrix

|||||||||||
|---|---|---|---|---|---|---|---|---|---|
|  | **Offload** | None | None | Model CPU | Model CPU | Sequential CPU | Sequential CPU | Group | Group |
| **Caching** | **Quantization** <br> \ <br> **torch.compile** | None | FP8 | None | FP8 | None | FP8 | None | FP8 |
| Off | Off | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Off | On | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| TeaCache | Off | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| TeaCache | On | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| TaylorSeer | Off | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| TaylorSeer | On | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cache-DiT | Off | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cache-DiT | On | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

## 14. Quick Checklist

- [ ] Set `export device=...` and `--device "$device"` to the same value.
- [ ] Use `--enable_inner_devices_manager` only for local rewriting when VRAM is tight
  and are not manually managing all module devices yourself (defaults to `False`).
- [ ] Start with all cache flags (TeaCache, TaylorSeer, Cache-DiT) disabled; enable
  one at a time. Priority: Cache-DiT > TaylorSeer > TeaCache.
- [ ] Use `--use_fp8_weights` only with a pre-quantized checkpoint from
  `utils/quantize_transformer.py`.
- [ ] Use `--enable_torch_compile` only after the baseline run is stable.
  If compiled runs occasionally return all-black images, disable it and fall
  back to the uncompiled baseline.
- [ ] Select `BooguImagePipeline` or `BooguImagePromptTuningPipeline` through
  `--use_prompt_tuning`.
- [ ] Set `--pretrained_pipeline_name_or_path`.
- [ ] Set `--custom_diffusion_transformer_path` if replacing the transformer checkpoint.
- [ ] Enable at most one offload flag.
- [ ] Use CUDA, not CPU, when any offload flag is enabled.
- [ ] Keep `rewriter_device == device` when the MLLM and rewriter are shared.
- [ ] Prefer a separate custom local rewriter or remote rewriting for advanced
  device/offload setups.
- [ ] Start with TeaCache/TaylorSeer disabled, then enable them only after the
  baseline is correct.
- [ ] For batch inference, put instructions and image paths in the YAML file.
- [ ] Set a unique `--output_image_path` for each run. In demo shell scripts,
  `case_name` is only a helper variable used to build that output path.
