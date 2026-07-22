# Ascend NPU BF16 Inference

The streamlined Ascend entry point supports all Boogu-Image-0.1 BF16
checkpoints with native PyTorch, Transformers, Diffusers, and torch_npu. It does
not enable FP8, FlashAttention, Triton, torch.compile, or CPU offload. Base and
Edit additionally support optional TeaCache and TaylorSeer acceleration for
offline inference. Use `inference.py` and `npu_demo_scripts/` for the complete
demo and CPU-offload feature set.

`inference_npu.py` reads `_class_name` from `model_index.json` to select the
standard or Turbo pipeline. Passing `--input-image` selects image editing;
without it, the script runs text-to-image generation. Model directory names
are not used to infer the task. The official Edit-Turbo hotfix currently
declares the standard pipeline in its metadata, so pass `--turbo` explicitly
for that checkpoint.

| Checkpoint type | Task | Steps | Text CFG | Image CFG | DMD sigma |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | T2I | 50 | 4.0 | 1.0 | disabled |
| Turbo | T2I | 4 | 1.0 | 1.0 | 0.001 |
| Edit | TI2I | 50 | 4.0 | 1.0 | disabled |
| Edit-Turbo | TI2I | 4 | 1.0 | 1.0 | 0.0 |

## Environment

Install the matching CANN toolkit first. The Ascend requirements pin the PTA
pair validated for this change (`torch==2.10.0`, `torch-npu==2.10.0`) and
install the remaining Python dependencies. Use the versions required by your
CANN release if they differ:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements/ascend.txt
.venv/bin/python -m pip install -e . --no-deps
```

Restrict physical devices with `ASCEND_RT_VISIBLE_DEVICES` before launching.
Device names passed to the script use the process-visible numbering.

## Checkpoint Validation

```bash
.venv/bin/python inference_npu.py \
  --model models/Boogu-Image-0.1-Base \
  --check-only
```

## Text-to-Image

Base and Turbo use the same command shape. Defaults are selected from the
checkpoint pipeline class:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 .venv/bin/python inference_npu.py \
  --device npu:0 \
  --model models/Boogu-Image-0.1-Turbo \
  --prompt "A cinematic mountain landscape illuminated by golden light." \
  --output outputs/turbo_npu.png
```

## Image Editing

Passing `--input-image` selects TI2I for Edit and Edit-Turbo checkpoints:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 .venv/bin/python inference_npu.py \
  --device npu:0 \
  --model models/Boogu-Image-0.1-Edit \
  --input-image input_image_examples/03.jpg \
  --prompt "Replace the background with a beach while preserving the subject." \
  --output outputs/edit_npu.png
```

Edit-Turbo uses the same command with an explicit pipeline override:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 .venv/bin/python inference_npu.py \
  --device npu:0 \
  --model models/Boogu-Image-0.1-Edit-Turbo \
  --input-image input_image_examples/03.jpg \
  --turbo \
  --output outputs/edit_turbo_npu.png
```

## NPU Operator Acceleration

### Fused RMSNorm

NPU inference automatically uses `torch_npu.npu_rms_norm` for affine RMSNorm
layers. No command-line option is required. CPU and CUDA execution keep their
existing implementations, and non-affine RMSNorm falls back to PyTorch.

The following 1024x1024 denoising times compare PyTorch RMSNorm with the fused
NPU operator in the validated environment. Loading and first-run compilation
time are excluded.

| Checkpoint type | PyTorch RMSNorm | NPU RMSNorm |
| --- | ---: | ---: |
| Base | 56 s | 52 s |
| Edit | 128 s | 118 s |
| Turbo | 2.17 s | 1.96 s |
| Edit-Turbo | 4.33 s | 4.04 s |

### Native GQA

NPU SDPA automatically uses native grouped-query attention instead of
materializing repeated key and value heads. CPU and CUDA keep the existing KV
expansion path. No command-line option is required.

The NPU operator microbenchmark improved from 0.354 ms to 0.243 ms for 32 query
heads, 8 KV heads, and sequence length 1024, and from 2.473 ms to 2.309 ms for
28 query heads, 7 KV heads, and sequence length 4096. A same-device Base run
improved 50-step denoising from 52 s to 51 s with identical output.

### Fused SwiGLU

NPU eager inference uses `torch_npu.npu_swiglu` in the transformer feed-forward
layers. CPU, CUDA, and compiled execution keep their existing implementations.
In the validated BF16 microbenchmark, the fused activation improved from
1.184 ms to 0.613 ms at 4096 tokens, while the complete feed-forward block
improved from 5.415 ms to 4.797 ms.

`torch_npu.npu_rotary_mul` is not used. The current batch-specific complex RoPE
layout requires per-call coefficient expansion, which made the fused path
slower than the existing implementation.

## Full Demo and CPU Offload Support

`npu_demo_scripts/` mirrors every script in `demo_scripts/` with NPU device
settings. These scripts use the feature-rich `inference.py` entry point for
batch inference, prompt rewriting, prompt tuning, TeaCache, TaylorSeer, BOG,
and CPU offload combinations.

Sequential, model, and group offload are mutually exclusive. NPU group offload
uses synchronous transfers because Diffusers 0.38 only exposes streamed group
offload for CUDA and XPU devices.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 bash npu_demo_scripts/demo_seq_offload_t2i.sh
ASCEND_RT_VISIBLE_DEVICES=0 bash npu_demo_scripts/demo_group_offload_t2i.sh
```

## Offline Cache Acceleration

TeaCache and TaylorSeer are optional and mutually exclusive. They are disabled
by default because cached inference changes the numerical path and can produce
small image differences. The cache lifecycle is implemented by the standard
50-step pipeline, so these options support Base and Edit only; Turbo and
Edit-Turbo reject them.

| Option | Effect |
| --- | --- |
| `--enable-teacache` | Enable TeaCache on single-stream layers. |
| `--enable-taylorseer` | Enable TaylorSeer on single-stream layers. |
| `--cache-all-layers` | Extend the selected method to double-stream layers. |
| `--teacache-rel-l1-thresh` | TeaCache relative L1 threshold; default `0.05`. |

For example, enable all-layer TeaCache for Edit:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 .venv/bin/python inference_npu.py \
  --device npu:0 \
  --model models/Boogu-Image-0.1-Edit \
  --input-image input_image_examples/03.jpg \
  --prompt "Replace the background with a beach while preserving the subject." \
  --enable-teacache \
  --cache-all-layers \
  --output outputs/edit_teacache_npu.png
```

The following 1024x1024, 50-step denoising times are indicative results from the
validated CANN 9.0.0, PyTorch 2.10.0, and torch-npu 2.10.0 environment. Exact
performance and image differences depend on the prompt and input image.

| Cache mode | Base T2I | Edit TI2I |
| --- | ---: | ---: |
| Disabled | 56 s | 128 s |
| TeaCache, single-stream | 56 s | 111 s |
| TeaCache, all layers | 40 s | 54 s |
| TaylorSeer, single-stream | 35 s | 79 s |
| TaylorSeer, all layers | 26 s | 58 s |

The official recommended Edit-Turbo checkpoint can be downloaded with:

```bash
huggingface-cli download Boogu/Boogu-Image-0.1-Edit-Turbo \
  --revision hotfix-1k-20260708 \
  --local-dir models/Boogu-Image-0.1-Edit-Turbo
```

In these examples, `npu:0` maps to the single physical device exposed by
`ASCEND_RT_VISIBLE_DEVICES`. The BF16 baseline requires about 40 GB of free
device memory and should be validated without offload first. Use `--steps`,
`--text-guidance-scale`, or `--image-guidance-scale` to override defaults.
