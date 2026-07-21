# Ascend NPU BF16 Inference

The Ascend entry point supports all Boogu-Image-0.1 BF16 checkpoints with
native PyTorch, Transformers, Diffusers, and torch_npu. It does not enable FP8,
FlashAttention, Triton, torch.compile, or CPU offload. Base and Edit additionally
support optional TeaCache and TaylorSeer acceleration for offline inference.

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
