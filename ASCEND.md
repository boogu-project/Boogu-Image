# Ascend NPU BF16 Inference

This adaptation targets Boogu-Image-0.1-Turbo BF16 with native PyTorch,
Transformers, Diffusers, and torch_npu. It does not enable FP8, FlashAttention,
Triton, torch.compile, or CPU offload.

Restrict physical devices with `ASCEND_RT_VISIBLE_DEVICES` before launching.
Device names passed to the script use the process-visible numbering.

## Environment

Install the matching PyTorch and torch_npu PTA packages first, then install the
remaining Python dependencies without replacing PyTorch:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements/ascend.txt
.venv/bin/python -m pip install -e . --no-deps
```

## Checkpoint Validation

```bash
.venv/bin/python inference_turbo_npu.py --check-only
```

## Inference

Run on one physical NPU:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 .venv/bin/python inference_turbo_npu.py \
  --device npu:0 \
  --model models/Boogu-Image-0.1-Turbo \
  --output outputs/turbo_npu.png
```

In this example, `npu:0` maps to the first physical device exposed by
`ASCEND_RT_VISIBLE_DEVICES`. The baseline requires about 40 GB of free device
memory and should be validated without offload first.
