#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export device="cuda:0"

mkdir -p outputs/test_ti2i_turbo_fp8/

python inference_turbo.py \
    --pretrained_pipeline_name_or_path "models/Boogu-Image-0.1-Edit-Turbo-fp8" \
    --use_fp8_weights True \
    --input_image_paths "input_image_examples/03.jpg" \
    --instruction "把背景替换到沙滩." \
    --dmd_conditioning_sigma 0.0 \
    --output_image_path "outputs/test_ti2i_turbo_fp8/out_1.png" \
    --device "$device"
