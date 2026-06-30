#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export device="cuda:0"

mkdir -p outputs/test_ti2i/

python inference.py \
    --pretrained_pipeline_name_or_path "models/Boogu-Image-0.1-Edit" \
    --input_image_paths "input_image_examples/03.jpg" \
    --instruction "把背景替换到沙滩." \
    --num_inference_steps 50 \
    --text_guidance_scale 4.0 --image_guidance_scale 1.0 \
    --output_image_path "outputs/test_ti2i/out_1.png" \
    --device "$device"
