#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export device="cuda:0"

mkdir -p outputs/test_base/

python inference.py \
    --pretrained_pipeline_name_or_path "models/Boogu-Image-0.1-Base" \
    --instruction "一幅国风琉金风格的山水画作，展现了桂林山水在金光普照下的壮丽景象。远山层叠，江水如镜，山峰边缘勾勒着发光的金色线条。画面采用石青石绿岩彩与鎏金质感相结合，局部有厚涂油画笔触，空中飘浮着金色粒子，营造出梦幻朦胧而又磅礴大气的意境。" \
    --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --text_guidance_scale 4.0 \
    --output_image_path "outputs/test_base/out_1.png" \
    --device "$device"
