# !/bin/bash

export HF_MODULES_CACHE="$(pwd)/.hf_modules_cache"

###########################################################
# Device settings (Important!)
export device="cuda:0"
export rewriter_device=$device
enable_inner_devices_manager=False
###########################################################

# Common settings
num_inference_steps=50
seed=0
text_guidance_scale=4.0
image_guidance_scale=1.0
pretrained_pipeline_name_or_path="models/Boogu-Image-0.1-Edit"
###############VLM input image size (Fixed)#################
max_vlm_input_pil_pixels="147456 147456 147456 147456"
max_vlm_input_pil_side_length=768
###########################################################

#####################2K resolution#########################
resolution=2K
width=2048
height=2048
max_input_image_pixels="4194304 4194304 4194304 4194304"
max_input_image_side_length=4096
##########################################################

# #####################1K resolution#########################
# resolution=1K
# width=1024
# height=1024
# max_input_image_pixels="1048576 1048576 1048576 1048576"
# max_input_image_side_length=2048
# ##########################################################

experiment_name=demo_ti2i_remote_reasoning

input_image_paths="./input_image_examples/04.jpg"
instruction="Change the color of the vehicle to blue."
use_rewrite_text_instruction=True
use_dashscope_remote_rewriting=True
dashscope_remote_rewriting_model="qwen-vl-max-latest"
dashscope_base_http_api_url="https://dashscope.aliyuncs.com/api/v1"
dashscope_api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxx"

save_rewritten_instruction=True
save_rewritten_instruction_path="outputs/${experiment_name}/${case_name}_rewritten_instruction.json"

case_name=ti2i_rm_car_remote_reasoning_${resolution}_steps${num_inference_steps}_tg${text_guidance_scale}_ig${image_guidance_scale}_seed${seed}

mkdir -p outputs/${experiment_name}

python inference.py \
    --seed $seed \
    --pretrained_pipeline_name_or_path $pretrained_pipeline_name_or_path \
    --num_inference_steps $num_inference_steps \
    --input_image_paths $input_image_paths \
    --use_rewrite_text_instruction $use_rewrite_text_instruction \
    --use_dashscope_remote_rewriting $use_dashscope_remote_rewriting \
    --dashscope_remote_rewriting_model $dashscope_remote_rewriting_model \
    --dashscope_base_http_api_url $dashscope_base_http_api_url \
    --dashscope_api_key $dashscope_api_key \
    --save_rewritten_instruction $save_rewritten_instruction \
    --save_rewritten_instruction_path $save_rewritten_instruction_path \
    --height $height \
    --width $width \
    --max_input_image_pixels $max_input_image_pixels \
    --max_input_image_side_length $max_input_image_side_length \
    --max_vlm_input_pil_pixels $max_vlm_input_pil_pixels \
    --max_vlm_input_pil_side_length $max_vlm_input_pil_side_length \
    --text_guidance_scale $text_guidance_scale \
    --image_guidance_scale $image_guidance_scale \
    --instruction "$instruction" \
    --output_image_path outputs/${experiment_name}/${case_name}.png \
    --num_images_per_instruction 1 \
    --enable_inner_devices_manager "$enable_inner_devices_manager" \
    --device "$device" \
    --rewriter_device "$rewriter_device"

echo "${instruction}" > outputs/${experiment_name}/${case_name}_prompt_text.txt
