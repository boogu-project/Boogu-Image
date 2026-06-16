# !/bin/bash

export HF_MODULES_CACHE="$(pwd)/.hf_modules_cache"

###########################################################
# Device settings (Important!)
export device="cuda:0"
export rewriter_device=$device # Must be the same as the main device
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

experiment_name=demo_ti2i_local_encoder_as_rewritter_reasoning

input_image_paths="./input_image_examples/01.png"
# instruction="Add three persimmons with leaves in the bottom right corner."
instruction="帮我在这幅画右下角加上三个带叶子的柿子"

use_rewrite_text_instruction=True
save_rewritten_instruction=True
save_rewritten_instruction_path="outputs/${experiment_name}/${case_name}_rewritten_instruction.json"
custom_local_instruction_rewriter_model=""
rewriter_system_prompt_type="default"
rewriter_max_new_tokens=800
do_sample_for_local_rewriter=False
merge_original_and_rewritten_instructions=True
unload_rewriter_level="keep"

case_name=ti2i_persimmons_local_encoder_as_rewritter_reasoning_${resolution}_steps${num_inference_steps}_tg${text_guidance_scale}_ig${image_guidance_scale}_seed${seed}

mkdir -p outputs/${experiment_name}

python inference.py \
    --seed $seed \
    --pretrained_pipeline_name_or_path $pretrained_pipeline_name_or_path \
    --num_inference_steps $num_inference_steps \
    --use_rewrite_text_instruction $use_rewrite_text_instruction \
    --save_rewritten_instruction $save_rewritten_instruction \
    --custom_local_instruction_rewriter_model "$custom_local_instruction_rewriter_model" \
    --rewriter_system_prompt_type $rewriter_system_prompt_type \
    --rewriter_max_new_tokens $rewriter_max_new_tokens \
    --do_sample_for_local_rewriter $do_sample_for_local_rewriter \
    --merge_original_and_rewritten_instructions $merge_original_and_rewritten_instructions \
    --save_rewritten_instruction_path $save_rewritten_instruction_path \
    --unload_rewriter_level $unload_rewriter_level \
    --input_image_paths $input_image_paths \
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
