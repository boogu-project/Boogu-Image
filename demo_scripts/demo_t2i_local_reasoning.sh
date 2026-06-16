# !/bin/bash

export HF_MODULES_CACHE="$(pwd)/.hf_modules_cache"

###########################################################
# Device settings (Important!)
export device="cuda:0"
export rewriter_device='cuda:1' # cuda:0 is the main device, cuda:1 is the rewriter device
enable_inner_devices_manager=False
###########################################################

# Common settings
num_inference_steps=50
seed=0
text_guidance_scale=4.0
image_guidance_scale=1.0
pretrained_pipeline_name_or_path="models/Boogu-Image-0.1-Base"
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

experiment_name=demo_t2i_local_reasoning
# instruction="A street photography shot of an elderly scavenger with a deeply weathered and wrinkled face in the center of the frame. The scene includes a trash can and a traffic light in the background. Shot on Leica camera, high photographic texture, classic street photography aesthetic, cinematic lighting, photorealistic."
instruction="生成一张节拍的照片，要有垃圾桶，红绿灯。画面中心有一个拾荒的老人，满脸沧桑。 照片要有摄影质感，以及徕卡相机的街拍质感。"

use_rewrite_text_instruction=True
save_rewritten_instruction=True
save_rewritten_instruction_path="outputs/${experiment_name}/${case_name}_rewritten_instruction.json"
custom_local_instruction_rewriter_model="/path/to/pipeline/weights/Qwen3-VL-32B-Instruct"
rewriter_system_prompt_type="default"
rewriter_max_new_tokens=800
do_sample_for_local_rewriter=False
merge_original_and_rewritten_instructions=True
unload_rewriter_level="destroy" # "destroy" or "keep" or 'cpu'

case_name=t2i_oldman_zh_local_reasoning_steps${num_inference_steps}_tg${text_guidance_scale}_ig${image_guidance_scale}_seed${seed}

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
