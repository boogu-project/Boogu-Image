# !/bin/bash

export HF_MODULES_CACHE="$(pwd)/.hf_modules_cache"

###########################################################
# Device settings (Important!)
export device="cuda:1"
export rewriter_device=$device
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

experiment_name=demo_group_offload_t2i
enable_group_offload_flag=True
# instruction="A street photography shot of an elderly scavenger with a deeply weathered and wrinkled face in the center of the frame. The scene includes a trash can and a traffic light in the background. Shot on Leica camera, high photographic texture, classic street photography aesthetic, cinematic lighting, photorealistic."
instruction="生成一张节拍的照片，要有垃圾桶，红绿灯。画面中心有一个拾荒的老人，满脸沧桑。 照片要有摄影质感，以及徕卡相机的街拍质感。"

# case_name=t2i_oldman_en_group_offload_${resolution}_steps${num_inference_steps}_tg${text_guidance_scale}_ig${image_guidance_scale}_seed${seed}
case_name=t2i_oldman_zh_group_offload_${resolution}_steps${num_inference_steps}_tg${text_guidance_scale}_ig${image_guidance_scale}_seed${seed}

mkdir -p outputs/${experiment_name}

python inference.py \
    --seed $seed \
    --pretrained_pipeline_name_or_path $pretrained_pipeline_name_or_path \
    --num_inference_steps $num_inference_steps \
    --enable_group_offload_flag $enable_group_offload_flag \
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
