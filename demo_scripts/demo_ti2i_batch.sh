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

# #####################2K resolution#########################
# resolution=2K
# width=2048
# height=2048
# max_input_image_pixels="4194304 4194304 4194304 4194304"
# max_input_image_side_length=4096
# ##########################################################

#####################1K resolution#########################
resolution=1K
width=1024
height=1024
max_input_image_pixels="1048576 1048576 1048576 1048576"
max_input_image_side_length=2048
##########################################################

experiment_name=demo_ti2i_batch
instruction=""

# ## Batch Inference or Not
use_batch_inference=True

# # If using `batch_data_config_path`, `instruction` and `input_image_paths` will be ignored.
batch_data_config_path="./batch_data_samples/ti2i_batch_data_sample.yml" # disable when use_batch_inference is False

case_name=ti2i_batch_example1_${resolution}_steps${num_inference_steps}_tg${text_guidance_scale}_ig${image_guidance_scale}_seed${seed}

mkdir -p outputs/${experiment_name}

python inference.py \
    --seed $seed \
    --pretrained_pipeline_name_or_path $pretrained_pipeline_name_or_path \
    --num_inference_steps $num_inference_steps \
    --use_batch_inference $use_batch_inference \
    --batch_data_config_path $batch_data_config_path \
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
