## =========================== Supported Task Scope ===========================
#  !! This is the few-step DMD-distilled (turbo) edit pipeline.
#  !! It focuses on editing with ONE reference image per sample.
#  !! DMD path runs without CFG: text/image_guidance_scale=1.0,
#     empty_instruction_guidance_scale=0.0.
## =========================== ++++++++++++++++++++ ===========================
import os

os.environ["device"] = (
    "cuda:0"  # Required environment variable for the pipeline to run on GPU.
)

import torch
from PIL import Image
from typing import List, Union
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline

# Load the turbo (DMD few-step) edit pipeline.
pipe = BooguImageTurboPipeline.from_pretrained(
    "models/Boogu-Image-0.1-Edit-Turbo",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.to(os.environ.get("device", "cuda:0"))

# Load input image(s)
input_image_paths: Union[List[List[str]], List[str]] = [
    ["./input_image_examples/01.png"]
]  # [batch_size, num_images_per_sample] or [num_images_per_sample] (batch_size=1)

input_images: Union[
    List[List[Image.Image]], List[Image.Image]
] = []  # [batch_size, num_images_per_sample] (batch_size>1) or [num_images_per_sample] (batch_size=1)
for paths in input_image_paths:
    input_images.append([Image.open(a_path).convert("RGB") for a_path in paths])

# Editing instruction.
instruction: Union[str, List[str]] = [
    "帮我在这幅画右下角加上三个带叶子的柿子。"
]
negative_instruction: Union[str, List[str]] = ""

# Edit image (few-step DMD student inference, no CFG).
image = pipe(
    instruction=instruction,
    input_image_paths=input_image_paths,
    input_images=input_images,
    negative_instruction=negative_instruction,
    # Final output size. For single-sample TI2I they may be None and follow
    # the reference image (with align_res=True). Resolved values are aligned
    # to multiples of 16.
    height=None,
    width=None,
    align_res=True,
    # Max input pixels / side length. Larger images are resized with aspect
    # ratio preserved and aligned to multiples of 16.
    max_input_image_pixels=2048 * 2048,  # Maximum value used during pretraining.
    max_input_image_side_length=2048 * 2,  # Maximum value used during pretraining.
    num_inference_steps=4,
    text_guidance_scale=1.0,
    image_guidance_scale=1.0,
    empty_instruction_guidance_scale=0.0,
    use_dmd_student_inference=True,
    dmd_conditioning_sigma=0.0,
    generator=torch.Generator(os.environ.get("device", "cuda:0")).manual_seed(42),
).images[0]

image.save("./outputs/ti2i_turbo_example.png")
