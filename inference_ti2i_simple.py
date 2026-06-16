# Resolution-related arguments:
#
# ## Common rules
# - All height/width-like values are adjusted internally to multiples of 16.
# - `height` and `width` describe the final output size. In practice, they are
#   the values passed to `F.interpolate(image, size=(height, width))` before
#   saving.
# - The hidden noise latents used for generation may be sized from several
#   internal factors, not only from `height` and `width`.
# - For both T2I and TI2I, the noise image for generation is treated as an input
#   image and is resized to satisfy both `max_input_image_pixels` and
#   `max_input_image_side_length`.
# - In all cases, set `max_input_image_pixels` and
#   `max_input_image_side_length` carefully. These limits are critical for
#   output sharpness and effective generation resolution.
#
# ## For T2I
# For any batch size, the final output images share one 16-aligned size, decided
# by `height` and `width`. Set `height` and `width` manually and correctly.
#
# If `height=width=2028`, `max_input_image_pixels=1024*1024`, and
# `max_input_image_side_length=1024`, generation runs on a `1024x1024` noise
# image and only upsamples to the 16-aligned final size (`2048x2048`) at the
# output stage (by "interpolate"). The result may therefore be less sharp than expected.
#
# ## For TI2I
# - With `align_res=True` and `batch_size=1` (usually one reference image per
#   sample), `height` and `width` may be `None`. The output aspect ratio follows
#   the first reference image. The final size is derived from that reference
#   image while respecting `max_input_image_pixels` and
#   `max_input_image_side_length`.
# - With `batch_size>1`, set `height` and `width` manually and correctly. The
#   whole batch will share the same final size, and the noise-sizing logic
#   matches T2I.
#
# ## Recommended manual setting
# Keep `max_input_image_pixels` aligned with the requested resolution:
# ```
# max_input_image_pixels=height*width
# max_input_image_side_length=2*max(height, width)
# ```
# These two limits are important in every mode. Setting them too low can produce
# a large but blurry upsampled image.

## =========================== Supported Task Scope ===========================
#  !! Our model focuses on editing with ONE reference image per sample.
## =========================== ++++++++++++++++++++ ===========================
import os

os.environ["device"] = (
    "cuda:0"  # Required environment variable for the pipeline to run on GPU.
)

import torch
from PIL import Image
from typing import List, Union
from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline

# Load the pipeline (the checkpoint for text+image-to-image generation)
pipe = BooguImagePipeline.from_pretrained(
    "models/Boogu-Image-0.1-Edit",
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


# Generate image
instruction: Union[str, List[str]] = [
    "帮我在这幅画右下角加上三个带叶子的柿子。"
]  # List[str] or str: [batch_size] or str (batch_size=1)
# instruction: Union[str, List[str]]= ["Add three persimmons with leaves in the bottom right corner."]
negative_instruction: Union[str, List[str]] = ""  # Optional, for CFG.

image = pipe(
    instruction=instruction,
    input_image_paths=input_image_paths,
    input_images=input_images,
    negative_instruction=negative_instruction,
    # Final output size. Set both manually for T2I or multi-sample TI2I.
    # For single-sample TI2I, they may be None and follow the reference image.
    # Resolved values are aligned to multiples of 16.
    height=None,
    width=None,
    # Max input pixels. Larger images are resized with aspect ratio preserved.
    # The resized sides are aligned to multiples of 16.
    max_input_image_pixels=2048 * 2048,  # Maximum value used during pretraining.
    # Max input side length. Larger images are resized with aspect ratio preserved.
    # The resized sides are aligned to multiples of 16.
    max_input_image_side_length=2048 * 2,  # Maximum value used during pretraining.
    # The output size is derived from the reference image while respecting
    # the  `max_input_image_pixels` and `max_input_image_side_length`
    # when `batch_size=1` (and only one reference image per sample) and `align_res=True`.
    align_res=True,
    num_inference_steps=50,
    text_guidance_scale=4.0,
    image_guidance_scale=1.0,
    generator=torch.Generator(os.environ.get("device", "cuda:0")).manual_seed(0),
).images[0]

image.save("./outputs/ti2i_example.png")
