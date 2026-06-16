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
#  !! The model's maximum native resolution is 2K.
## =========================== ++++++++++++++++++++ ===========================

import os

os.environ["device"] = (
    "cuda:0"  # Required environment variable for the pipeline to run on GPU.
)

import torch
from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline

# Load the pipeline
pipe = BooguImagePipeline.from_pretrained(
    "models/Boogu-Image-0.1-Base",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.to(os.environ.get("device", "cuda:0"))

# Generate image
# instruction="A street photography shot of an elderly scavenger with a deeply weathered and wrinkled face in the center of the frame. The scene includes a trash can and a traffic light in the background. Shot on Leica camera, high photographic texture, classic street photography aesthetic, cinematic lighting, photorealistic."
instruction = "生成一张节拍的照片，要有垃圾桶，红绿灯。画面中心有一个拾荒的老人，满脸沧桑。 照片要有摄影质感，以及徕卡相机的街拍质感。"
negative_instruction = ""  # Optional, for CFG.

# The output image size. Must be a multiple of 16.
output_image_height = 2048
output_image_width = 2048


image = pipe(
    instruction=instruction,
    negative_instruction=negative_instruction,
    height=output_image_height,
    width=output_image_width,
    # The noise image used for generation is treated as an input image, so it
    # must satisfy `max_input_image_pixels` and `max_input_image_side_length`.
    # The settings below match the requested output size. If these limits are
    # too small, generation runs at a lower native resolution and may look
    # blurry after upsampling.
    max_input_image_pixels=output_image_height * output_image_width,
    max_input_image_side_length=2 * max(output_image_height, output_image_width),
    num_inference_steps=50,
    text_guidance_scale=4.0,
    generator=torch.Generator(os.environ.get("device", "cuda:0")).manual_seed(0),
).images[0]

image.save("./outputs/t2i_example.png")
