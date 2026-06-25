import os

os.environ["device"] = "cuda:0"

import torch

from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline

# Load the pipeline
pipe = BooguImageTurboPipeline.from_pretrained(
    "models/Boogu-Image-0.1-Turbo",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.to(os.environ.get("device", "cuda:0"))

# Generate image (few-step DMD student inference, no CFG)
instruction = [
    "一幅国风琉金风格的山水画作，展现了桂林山水在金光普照下的壮丽景象。远山层叠，江水如镜，山峰边缘勾勒着发光的金色线条。画面采用石青石绿岩彩与鎏金质感相结合，局部有厚涂油画笔触，空中飘浮着金色粒子，营造出梦幻朦胧而又磅礴大气的意境。"
]

image = pipe(
    instruction=instruction,
    negative_instruction="",
    empty_instruction="",
    height=1024,
    width=1024,
    num_inference_steps=4,
    text_guidance_scale=1.0,
    image_guidance_scale=1.0,
    empty_instruction_guidance_scale=0.0,
    use_dmd_student_inference=True,
    dmd_conditioning_sigma=0.001,
    generator=torch.Generator(os.environ.get("device", "cuda:0")).manual_seed(42),
).images[0]

image.save("./outputs/turbo_t2i_example.png")
