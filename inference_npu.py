import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


STANDARD_PIPELINE = "BooguImagePipeline"
TURBO_PIPELINE = "BooguImageTurboPipeline"


@dataclass(frozen=True)
class InferenceConfig:
    task: str
    num_inference_steps: int
    text_guidance_scale: float
    image_guidance_scale: float
    dmd_conditioning_sigma: float | None


def parse_npu_device(value: str) -> str:
    if re.fullmatch(r"npu(?::\d+)?", value):
        return value
    raise argparse.ArgumentTypeError("device must be 'npu' or 'npu:<index>'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Boogu BF16 inference on Ascend NPU."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Use the Turbo pipeline when checkpoint metadata does not declare it.",
    )
    parser.add_argument("--device", type=parse_npu_device, default="npu:0")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", type=Path, default=Path("outputs/npu.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--text-guidance-scale", type=float)
    parser.add_argument("--image-guidance-scale", type=float)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_pipeline_class_name(model_path: Path) -> str:
    model_index_path = model_path / "model_index.json"
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    class_name = model_index.get("_class_name")
    if class_name not in {STANDARD_PIPELINE, TURBO_PIPELINE}:
        raise ValueError(
            f"Unsupported pipeline class in {model_index_path}: {class_name!r}"
        )
    return class_name


def resolve_inference_config(
    pipeline_class_name: str, has_input_image: bool
) -> InferenceConfig:
    if pipeline_class_name not in {STANDARD_PIPELINE, TURBO_PIPELINE}:
        raise ValueError(
            f"Unsupported pipeline/task combination: {pipeline_class_name}, "
            f"input_image={has_input_image}"
        )

    is_turbo = pipeline_class_name == TURBO_PIPELINE
    return InferenceConfig(
        task="ti2i" if has_input_image else "t2i",
        num_inference_steps=4 if is_turbo else 50,
        text_guidance_scale=1.0 if is_turbo else 4.0,
        image_guidance_scale=1.0,
        dmd_conditioning_sigma=(0.0 if has_input_image else 0.001)
        if is_turbo
        else None,
    )


def select_pipeline_class_name(model_path: Path, force_turbo: bool) -> str:
    pipeline_class_name = read_pipeline_class_name(model_path)
    if force_turbo:
        return TURBO_PIPELINE
    return pipeline_class_name


def verify_checkpoint(model_path: Path) -> None:
    from safetensors import safe_open

    checkpoint_files = []
    index_files = [
        model_path / "mllm/model.safetensors.index.json",
        model_path / "transformer/diffusion_pytorch_model.safetensors.index.json",
    ]
    for index_file in index_files:
        index = json.loads(index_file.read_text(encoding="utf-8"))
        checkpoint_files.extend(
            index_file.parent / filename
            for filename in sorted(set(index["weight_map"].values()))
        )
    checkpoint_files.append(model_path / "vae/diffusion_pytorch_model.safetensors")

    for checkpoint_file in checkpoint_files:
        try:
            with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
                next(iter(handle.keys()))
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint is incomplete or invalid: {checkpoint_file}"
            ) from exc


def main():
    args = parse_args()
    model_path = args.model.resolve()
    input_image_path = args.input_image.resolve() if args.input_image else None
    output_path = args.output.resolve()

    pipeline_class_name = select_pipeline_class_name(model_path, args.turbo)
    config = resolve_inference_config(
        pipeline_class_name, has_input_image=input_image_path is not None
    )
    verify_checkpoint(model_path)
    if args.check_only:
        print(
            f"Checkpoint is valid: {model_path} "
            f"(pipeline={pipeline_class_name}, task={config.task})"
        )
        return

    if input_image_path is not None and not input_image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_image_path}")

    os.environ["device"] = args.device

    import torch
    import torch_npu  # noqa: F401
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration

    if pipeline_class_name == TURBO_PIPELINE:
        from boogu.pipelines.boogu.pipeline_boogu_turbo import (
            BooguImageTurboPipeline as PipelineClass,
        )
    else:
        from boogu.pipelines.boogu.pipeline_boogu import (
            BooguImagePipeline as PipelineClass,
        )

    torch.npu.set_device(args.device)
    print(
        f"Loading {pipeline_class_name} for {config.task.upper()} on {args.device}: "
        f"{model_path}"
    )

    mllm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path / "mllm",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    pipe = PipelineClass.from_pretrained(
        model_path,
        mllm=mllm,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    pipe.to(args.device)

    prompt = args.prompt
    if prompt is None:
        if input_image_path is None:
            prompt = "A cinematic mountain landscape illuminated by golden light."
        else:
            prompt = "Replace the background with a beach while preserving the subject."

    num_inference_steps = (
        args.steps if args.steps is not None else config.num_inference_steps
    )
    text_guidance_scale = (
        args.text_guidance_scale
        if args.text_guidance_scale is not None
        else config.text_guidance_scale
    )
    image_guidance_scale = (
        args.image_guidance_scale
        if args.image_guidance_scale is not None
        else config.image_guidance_scale
    )

    call_kwargs = {
        "instruction": [prompt],
        "negative_instruction": args.negative_prompt,
        "empty_instruction": "",
        "height": args.height,
        "width": args.width,
        "max_input_image_pixels": args.height * args.width,
        "max_input_image_side_length": 2 * max(args.height, args.width),
        "num_inference_steps": num_inference_steps,
        "text_guidance_scale": text_guidance_scale,
        "image_guidance_scale": image_guidance_scale,
        "empty_instruction_guidance_scale": 0.0,
        "generator": torch.Generator(args.device).manual_seed(args.seed),
        "device": args.device,
    }

    if input_image_path is not None:
        with Image.open(input_image_path) as input_image:
            call_kwargs["input_images"] = [[input_image.convert("RGB")]]
        call_kwargs["input_image_paths"] = [[str(input_image_path)]]

    if config.dmd_conditioning_sigma is not None:
        call_kwargs["use_dmd_student_inference"] = True
        call_kwargs["dmd_conditioning_sigma"] = config.dmd_conditioning_sigma

    try:
        with torch.inference_mode():
            image = pipe(**call_kwargs).images[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(output_path)
    finally:
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
