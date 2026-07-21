import argparse
import json
import os
from pathlib import Path


DEFAULT_MODEL_PATH = "models/Boogu-Image-0.1-Turbo"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Boogu Turbo BF16 on Ascend NPU.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--prompt",
        default="一幅国风鎏金风格的桂林山水画，金色光线勾勒群山与江面。",
    )
    parser.add_argument("--output", default="outputs/turbo_npu.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def verify_checkpoint(model_path: Path) -> None:
    from safetensors import safe_open

    checkpoint_files = []
    index_files = [
        model_path / "mllm/model.safetensors.index.json",
        model_path / "transformer/diffusion_pytorch_model.safetensors.index.json",
    ]
    for index_file in index_files:
        index = json.loads(index_file.read_text())
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
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()

    os.environ["device"] = args.device

    verify_checkpoint(model_path)
    if args.check_only:
        print(f"Checkpoint is valid: {model_path}")
        return

    import torch
    import torch_npu  # noqa: F401
    from transformers import Qwen3VLForConditionalGeneration

    from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline

    torch.npu.set_device(args.device)
    mllm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path / "mllm",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    pipe = BooguImageTurboPipeline.from_pretrained(
        model_path,
        mllm=mllm,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    pipe.to(args.device)

    generator = torch.Generator(args.device).manual_seed(args.seed)
    with torch.inference_mode():
        image = pipe(
            instruction=[args.prompt],
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
            generator=generator,
            device=args.device,
        ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
