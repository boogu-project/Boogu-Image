import argparse
import os

import cache_dit
import torch

os.environ["device"] = "cuda"
from boogu.models.transformers.transformer_boogu import (
    BooguImageTransformer2DModel,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a transformer checkpoint.")
    parser.add_argument(
        "--model_path", required=True, help="Path to the original model weights."
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Path to save the quantized weights.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bf16",
        choices=["bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype to load the original model with. Default: bf16",
    )
    parser.add_argument(
        "--quant_type",
        default="float8_weight_only",
        help="Quantization type. Default: float8_weight_only",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name in ("float16", "fp16"):
        return torch.float16
    if dtype_name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")


def main():
    args = parse_args()
    torch_dtype = resolve_torch_dtype(args.torch_dtype)

    transformer = BooguImageTransformer2DModel.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    )
    quantize_config = cache_dit.QuantizeConfig(
        quant_type=args.quant_type,
        exclude_layers=["embedder", "embed", "lm_head"],
        regional_quantize=False,
        per_tensor_fallback=True,
    )
    transformer = cache_dit.quantize(
        transformer,
        quantize_config=quantize_config,
    )
    transformer.save_pretrained(args.save_path, safe_serialization=False)


if __name__ == "__main__":
    main()
