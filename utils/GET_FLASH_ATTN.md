# get_flash_attn.py

Automatically installs the correct `flash-attn` wheel for your environment.

## Requirements
- Python, PyTorch with CUDA already installed
- Linux x86_64

## Usage

```bash
# Auto: detect environment → download prebuilt wheel → fallback to source compile
python utils/get_flash_attn.py

# Force source compilation (skip download attempts)
python utils/get_flash_attn.py --build
```

## How it works

1. **mjun0812 prebuild wheels** — searches [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) for a wheel matching your torch / CUDA / Python version
2. **Dao-AILab official releases** — tries the official [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) releases (both cxx11abi variants)
3. **Source compilation** — if no prebuilt wheel is found, compiles from source via `pip install flash-attn --no-build-isolation` (longer time)

