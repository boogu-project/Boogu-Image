#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-boogu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements/torch2.7-cu126.txt}"
CONDA_PYTHONWARNINGS="${CONDA_PYTHONWARNINGS:-ignore}"

if ! command -v conda > /dev/null 2>&1; then
    echo "conda not found on PATH. Install Miniconda/Miniforge first." >&2
    exit 1
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "Requirements file not found: $REQUIREMENTS_FILE" >&2
    exit 1
fi

BASE_ENV_NAME="$ENV_NAME"
EXISTING_ENVS="$(PYTHONWARNINGS="$CONDA_PYTHONWARNINGS" conda env list | awk '{print $1}')"
suffix=0
while grep -qx "$ENV_NAME" <<< "$EXISTING_ENVS"; do
    suffix=$((suffix + 1))
    ENV_NAME="${BASE_ENV_NAME}${suffix}"
done

if [[ "$ENV_NAME" != "$BASE_ENV_NAME" ]]; then
    echo "Environment already exists: $BASE_ENV_NAME; creating $ENV_NAME instead."
fi

PYTHONWARNINGS="$CONDA_PYTHONWARNINGS" conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y

PYTHONWARNINGS="$CONDA_PYTHONWARNINGS" conda run -s -n "$ENV_NAME" python -m pip install --upgrade pip
PYTHONWARNINGS="$CONDA_PYTHONWARNINGS" conda run -s -n "$ENV_NAME" python -m pip install -r "$REQUIREMENTS_FILE"
PYTHONWARNINGS="$CONDA_PYTHONWARNINGS" conda run -s -n "$ENV_NAME" python -m pip install -e .

if [[ -f utils/get_flash_attn.py ]]; then
    PYTHONWARNINGS="$CONDA_PYTHONWARNINGS" conda run -s -n "$ENV_NAME" python utils/get_flash_attn.py
fi

echo
echo "Done."
echo "Activate it with: conda activate $ENV_NAME"
