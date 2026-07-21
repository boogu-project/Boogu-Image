import argparse
import re
from typing import List, Optional


_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(?::\d+)?|npu(?::\d+)?)$")


def get_device_validator(additional_types: Optional[List[str]] = None):
    """
    Factory function that returns a validator for device arguments.

    Base supported formats: 'cpu', 'cuda[:x]', or 'npu[:x]'.
    Additional formats can be provided via `additional_types` (e.g., ['auto']).
    """
    # Initialize as an empty list if None is provided
    if additional_types is None:
        additional_types = []

    def validate_device_format(value: str):
        """
        Validates if the device parameter format is correct.
        """
        # If the user input is an empty string, return None (preserves original logic)
        if not value:
            return None

        value = value.lower()
        if _DEVICE_PATTERN.fullmatch(value):
            return value

        # Check if the value is in the additionally allowed types (e.g., 'auto')
        if value in additional_types:
            return value

        # If it doesn't match any allowed format, raise ArgumentTypeError.
        # argparse will automatically catch this and print a user-friendly error message.
        allowed_msg = "'cpu', 'cuda[:x]', or 'npu[:x]'"
        if additional_types:
            allowed_msg += f", or one of {additional_types}"

        raise argparse.ArgumentTypeError(
            f"Invalid device format: '{value}'. Must be {allowed_msg}."
        )

    return validate_device_format


def validate_device_and_offload_strategy_compatibility(
    device: str,
    enable_sequential_cpu_offload_flag: bool,
    enable_model_cpu_offload_flag: bool,
    enable_group_offload_flag: bool,
) -> bool:
    """
    Validate whether the device and offload strategy are compatible.
    """
    if device is None:
        return False

    def _normalize_bool_flag(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"true", "t", "1", "yes", "y", "on"}:
                return True
            if value in {"false", "f", "0", "no", "n", "off"}:
                return False
        return None

    offload_flags = [
        _normalize_bool_flag(enable_sequential_cpu_offload_flag),
        _normalize_bool_flag(enable_model_cpu_offload_flag),
        _normalize_bool_flag(enable_group_offload_flag),
    ]

    # All offload flags must be explicitly set to valid boolean values.
    if any(flag is None for flag in offload_flags):
        return False

    # Only one automatic offload strategy can be active at a time.
    if sum(int(flag) for flag in offload_flags) > 1:
        return False

    device = str(device).strip().lower()
    if not _DEVICE_PATTERN.fullmatch(device):
        return False

    # The existing Diffusers/Accelerate offload path is only validated on CUDA.
    if any(offload_flags) and not device.startswith("cuda"):
        return False

    return True
