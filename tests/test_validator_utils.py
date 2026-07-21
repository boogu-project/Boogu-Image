import argparse

import pytest

from boogu.utils.validator_utils import (
    get_device_validator,
    validate_device_and_offload_strategy_compatibility,
)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:3", "npu", "npu:1"])
def test_device_validator_accepts_supported_devices(device):
    assert get_device_validator()(device) == device


def test_device_validator_rejects_unknown_device():
    with pytest.raises(argparse.ArgumentTypeError):
        get_device_validator()("xpu:0")


def test_npu_is_valid_without_offload():
    assert validate_device_and_offload_strategy_compatibility(
        "npu:0", False, False, False
    )


def test_npu_offload_is_rejected_until_validated():
    assert not validate_device_and_offload_strategy_compatibility(
        "npu:0", False, True, False
    )
