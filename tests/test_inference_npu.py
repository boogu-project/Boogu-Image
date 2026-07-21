import argparse
import json

import pytest

from inference_npu import (
    STANDARD_PIPELINE,
    TURBO_PIPELINE,
    parse_npu_device,
    read_pipeline_class_name,
    resolve_inference_config,
    select_pipeline_class_name,
)


@pytest.mark.parametrize(
    ("pipeline_class_name", "has_input_image", "task", "steps", "text_cfg", "sigma"),
    [
        (STANDARD_PIPELINE, False, "t2i", 50, 4.0, None),
        (STANDARD_PIPELINE, True, "ti2i", 50, 4.0, None),
        (TURBO_PIPELINE, False, "t2i", 4, 1.0, 0.001),
        (TURBO_PIPELINE, True, "ti2i", 4, 1.0, 0.0),
    ],
)
def test_resolve_inference_config(
    pipeline_class_name, has_input_image, task, steps, text_cfg, sigma
):
    config = resolve_inference_config(pipeline_class_name, has_input_image)

    assert config.task == task
    assert config.num_inference_steps == steps
    assert config.text_guidance_scale == text_cfg
    assert config.image_guidance_scale == 1.0
    assert config.dmd_conditioning_sigma == sigma


def test_read_pipeline_class_name(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": TURBO_PIPELINE}), encoding="utf-8"
    )

    assert read_pipeline_class_name(tmp_path) == TURBO_PIPELINE


def test_read_pipeline_class_name_rejects_unknown_class(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "UnknownPipeline"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported pipeline class"):
        read_pipeline_class_name(tmp_path)


def test_select_pipeline_class_name_can_force_turbo(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": STANDARD_PIPELINE}), encoding="utf-8"
    )

    assert select_pipeline_class_name(tmp_path, force_turbo=True) == TURBO_PIPELINE


@pytest.mark.parametrize("device", ["npu", "npu:0", "npu:15"])
def test_parse_npu_device_accepts_npu(device):
    assert parse_npu_device(device) == device


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "npu:-1", "npu:x"])
def test_parse_npu_device_rejects_other_devices(device):
    with pytest.raises(argparse.ArgumentTypeError, match="device must be"):
        parse_npu_device(device)
