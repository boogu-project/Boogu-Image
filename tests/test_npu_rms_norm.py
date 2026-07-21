import torch

from boogu.ops.simple_layer_norm import NpuRMSNorm


def test_npu_rms_norm_cpu_fallback_matches_torch():
    expected = torch.nn.RMSNorm(8, eps=1e-6)
    actual = NpuRMSNorm(8, eps=1e-6)
    actual.load_state_dict(expected.state_dict())
    inputs = torch.randn(2, 4, 8)

    torch.testing.assert_close(actual(inputs), expected(inputs))
    assert actual.state_dict().keys() == expected.state_dict().keys()


def test_npu_rms_norm_without_weight_uses_torch_fallback():
    expected = torch.nn.RMSNorm(8, eps=1e-6, elementwise_affine=False)
    actual = NpuRMSNorm(8, eps=1e-6, elementwise_affine=False)
    inputs = torch.randn(2, 4, 8)

    torch.testing.assert_close(actual(inputs), expected(inputs))
