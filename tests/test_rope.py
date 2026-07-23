import torch

from boogu.models.transformers.rope import (
    _gather_rotary_freqs_as_real,
    _rotary_freqs_dtype,
)


def test_npu_rotary_frequencies_use_float32(monkeypatch):
    monkeypatch.setenv("device", "npu:0")

    assert _rotary_freqs_dtype() == torch.float32


def test_real_view_gather_matches_complex_gather():
    freqs = [
        torch.polar(torch.ones(5, 2), torch.randn(5, 2)),
        torch.polar(torch.ones(6, 2), torch.randn(6, 2)),
    ]
    ids = torch.tensor([[[0, 1], [3, 4]]], dtype=torch.int64)

    actual = _gather_rotary_freqs_as_real(freqs, ids, (4, 4))
    expected = torch.cat(
        [freqs[0][ids[:, :, 0]], freqs[1][ids[:, :, 1]]], dim=-1
    )

    torch.testing.assert_close(actual, expected)
