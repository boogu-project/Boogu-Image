import torch

from boogu.models.attention_processor import _prepare_sdpa_padding_mask


def test_prepare_sdpa_padding_mask_expands_query_dimension():
    mask = torch.tensor([[True, True, False], [True, False, False]])

    expanded = _prepare_sdpa_padding_mask(mask, batch_size=2, query_length=4)

    assert expanded.shape == (2, 1, 4, 3)
    assert torch.equal(expanded[:, 0, 0], mask)
    assert torch.equal(expanded[:, 0, 3], mask)


def test_expanded_padding_mask_matches_broadcast_sdpa_mask():
    torch.manual_seed(0)
    query = torch.randn(2, 2, 4, 8)
    key = torch.randn(2, 2, 3, 8)
    value = torch.randn(2, 2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    broadcast_mask = mask.view(2, 1, 1, 3)
    expanded_mask = _prepare_sdpa_padding_mask(mask, 2, 4)

    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=broadcast_mask
    )
    actual = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=expanded_mask
    )

    torch.testing.assert_close(actual, expected)
