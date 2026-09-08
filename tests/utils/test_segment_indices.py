"""Tests for `get_segment_indices_from_n`.

Kept separate from `test_maths.py`, which currently fails to import (it asks for
`decompose_tensor` and friends, which `diep.utils.maths` does not define). That breakage
predates the three-body index-space fix.

The helper indexes the three-body scatter. Its previous cumsum-based implementation was
correct only when every segment was non-empty; after the index-space fix `n_triple_ij` has
one entry per parent bond and is zero for every bond in no triple, so empty segments are
routine and the old form would either raise or scatter into the wrong bonds.
"""

from __future__ import annotations

import torch
from diep.utils.maths import get_segment_indices_from_n


def test_basic():
    assert get_segment_indices_from_n(torch.tensor([2, 3])).tolist() == [0, 0, 1, 1, 1]
    assert get_segment_indices_from_n(torch.tensor([2, 3, 1])).tolist() == [0, 0, 1, 1, 1, 2]


def test_empty_segments():
    assert get_segment_indices_from_n(torch.tensor([2, 0, 3])).tolist() == [0, 0, 2, 2, 2]  # interior zero
    assert get_segment_indices_from_n(torch.tensor([2, 3, 0])).tolist() == [0, 0, 1, 1, 1]  # trailing zero
    assert get_segment_indices_from_n(torch.tensor([0, 2])).tolist() == [1, 1]  # leading zero
    assert get_segment_indices_from_n(torch.tensor([0, 0, 0])).tolist() == []  # all empty
    assert get_segment_indices_from_n(torch.tensor([], dtype=torch.int32)).tolist() == []


def test_returns_int64():
    """Tensor.scatter_add_ rejects int32 index tensors, and n_triple_ij is int32."""
    assert get_segment_indices_from_n(torch.tensor([2, 3], dtype=torch.int32)).dtype == torch.int64
    assert get_segment_indices_from_n(torch.tensor([2, 0, 3], dtype=torch.int32)).dtype == torch.int64


def test_usable_as_a_scatter_add_index():
    ns = torch.tensor([2, 0, 3], dtype=torch.int32)
    segment_ids = get_segment_indices_from_n(ns)
    src = torch.arange(int(ns.sum()), dtype=torch.float32)
    out = torch.zeros(ns.numel()).scatter_add_(0, segment_ids, src)
    assert out.tolist() == [1.0, 0.0, 9.0]


def test_identical_to_the_cumsum_form_when_no_segment_is_empty():
    """The replacement is provably identical to the original wherever the original was
    correct, so it cannot change any result that was previously right."""
    torch.manual_seed(0)
    for _ in range(50):
        length = int(torch.randint(1, 20, (1,)).item())
        ns = torch.randint(1, 6, (length,), dtype=torch.int32)
        segments = torch.zeros(int(ns.sum()), dtype=torch.int32)
        segments[ns.cumsum(0)[:-1]] = 1
        assert torch.equal(segments.cumsum(0), get_segment_indices_from_n(ns))


def test_device_is_preserved():
    ns = torch.tensor([2, 0, 3])
    assert get_segment_indices_from_n(ns).device == ns.device
    if torch.cuda.is_available():
        ns_cuda = ns.cuda()
        assert get_segment_indices_from_n(ns_cuda).device == ns_cuda.device
