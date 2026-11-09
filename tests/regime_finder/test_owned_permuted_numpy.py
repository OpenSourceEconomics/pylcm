"""Paper-independent packed NumPy ownership regression and refusal controls."""
import itertools
import struct

import numpy as np
import pytest

from _lcm_regime_finder.structural_arrays import NativeWriter
from _lcm_regime_finder.structural_check import Reader
from _lcm_regime_finder.structural_events import compare_source, source_anchor


@pytest.mark.parametrize('axes', list(itertools.permutations(range(3))))
def test_owned_axis_permutation_preserves_values_layout_and_aliases(axes: tuple[int, ...]) -> None:
    source = np.arange(24, dtype='<f8').reshape(2, 3, 4).transpose(axes).copy(order='K')
    assert source.base is None and source.flags.owndata
    view = source[::-1]
    roots = [source, view, source]
    anchor = source_anchor(roots, source_context={}, transaction={'kind': 'exclusive-owner', 'authority': 'test-owned arrays'})
    bundle = NativeWriter().finish(roots, anchor)
    reader = Reader(bundle)
    result = reader.decode()
    assert result[0] is result[2]
    assert result[0].buffer is result[1].buffer
    assert result[0].shape == source.shape
    assert result[0].strides == source.strides
    assert result[1].strides == view.strides
    for original, decoded in zip((source, view), result[:2], strict=True):
        raw = b''.join(decoded.buffer.iter_bytes())
        for index in np.ndindex(original.shape):
            offset = decoded.offset + sum(i * stride for i, stride in zip(index, decoded.strides, strict=True))
            assert struct.unpack_from('<d', raw, offset)[0] == float(original[index])
    compare_source(reader, anchor)


@pytest.mark.parametrize('strides', [(16, 0), (8, 8), (0, 24)])
def test_owned_holey_or_overlapping_storage_still_refuses(strides: tuple[int, int]) -> None:
    source = np.ndarray((2, 2), dtype='<f8', strides=strides)
    assert source.base is None and source.flags.owndata
    with pytest.raises(ValueError, match='unsupported native array storage'):
        NativeWriter().finish(source, {})
    with pytest.raises(ValueError, match='Unsupported source buffer ownership'):
        source_anchor(source, source_context={}, transaction={})


def test_external_storage_still_refuses() -> None:
    source = np.frombuffer(bytearray(32), dtype='<f8')
    with pytest.raises(ValueError, match='unsupported native array storage'):
        NativeWriter().finish(source, {})
    with pytest.raises(ValueError, match='Unsupported source buffer ownership'):
        source_anchor(source, source_context={}, transaction={})


def test_reinterpreted_view_uses_the_owners_itemsize() -> None:
    owner = np.arange(24, dtype='<f8').reshape(2, 3, 4).transpose(1, 0, 2).copy(order='K')
    view = owner.view('<u4')
    roots = [owner, view]
    anchor = source_anchor(roots, source_context={}, transaction={})
    reader = Reader(NativeWriter().finish(roots, anchor))
    result = reader.decode()
    assert result[0].buffer is result[1].buffer
    raw = b''.join(result[1].buffer.iter_bytes())
    for index in np.ndindex(view.shape):
        offset = result[1].offset + sum(i * stride for i, stride in zip(index, result[1].strides, strict=True))
        assert struct.unpack_from('<I', raw, offset)[0] == int(view[index])
    compare_source(reader, anchor)


def test_owned_permutation_mutation_still_refuses() -> None:
    source = np.arange(24, dtype='<f8').reshape(2, 3, 4).transpose(1, 2, 0).copy(order='K')
    def mutate() -> None:
        source[0, 0, 0] = -1
    with pytest.raises(ValueError, match='changed'):
        NativeWriter().finish(source, {}, mutate)
