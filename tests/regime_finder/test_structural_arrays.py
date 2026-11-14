"""Exact native bytes, shared buffers, view bounds and mutation observations."""

import pytest
import importlib.util

from _lcm_regime_finder.structural_arrays import NativeWriter
from _lcm_regime_finder.structural_check import decode
from _lcm_regime_finder.structural_transport import reseal


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="native NumPy environment required"
)
def test_shared_contiguous_and_negative_stride_views() -> None:
    import numpy as np

    base = np.arange(12, dtype="<i4")
    source = [base, base[2:10:2], base[::-1], base]
    bundle = NativeWriter().finish(source, {})
    result = decode(bundle=bundle)
    assert (result[0]) is (result[3])
    assert (result[0].buffer) is (result[1].buffer)
    assert (result[0].buffer) is (result[2].buffer)
    assert (result[1].offset) == (8)
    assert (result[2].strides) == ((-4,))
    assert (b"".join(result[0].buffer.iter_bytes())) == (base.tobytes())


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="native NumPy environment required"
)
def test_nan_signed_zero_and_endianness_are_not_json_numbers() -> None:
    import numpy as np

    data = bytes.fromhex("00000000000000800100c07f")
    source = np.frombuffer(bytearray(data), dtype="<f4").copy()
    bundle = NativeWriter().finish(source, {})
    result = decode(bundle=bundle)
    assert (b"".join(result.buffer.iter_bytes())) == (data)
    assert (result.dtype) == ("<f4")


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="native NumPy environment required"
)
def test_real_buffer_mutation_after_read_is_rejected() -> None:
    import numpy as np

    source = np.arange(20000, dtype="<i4")

    def mutate():
        source[-1] = -1

    with pytest.raises(ValueError, match="changed"):
        NativeWriter().finish(source, {}, mutate)


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="native NumPy environment required"
)
def test_rehashed_layout_cannot_escape_buffer() -> None:
    import numpy as np

    bundle = NativeWriter().finish(np.arange(3, dtype="<i4"), {})
    bundle["manifest"]["records"][0][5] = 10**9
    reseal(bundle=bundle)
    with pytest.raises(ValueError, match="escapes"):
        decode(bundle=bundle)


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="native NumPy environment required"
)
def test_equal_distinct_arrays_do_not_acquire_shared_owners() -> None:
    import numpy as np

    first = np.arange(4, dtype="<i4")
    result = decode(bundle=NativeWriter().finish([first, first.copy()], {}))
    assert (result[0].buffer) is not (result[1].buffer)


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="native NumPy environment required"
)
def test_object_dtype_is_explicitly_unsupported() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="dtype"):
        NativeWriter().finish(np.array([object()]), {})


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="native NumPy environment required"
)
@pytest.mark.skipif(
    not (importlib.util.find_spec("jax")), reason="actual JAX runtime required"
)
def test_actual_jax_array_retains_exact_value_and_object_alias() -> None:
    import numpy as np
    import jax.numpy as jnp

    source = jnp.arange(12000, dtype=jnp.int32)
    bundle = NativeWriter().finish([source, source], {})
    result = decode(bundle=bundle)
    assert (result[0]) is (result[1])
    assert (b"".join(result[0].buffer.iter_bytes())) == (np.asarray(source).tobytes())
    assert bundle["manifest"]["storage_obligations"]
