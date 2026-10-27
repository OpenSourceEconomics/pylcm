"""A subject population that is not a multiple of the tile width runs one body shape."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.programs import SUBJECT_WIDTH_KEYWORD, _SubjectTiled
from lcm.typing import FloatND, IntND


def test_subject_tiled_traces_its_body_at_the_tile_width_only() -> None:
    """The remainder tile is padded to the tile width, so the body is traced once.

    A second trace at the remainder's shape would compile a second program for
    the last subjects, whose rounding can differ from the full tiles'.
    """
    shapes = []

    def body(*, x: jax.Array, scale: jax.Array) -> jax.Array:
        shapes.append(x.shape)
        return x * scale + 1.0

    tiled = _SubjectTiled(func=body, subject_arg_names=("x",))
    x = jnp.arange(10, dtype=jnp.float32)
    jax.jit(lambda x, s: tiled(x=x, scale=s, **{SUBJECT_WIDTH_KEYWORD: 4}))(
        x, jnp.float32(3.0)
    )
    assert shapes == [()]


def test_subject_tiled_returns_every_subject_once_with_a_padded_remainder() -> None:
    """Padding the remainder leaves exactly the population's rows, in order."""
    tiled = _SubjectTiled(
        func=lambda *, x, scale: x * scale + 1.0, subject_arg_names=("x",)
    )
    x = jnp.arange(10, dtype=jnp.float32)
    out = tiled(x=x, scale=jnp.float32(3.0), **{SUBJECT_WIDTH_KEYWORD: 4})
    np.testing.assert_array_equal(np.asarray(out), np.arange(10) * 3.0 + 1.0)


@pytest.mark.parametrize("width", [1, 2, 8])
@pytest.mark.parametrize("compiled", [False, True])
def test_subject_tiled_preserves_empty_payload_axes_and_original_rows(
    *, width: int, compiled: bool
) -> None:
    """Empty count rosters retain their subject rows beside ordinary outputs."""

    def body(*, x: FloatND) -> tuple[FloatND, IntND, IntND]:
        code = x.astype(jnp.int32)
        return x + 1.0, jnp.zeros((0,), dtype=jnp.int32), jnp.stack((code, code + 10))

    tiled = _SubjectTiled(func=body, subject_arg_names=("x",))

    def evaluate(x: FloatND) -> tuple[FloatND, IntND, IntND]:
        return cast(
            "tuple[FloatND, IntND, IntND]",
            tiled(x=x, **{SUBJECT_WIDTH_KEYWORD: width}),
        )

    x = jnp.arange(5, dtype=jnp.float32)
    values, empty_counts, codes = (jax.jit(evaluate) if compiled else evaluate)(x)
    assert values.dtype == x.dtype
    assert empty_counts.dtype == codes.dtype == jnp.int32
    assert empty_counts.shape == (5, 0)
    expected = np.arange(5, dtype=np.int32)
    np.testing.assert_array_equal(values, expected + 1.0)
    np.testing.assert_array_equal(codes, np.stack((expected, expected + 10), axis=1))
