"""A subject population that is not a multiple of the tile width runs one body shape."""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.simulation.programs import SUBJECT_WIDTH_KEYWORD, _SubjectTiled


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
