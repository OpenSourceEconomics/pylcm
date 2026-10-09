from functools import partial

import jax.numpy as jnp
import pytest
from jax import grad, jit, make_jaxpr, vmap
from numpy.testing import assert_array_equal

from _lcm.regime_building.argmax import (
    _flatten_last_n_axes,
    _move_axes_to_back,
    argmax_and_max,
)
from tests.solution._reduced_maximum_probe import (
    count_equalities_with_a_reduced_maximum,
)

# Test jitted functions
jitted_argmax = jit(argmax_and_max, static_argnames=["axis", "initial"])


def test_argmax_1d_with_mask():
    a = jnp.arange(10, dtype=jnp.int32)
    mask = jnp.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    _argmax, _max = jitted_argmax(a=a, where=mask, initial=-1)
    assert _argmax == 4
    assert _max == 4


def test_argmax_2d_with_mask():
    a = jnp.arange(10, dtype=jnp.int32).reshape(2, 5)
    mask = jnp.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool).reshape(a.shape)

    _argmax, _max = jitted_argmax(a=a, axis=None, where=mask, initial=-1)
    assert _argmax == 4
    assert _max == 4

    _argmax, _max = jitted_argmax(a=a, axis=0, where=mask, initial=-1)
    assert_array_equal(_argmax, jnp.array([0, 0, 0, 0, 0]))
    assert_array_equal(_max, jnp.array([0, -1, -1, 3, 4]))

    _argmax, _max = jitted_argmax(a=a, axis=1, where=mask, initial=-1)
    assert_array_equal(_argmax, jnp.array([4, 0]))
    assert_array_equal(_max, jnp.array([4, -1]))


def test_argmax_1d_no_mask():
    a = jnp.arange(10, dtype=jnp.int32)
    _argmax, _max = jitted_argmax(a=a)
    assert _argmax == 9
    assert _max == 9


def test_argmax_2d_no_mask():
    a = jnp.arange(10, dtype=jnp.int32).reshape(2, 5)

    _argmax, _max = jitted_argmax(a=a, axis=None)
    assert _argmax == 9
    assert _max == 9

    _argmax, _max = jitted_argmax(a=a, axis=0)
    assert_array_equal(_argmax, jnp.array([1, 1, 1, 1, 1]))
    assert_array_equal(_max, jnp.array([5, 6, 7, 8, 9]))

    _argmax, _max = jitted_argmax(a=a, axis=1)
    assert_array_equal(_argmax, jnp.array([4, 4]))
    assert_array_equal(_max, jnp.array([4, 9]))

    _argmax, _max = jitted_argmax(a=a, axis=(0, 1))
    assert _argmax == 9
    assert _max == 9


def test_argmax_3d_no_mask():
    a = jnp.arange(24, dtype=jnp.int32).reshape(2, 3, 4)

    _argmax, _max = jitted_argmax(a=a, axis=None)
    assert _argmax == 23
    assert _max == 23

    _argmax, _max = jitted_argmax(a=a, axis=0)
    assert_array_equal(_argmax, jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))
    assert_array_equal(
        _max,
        jnp.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]),
    )

    _argmax, _max = jitted_argmax(a=a, axis=1)
    assert_array_equal(_argmax, jnp.array([[2, 2, 2, 2], [2, 2, 2, 2]]))
    assert_array_equal(_max, jnp.array([[8, 9, 10, 11], [20, 21, 22, 23]]))

    _argmax, _max = jitted_argmax(a=a, axis=2)
    assert_array_equal(_argmax, jnp.array([[3, 3, 3], [3, 3, 3]]))
    assert_array_equal(_max, jnp.array([[3, 7, 11], [15, 19, 23]]))

    _argmax, _max = jitted_argmax(a=a, axis=(0, 1))
    assert_array_equal(_argmax, jnp.array([5, 5, 5, 5]))
    assert_array_equal(_max, jnp.array([20, 21, 22, 23]))

    _argmax, _max = jitted_argmax(a=a, axis=(0, 2))
    assert_array_equal(_argmax, jnp.array([7, 7, 7]))
    assert_array_equal(_max, jnp.array([15, 19, 23]))

    _argmax, _max = jitted_argmax(a=a, axis=(1, 2))
    assert_array_equal(_argmax, jnp.array([11, 11]))
    assert_array_equal(_max, jnp.array([11, 23]))


def test_argmax_with_ties():
    # If multiple maxima exist, argmax will select the first index.
    a = jnp.zeros((2, 2, 2))
    _argmax, _ = jitted_argmax(a=a, axis=(1, 2))
    assert_array_equal(_argmax, jnp.array([0, 0]))


def test_move_axes_to_back_1d():
    a = jnp.arange(4, dtype=jnp.int32)
    got = _move_axes_to_back(a=a, axes=(0,))
    assert_array_equal(got, a)


def test_move_axes_to_back_2d():
    a = jnp.arange(4, dtype=jnp.int32).reshape(2, 2)
    got = _move_axes_to_back(a=a, axes=(0,))
    assert_array_equal(got, a.transpose(1, 0))


def test_move_axes_to_back_3d():
    # 2 dimensions in back
    a = jnp.arange(8, dtype=jnp.int32).reshape(2, 2, 2)
    got = _move_axes_to_back(a=a, axes=(0, 1))
    assert_array_equal(got, a.transpose(2, 0, 1))

    # 2 dimensions in front
    a = jnp.arange(8, dtype=jnp.int32).reshape(2, 2, 2)
    got = _move_axes_to_back(a=a, axes=(1,))
    assert_array_equal(got, a.transpose(0, 2, 1))


def test_flatten_last_n_axes_1d():
    a = jnp.arange(4, dtype=jnp.int32)
    got = _flatten_last_n_axes(a=a, n=1)
    assert_array_equal(got, a)


def test_flatten_last_n_axes_2d():
    a = jnp.arange(4, dtype=jnp.int32).reshape(2, 2)

    got = _flatten_last_n_axes(a=a, n=1)
    assert_array_equal(got, a)

    got = _flatten_last_n_axes(a=a, n=2)
    assert_array_equal(got, a.reshape(4))


def test_flatten_last_n_axes_3d():
    a = jnp.arange(8, dtype=jnp.int32).reshape(2, 2, 2)

    got = _flatten_last_n_axes(a=a, n=1)
    assert_array_equal(got, a)

    got = _flatten_last_n_axes(a=a, n=2)
    assert_array_equal(got, a.reshape(2, 4))

    got = _flatten_last_n_axes(a=a, n=3)
    assert_array_equal(got, a.reshape(8))


def test_argmax_and_max_identity_is_not_matched_against_a_separately_reduced_max():
    """The argmax and the max it reports come out of one reduction."""
    a = jnp.array([[-1650.6389, -14.865698, -14.989168, -7401.933]])
    where = jnp.array([[True, True, True, False]])

    jaxpr = make_jaxpr(
        partial(argmax_and_max, axis=1, initial=-jnp.inf),
    )(a=a, where=where).jaxpr

    assert count_equalities_with_a_reduced_maximum(jaxpr) == 0


@pytest.mark.parametrize(
    ("a", "where", "expected_index"),
    [
        ([10.0, 3.0, 7.0], [False, True, True], 2),
        ([-jnp.inf, 4.0, 9.0, 9.0], [True, True, True, True], 2),
        ([0.5, 8.0, -1.0], [False, True, False], 1),
    ],
    ids=["position-zero-infeasible", "tie-after-position-zero", "single-feasible"],
)
def test_argmax_and_max_index_attains_the_reported_max_under_jit_and_vmap(
    *, a: list[float], where: list[bool], expected_index: int
) -> None:
    """In a batch, each row's index attains its max; it is never a stray zero."""
    rows = jnp.asarray([a, a], dtype=jnp.float32)
    masks = jnp.asarray([where, where])

    index, _ = vmap(
        jit(partial(argmax_and_max, axis=0, initial=-jnp.inf)),
    )(a=rows, where=masks)

    assert_array_equal(index, jnp.array([expected_index, expected_index]))


@pytest.mark.parametrize(
    ("values", "initial"),
    [
        ([1.0, 3.0], 3.0),
        ([3.0, 3.0], 3.0),
        ([3.0, 3.0], -jnp.inf),
        ([1.0, 3.0], 5.0),
    ],
)
def test_argmax_and_max_differentiates_like_jnp_max_when_initial_ties(
    *, values: list[float], initial: float
) -> None:
    """The max's gradient equals `jnp.max`'s, including a finite tying `initial`."""
    a = jnp.array(values)
    got = grad(lambda a: argmax_and_max(a=a, axis=0, initial=initial)[1])(a)
    expected = grad(lambda a: jnp.max(a, axis=0, initial=initial))(a)
    assert_array_equal(got, expected)
