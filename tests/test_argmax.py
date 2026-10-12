import itertools
import math
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, grad, jit, make_jaxpr, vmap
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


def _admitted_maximum(*, value: float, feasible: bool, initial: float | None) -> float:
    """Maximum over one candidate and `initial`, written out without tensor code.

    An infeasible candidate is not admitted; `initial` always is, and a missing
    `initial` stands for minus infinity. A feasible NaN candidate is the result.
    """
    if feasible and math.isnan(value):
        return math.nan
    admitted = [value] if feasible else []
    admitted.append(-math.inf if initial is None else initial)
    return max(admitted)


def _expected_one_candidate_maxima(
    *,
    a: np.ndarray,
    where: np.ndarray | bool | list[list[bool]] | None,
    initial: float | None,
) -> np.ndarray:
    feasible = np.broadcast_to(True if where is None else np.asarray(where), a.shape)
    expected = np.empty_like(a)
    for cell in np.ndindex(a.shape):
        expected[cell] = _admitted_maximum(
            value=float(a[cell]), feasible=bool(feasible[cell]), initial=initial
        )
    return expected


ONE_CANDIDATE_CASES = list(
    itertools.product(
        [-8.0, -0.0, 0.0, 1.0, 8.0, float("nan")],
        [None, False, True],
        [None, -math.inf, -2.0, 0.0, 2.0],
    )
)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("axis", [None, ()], ids=["scalar", "empty-axes"])
@pytest.mark.parametrize(("value", "where", "initial"), ONE_CANDIDATE_CASES)
def test_argmax_and_max_of_one_candidate_respects_where_and_initial(
    *,
    compiled: bool,
    axis: tuple[int, ...] | None,
    value: float,
    where: bool | None,
    initial: float | None,
) -> None:
    """A cell with no axis to reduce has one candidate, admitted only where feasible."""
    reduce = jitted_argmax if compiled else argmax_and_max
    a = jnp.asarray(value)
    mask = None if where is None else jnp.asarray(where)
    index, maximum = reduce(a=a, axis=axis, where=mask, initial=initial)
    expected = _expected_one_candidate_maxima(
        a=np.asarray(a), where=where, initial=initial
    )
    assert_array_equal(
        np.stack([np.asarray(index, dtype=float), np.asarray(maximum)]),
        np.stack([np.zeros_like(expected), expected]),
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("where", [None, False, True, [[True, False, True]]], ids=str)
def test_argmax_and_max_over_empty_axes_keeps_each_cell_and_broadcasts_where(
    *, compiled: bool, where: bool | list[list[bool]] | None
) -> None:
    """Reducing over no axes leaves every cell, each masked by its own feasibility."""
    reduce = jitted_argmax if compiled else argmax_and_max
    a = jnp.asarray([[-2.0, 1.0, 4.0], [7.0, float("nan"), -0.0]])
    mask = None if where is None else jnp.asarray(where)
    index, maximum = reduce(a=a, axis=(), where=mask, initial=0.0)
    expected = _expected_one_candidate_maxima(a=np.asarray(a), where=where, initial=0.0)
    assert_array_equal(
        np.stack([np.asarray(index, dtype=float), np.asarray(maximum)]),
        np.stack([np.zeros_like(expected), expected]),
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_argmax_and_max_of_mapped_scalar_cells_masks_at_the_feasibility_boundary(
    *, compiled: bool
) -> None:
    """Mapped over cells, each scalar candidate is admitted exactly when `x >= 0`.

    The smallest normal numbers on either side of zero sit at the boundary;
    subnormals are avoided because the backend may flush them to zero.
    """
    tiny = jnp.finfo(jnp.asarray(0.0).dtype).tiny
    x = jnp.asarray([-1.0, 0.0, 1.0, -tiny, tiny])
    assert bool(x[3] < 0)
    assert bool(x[4] > 0)

    # keyword-only-exempt: library-callback=jax.vmap
    def one_cell(value: Array, feasible: Array) -> tuple[Array, Array]:
        return argmax_and_max(a=value, where=feasible, initial=-jnp.inf)

    mapped = vmap(one_cell)
    reduce = jit(mapped) if compiled else mapped
    index, maximum = reduce(x + 2, x >= 0)
    expected = _expected_one_candidate_maxima(
        a=np.asarray(x + 2), where=np.asarray(x >= 0), initial=-math.inf
    )
    assert_array_equal(
        np.stack([np.asarray(index, dtype=float), np.asarray(maximum)]),
        np.stack([np.zeros_like(expected), expected]),
    )


@pytest.mark.parametrize(
    ("feasible", "initial", "expected"),
    [
        (False, -math.inf, 0.0),
        (True, -math.inf, 1.0),
        (False, -7.0, 0.0),
        (True, -7.0, 1.0),
        (True, 3.0, 0.0),
        (True, 1.0, 0.5),
    ],
)
def test_argmax_and_max_of_one_candidate_differentiates_like_jnp_max(
    *, feasible: bool, initial: float, expected: float
) -> None:
    """The candidate's derivative is 1 above `initial`, 0 below, 1/2 at a tie."""
    derivative = grad(
        lambda x: argmax_and_max(a=x, where=jnp.asarray(feasible), initial=initial)[1]
    )
    assert_array_equal(np.asarray(derivative(jnp.asarray(1.0))), expected)
