import jax.numpy as jnp
from jax import jit
from numpy.testing import assert_array_equal

from lcm.argmax import _flatten_last_n_axes, _move_axes_to_back, argmax_and_max

# Test jitted functions
# ======================================================================================
jitted_argmax = jit(argmax_and_max, static_argnums=[1, 2])


# ======================================================================================
# argmax
# ======================================================================================


def test_argmax_1d_with_mask():
    a = jnp.arange(10)
    mask = jnp.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    _argmax, _max = jitted_argmax(a, where=mask, initial=-1)
    assert _argmax == 4
    assert _max == 4


def test_argmax_2d_with_mask():
    a = jnp.arange(10).reshape(2, 5)
    mask = jnp.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool).reshape(a.shape)

    _argmax, _max = jitted_argmax(a, axis=None, where=mask, initial=-1)
    assert _argmax == 4
    assert _max == 4

    _argmax, _max = jitted_argmax(a, axis=0, where=mask, initial=-1)
    assert_array_equal(_argmax, jnp.array([0, 0, 0, 0, 0]))
    assert_array_equal(_max, jnp.array([0, -1, -1, 3, 4]))

    _argmax, _max = jitted_argmax(a, axis=1, where=mask, initial=-1)
    assert_array_equal(_argmax, jnp.array([4, 0]))
    assert_array_equal(_max, jnp.array([4, -1]))


def test_argmax_1d_no_mask():
    a = jnp.arange(10)
    _argmax, _max = jitted_argmax(a)
    assert _argmax == 9
    assert _max == 9


def test_argmax_2d_no_mask():
    a = jnp.arange(10).reshape(2, 5)

    _argmax, _max = jitted_argmax(a, axis=None)
    assert _argmax == 9
    assert _max == 9

    _argmax, _max = jitted_argmax(a, axis=0)
    assert_array_equal(_argmax, jnp.array([1, 1, 1, 1, 1]))
    assert_array_equal(_max, jnp.array([5, 6, 7, 8, 9]))

    _argmax, _max = jitted_argmax(a, axis=1)
    assert_array_equal(_argmax, jnp.array([4, 4]))
    assert_array_equal(_max, jnp.array([4, 9]))

    _argmax, _max = jitted_argmax(a, axis=(0, 1))
    assert _argmax == 9
    assert _max == 9


def test_argmax_3d_no_mask():
    a = jnp.arange(24).reshape(2, 3, 4)

    _argmax, _max = jitted_argmax(a, axis=None)
    assert _argmax == 23
    assert _max == 23

    _argmax, _max = jitted_argmax(a, axis=0)
    assert_array_equal(_argmax, jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))
    assert_array_equal(
        _max,
        jnp.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]),
    )

    _argmax, _max = jitted_argmax(a, axis=1)
    assert_array_equal(_argmax, jnp.array([[2, 2, 2, 2], [2, 2, 2, 2]]))
    assert_array_equal(_max, jnp.array([[8, 9, 10, 11], [20, 21, 22, 23]]))

    _argmax, _max = jitted_argmax(a, axis=2)
    assert_array_equal(_argmax, jnp.array([[3, 3, 3], [3, 3, 3]]))
    assert_array_equal(_max, jnp.array([[3, 7, 11], [15, 19, 23]]))

    _argmax, _max = jitted_argmax(a, axis=(0, 1))
    assert_array_equal(_argmax, jnp.array([5, 5, 5, 5]))
    assert_array_equal(_max, jnp.array([20, 21, 22, 23]))

    _argmax, _max = jitted_argmax(a, axis=(0, 2))
    assert_array_equal(_argmax, jnp.array([7, 7, 7]))
    assert_array_equal(_max, jnp.array([15, 19, 23]))

    _argmax, _max = jitted_argmax(a, axis=(1, 2))
    assert_array_equal(_argmax, jnp.array([11, 11]))
    assert_array_equal(_max, jnp.array([11, 23]))


def test_argmax_with_ties():
    # If multiple maxima exist, argmax will select the first index.
    a = jnp.zeros((2, 2, 2))
    _argmax, _ = jitted_argmax(a, axis=(1, 2))
    assert_array_equal(_argmax, jnp.array([0, 0]))


# ======================================================================================
# Move axes to back
# ======================================================================================


def test_move_axes_to_back_1d():
    a = jnp.arange(4)
    got = _move_axes_to_back(a, axes=(0,))
    assert_array_equal(got, a)


def test_move_axes_to_back_2d():
    a = jnp.arange(4).reshape(2, 2)
    got = _move_axes_to_back(a, axes=(0,))
    assert_array_equal(got, a.transpose(1, 0))


def test_move_axes_to_back_3d():
    # 2 dimensions in back
    a = jnp.arange(8).reshape(2, 2, 2)
    got = _move_axes_to_back(a, axes=(0, 1))
    assert_array_equal(got, a.transpose(2, 0, 1))

    # 2 dimensions in front
    a = jnp.arange(8).reshape(2, 2, 2)
    got = _move_axes_to_back(a, axes=(1,))
    assert_array_equal(got, a.transpose(0, 2, 1))


# ======================================================================================
# Flatten last n axes
# ======================================================================================


def test_flatten_last_n_axes_1d():
    a = jnp.arange(4)
    got = _flatten_last_n_axes(a, n=1)
    assert_array_equal(got, a)


def test_flatten_last_n_axes_2d():
    a = jnp.arange(4).reshape(2, 2)

    got = _flatten_last_n_axes(a, n=1)
    assert_array_equal(got, a)

    got = _flatten_last_n_axes(a, n=2)
    assert_array_equal(got, a.reshape(4))


def test_flatten_last_n_axes_3d():
    a = jnp.arange(8).reshape(2, 2, 2)

    got = _flatten_last_n_axes(a, n=1)
    assert_array_equal(got, a)

    got = _flatten_last_n_axes(a, n=2)
    assert_array_equal(got, a.reshape(2, 4))

    got = _flatten_last_n_axes(a, n=3)
    assert_array_equal(got, a.reshape(8))
