import itertools

import jax.numpy as jnp
import pytest
from lcm.dispatchers import (
    allow_args,
    allow_kwargs,
    convert_kwargs_to_args,
    productmap,
    spacemap,
    vmap_1d,
)
from numpy.testing import assert_array_almost_equal as aaae


def f(a, b, c):
    return jnp.sin(a) + jnp.cos(b) + jnp.tan(c)


def f2(b, a, c):
    return jnp.sin(a) + jnp.cos(b) + jnp.tan(c)


def g(a, b, c, d):
    return f(a, b, c) + jnp.log(d)


# ======================================================================================
# productmap
# ======================================================================================


@pytest.fixture()
def setup_productmap_f():
    return {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }


@pytest.fixture()
def expected_productmap_f():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T
    return f(*helper).reshape(10, 7, 5)


@pytest.fixture()
def setup_productmap_g():
    return {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
        "d": jnp.linspace(1, 3, 4),
    }


@pytest.fixture()
def expected_productmap_g():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
        "d": jnp.linspace(1, 3, 4),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T
    return g(*helper).reshape(10, 7, 5, 4)


@pytest.mark.parametrize(
    ("func", "args", "grids", "expected"),
    [
        (f, ["a", "b", "c"], "setup_productmap_f", "expected_productmap_f"),
        (g, ["a", "b", "c", "d"], "setup_productmap_g", "expected_productmap_g"),
    ],
)
def test_productmap_with_all_arguments_mapped(func, args, grids, expected, request):
    grids = request.getfixturevalue(grids)
    expected = request.getfixturevalue(expected)

    decorated = productmap(func, args)
    calculated_args = decorated(*grids.values())
    calculated_kwargs = decorated(**grids)

    aaae(calculated_args, expected)
    aaae(calculated_kwargs, expected)


def test_productmap_different_func_order(setup_productmap_f):
    decorated_f = productmap(f, ["a", "b", "c"])
    expected = decorated_f(*setup_productmap_f.values())

    decorated_f2 = productmap(f2, ["a", "b", "c"])
    calculated_f2_kwargs = decorated_f2(**setup_productmap_f)

    aaae(calculated_f2_kwargs, expected)


def test_productmap_change_arg_order(setup_productmap_f, expected_productmap_f):
    expected = jnp.transpose(expected_productmap_f, (1, 0, 2))

    decorated = productmap(f, ["b", "a", "c"])
    calculated = decorated(**setup_productmap_f)

    aaae(calculated, expected)


def test_productmap_with_all_arguments_mapped_some_len_one():
    grids = {
        "a": jnp.array([1]),
        "b": jnp.array([2]),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T

    expected = f(*helper).reshape(1, 1, 5)

    decorated = productmap(f, ["a", "b", "c"])
    calculated = decorated(*grids.values())
    aaae(calculated, expected)


def test_productmap_with_all_arguments_mapped_some_scalar():
    grids = {
        "a": 1,
        "b": 2,
        "c": jnp.linspace(1, 5, 5),
    }

    decorated = productmap(f, ["a", "b", "c"])
    with pytest.raises(ValueError, match="vmap was requested to map its argument"):
        decorated(*grids.values())


def test_productmap_with_some_arguments_mapped():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": 1,
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(grids["a"], [grids["b"]], grids["c"]))).T

    expected = f(*helper).reshape(10, 5)

    decorated = productmap(f, ["a", "c"])
    calculated = decorated(*grids.values())
    aaae(calculated, expected)


def test_productmap_with_some_argument_mapped_twice():
    error_msg = "Same argument provided more than once."
    with pytest.raises(ValueError, match=error_msg):
        productmap(f, ["a", "a", "c"])


# ======================================================================================
# spacemap
# ======================================================================================


@pytest.fixture()
def setup_spacemap():
    value_grid = {
        "a": jnp.array([1.0, 2, 3]),
        "b": jnp.array([3.0, 4]),
    }

    sparse_values = {
        "c": jnp.array([7.0, 8, 9, 10]),
        "d": jnp.array([9.0, 10, 11, 12, 13]),
    }

    helper = jnp.array(list(itertools.product(*sparse_values.values()))).T

    combination_grid = {
        "c": helper[0],
        "d": helper[1],
    }
    return value_grid, combination_grid


@pytest.fixture()
def expected_spacemap():
    value_grid = {
        "a": jnp.array([1.0, 2, 3]),
        "b": jnp.array([3.0, 4]),
    }

    combination_grid = {
        "c": jnp.array([7.0, 8, 9, 10]),
        "d": jnp.array([9.0, 10, 11, 12, 13]),
    }

    all_grids = {**value_grid, **combination_grid}
    helper = jnp.array(list(itertools.product(*all_grids.values()))).T

    return g(*helper).reshape(3, 2, 4 * 5)


@pytest.mark.parametrize("dense_first", [True, False])
def test_spacemap_all_arguments_mapped(setup_spacemap, expected_spacemap, dense_first):
    dense_vars, sparse_vars = setup_spacemap

    decorated = spacemap(
        g,
        list(dense_vars),
        list(sparse_vars),
        dense_first=dense_first,
    )
    calculated = decorated(**dense_vars, **sparse_vars)

    if dense_first:
        aaae(calculated, expected_spacemap)
    else:
        aaae(calculated, jnp.transpose(expected_spacemap, axes=(2, 0, 1)))


@pytest.mark.parametrize(
    ("error_msg", "dense_vars", "sparse_vars"),
    [
        (
            "dense_vars and sparse_vars overlap",
            ["a", "b"],
            ["a", "c", "d"],
        ),
        (
            "Same argument provided more than once.",
            ["a", "a", "b"],
            ["c", "d"],
        ),
    ],
)
def test_spacemap_arguments_overlap(error_msg, dense_vars, sparse_vars):
    with pytest.raises(ValueError, match=error_msg):
        spacemap(g, dense_vars, sparse_vars, dense_first=True)


# ======================================================================================
# convert kwargs to args
# ======================================================================================


def test_convert_kwargs_to_args():
    kwargs = {"a": 1, "b": 2, "c": 3}
    parameters = ["c", "a", "b"]
    exp = [3, 1, 2]
    got = convert_kwargs_to_args(kwargs, parameters)
    assert got == exp


# ======================================================================================
# allow kwargs
# ======================================================================================


def test_allow_kwargs():
    def f(a, /, b):
        # a is positional-only
        return a + b

    with pytest.raises(TypeError):
        f(a=1, b=2)

    assert allow_kwargs(f)(a=1, b=2) == 3


def test_allow_kwargs_with_keyword_only_args():
    def f(a, /, *, b):
        return a + b

    with pytest.raises(TypeError):
        f(a=1, b=2)

    assert allow_kwargs(f)(a=1, b=2) == 3


def test_allow_kwargs_incorrect_number_of_args():
    def f(a, /, b):
        return a + b

    with pytest.raises(ValueError, match="Not enough or too many arguments"):
        allow_kwargs(f)(a=1, b=2, c=3)

    with pytest.raises(ValueError, match="Not enough or too many arguments"):
        allow_kwargs(f)(a=1)


def test_allow_kwargs_with_productmap():
    def f(a, /, b):
        # a is positional-only
        return a + b

    # productmap calls allow_kwargs internally
    decorated = productmap(f, ["a", "b"])

    a = jnp.arange(2)
    b = jnp.arange(2)

    with pytest.raises(TypeError):
        # TypeError since a is positional-only
        f(a=a, b=b)

    aaae(decorated(a=a, b=b), jnp.array([[0, 1], [1, 2]]))
    aaae(decorated(a, b=b), jnp.array([[0, 1], [1, 2]]))
    aaae(decorated(a, b), jnp.array([[0, 1], [1, 2]]))


# ======================================================================================
# allow args
# ======================================================================================


def test_allow_args():
    def f(a, *, b):
        # b is keyword-only
        return a + b

    with pytest.raises(TypeError):
        f(1, 2)

    assert allow_args(f)(1, 2) == 3
    assert allow_args(f)(1, b=2) == 3
    assert allow_args(f)(b=2, a=1) == 3


def test_allow_args_different_kwargs_order():
    def f(a, b, c, *, d):
        return a + b + c + d

    with pytest.raises(TypeError):
        f(1, 2, 3, 4)

    assert allow_args(f)(1, 2, 3, 4) == 10
    assert allow_args(f)(1, 2, d=4, c=3) == 10


def test_allow_args_incorrect_number_of_args():
    def f(a, *, b):
        return a + b

    with pytest.raises(ValueError, match="Not enough or too many arguments"):
        allow_args(f)(1, 2, b=3)

    with pytest.raises(ValueError, match="Not enough or too many arguments"):
        allow_args(f)(1)


def test_allow_args_with_productmap():
    def f(a, *, b):
        # b is keyword-only
        return a + b

    decorated_f = productmap(f, ["a", "b"])
    decorated_f_with_allow_args = productmap(allow_args(f), ["a", "b"])

    a = jnp.arange(2)
    b = jnp.arange(2)

    with pytest.raises(ValueError, match="vmap in_axes specification must be a tree"):
        # ValueError since vmap is applied to a function with keyword-only argument
        decorated_f(a=a, b=b)

    with pytest.raises(ValueError, match="vmap in_axes specification must be a tree"):
        # ValueError since vmap is applied to a function with keyword-only argument
        decorated_f(a, b=b)

    with pytest.raises(KeyError):
        # KeyError since f expects b as keyword argument
        decorated_f(a, b)

    aaae(decorated_f_with_allow_args(a, b), jnp.array([[0, 1], [1, 2]]))
    aaae(decorated_f_with_allow_args(a, b=b), jnp.array([[0, 1], [1, 2]]))
    aaae(decorated_f_with_allow_args(a=a, b=b), jnp.array([[0, 1], [1, 2]]))


# ======================================================================================
# vmap_1d
# ======================================================================================


def test_vmap_1d():
    def func(a, b, c):
        return c * (a + b)

    vmapped = vmap_1d(func, variables=["a", "b"])
    a = jnp.array([1, 2])
    got = vmapped(a=a, b=a, c=-1)
    exp = jnp.array([-2, -4])

    aaae(got, exp)


def test_vmap_1d_error():
    with pytest.raises(ValueError, match="Same argument provided more than once."):
        vmap_1d(None, variables=["a", "a"])
