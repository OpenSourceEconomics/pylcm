from __future__ import annotations

import inspect
import re

import jax.numpy as jnp
import pytest
from jax import vmap
from numpy.testing import assert_array_almost_equal as aaae

from lcm.functools import (
    all_as_args,
    all_as_kwargs,
    allow_args,
    allow_only_kwargs,
    convert_kwargs_to_args,
    get_union_of_arguments,
)

# ======================================================================================
# get_union_of_arguments
# ======================================================================================


def test_get_union_of_arguments():
    def f(a, b):
        pass

    def g(b, c):
        pass

    got = get_union_of_arguments([f, g])
    assert got == {"a", "b", "c"}


def test_get_union_of_arguments_no_args():
    def f():
        pass

    got = get_union_of_arguments([f])
    assert got == set()


# ======================================================================================
# all_as_kwargs
# ======================================================================================


def test_all_as_kwargs():
    got = all_as_kwargs(
        args=(1, 2),
        kwargs={"c": 3},
        arg_names=["a", "b", "c"],
    )
    assert got == {"a": 1, "b": 2, "c": 3}


def test_all_as_kwargs_empty_args():
    got = all_as_kwargs(
        args=(),
        kwargs={"a": 1, "b": 2, "c": 3},
        arg_names=["a", "b", "c"],
    )
    assert got == {"a": 1, "b": 2, "c": 3}


def test_all_as_kwargs_empty_kwargs():
    got = all_as_kwargs(
        args=(1, 2, 3),
        kwargs={},
        arg_names=["a", "b", "c"],
    )
    assert got == {"a": 1, "b": 2, "c": 3}


# ======================================================================================
# all_as_args
# ======================================================================================


def test_all_as_args():
    got = all_as_args(
        args=(1, 2),
        kwargs={"c": 3},
        arg_names=["a", "b", "c"],
    )
    assert got == (1, 2, 3)


def test_all_as_args_empty_args():
    got = all_as_args(
        args=(),
        kwargs={"a": 1, "b": 2, "c": 3},
        arg_names=["a", "b", "c"],
    )
    assert got == (1, 2, 3)


def test_all_as_args_empty_kwargs():
    got = all_as_args(
        args=(1, 2, 3),
        kwargs={},
        arg_names=["a", "b", "c"],
    )
    assert got == (1, 2, 3)


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


def test_allow_only_kwargs():
    def f(a, /, b):
        # a is positional-only
        return a + b

    with pytest.raises(TypeError):
        f(a=1, b=2)  # type: ignore[call-arg]

    assert allow_only_kwargs(f)(a=1, b=2) == 3


def test_allow_only_kwargs_with_keyword_only_args():
    def f(a, /, *, b):
        return a + b

    with pytest.raises(TypeError):
        f(a=1, b=2)  # type: ignore[call-arg]

    assert allow_only_kwargs(f)(a=1, b=2) == 3


def test_allow_only_kwargs_too_many_args():
    def f(a, /, b):
        return a + b

    too_many_match = re.escape("Expected arguments: ['a', 'b'], got extra: {'c'}")
    with pytest.raises(ValueError, match=too_many_match):
        allow_only_kwargs(f)(a=1, b=2, c=3)


def test_allow_only_kwargs_too_few_args():
    def f(a, /, b):
        return a + b

    too_few_match = re.escape("Expected arguments: ['a', 'b'], missing: {'b'}")
    with pytest.raises(ValueError, match=too_few_match):
        allow_only_kwargs(f)(a=1)


def test_allow_only_kwargs_signature_change():
    def f(a, /, b, *, c):
        pass

    decorated = allow_only_kwargs(f)
    parameters = inspect.signature(decorated).parameters

    assert parameters["a"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["b"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["c"].kind == inspect.Parameter.KEYWORD_ONLY


# ======================================================================================
# allow args
# ======================================================================================


def test_allow_args():
    def f(a, *, b):
        # b is keyword-only
        return a + b

    with pytest.raises(TypeError):
        f(1, 2)  # type: ignore[misc]

    assert allow_args(f)(1, 2) == 3
    assert allow_args(f)(1, b=2) == 3
    assert allow_args(f)(b=2, a=1) == 3


def test_allow_args_different_kwargs_order():
    def f(a, b, c, *, d):
        return a + b + c + d

    with pytest.raises(TypeError):
        f(1, 2, 3, 4)  # type: ignore[misc]

    assert allow_args(f)(1, 2, 3, 4) == 10
    assert allow_args(f)(1, 2, d=4, c=3) == 10


def test_allow_args_too_many_args():
    def f(a, *, b):
        return a + b

    with pytest.raises(ValueError, match=r"Too many arguments provided."):
        allow_args(f)(1, 2, b=3)


def test_allow_args_too_few_args():
    def f(a, *, b):
        return a + b

    with pytest.raises(ValueError, match=r"Not all arguments provided."):
        allow_args(f)(1)


def test_allow_args_with_vmap():
    def f(a, *, b):
        # b is keyword-only
        return a + b

    f_vmapped = vmap(f, in_axes=(0, 0))
    f_allow_args_vmapped = vmap(allow_args(f), in_axes=(0, 0))

    a = jnp.arange(2)
    b = jnp.arange(2)

    with pytest.raises(TypeError):
        # TypeError since b is keyword-only
        f_vmapped(a, b)  # type: ignore[misc]

    with pytest.raises(ValueError, match="vmap in_axes must be an int"):
        # ValueError since vmap doesn't support keyword arguments
        f_vmapped(a, b=b)

    aaae(f_allow_args_vmapped(a, b), jnp.array([0, 2]))


def test_allow_args_signature_change():
    def f(a, /, b, *, c):
        pass

    decorated = allow_args(f)
    parameters = inspect.signature(decorated).parameters

    assert parameters["a"].kind == inspect.Parameter.POSITIONAL_ONLY
    assert parameters["b"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["c"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
