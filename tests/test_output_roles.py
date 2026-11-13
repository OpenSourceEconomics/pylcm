"""One wrapper converts a user function's output to the role its consumer needs."""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from dags import get_annotations

from _lcm.output_roles import OutputRole, in_role, target_role
from lcm.exceptions import InvalidFunctionOutputError
from lcm.grids import DiscreteGrid, LinSpacedGrid
from lcm.typing import ContinuousState, FloatND


def next_wealth(*, wealth: ContinuousState, rate: float) -> FloatND:
    """Grow wealth at `rate`."""
    return wealth * rate


def returns(value):
    def function() -> FloatND:
        return value

    return function


def test_wrapper_keeps_the_signature_name_and_annotations():
    """The engine reads the same signature, name and annotations as before."""
    wrapped = in_role(func=next_wealth, role=OutputRole.FLOAT, kind="Law")
    assert inspect.signature(wrapped) == inspect.signature(next_wealth)
    assert wrapped.__name__ == "next_wealth"
    assert get_annotations(wrapped) == get_annotations(next_wealth)


def test_wrapping_twice_in_one_role_returns_the_wrapper():
    wrapped = in_role(func=next_wealth, role=OutputRole.FLOAT, kind="Law")
    assert in_role(func=wrapped, role=OutputRole.FLOAT, kind="Law") is wrapped


def test_wrapping_in_another_role_is_a_bug():
    wrapped = in_role(func=next_wealth, role=OutputRole.FLOAT, kind="Law")
    with pytest.raises(TypeError, match="already in role"):
        in_role(func=wrapped, role=OutputRole.CODE, kind="Law")


@pytest.mark.parametrize(
    ("role", "value", "dtype_kind"),
    [
        (OutputRole.FLOAT, 1, "f"),
        (OutputRole.FLOAT, np.int64(2**40), "f"),
        (OutputRole.BOOL, True, "b"),
        (OutputRole.CODE, True, "i"),
        (OutputRole.CODE, np.uint8(3), "i"),
    ],
)
def test_output_takes_its_roles_kind(*, role, value, dtype_kind):
    out = in_role(func=returns(value), role=role, kind="Function")()
    assert jnp.asarray(out).dtype.kind == dtype_kind


@pytest.mark.parametrize(
    ("role", "value"),
    [
        (OutputRole.FLOAT, True),
        (OutputRole.BOOL, 1),
        (OutputRole.CODE, 1.0),
    ],
)
def test_wrong_kind_names_the_function(*, role, value):
    function = returns(value)
    with pytest.raises(InvalidFunctionOutputError, match=r"Gate 'function' returned"):
        in_role(func=function, role=role, kind="Gate")()


def test_a_discrete_grid_target_is_a_code_and_any_other_a_float():
    assert target_role(DiscreteGrid(values=np.array([0.5, 1.5]))) is OutputRole.CODE
    assert (
        target_role(LinSpacedGrid(start=0.0, stop=1.0, n_points=3)) is OutputRole.FLOAT
    )
