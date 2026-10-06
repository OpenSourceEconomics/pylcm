"""A DC-EGM child's resources map is the one its own age resolves to.

The child's resources map is `R_a(x) = (1 + slope * a) * x + transfer * a`, an
age-specialized function closing over its age rather than reading `age` as an
argument. It is declared either as `resources` itself or as `available_wealth`,
a function the resources DAG reads. The parent at period `t` reads the
child at period `t + 1`, so the carry is queried at the resources the child's age
produces, and the savings derivative of that query is the child's slope.

Ages are the periods `0, 1, 2, 3`, so the child of period `t` has age `t + 1`.
Integer coefficients and inputs keep every expected value exact at either
precision.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import core_program_graph
from lcm import AgeSpecializedFunction, LinSpacedGrid, Model
from lcm.solvers import DCEGM
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import make_alive_dead_model, savings, utility


def identity_liquid(*, savings: FloatND) -> ContinuousState:
    """Liquid wealth next period equals savings."""
    return savings


def zero_bequest(*, liquid: ContinuousState) -> FloatND:
    """The terminal regime values remaining wealth at zero."""
    return jnp.zeros_like(liquid)


def resources(*, available_wealth: FloatND) -> FloatND:
    """Resources are the age-specialized available wealth."""
    return available_wealth


def _available_wealth_factory(*, slope: float, transfer: float) -> Any:
    def build(age: float) -> Any:
        def available_wealth(*, liquid: ContinuousState) -> FloatND:
            return (1.0 + slope * age) * liquid + transfer * age

        return available_wealth

    return build


def _resources_factory(*, slope: float, transfer: float) -> Any:
    def build(age: float) -> Any:
        def resources(*, liquid: ContinuousState) -> FloatND:
            return (1.0 + slope * age) * liquid + transfer * age

        return resources

    return build


def _model(*, slope: float, transfer: float, declared_as: str) -> Model:
    def signature(age: float) -> tuple[float, float]:
        return (slope * age, transfer * age)

    specialized = (
        {
            "resources": AgeSpecializedFunction(
                build=_resources_factory(slope=slope, transfer=transfer),
                signature=signature,
            )
        }
        if declared_as == "resources"
        else {
            "resources": resources,
            "available_wealth": AgeSpecializedFunction(
                build=_available_wealth_factory(slope=slope, transfer=transfer),
                signature=signature,
            ),
        }
    )
    return make_alive_dead_model(
        n_periods=4,
        n_liquid=9,
        liquid_max=60.0,
        n_consumption=9,
        alive_functions={"utility": utility, "savings": savings, **specialized},
        liquid_law=identity_liquid,
        alive_solver=DCEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=60.0, n_points=8)
        ),
        constraints={},
        dead_functions={"utility": zero_bequest},
    )


_CASES = {
    "age-invariant": (0.0, 0.0),
    "transfer": (0.0, 10.0),
    "slope": (1.0, 0.0),
    "slope-and-transfer": (1.0, 10.0),
}


@pytest.fixture(
    scope="module",
    params=[(case, where) for case in _CASES for where in ("resources", "dependency")],
    ids=lambda param: f"{param[0]}-as-{param[1]}",
)
def resources_case(request: pytest.FixtureRequest) -> tuple[Model, float, float]:
    case, declared_as = request.param
    slope, transfer = _CASES[case]
    return (
        _model(slope=slope, transfer=transfer, declared_as=declared_as),
        slope,
        transfer,
    )


def _child_read(*, model: Model, period: int) -> Any:
    """The parent's read of its `alive` child, from the period's replay step."""
    kernel = model._regimes["alive"].solution.period_kernels[period]
    step: Any = core_program_graph(kernel=kernel)["replay"].function
    seen: set[int] = set()
    while not hasattr(step, "pieces"):
        assert id(step) not in seen, "the replay program wraps no EGM step"
        seen.add(id(step))
        step = step.func if isinstance(step, functools.partial) else step.__wrapped__
    return step.pieces.continuation_plan.child_reads["alive"]


@pytest.mark.parametrize("period", [0, 1])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_child_resources_and_slope_are_the_child_ages(
    *, resources_case: tuple[Model, float, float], period: int, jit: bool
) -> None:
    """The child of period `t` reads `(1 + slope k) x + transfer k` at age `k = t + 1`.

    At `x = 3` with slope 1 and transfer 10, the child of period 0 reads 16 with
    slope 2, and the child of period 1 reads resources 29 with slope 3.
    """
    model, slope, transfer = resources_case
    read = _child_read(model=model, period=period)
    child_age = period + 1
    evaluate = jax.value_and_grad(lambda x: read.resources_func(liquid=x))
    if jit:
        evaluate = jax.jit(evaluate)
    actual = np.asarray(evaluate(jnp.asarray(3.0)))
    expected = np.asarray(
        [
            (1.0 + slope * child_age) * 3.0 + transfer * child_age,
            1.0 + slope * child_age,
        ],
        dtype=actual.dtype,
    )
    np.testing.assert_array_equal(actual, expected)


def test_periods_share_a_child_read_exactly_when_the_child_maps_agree(
    *, resources_case: tuple[Model, float, float]
) -> None:
    """Periods 0 and 1 share one child read iff the child's map is age-invariant."""
    model, slope, transfer = resources_case
    shared = _child_read(model=model, period=0) is _child_read(model=model, period=1)
    assert shared == (not slope and not transfer)
