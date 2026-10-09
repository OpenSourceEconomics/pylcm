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

A second model separates the child from the source: an age-invariant `source`
regime moves into a `child` regime carrying the age-specialized resources map,
so the child's age is the only thing distinguishing the source's two periods.
The source's liquid law is either the identity or `2 s + 1`; the read composes
it with the child's map, so its savings derivative is the law's slope times the
child's.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.continuation import (
    _ChildEulerState,
    _RowQueriesAndGradients,
    child_resources_params,
)
from _lcm.execution.core_program import core_program_graph
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ConsumptionSavingsRegime,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
)
from lcm.solvers import DCEGM, GridSearch
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.test_models.nbegm_common import make_alive_dead_model, savings, utility


def identity_liquid(*, savings: FloatND) -> ContinuousState:
    """Liquid wealth next period equals savings."""
    return savings


def doubled_liquid(*, savings: FloatND) -> ContinuousState:
    """Liquid wealth next period is twice savings plus one."""
    return 2.0 * savings + 1.0


def liquid_resources(*, liquid: ContinuousState) -> FloatND:
    """Resources are the liquid state itself."""
    return liquid


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


def _continuation_plan(*, model: Model, regime: str, period: int) -> Any:
    """The continuation plan of `regime` at `period`, from the replay step."""
    kernel = model._regimes[regime].solution.period_kernels[period]
    step: Any = core_program_graph(kernel=kernel)["replay"].function
    seen: set[int] = set()
    while not hasattr(step, "pieces"):
        assert id(step) not in seen, "the replay program wraps no EGM step"
        seen.add(id(step))
        step = step.func if isinstance(step, functools.partial) else step.__wrapped__
    return step.pieces.continuation_plan


def _child_read(*, model: Model, period: int) -> Any:
    """The parent's read of its `alive` child, from the period's replay step."""
    plan = _continuation_plan(model=model, regime="alive", period=period)
    return plan.child_reads["alive"]


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


@categorical(ordered=False)
class _SourceChildId:
    source: ScalarInt
    child: ScalarInt
    dead: ScalarInt


def _source_child_model(*, slope: float, transfer: float, liquid_law: Any) -> Model:
    """An age-invariant `source` at ages 0 and 1 moving into an age-varying `child`.

    The child, active at ages 1 and 2, carries `R_a(x) = (1 + slope a) x +
    transfer a` as its resources. The source's resources are its liquid state and
    its liquid law toward the child is `liquid_law`.
    """
    liquid_grid = LinSpacedGrid(start=0.1, stop=60.0, n_points=9)
    consumption_grid = LinSpacedGrid(start=0.1, stop=60.0, n_points=9)
    solver = DCEGM(savings_grid=LinSpacedGrid(start=0.0, stop=60.0, n_points=8))
    source = ConsumptionSavingsRegime(
        actions={"consumption": consumption_grid},
        states={"liquid": liquid_grid},
        state_transitions={"liquid": {"child": liquid_law}},
        constraints={},
        functions={
            "utility": utility,
            "savings": savings,
            "resources": liquid_resources,
        },
        solver=solver,
        liquid=LiquidMargin(
            state="liquid",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
    )
    child = ConsumptionSavingsRegime(
        actions={"consumption": consumption_grid},
        states={"liquid": liquid_grid},
        state_transitions={
            "liquid": {"child": identity_liquid, "dead": identity_liquid}
        },
        constraints={},
        functions={
            "utility": utility,
            "savings": savings,
            "resources": AgeSpecializedFunction(
                build=_resources_factory(slope=slope, transfer=transfer),
                signature=lambda age: (slope * age, transfer * age),
            ),
        },
        solver=solver,
        liquid=LiquidMargin(
            state="liquid",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
    )
    dead = Regime(
        states={"liquid": liquid_grid},
        functions={"utility": zero_bequest},
        solver=GridSearch(),
    )
    return Model(
        regimes={"source": source, "child": child, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        edges={"source": {"child": (0, 1)}, "child": {"child": 1, "dead": 2}},
        regime_id_class=_SourceChildId,
        initial_nodes={0: "source", 1: "source"},
    )


# Each law: the function, its savings slope, and the savings it maps to `x = 3`.
_SOURCE_LAWS = {
    "identity-law": (identity_liquid, 1.0, 3.0),
    "doubled-law": (doubled_liquid, 2.0, 1.0),
}
_CHILD_CASES = {
    "age-varying-child": (1.0, 10.0),
    "age-invariant-child": (0.0, 0.0),
}


@pytest.fixture(
    scope="module",
    params=[(law, child) for law in _SOURCE_LAWS for child in _CHILD_CASES],
    ids=lambda param: f"{param[0]}-{param[1]}",
)
def source_child_case(
    request: pytest.FixtureRequest,
) -> tuple[Model, float, float, float, float]:
    law_name, child_name = request.param
    liquid_law, law_slope, savings_at_three = _SOURCE_LAWS[law_name]
    slope, transfer = _CHILD_CASES[child_name]
    model = _source_child_model(slope=slope, transfer=transfer, liquid_law=liquid_law)
    return model, slope, transfer, law_slope, savings_at_three


def _source_reads_child(*, model: Model, period: int, savings_value: Any) -> Any:
    """The source's composed child-resources query and savings gradient."""
    plan = _continuation_plan(model=model, regime="source", period=period)
    read = plan.child_reads["child"]
    return _RowQueriesAndGradients(
        read=read,
        child_euler_state=_ChildEulerState(
            next_state_func=read.euler_state_func,
            combo_pool={},
            post_decision_name=plan.post_decision_name,
            next_state_key=read.next_state_key,
        ),
        deterministic_resources_kwargs={},
        resources_param_kwargs=child_resources_params(
            read=read, combo_pool={"period": jnp.int32(period)}
        ),
        savings_value=savings_value,
    )(())


@pytest.mark.parametrize("period", [0, 1])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_age_invariant_source_reads_the_child_regimes_age(
    *,
    source_child_case: tuple[Model, float, float, float, float],
    period: int,
    jit: bool,
) -> None:
    """An age-invariant source reads its child's resources at the child's age.

    The savings landing on child liquid `x = 3` yields `R_k(3)` and the savings
    slope `law slope * (1 + slope k)` at child age `k = t + 1`: with slope 1 and
    transfer 10, `(16, 2)` from period 0 and `(29, 3)` from period 1 under the
    identity law, and `(16, 4)` and `(29, 6)` under `2 s + 1`.
    """
    model, slope, transfer, law_slope, savings_at_three = source_child_case

    def evaluate(savings_value: Any) -> Any:
        return jnp.stack(
            _source_reads_child(model=model, period=period, savings_value=savings_value)
        )

    run = jax.jit(evaluate) if jit else evaluate
    actual = np.asarray(run(jnp.asarray(savings_at_three)))
    child_age = period + 1
    expected = np.asarray(
        [
            (1.0 + slope * child_age) * 3.0 + transfer * child_age,
            law_slope * (1.0 + slope * child_age),
        ],
        dtype=actual.dtype,
    )
    np.testing.assert_array_equal(actual, expected)


def test_age_invariant_source_shares_a_child_read_only_for_an_invariant_child(
    *, source_child_case: tuple[Model, float, float, float, float]
) -> None:
    """The source's periods share one child read iff the child's map is invariant.

    The source itself is the same at both periods, so the child's age is the only
    part of the context that can keep its two reads apart.
    """
    model, slope, transfer, _, _ = source_child_case
    reads = [
        _continuation_plan(model=model, regime="source", period=period).child_reads[
            "child"
        ]
        for period in (0, 1)
    ]
    assert (reads[0] is reads[1]) == (not slope and not transfer)
