"""Scalar borrowing floors move EGM nodes before continuation evaluation."""

import logging
from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from dags import rename_arguments

from _lcm.egm import one_asset_egm_step as step_module
from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.scalar_query import ScalarEGMQuery
from _lcm.execution.core_program import CoreBuildContext
from _lcm.regime_building.processing import _fail_if_a_continuation_demand_is_unmet
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.egm import (
    OneRowOutput,
    _BoundEGM,
    _EGMArgumentBuilder,
    _EGMCore,
    _EGMPeriodKernel,
    _RegimeMassGuardedCore,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    FeasibilityKernelOutput,
)
from _lcm.typing import EconFunctionKwargs, PytreeValue
from benchmarks.asv._compile_counters import count_compile_requests
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
    post_decision_lower_bound,
)
from lcm.exceptions import (
    GridInitializationError,
    InvalidValueFunctionError,
    NoFeasibleActionError,
    RegimeInitializationError,
)
from lcm.solver_api import EGM_CONTINUATION, ContinuationCapabilities
from lcm.solvers import EGM, NBEGM
from lcm.typing import (
    Age,
    ContinuousAction,
    ContinuousState,
    Float1D,
    FloatND,
    ScalarFloat,
    ScalarInt,
    UserFunction,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION
from tests.solution._nbegm_direct_oracle import OracleContext, ride_along_kernel


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def inverse_marginal_utility(marginal_continuation: FloatND) -> ContinuousAction:
    return 1.0 / marginal_continuation


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def savings(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def next_wealth(savings: FloatND) -> ContinuousState:
    return 3.0 * savings + 4.0


def lower(floor: float) -> FloatND:
    return jnp.asarray(floor)


def build_dynamic_egm_model(
    *,
    literal: float | None = None,
    lower_function: UserFunction = lower,
    source_nodes: tuple[float, ...] = (0.0, 0.5, 1.0),
    current_nodes: tuple[float, ...] = (0.75, 1.0, 2.0, 3.0),
    terminal_nodes: tuple[float, ...] = (4.0, 5.5, 7.0, 8.5),
    law: UserFunction = next_wealth,
    analytic_inverse: bool = True,
    late_start: bool = False,
) -> Model:
    """Build a log problem with exactly represented corner continuation queries."""
    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )
    return Model(
        regimes={
            "saving": ConsumptionSavingsRegime(
                liquid=margin,
                states={"wealth": IrregSpacedGrid(points=np.array(current_nodes))},
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.25, 0.5, 1.0, 2.0])
                    )
                },
                functions={
                    "utility": utility,
                    **(
                        {"inverse_marginal_utility": inverse_marginal_utility}
                        if analytic_inverse
                        else {}
                    ),
                    "savings": savings,
                },
                state_transitions={"wealth": law},
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=margin,
                        lower=lower_function if literal is None else literal,
                    )
                },
                solver=EGM(
                    savings_grid=IrregSpacedGrid(
                        points=np.array(
                            source_nodes
                            if literal is None
                            else tuple(
                                literal + (node - source_nodes[0])
                                for node in source_nodes
                            )
                        ),
                    )
                ),
            ),
            "done": Regime(
                states={"wealth": IrregSpacedGrid(points=np.array(terminal_nodes))},
                functions={"utility": terminal_utility},
            ),
        },
        ages=AgeGrid(exact_values=(10, 20, 30))
        if late_start
        else AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"saving": {"done": 20 if late_start else 0}},
        initial_nodes={20 if late_start else 0: "saving"},
        regime_id_class=RegimeId,
    )


def dynamic_egm_params(*, floor: float, literal: bool = False) -> UserParams:
    """Bind unit discounting and the callable floor's own parameter namespace."""
    return {
        "saving": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            **({} if literal else {"borrowing_lower": {"floor": jnp.asarray(floor)}}),
        },
        "done": {},
    }


@pytest.mark.parametrize(
    ("floor", "expected"), [(0.0, np.log(4.0)), (0.5, np.log(2.75))]
)
def test_dynamic_egm_binding_corner_has_analytic_value(
    *, floor: float, expected: float
) -> None:
    """Strict concavity and negative saving slope make the declared floor binding."""
    model = build_dynamic_egm_model()
    solution = model.solve(params=dynamic_egm_params(floor=floor), log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["saving"][1],
        expected,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_dynamic_egm_constant_callable_matches_literal(*, floor: float) -> None:
    """Equivalent scalar floors use the same translated saving nodes and values."""
    dynamic = build_dynamic_egm_model().solve(
        params=dynamic_egm_params(floor=floor),
        log_level="off",
    )
    literal = build_dynamic_egm_model(literal=floor).solve(
        params=dynamic_egm_params(floor=floor, literal=True),
        log_level="off",
    )
    np.testing.assert_allclose(
        dynamic.values[0]["saving"],
        literal.values[0]["saving"],
        rtol=1000 * np.finfo(np.asarray(dynamic.values[0]["saving"]).dtype).eps,
        atol=1000 * np.finfo(np.asarray(dynamic.values[0]["saving"]).dtype).eps,
    )


def scalar_context_output(
    *, floor: float, model: Model | None = None, period: int = 0
) -> tuple[Model, _EGMPeriodKernel, OracleContext, FeasibilityKernelOutput]:
    """Run the installed kernel with its actual captured solved child inputs."""
    model = build_dynamic_egm_model() if model is None else model
    kernel, context = ride_along_kernel(
        model=model,
        params=dynamic_egm_params(floor=floor),
        regime_name="saving",
        period=period,
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    program = kernel.core_programs()["main"]
    output = kernel(
        compiled_cores={"main": program.function},
        **context,
        logger=logging.getLogger(__name__),
    )
    assert isinstance(output, FeasibilityKernelOutput)
    return model, kernel, context, output


@pytest.mark.parametrize(("floor", "marginal"), [(0.0, 1.0), (0.5, 2.0)])
def test_dynamic_egm_corner_publishes_analytic_marginal(
    *, floor: float, marginal: float
) -> None:
    """A binding corner carries the marginal of its actual consumption."""
    _, _, _, output = scalar_context_output(floor=floor)
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    np.testing.assert_array_almost_equal(
        carry.leaves()[("marginal_utility",)][1],
        marginal,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_dynamic_egm_corner_keeps_original_saving(*, floor: float) -> None:
    """The carried policy retains the selected original saving candidate."""
    _, _, _, output = scalar_context_output(floor=floor)
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    np.testing.assert_array_equal(carry.leaves()[("query_data", "savings")][1], floor)


@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_dynamic_egm_corner_continuation_survives_current_mask(*, floor: float) -> None:
    """The compact corner stores continuation separately from current utility."""
    _, _, _, output = scalar_context_output(floor=floor)
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    np.testing.assert_array_almost_equal(
        carry.leaves()[("query_data", "corner_continuation")],
        [np.log(3 * floor + 4)],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("runtime_checks", [True, False])
@pytest.mark.parametrize("floor", [np.nan, np.inf])
def test_dynamic_egm_nonfinite_floor_is_mandatory(
    *, floor: float, runtime_checks: bool
) -> None:
    """Invalid floor arithmetic raises before accepting an economic result."""

    def invalid_lower(floor: float) -> FloatND:
        return jnp.asarray(floor) * jnp.asarray(invalid)

    invalid = floor
    with pytest.raises(
        InvalidValueFunctionError,
        match=r"borrowing.*nonfinite borrowing floor.*saving.*period 0",
    ):
        build_dynamic_egm_model(lower_function=invalid_lower).solve(
            params=dynamic_egm_params(floor=1.0),
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("runtime_checks", [True, False])
def test_dynamic_egm_collapsed_shifted_grid_is_mandatory(
    *, runtime_checks: bool
) -> None:
    """A finite floor cannot collapse strict source offsets into repeated nodes."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype
    floor = float(np.finfo(dtype).max / 2)
    with pytest.raises(
        InvalidValueFunctionError, match=r"borrowing.*invalid shifted savings grid"
    ):
        build_dynamic_egm_model().solve(
            params=dynamic_egm_params(floor=floor),
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("runtime_checks", [True, False])
@pytest.mark.parametrize("offset_failure", ["collapse", "overflow"])
def test_dynamic_egm_invalid_original_offsets_keep_bound_attribution(
    *, runtime_checks: bool, offset_failure: Literal["collapse", "overflow"]
) -> None:
    """Invalid represented offsets raise the mandatory bound error before law reads."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype
    magnitude = 1e20 if offset_failure == "collapse" else np.finfo(dtype).max
    nodes = (-float(magnitude), 1.0, 2.0)
    if offset_failure == "overflow":
        nodes = (-float(magnitude), 0.0, float(magnitude))
    model = build_dynamic_egm_model(source_nodes=nodes)
    with pytest.raises(
        InvalidValueFunctionError,
        match=r"borrowing.*invalid shifted savings grid.*saving.*period 0",
    ):
        model.solve(
            params=dynamic_egm_params(floor=0.5),
            log_level="off",
            runtime_checks=runtime_checks,
        )


def scalar_reader(
    *, floor: float
) -> tuple[ScalarEGMQuery, EGMCarry, EGMQueryData, EconFunctionKwargs]:
    """Bind the actual installed row reader and its canonical runtime inputs."""
    _, kernel, _, output = scalar_context_output(floor=floor)
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    builder = kernel.core_programs()["main"].argument_builder
    assert isinstance(builder, _EGMArgumentBuilder)
    reader = builder.current_query
    assert reader is not None
    data = carry.query_data_for_transport()
    assert data is not None
    return (
        reader,
        carry,
        data,
        {
            "borrowing_lower__floor": jnp.asarray(floor),
            "koopmans_aggregator__discount_factor": jnp.asarray(1.0),
            "period": jnp.int32(0),
            "age": jnp.int32(0),
        },
    )


@pytest.mark.parametrize("query", [1.0, 0.5, 0.25])
def test_dynamic_egm_query_rejects_foreign_floor_provenance(*, query: float) -> None:
    """Foreign same-shaped rows cannot supply a different generation floor."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    result = reader(
        query=jnp.array([query]),
        grid=carry.leaves()[("endog_grid",)],
        value=carry.leaves()[("value",)],
        marginal=carry.leaves()[("marginal_utility",)],
        data=replace(data, floor_used=jnp.asarray(0.0)),
        pool=pool,
    )
    assert bool(jnp.any(result.coverage_missing))


@pytest.mark.parametrize("query", [0.5, 0.25])
def test_dynamic_egm_valid_empty_query_keeps_economic_minus_infinity(
    *, query: float
) -> None:
    """A correctly attributed empty query is economic rather than missing support."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    result = reader(
        query=jnp.array([query]),
        grid=carry.endog_grid,
        value=carry.value,
        marginal=carry.marginal_utility,
        data=data,
        pool=pool,
    )
    np.testing.assert_array_equal(result.value, [-np.inf])
    np.testing.assert_array_equal(result.marginal, [0.0])
    np.testing.assert_array_equal(result.coverage_missing, [False])


@pytest.mark.parametrize("supported", [True, False])
def test_dynamic_egm_economic_row_requires_ordinary_support(*, supported: bool) -> None:
    """A represented impossible row is distinct from an all-padding interior."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    kinds = jnp.full(data.node_kind.shape, QueryNodeKind.PAD, dtype=jnp.int32)
    if supported:
        kinds = kinds.at[1].set(QueryNodeKind.ECONOMIC_MINUS_INF)
    result = reader(
        query=jnp.array([1.0]),
        grid=carry.endog_grid,
        value=jnp.full(carry.value.shape, -jnp.inf),
        marginal=jnp.zeros(carry.marginal_utility.shape),
        data=replace(
            data,
            node_kind=kinds,
            link_valid=jnp.zeros(data.link_valid.shape, dtype=bool),
            corner_kind=jnp.array([QueryNodeKind.ECONOMIC_MINUS_INF], dtype=jnp.int32),
            corner_continuation=jnp.array([-jnp.inf]),
        ),
        pool=pool,
    )
    np.testing.assert_array_equal(result.value, [-np.inf])
    np.testing.assert_array_equal(result.marginal, [0.0])
    np.testing.assert_array_equal(result.coverage_missing, [not supported])


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_dynamic_egm_finite_corner_cannot_hide_invalid_active_row(
    *, invalid: float
) -> None:
    """An attempted live numerical read reports its invalid value before recovery."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    result = reader(
        query=jnp.array([1.0]),
        grid=carry.endog_grid,
        value=carry.value.at[1].set(invalid),
        marginal=carry.marginal_utility,
        data=data,
        pool=pool,
    )
    np.testing.assert_array_equal(result.coverage_missing, [True])


def test_dynamic_egm_floor_descriptor_seals_discounting() -> None:
    """Query discounting belongs to the independently bound evaluator identity."""
    model = build_dynamic_egm_model()
    authority = model._regimes["saving"].solution.query_authority
    assert authority is not None
    assert "koopmans_aggregator" in dict(authority.evaluator_targets)


def period_lower(*, period: ScalarInt, floor: float) -> FloatND:
    return floor + 0.5 * period


def age_lower(*, age: Age, floor: float) -> FloatND:
    return floor + age / 40.0


@pytest.mark.parametrize("lower_function", [period_lower, age_lower])
def test_dynamic_egm_floor_uses_actual_period_and_age(
    *, lower_function: UserFunction
) -> None:
    """Period one at age twenty evaluates the declared floor as one half."""
    model = build_dynamic_egm_model(lower_function=lower_function, late_start=True)
    solution = model.solve(params=dynamic_egm_params(floor=0.0), log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[1]["saving"][1], np.log(2.75), decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("floor", [-0.1, 0.25])
def test_dynamic_egm_negative_and_positive_floors_have_analytic_corners(
    *, floor: float
) -> None:
    """A binding floor permits exactly consumption one minus the declared floor."""
    terminal_nodes = tuple(3 * (floor + offset) + 4 for offset in (0, 0.5, 1, 1.5))
    model = build_dynamic_egm_model(terminal_nodes=terminal_nodes)
    solution = model.solve(params=dynamic_egm_params(floor=floor), log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["saving"][1],
        np.log((1 - floor) * (3 * floor + 4)),
        decimal=DECIMAL_PRECISION,
    )


def long_next_wealth(savings: FloatND) -> ContinuousState:
    return 3.0 * savings + 30.0


def test_dynamic_egm_numeric_inverse_uses_translated_offsets() -> None:
    """The interior optimum is saving 2.5 and consumption 12.5 at liquid 15."""
    model = build_dynamic_egm_model(
        source_nodes=(-10.0, -7.5, -5.0),
        current_nodes=(10.0, 15.0, 20.0),
        terminal_nodes=(30.0, 37.5, 45.0, 60.0),
        law=long_next_wealth,
        analytic_inverse=False,
    )
    _, _, _, output = scalar_context_output(floor=0.0, model=model)
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    np.testing.assert_array_almost_equal(
        carry.leaves()[("value",)][1],
        np.log(468.75),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        carry.leaves()[("query_data", "savings")][1],
        2.5,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        carry.leaves()[("marginal_utility",)][1],
        0.08,
        decimal=DECIMAL_PRECISION,
    )


def test_dynamic_egm_large_origin_preserves_offset_arithmetic() -> None:
    """Original spacing four remains shifted nodes one, five and nine."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype
    origin = float(2 ** (25 if dtype == np.float32 else 54))
    model = build_dynamic_egm_model(
        source_nodes=(origin, origin + 4, origin + 8),
        current_nodes=(2.0, 3.0, 4.0),
        terminal_nodes=(7.0, 19.0, 31.0, 43.0),
    )
    _, kernel, context, output = scalar_context_output(floor=1.0, model=model)
    arguments = kernel.core_programs()["main"].argument_builder(
        CoreBuildContext(**context)
    )
    np.testing.assert_array_equal(arguments["effective_savings_grid"], [1, 5, 9])
    np.testing.assert_array_almost_equal(
        output.value[0],
        np.log(7.0),
        decimal=DECIMAL_PRECISION,
    )


def test_dynamic_egm_domain_uses_exact_adjacent_float_comparisons() -> None:
    """The floor and its lower neighbor are empty; its upper neighbor consumes."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype
    floor = dtype.type(0.5)
    below = np.nextafter(floor, dtype.type(-np.inf))
    above = np.nextafter(floor, dtype.type(np.inf))
    model = build_dynamic_egm_model(
        current_nodes=(float(below), float(floor), float(above), 1.0),
    )
    _, _, _, output = scalar_context_output(floor=float(floor), model=model)
    np.testing.assert_array_equal(np.isneginf(output.value[:2]), [True, True])
    np.testing.assert_array_almost_equal(
        output.value[2],
        np.log(float(above - floor) * 5.5),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(output.feasibility_counts, [2, 0])


@categorical(ordered=False)
class ParentRegimeId:
    parent: ScalarInt
    saving: ScalarInt
    done: ScalarInt


def parent_next_wealth(savings: FloatND) -> ContinuousState:
    return 5.0 * savings - 1.0


def build_parent_model(
    *,
    lower_function: UserFunction = lower,
    own_floor: bool = False,
    parent_law: UserFunction = parent_next_wealth,
    current_nodes: tuple[float, ...] = (0.5, 0.6, 0.8, 1.0),
    enable_jit: bool = True,
    literal_child: bool = False,
    nb_parent: bool = False,
) -> Model:
    """Build a parent with no DAG floor and a separately attributed scalar child."""
    state_name = "liquid" if nb_parent else "wealth"
    margin = LiquidMargin(
        state=state_name,
        action="consumption",
        resources=state_name,
        post_decision_state="savings",
    )
    functions = {
        "utility": utility,
        "inverse_marginal_utility": inverse_marginal_utility,
        "savings": rename_arguments(savings, mapper={"wealth": "liquid"})
        if nb_parent
        else savings,
    }
    return Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={state_name: IrregSpacedGrid(points=np.array(current_nodes))},
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.05, 0.1, 0.2, 0.4])
                    )
                },
                functions=functions,
                state_transitions={state_name: parent_law},
                constraints={
                    "parent_borrowing": post_decision_lower_bound(
                        margin=margin, lower=lower
                    )
                }
                if own_floor
                else {},
                solver=(NBEGM if nb_parent else EGM)(
                    savings_grid=IrregSpacedGrid(points=np.array([0.4, 0.5, 0.6]))
                ),
            ),
            "saving": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    state_name: IrregSpacedGrid(points=np.array([0.75, 1.0, 2.0, 3.0]))
                },
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.25, 0.5, 1.0, 2.0])
                    )
                },
                functions=functions,
                state_transitions={state_name: next_wealth},
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=margin, lower=0.5 if literal_child else lower_function
                    )
                },
                solver=EGM(
                    savings_grid=IrregSpacedGrid(
                        points=np.array(
                            (0.5, 1.0, 1.5) if literal_child else (0.0, 0.5, 1.0)
                        )
                    )
                ),
            ),
            "done": Regime(
                states={
                    state_name: IrregSpacedGrid(points=np.array([4.0, 5.5, 7.0, 8.5]))
                },
                functions={
                    "utility": rename_arguments(
                        terminal_utility, mapper={"wealth": "liquid"}
                    )
                    if nb_parent
                    else terminal_utility
                },
            ),
        },
        ages=AgeGrid(exact_values=(10, 20, 30)),
        edges={"parent": {"saving": 10}, "saving": {"done": 20}},
        initial_nodes={10: "parent"},
        regime_id_class=ParentRegimeId,
        enable_jit=enable_jit,
    )


def parent_params(
    *, floor: float, own_floor: float | None = None, literal_child: bool = False
) -> UserParams:
    return {
        "parent": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            **(
                {}
                if own_floor is None
                else {"parent_borrowing_lower": {"floor": own_floor}}
            ),
        },
        **dynamic_egm_params(floor=floor, literal=literal_child),
    }


def parent_context(
    *, own_floor: bool = False, floor: float = 0.5
) -> tuple[Model, _EGMPeriodKernel, OracleContext]:
    """Capture a valid child before exercising its actual parent adapter."""
    model = build_parent_model(own_floor=own_floor, enable_jit=False)
    kernel, context = ride_along_kernel(
        model=model,
        params=parent_params(floor=floor, own_floor=0.4 if own_floor else None),
        regime_name="parent",
        period=0,
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    return model, kernel, context


@pytest.mark.parametrize(
    ("floor", "index", "expected"), [(0.0, 1, np.log(0.8)), (0.5, 0, np.log(0.275))]
)
def test_dynamic_egm_parent_value_uses_child_floor(
    *, floor: float, index: int, expected: float
) -> None:
    """The parent continuation uses its child's canonical namespace and corner."""
    model = build_parent_model()
    solution = model.solve(params=parent_params(floor=floor), log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"][index], expected, decimal=DECIMAL_PRECISION
    )


def offgrid_parent_next_wealth(savings: FloatND) -> ContinuousState:
    return 5.0 * savings - 0.75


def test_dynamic_egm_parent_corner_uses_actual_child_query() -> None:
    """The child corner at liquid 1.25 enters the constrained parent value."""
    model = build_parent_model(parent_law=offgrid_parent_next_wealth)
    solution = model.solve(params=parent_params(floor=0.5), log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"][0],
        np.log(0.4125),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("runtime_checks", [True, False])
def test_dynamic_egm_parent_accepts_child_integrity_before_economics(
    *, runtime_checks: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child's invalid actual query is attributed independently of parent bounds."""
    model = build_parent_model(enable_jit=False)
    kernel, _ = ride_along_kernel(
        model=model, params=parent_params(floor=0.5), regime_name="parent", period=0
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    guard = kernel.core_programs()["main"].function
    assert isinstance(guard, _RegimeMassGuardedCore)
    core = guard.core
    assert isinstance(core, _EGMCore)
    target_reader = core.next_query
    assert target_reader is not None
    original = ScalarEGMQuery.evaluate_floor
    calls = []

    # keyword-only-exempt: primary-argument=self
    def seeded_floor(
        self: ScalarEGMQuery, *, pool: EconFunctionKwargs
    ) -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
        floor, nodes, status = original(self, pool=pool)
        if self is target_reader:
            calls.append(float(floor))
            status = BoundInputStatus(
                nonfinite_floor=jnp.asarray(a=True),
                invalid_shifted_grid=status.invalid_shifted_grid,
            )
        return floor, nodes, status

    monkeypatch.setattr(ScalarEGMQuery, "evaluate_floor", seeded_floor)
    with pytest.raises(
        InvalidValueFunctionError,
        match=r"borrowing.*nonfinite borrowing floor.*saving.*period 1",
    ):
        model.solve(
            params=parent_params(floor=0.5),
            runtime_checks=runtime_checks,
            log_level="off",
        )
    assert calls == [0.5]


def test_dynamic_egm_parent_requires_feature_child_record() -> None:
    """Trusted feature-child authority cannot fall back to an ordinary reader."""
    _, kernel, context = parent_context()
    child = context["next_regime_to_continuation"]["saving"]
    assert isinstance(child, EGMCarry)
    context["next_regime_to_continuation"] = {"saving": replace(child, query_data=None)}
    with pytest.raises(TypeError, match="model-bound scalar query lacks"):
        kernel(
            compiled_cores={"main": kernel.core_programs()["main"].function},
            **context,
            logger=logging.getLogger(__name__),
        )


def test_dynamic_egm_parent_requires_inherited_integrity_record() -> None:
    """A parent without a DAG floor still requires its child's mandatory record."""
    _, kernel, context = parent_context()
    original = kernel.core_programs()["main"].function

    def drop_records(**arguments: PytreeValue) -> OneRowOutput:
        evaluated = original(**arguments)
        assert isinstance(evaluated, BoundedResult)
        return replace(evaluated, bound_status=())

    with pytest.raises(TypeError, match="model-owned declaration slots"):
        kernel(
            compiled_cores={"main": drop_records},
            **context,
            logger=logging.getLogger(__name__),
        )


def test_dynamic_egm_source_integrity_precedes_child_coverage(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid source input is accepted before a child's representation failure."""
    _, kernel, context = parent_context(own_floor=True)
    builder = kernel.core_programs()["main"].argument_builder
    assert isinstance(builder, _EGMArgumentBuilder)
    source_reader = builder.current_query
    assert source_reader is not None
    child = context["next_regime_to_continuation"]["saving"]
    assert isinstance(child, EGMCarry)
    data = child.query_data_for_transport()
    assert data is not None
    assert data.corner_kind is not None
    damaged = replace(
        data,
        node_kind=jnp.full(data.node_kind.shape, QueryNodeKind.PAD, dtype=jnp.int32),
        corner_kind=jnp.full(
            data.corner_kind.shape, QueryNodeKind.PAD, dtype=jnp.int32
        ),
    )
    context["next_regime_to_continuation"] = {
        "saving": replace(child, query_data=damaged)
    }
    original = ScalarEGMQuery.evaluate_floor
    calls = []

    # keyword-only-exempt: primary-argument=self
    def seeded_floor(
        self: ScalarEGMQuery, *, pool: EconFunctionKwargs
    ) -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
        floor, nodes, status = original(self, pool=pool)
        if self is source_reader:
            calls.append(float(floor))
            status = BoundInputStatus(
                nonfinite_floor=jnp.asarray(a=True),
                invalid_shifted_grid=status.invalid_shifted_grid,
            )
        return floor, nodes, status

    monkeypatch.setattr(ScalarEGMQuery, "evaluate_floor", seeded_floor)
    with pytest.raises(
        InvalidValueFunctionError,
        match=r"parent_borrowing.*nonfinite borrowing floor.*parent.*period 0",
    ):
        kernel(
            compiled_cores={"main": kernel.core_programs()["main"].function},
            **context,
            logger=logging.getLogger(__name__),
        )
    assert calls == [float(jnp.asarray(0.4))]


@pytest.mark.parametrize(
    "damage", ["drop", "foreign_target", "foreign_period", "value_only"]
)
def test_dynamic_egm_parent_requires_its_installed_reader_declaration(
    *,
    damage: Literal["drop", "foreign_target", "foreign_period", "value_only"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a complete declaration from the consuming adapter serves a feature child."""
    original = _BoundEGM.build_period_kernels
    calls = []

    # keyword-only-exempt: primary-argument=self
    def damaged_build(
        self: _BoundEGM, *, context: SolverBuildContext
    ) -> SolutionKernels:
        built = original(self, context=context)
        if context.regime_name != "parent":
            return built
        calls.append(context.regime_name)
        declared = built.continuation_reader_capabilities
        assert declared is not None
        offered = declared[0]["saving"]
        if damage == "drop":
            replacement = None
        elif damage == "foreign_target":
            replacement = {0: {"done": offered}}
        elif damage == "foreign_period":
            replacement = {1: {"saving": offered}}
        else:
            replacement = {0: {"saving": ContinuationCapabilities(value=True)}}
        return replace(built, continuation_reader_capabilities=replacement)

    monkeypatch.setattr(_BoundEGM, "build_period_kernels", damaged_build)
    with pytest.raises(
        RegimeInitializationError, match=r"parent.*saving.*reader answers no"
    ):
        build_parent_model()
    assert calls == ["parent"]


def test_dynamic_egm_parent_literal_keeps_linear_query_approximation() -> None:
    """The literal child retains its ordinary interpolated value at liquid 1.25."""
    model = build_parent_model(
        parent_law=offgrid_parent_next_wealth, literal_child=True
    )
    solution = model.solve(
        params=parent_params(floor=0.5, literal_child=True),
        log_level="off",
    )
    expected = np.log(0.1) + 0.75 * np.log(2.75) + 0.25 * np.log(8.25)
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"][0],
        expected,
        decimal=DECIMAL_PRECISION,
    )
    assert model._regimes["parent"].solution.continuation_reader_capabilities is None


@pytest.mark.parametrize(("floor", "expected"), [(0.0, 5.0), (0.5, 10.0)])
def test_dynamic_egm_parent_composes_child_marginal_once(
    *, floor: float, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Euler target contains exactly one parent-to-child derivative factor."""
    _, kernel, context = parent_context(floor=floor)
    original = step_module.invert_euler
    calls: list[FloatND] = []

    def measured_invert(
        *,
        expected_marginal_continuation: FloatND,
        discount_factor: ScalarFloat | float,
        inverse_marginal_utility: Callable[[FloatND], FloatND],
    ) -> FloatND:
        calls.append(expected_marginal_continuation)
        return original(
            expected_marginal_continuation=expected_marginal_continuation,
            discount_factor=discount_factor,
            inverse_marginal_utility=inverse_marginal_utility,
        )

    monkeypatch.setattr(step_module, "invert_euler", measured_invert)
    kernel(
        compiled_cores={"main": kernel.core_programs()["main"].function},
        **context,
        logger=logging.getLogger(__name__),
    )
    assert len(calls) == 1
    np.testing.assert_array_almost_equal(
        calls[0][0], expected, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("lower_function", [period_lower, age_lower])
def test_dynamic_egm_parent_resolves_target_time(
    *, lower_function: UserFunction
) -> None:
    """A parent at age ten reads the child floor at period one and age twenty."""
    model = build_parent_model(lower_function=lower_function)
    solution = model.solve(params=parent_params(floor=0.0), log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"][0],
        np.log(0.275),
        decimal=DECIMAL_PRECISION,
    )


def test_dynamic_egm_parent_rejects_foreign_runtime_period() -> None:
    """A child reader's target period cannot be supplied by a foreign replay context."""
    _, kernel, context = parent_context()
    context["period"] = 1
    with pytest.raises(TypeError, match=r"period.*authority"):
        kernel(
            compiled_cores={"main": kernel.core_programs()["main"].function},
            **context,
            logger=logging.getLogger(__name__),
        )


def test_dynamic_egm_parent_current_empty_domains_are_economic() -> None:
    """A parent reading a feature child still masks its own nonpositive consumption."""
    model = build_parent_model(current_nodes=(0.3, 0.4, 0.5, 0.6))
    solution = model.solve(
        params=parent_params(floor=0.5),
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_array_equal(
        solution.values[0]["parent"][:2],
        [-np.inf, -np.inf],
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"][2],
        np.log(0.275),
        decimal=DECIMAL_PRECISION,
    )


def test_dynamic_egm_read_hook_cannot_change_adapter_capabilities(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaf declaration preserves the capabilities of the original built adapter."""
    original = _BoundEGM.declare_continuation_reads

    # keyword-only-exempt: primary-argument=self
    def damaged_declaration(
        self: _BoundEGM, *, kernels: SolutionKernels, context: SolverBuildContext
    ) -> SolutionKernels:
        declared = original(self, kernels=kernels, context=context)
        return (
            replace(declared, continuation_reader_capabilities=None)
            if context.regime_name == "parent"
            else declared
        )

    monkeypatch.setattr(_BoundEGM, "declare_continuation_reads", damaged_declaration)
    with pytest.raises(
        RegimeInitializationError, match="continuation_reader_capabilities"
    ):
        build_parent_model()


def test_dynamic_egm_unreached_demand_keeps_raw_capability_requirement() -> None:
    """A reader for another edge cannot satisfy an unreached feature-child demand."""
    model = build_parent_model(enable_jit=False)
    parent = model._regimes["parent"]
    graph = parent.solution.reachability
    declared = parent.solution.continuation_reader_capabilities
    assert declared is not None
    other_edge = {0: {"done": declared[0]["saving"]}}
    graph = replace(
        graph,
        targets_by_period=(
            MappingProxyType({**graph.targets_by_period[0], "parent": ("done",)}),
            MappingProxyType({**graph.targets_by_period[1], "parent": ("saving",)}),
        ),
    )
    regimes = {
        **model._regimes,
        "parent": replace(
            parent,
            solution=replace(
                parent.solution,
                reachability=graph,
                continuation_reader_capabilities=other_edge,
            ),
        ),
    }
    with pytest.raises(
        RegimeInitializationError, match=r"parent.*saving.*reader answers no"
    ):
        _fail_if_a_continuation_demand_is_unmet(
            demands=(("parent", "saving", EGM_CONTINUATION),),
            canonical_regimes=regimes,
            user_regimes=model.user_regimes,
        )


def test_dynamic_egm_child_does_not_activate_an_nb_parent_reader() -> None:
    """A feature child cannot grant its scalar reader to another solver family."""
    with pytest.raises(
        RegimeInitializationError, match=r"parent.*saving.*reader answers no"
    ):
        build_parent_model(enable_jit=False, nb_parent=True)


@pytest.mark.parametrize("location", ["row", "corner"])
@pytest.mark.parametrize("label", ["owner", "side"])
def test_dynamic_egm_query_requires_its_owned_piece(
    *, location: Literal["row", "corner"], label: Literal["owner", "side"]
) -> None:
    """Foreign interval or one-sided labels cannot supply owned query support."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    if location == "row":
        change = (
            {"owner_id": data.owner_id.at[1].set(1)}
            if label == "owner"
            else {"side_id": data.side_id.at[1].set(1)}
        )
        data = replace(
            data,
            **change,
            corner_kind=jnp.array([QueryNodeKind.PAD], dtype=jnp.int32),
        )
    else:
        change = (
            {"corner_interval_id": jnp.array([1], dtype=jnp.int32)}
            if label == "owner"
            else {"corner_side_id": jnp.array([1], dtype=jnp.int32)}
        )
        data = replace(
            data,
            **change,
            node_kind=jnp.full(
                data.node_kind.shape, QueryNodeKind.PAD, dtype=jnp.int32
            ),
        )
    result = reader(
        query=jnp.array([1.0]),
        grid=carry.endog_grid,
        value=carry.value,
        marginal=carry.marginal_utility,
        data=data,
        pool=pool,
    )
    np.testing.assert_array_equal(result.coverage_missing, [True])


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_dynamic_egm_owned_corner_recovers_an_unowned_row(*, invalid: float) -> None:
    """An unowned row supplies no arithmetic, while the owned finite corner remains."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    result = reader(
        query=jnp.array([1.0]),
        grid=carry.endog_grid,
        value=carry.value.at[1].set(invalid),
        marginal=carry.marginal_utility,
        data=replace(data, owner_id=data.owner_id.at[1].set(1)),
        pool=pool,
    )
    np.testing.assert_array_almost_equal(
        result.value, [np.log(2.75)], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result.marginal, [2.0])
    np.testing.assert_array_equal(result.coverage_missing, [False])


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_dynamic_egm_empty_domain_keeps_invalid_active_read(*, invalid: float) -> None:
    """Economic emptiness cannot erase an invalid attempted finite-node read."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    result = reader(
        query=jnp.array([0.25]),
        grid=carry.endog_grid,
        value=carry.value.at[0].set(invalid),
        marginal=carry.marginal_utility,
        data=data,
        pool=pool,
    )
    np.testing.assert_array_equal(result.coverage_missing, [True])


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_dynamic_egm_valid_row_cannot_hide_invalid_finite_corner(
    *, invalid: float
) -> None:
    """Every attempted finite corner record must carry valid continuation arithmetic."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    result = reader(
        query=jnp.array([1.0]),
        grid=carry.endog_grid,
        value=carry.value,
        marginal=carry.marginal_utility,
        data=replace(data, corner_continuation=jnp.array([invalid])),
        pool=pool,
    )
    np.testing.assert_array_equal(result.coverage_missing, [True])


@pytest.mark.parametrize(
    ("location", "kind", "value", "marginal"),
    [
        ("row", QueryNodeKind.FINITE, -np.inf, 0.0),
        ("row", QueryNodeKind.ECONOMIC_MINUS_INF, 0.0, 0.0),
        ("row", QueryNodeKind.ECONOMIC_MINUS_INF, -np.inf, 1.0),
        ("corner", QueryNodeKind.FINITE, -np.inf, 0.0),
        ("corner", QueryNodeKind.ECONOMIC_MINUS_INF, 0.0, 0.0),
    ],
)
def test_dynamic_egm_query_kind_matches_its_numerical_record(
    *,
    location: Literal["row", "corner"],
    kind: QueryNodeKind,
    value: float,
    marginal: float,
) -> None:
    """Finite and economic-minus-infinity labels describe their associated numbers."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    row_value, row_marginal = carry.value, carry.marginal_utility
    if location == "row":
        data = replace(data, node_kind=data.node_kind.at[1].set(kind))
        row_value = row_value.at[1].set(value)
        row_marginal = row_marginal.at[1].set(marginal)
    else:
        data = replace(
            data,
            corner_kind=jnp.array([kind], dtype=jnp.int32),
            corner_continuation=jnp.array([value]),
        )
    result = reader(
        query=jnp.array([1.0]),
        grid=carry.endog_grid,
        value=row_value,
        marginal=row_marginal,
        data=data,
        pool=pool,
    )
    np.testing.assert_array_equal(result.coverage_missing, [True])


@pytest.mark.parametrize("location", ["row", "corner"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_dynamic_egm_padding_poison_is_not_a_finite_read(
    *, location: Literal["row", "corner"], invalid: float
) -> None:
    """Padding never supplies numerical evidence or an active invalid-input flag."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    row_value, row_marginal = carry.value, carry.marginal_utility
    if location == "row":
        data = replace(
            data,
            node_kind=jnp.full(
                data.node_kind.shape, QueryNodeKind.PAD, dtype=jnp.int32
            ),
            link_valid=jnp.zeros(data.link_valid.shape, dtype=bool),
        )
        row_value = jnp.full(row_value.shape, invalid)
        row_marginal = jnp.full(row_marginal.shape, invalid)
    else:
        data = replace(
            data,
            corner_kind=jnp.array([QueryNodeKind.PAD], dtype=jnp.int32),
            corner_continuation=jnp.array([invalid]),
        )
    result = reader(
        query=jnp.array([1.0]),
        grid=carry.endog_grid,
        value=row_value,
        marginal=row_marginal,
        data=data,
        pool=pool,
    )
    np.testing.assert_array_almost_equal(
        result.value, [np.log(2.75)], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result.marginal, [2.0])
    np.testing.assert_array_equal(result.coverage_missing, [False])


def test_dynamic_egm_positive_economic_weight_has_neutral_marginal() -> None:
    """A supported impossible continuation yields minus infinity and zero."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    data = replace(
        data,
        node_kind=data.node_kind.at[2].set(QueryNodeKind.ECONOMIC_MINUS_INF),
        corner_kind=jnp.array([QueryNodeKind.ECONOMIC_MINUS_INF], dtype=jnp.int32),
        corner_continuation=jnp.array([-jnp.inf]),
    )
    result = reader(
        query=jnp.array([1.5]),
        grid=carry.endog_grid,
        value=carry.value.at[2].set(-jnp.inf),
        marginal=carry.marginal_utility.at[2].set(0.0),
        data=data,
        pool=pool,
    )
    np.testing.assert_array_equal(result.value, [-np.inf])
    np.testing.assert_array_equal(result.marginal, [0.0])
    np.testing.assert_array_equal(result.coverage_missing, [False])


def test_dynamic_egm_negative_economic_weight_recovers_finite_corner() -> None:
    """Undefined extrapolation of an economic infinity leaves a finite legal corner."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    result = reader(
        query=jnp.array([0.6]),
        grid=carry.endog_grid,
        value=carry.value.at[1].set(-jnp.inf),
        marginal=carry.marginal_utility.at[1].set(0.0),
        data=replace(
            data, node_kind=data.node_kind.at[1].set(QueryNodeKind.ECONOMIC_MINUS_INF)
        ),
        pool=pool,
    )
    np.testing.assert_array_almost_equal(
        result.value, [np.log(0.55)], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.marginal, [10.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result.coverage_missing, [False])


def test_dynamic_egm_single_node_state_grid_is_refused() -> None:
    """The public irregular-grid contract requires at least two state nodes."""
    with pytest.raises(GridInitializationError, match="at least 2 elements"):
        build_dynamic_egm_model(current_nodes=(1.0,))


def test_dynamic_egm_floor_value_changes_reuse_compiled_programs() -> None:
    """A changed floor adds no lowering or compilation beyond a same-floor solve."""
    model = build_dynamic_egm_model()
    with count_compile_requests() as cold:
        first = model.solve(params=dynamic_egm_params(floor=0.0), log_level="off")
        first.values[0]["saving"].block_until_ready()
    assert cold.compile_requests > 0
    with count_compile_requests() as repeated:
        same = model.solve(params=dynamic_egm_params(floor=0.0), log_level="off")
        same.values[0]["saving"].block_until_ready()
    with count_compile_requests() as warm:
        changed = model.solve(params=dynamic_egm_params(floor=0.5), log_level="off")
        changed.values[0]["saving"].block_until_ready()
    assert warm.trace_requests == repeated.trace_requests
    assert (repeated.lowering_requests, repeated.compile_requests) == (0, 0)
    assert (warm.lowering_requests, warm.compile_requests) == (0, 0)
    np.testing.assert_array_almost_equal(
        changed.values[0]["saving"][1],
        np.log(2.75),
        decimal=DECIMAL_PRECISION,
    )


def test_dynamic_egm_simulation_uses_actual_subject_floor() -> None:
    """A finite illegal action loses to the best legal grid action at actual wealth."""
    model = build_dynamic_egm_model()
    params = dynamic_egm_params(floor=0.5)
    frame = model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions={
            "wealth": jnp.array([1.25]),
            "age": jnp.array([0]),
            "regime_id": jnp.array([RegimeId.saving], dtype=jnp.int32),
        },
        log_level="off",
        seed=42,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame.query("period == 0")["consumption"], [0.5])


@pytest.mark.parametrize("runtime_checks", [True, False])
def test_dynamic_egm_simulation_rejects_empty_actual_subject(
    *, runtime_checks: bool
) -> None:
    """No declared positive action is feasible at actual wealth equal to the floor."""
    model = build_dynamic_egm_model()
    params = dynamic_egm_params(floor=0.5)
    solution = model.solve(params=params, log_level="off")
    with pytest.raises(NoFeasibleActionError, match=r"saving.*period 0"):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "wealth": jnp.array([0.5]),
                "age": jnp.array([0]),
                "regime_id": jnp.array([RegimeId.saving], dtype=jnp.int32),
            },
            log_level="off",
            runtime_checks=runtime_checks,
            seed=42,
        )


def test_dynamic_egm_offgrid_corner_keeps_its_associated_marginal() -> None:
    """The exact lower corner supplies its own value, marginal and saving tuple."""
    reader, carry, data, pool = scalar_reader(floor=0.5)
    result = reader(
        query=jnp.array([1.25]),
        grid=carry.endog_grid,
        value=carry.value,
        marginal=carry.marginal_utility,
        data=data,
        pool=pool,
    )
    np.testing.assert_array_almost_equal(
        result.value, [np.log(4.125)], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.marginal, [4.0 / 3.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result.savings, [0.5])
    np.testing.assert_array_equal(result.coverage_missing, [False])


def test_dynamic_egm_parent_corner_keeps_derivative_channels_distinct(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent composes the child's corner slope once and retains its own slope."""
    model = build_parent_model(parent_law=offgrid_parent_next_wealth, enable_jit=False)
    kernel, context = ride_along_kernel(
        model=model, params=parent_params(floor=0.5), regime_name="parent", period=0
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    original = step_module.invert_euler
    calls: list[FloatND] = []

    def measured_invert(
        *,
        expected_marginal_continuation: FloatND,
        discount_factor: ScalarFloat | float,
        inverse_marginal_utility: Callable[[FloatND], FloatND],
    ) -> FloatND:
        calls.append(expected_marginal_continuation)
        return original(
            expected_marginal_continuation=expected_marginal_continuation,
            discount_factor=discount_factor,
            inverse_marginal_utility=inverse_marginal_utility,
        )

    monkeypatch.setattr(step_module, "invert_euler", measured_invert)
    output = kernel(
        compiled_cores={"main": kernel.core_programs()["main"].function},
        **context,
        logger=logging.getLogger(__name__),
    )
    assert len(calls) == 1
    np.testing.assert_array_almost_equal(
        calls[0][0], 20.0 / 3.0, decimal=DECIMAL_PRECISION
    )
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    np.testing.assert_array_almost_equal(
        carry.leaves()[("marginal_utility",)][0], 10.0, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        output.value[0], np.log(0.4125), decimal=DECIMAL_PRECISION
    )
