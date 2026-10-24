"""A plain parent's literal DC row uses target resources and target dependencies."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NotRequired, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from _lcm.egm import one_asset_egm_step as step_module
from _lcm.egm.one_asset_egm_step import RetiredEGMResult
from _lcm.egm.preferences import Preferences
from _lcm.execution.core_program import CoreBuildContext
from _lcm.solution.egm import _EGMArgumentBuilder, _EGMPeriodKernel
from _lcm.typing import PytreeValue, QualifiedName, ReferenceName, ShapeDtypePytree
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import KernelOutput
from lcm.solvers import DCEGM, EGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    Float1D,
    FloatND,
    ScalarFloat,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    RegimeId,
    _inverse,
    _ordinary_law,
    _parent_resources,
    _savings,
    _terminal,
    _utility,
)
from tests.solution._nbegm_direct_oracle import OracleContext, interpolate_row

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _available_at_age(age: float) -> UserFunction:
    slope = 1.0 + age - 40.0

    def available_wealth(*, liquid: ContinuousState, fixed_shift: float) -> FloatND:
        return slope * liquid + fixed_shift

    return available_wealth


def _target_resources(
    *, available_wealth: FloatND, free_shift: float, period: ScalarInt, age: float
) -> FloatND:
    return available_wealth + free_shift * period + 0.125 * (age - 40.0)


def _source_law(*, savings: FloatND, age: float, period: ScalarInt) -> ContinuousState:
    return 5.0 * savings - 1.0 + 0.125 * (age - 40.0) + 0.125 * period


def _source_utility(
    *, consumption: ContinuousAction, free_shift: float, fixed_shift: float
) -> FloatND:
    return jnp.log(consumption) + free_shift + fixed_shift


def _model(
    *,
    parent_liquid: tuple[float, float] = (0.225, 0.25),
    parent_savings: tuple[float, float, float] = (0.2, 0.3, 0.4),
) -> Model:
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    return Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=np.array(parent_liquid))},
                actions={
                    "consumption": IrregSpacedGrid(points=np.array([0.025, 0.05, 0.1]))
                },
                functions={
                    "utility": _source_utility,
                    "resources": _parent_resources,
                    "savings": _savings,
                    "inverse_marginal_utility": _inverse,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.2)
                },
                state_transitions={"liquid": _source_law},
                solver=EGM(
                    savings_grid=IrregSpacedGrid(points=np.array(parent_savings))
                ),
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0, 1.5]))
                },
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.025, 0.05, 0.25, 0.5, 1.0])
                    )
                },
                functions={
                    "utility": _utility,
                    "resources": _target_resources,
                    "available_wealth": AgeSpecializedFunction(
                        build=_available_at_age, signature=lambda age: age
                    ),
                    "savings": _savings,
                    "inverse_marginal_utility": _inverse,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.5)
                },
                state_transitions={"liquid": _ordinary_law},
                solver=DCEGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.5, 1.0, 1.5])),
                    n_constrained_points=4,
                ),
            ),
            "done": Regime(
                states={"liquid": IrregSpacedGrid(points=np.array([5.5, 7.0, 8.5]))},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=43, step="Y"),
        edges={"parent": {"child": (40, 41)}, "child": {"done": (41, 42)}},
        initial_nodes={40: "parent", 41: "parent"},
        regime_id_class=RegimeId,
        fixed_params={
            "parent": {"utility": {"fixed_shift": 17.0}},
            "child": {"available_wealth": {"fixed_shift": 0.5}},
        },
    )


@dataclass(frozen=True, kw_only=True)
class _ObservedCall:
    """Actual call ownership and independent numeric observations."""

    builder: _EGMArgumentBuilder
    """Original builder whose period-local transport owns the call."""

    context: CoreBuildContext
    """Original context identifying the source computational period."""

    arguments: Mapping[ReferenceName, NDArray[np.generic] | None]
    """Host copies of exact argument leaves before native buffer consumption."""

    savings: NDArray[np.float32 | np.float64]
    """Host copy of the actual source saving nodes."""

    bound_params: Mapping[QualifiedName, NDArray[np.generic]]
    """Host copies of fixed numeric leaves used by the independent assertions."""


def _snapshot_call(
    *,
    builder: _EGMArgumentBuilder,
    context: CoreBuildContext,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
) -> _ObservedCall:
    """Copy observed numeric leaves before returning their original owners."""
    copied: dict[ReferenceName, NDArray[np.generic] | None] = {}
    for name, value in arguments.items():
        if value is None:
            copied[name] = None
        else:
            assert isinstance(value, jax.Array | np.ndarray | float | int)
            copied[name] = np.array(value, copy=True)
    return _ObservedCall(
        builder=builder,
        context=context,
        arguments=MappingProxyType(copied),
        savings=np.array(builder.savings_grid, copy=True),
        bound_params=MappingProxyType(
            {
                name: np.array(builder.bound_params[name], copy=True)
                for name in (
                    "child__available_wealth__fixed_shift",
                    "utility__fixed_shift",
                )
            }
        ),
    )


class _StepInputs(TypedDict):
    next_value: Float1D
    next_marginal: Float1D
    liquid_grid: Float1D
    next_liquid_grid: Float1D
    savings_grid: Float1D
    discount_factor: ScalarFloat | float
    preferences: Preferences
    next_liquid: Float1D
    marginal_return: Float1D
    continuation_value: NotRequired[Float1D | None]
    continuation_marginal: NotRequired[Float1D | None]
    feature_floor: NotRequired[bool]


def _observe_installed_step(
    *, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Record original step inputs, numeric reads and complete unchanged outputs."""
    reads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    steps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    original_step = step_module.egm_one_asset_step
    original_read = step_module.interp_on_regular_grid

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_read(
        query: np.ndarray, grid: np.ndarray, row: np.ndarray, result: np.ndarray
    ) -> None:
        reads.append(
            (
                np.array(query, copy=True),
                np.array(grid, copy=True),
                np.array(row, copy=True),
                np.array(result, copy=True),
            )
        )

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_step(query: np.ndarray, liquid: np.ndarray, result: np.ndarray) -> None:
        steps.append(
            (
                np.array(query, copy=True),
                np.array(liquid, copy=True),
                np.array(result, copy=True),
            )
        )

    def observe_read(*, x_query: FloatND, xp: Float1D, fp: Float1D) -> FloatND:
        result = original_read(x_query=x_query, xp=xp, fp=fp)
        jax.debug.callback(record_read, x_query, xp, fp, result)
        return result

    def observe_step(**kwargs: Unpack[_StepInputs]) -> RetiredEGMResult:
        result = original_step(**kwargs)
        for numeric, row in (
            (kwargs.get("continuation_value"), kwargs["next_value"]),
            (kwargs.get("continuation_marginal"), kwargs["next_marginal"]),
        ):
            if numeric is not None:
                jax.debug.callback(
                    record_read,
                    kwargs["next_liquid"],
                    kwargs["next_liquid_grid"],
                    row,
                    numeric,
                )
        jax.debug.callback(
            record_step,
            kwargs["next_liquid"],
            kwargs["liquid_grid"],
            jnp.stack((result.value, result.marginal, result.consumption)),
        )
        return result

    monkeypatch.setattr(step_module, "interp_on_regular_grid", observe_read)
    monkeypatch.setattr(step_module, "egm_one_asset_step", observe_step)
    return reads, steps


class _KernelInputs(OracleContext):
    compiled_cores: Mapping[str, Callable[..., PytreeValue]]
    logger: logging.Logger


def _observe_runtime_dispatches(
    *, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[int], list[int]]:
    """Identify actual kernel dispatches separately from shape-lowering builders."""
    active: list[int] = []
    executed: list[int] = []
    original = _EGMPeriodKernel.__call__

    def observe(
        self: _EGMPeriodKernel, **kwargs: Unpack[_KernelInputs]
    ) -> KernelOutput:
        active.append(kwargs["period"])
        executed.append(kwargs["period"])
        try:
            return original(self, **kwargs)
        finally:
            active.pop()

    monkeypatch.setattr(_EGMPeriodKernel, "__call__", observe)
    return active, executed


def test_plain_literal_dc_transport_binds_target_resources_time_and_params(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read distinct target resource maps while keeping source-time physical laws."""
    observed: list[_ObservedCall] = []
    reads, _steps = _observe_installed_step(monkeypatch=monkeypatch)
    active_dispatches, executed_dispatches = _observe_runtime_dispatches(
        monkeypatch=monkeypatch
    )
    original = _EGMArgumentBuilder.__call__

    # keyword-only-exempt: library-callback=_EGMArgumentBuilder.__call__
    def observe(
        self: _EGMArgumentBuilder, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        arguments = original(self, context)
        if (
            self.regime_name == "parent"
            and context.period in active_dispatches
            and isinstance(arguments["next_liquid_grid"], jax.Array)
        ):
            grid = np.asarray(arguments["next_liquid_grid"])
            if np.count_nonzero(np.isfinite(grid)) >= 2 and grid[0] != grid[1]:
                observed.append(
                    _snapshot_call(builder=self, context=context, arguments=arguments)
                )
        return arguments

    monkeypatch.setattr(_EGMArgumentBuilder, "__call__", observe)
    model = _model()
    model.solve(
        params={
            "parent": {
                "utility": {"free_shift": 5.0},
                "koopmans_aggregator": {"discount_factor": 1.0},
            },
            "child": {
                "resources": {"free_shift": 0.25},
                "koopmans_aggregator": {"discount_factor": 1.0},
            },
            "done": {},
        },
        log_level="off",
        runtime_checks=False,
    )
    jax.effects_barrier()
    assert set(executed_dispatches) == {0, 1}
    assert {call.context.period for call in observed} == {0, 1}
    builders: dict[int, set[int]] = {0: set(), 1: set()}
    for call in observed:
        builder, context, arguments = call.builder, call.context, call.arguments
        period = context.period
        savings = call.savings
        physical = 5.0 * savings - 1.0 + 0.25 * period
        slope = 2.0 + period
        expected_resources = slope * physical + 0.5 + 0.375 * (period + 1)
        expected_return = 5.0 * slope
        np.testing.assert_allclose(
            cast("FloatND", arguments["next_liquid"]),
            expected_resources,
            rtol=10**-DECIMAL_PRECISION,
            atol=0,
        )
        np.testing.assert_array_equal(
            arguments["marginal_return"], np.full(savings.shape, expected_return)
        )
        assert arguments.get("next_query_data") is None
        assert arguments.get("next_query_pool") is None
        transport = getattr(builder, "literal_dc_resources", None)
        assert transport is not None
        assert transport.target_period == period + 1
        assert transport.input_names == frozenset(
            {
                "liquid",
                "child__available_wealth__fixed_shift",
                "child__resources__free_shift",
                "period",
                "age",
            }
        )
        np.testing.assert_array_equal(
            call.bound_params["child__available_wealth__fixed_shift"], 0.5
        )
        np.testing.assert_array_equal(call.bound_params["utility__fixed_shift"], 17.0)
        np.testing.assert_array_equal(arguments["child__resources__free_shift"], 0.25)
        np.testing.assert_array_equal(arguments["utility__free_shift"], 5.0)
        builders[period].add(id(builder))
        grid = np.asarray(arguments["next_liquid_grid"])
        for channel, multiplier in (
            ("next_value", 1.0),
            ("next_marginal", expected_return),
        ):
            matching = [
                result
                for query, row_grid, row_value, result in reads
                if np.array_equal(query, np.asarray(arguments["next_liquid"]))
                and np.array_equal(row_grid, grid, equal_nan=True)
                and np.array_equal(
                    row_value, np.asarray(arguments[channel]), equal_nan=True
                )
            ]
            assert matching, {
                "period": period,
                "channel": channel,
                "requested_query": np.asarray(arguments["next_liquid"]).tolist(),
                "requested_grid": grid.tolist(),
                "requested_row": np.asarray(arguments[channel]).tolist(),
                "requested_dtype": str(grid.dtype),
                "read_count": len(reads),
                "last_reads": [
                    (
                        query.shape,
                        str(query.dtype),
                        query.tolist(),
                        row_grid.tolist(),
                        row_value.tolist(),
                    )
                    for query, row_grid, row_value, _ in reads[-12:]
                ],
            }
            actual = matching[0] * multiplier
            expected = [
                multiplier
                * interpolate_row(
                    query=float(point),
                    xp=grid,
                    fp=np.asarray(arguments[channel]),
                    slopes=None,
                )
                for point in expected_resources
            ]
            np.testing.assert_allclose(
                actual, expected, rtol=10**-DECIMAL_PRECISION, atol=0
            )
    assert builders[0].isdisjoint(builders[1])


def _high_parent_model() -> Model:
    """Keep the candidate counts while publishing a high last-live Euler knot."""
    return _model(parent_liquid=(0.225, 137.0 / 60.0), parent_savings=(0.2, 2.0, 2.5))


def test_public_plain_parent_uses_last_live_literal_dc_candidate(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish the finite Euler-knot value when child queries exceed a padded row."""
    observed: list[_ObservedCall] = []
    _reads, steps = _observe_installed_step(monkeypatch=monkeypatch)
    original = _EGMArgumentBuilder.__call__

    # keyword-only-exempt: library-callback=_EGMArgumentBuilder.__call__
    def observe(
        self: _EGMArgumentBuilder, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        arguments = original(self, context)
        grid = arguments.get("next_liquid_grid")
        if self.regime_name == "parent" and isinstance(grid, jax.Array):
            numeric = np.asarray(grid)
            if np.count_nonzero(np.isfinite(numeric)) >= 2 and numeric[0] != numeric[1]:
                observed.append(
                    _snapshot_call(builder=self, context=context, arguments=arguments)
                )
        return arguments

    monkeypatch.setattr(_EGMArgumentBuilder, "__call__", observe)
    model = _high_parent_model()
    solution = None
    public_failure = None
    try:
        solution = model.solve(
            params={
                "parent": {
                    "utility": {"free_shift": 5.0},
                    "koopmans_aggregator": {"discount_factor": 1.0},
                },
                "child": {
                    "resources": {"free_shift": 0.25},
                    "koopmans_aggregator": {"discount_factor": 1.0},
                },
                "done": {},
            },
            log_level="off",
            runtime_checks=False,
        )
    except InvalidValueFunctionError as error:
        public_failure = error
    jax.effects_barrier()
    calls = [call for call in observed if call.context.period == 0]
    assert calls
    call = calls[-1]
    arguments = call.arguments
    query = np.asarray(arguments["next_liquid"])
    grid = np.asarray(arguments["next_liquid_grid"])
    assert np.isnan(grid).any()
    count = int(np.count_nonzero(np.isfinite(grid)))
    assert query[1] > grid[count - 1]
    values = np.asarray(
        [
            interpolate_row(
                query=float(point),
                xp=grid,
                fp=np.asarray(arguments["next_value"]),
                slopes=None,
            )
            for point in query
        ]
    )
    marginals = np.asarray(
        [
            interpolate_row(
                query=float(point),
                xp=grid,
                fp=np.asarray(arguments["next_marginal"]),
                slopes=None,
            )
            for point in query
        ]
    )
    consumption = 1.0 / (np.asarray(arguments["marginal_return"]) * marginals)
    endogenous = call.savings + consumption
    expected_value = float(np.log(consumption[1]) + 22.0 + values[1])
    np.testing.assert_allclose(
        values[1],
        np.log(17.0 / 6.0) + np.log(17.0 / 2.0),
        rtol=10**-DECIMAL_PRECISION,
        atol=0,
    )
    np.testing.assert_allclose(
        marginals[1], 6.0 / 17.0, rtol=10**-DECIMAL_PRECISION, atol=0
    )
    np.testing.assert_allclose(
        endogenous[1], 137.0 / 60.0, rtol=10**-DECIMAL_PRECISION, atol=0
    )
    selected = [
        (liquid, result)
        for actual_query, liquid, result in steps
        if np.array_equal(actual_query, query)
    ]
    assert selected
    for liquid, actual in selected:
        np.testing.assert_allclose(
            liquid[-1], endogenous[1], rtol=10**-DECIMAL_PRECISION, atol=0
        )
        np.testing.assert_allclose(
            actual[:, -1],
            [expected_value, 1.0 / consumption[1], consumption[1]],
            rtol=10**-DECIMAL_PRECISION,
            atol=0,
        )
    assert public_failure is None
    assert solution is not None
    np.testing.assert_allclose(
        solution.values[0]["parent"][-1],
        expected_value,
        rtol=10**-DECIMAL_PRECISION,
        atol=0,
    )
