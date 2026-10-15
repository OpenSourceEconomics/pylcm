"""A plain EGM row values its reconstructed action at the declared-law landing."""

import math
from collections.abc import Sequence
from typing import cast
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.one_asset_egm_step import RetiredEGMResult
from _lcm.egm.query_data import EGMQueryData
from _lcm.egm.scalar_query import (
    PreparedScalarEGMQuery,
    ScalarEGMQuery,
    ScalarEGMQueryResult,
)
from _lcm.solution.egm import (
    BoundedOneRowOutput,
    OneRowOutput,
    _EGMCore,
    _EGMPeriodKernel,
    _RegimeMassGuardedCore,
)
from _lcm.solution.feasibility import BoundedResult
from _lcm.typing import PytreeValue
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
from lcm.exceptions import InvalidValueFunctionError
from lcm.solvers import EGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    Float1D,
    FloatND,
    ScalarInt,
)
from tests.egm.test_dynamic_egm_floor import (
    build_parent_model,
    parent_params,
)
from tests.solution._nbegm_direct_oracle import ride_along_kernel


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
    return 1.0 + savings + savings**2


def lower() -> FloatND:
    return jnp.asarray(0.0)


def _represented_child_value(wealth: float) -> float:
    """Read the declared three-node terminal value without solver interpolation."""
    if wealth <= 2.0:
        return (wealth - 1.0) * math.log(2.0)
    return math.log(2.0) + (wealth - 2.0) / 5.0 * math.log(7.0 / 2.0)


@pytest.mark.parametrize("literal", [True, False], ids=["literal", "callable"])
def test_parent_values_the_reconstructed_action_through_its_nonlinear_law(
    *, literal: bool
) -> None:
    """An interior action reads child node values at its actual nonlinear landing."""
    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )
    model = Model(
        regimes={
            "saving": ConsumptionSavingsRegime(
                liquid=margin,
                states={"wealth": IrregSpacedGrid(points=(2.5, 3.0))},
                actions={"consumption": IrregSpacedGrid(points=(0.5, 1.0))},
                functions={
                    "utility": utility,
                    "inverse_marginal_utility": inverse_marginal_utility,
                    "savings": savings,
                },
                state_transitions={"wealth": next_wealth},
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=margin, lower=0.0 if literal else lower
                    )
                },
                solver=EGM(savings_grid=IrregSpacedGrid(points=(0.0, 2.0))),
            ),
            "done": Regime(
                states={"wealth": IrregSpacedGrid(points=(1.0, 2.0, 7.0))},
                functions={"utility": terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"saving": {"done": 0}},
        initial_nodes={0: "saving"},
        regime_id_class=RegimeId,
    )
    params = {
        "saving": {"koopmans_aggregator": {"discount_factor": 0.5}},
        "done": {},
    }
    got = np.asarray(model.solve(params=params, log_level="off").values[0]["saving"])
    # Terminal marginals at landings 1 and 7 are 1 and 1/7. The source law's
    # derivatives are 1 and 5, so Euler consumptions are 2 and 14/5.
    expected = []
    chord = []
    cached = []
    wrong_law = []
    for wealth in (2.5, 3.0):
        weight = (wealth - 2.0) / (24.0 / 5.0 - 2.0)
        consumption = 2.0 + weight * (14.0 / 5.0 - 2.0)
        saving = weight * 2.0
        landing = 1.0 + saving + saving**2
        expected.append(math.log(consumption) + 0.5 * _represented_child_value(landing))
        chord.append(
            (1.0 - weight) * math.log(2.0)
            + weight * (math.log(14.0 / 5.0) + 0.5 * math.log(7.0))
        )
        cached.append(math.log(consumption) + 0.5 * weight * math.log(7.0))
        wrong_law.append(
            math.log(consumption) + 0.5 * _represented_child_value(1.0 + 6.0 * weight)
        )
    # Each alternative changes the represented numerical object, even with the
    # identical original Euler actions and saving candidates.
    for alternative in (chord, cached, wrong_law):
        assert np.max(np.abs(np.asarray(expected) - alternative)) > 0.01
    tolerance = 100.0 * np.finfo(got.dtype).eps
    np.testing.assert_allclose(got, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("active", [True, False])
def test_scalar_valuation_coverage_obeys_the_reconstructed_action_mask(
    *, active: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only active reconstructed actions propagate their second-read coverage fault."""
    # The child's binding floor gives W(5A-1)=log(5A-1.5)+log(5.5).
    # Original parent Euler tuples have C=A-.3 and X=2A-.3 at A=.4,.5,.6.
    # The active nodes lie strictly between knots. The inactive specimen has
    # one legal constrained corner and one complete upper-endpoint tuple.
    current_nodes = (0.6, 0.8) if active else (0.45, 1.0)
    model = build_parent_model(current_nodes=current_nodes, enable_jit=False)
    params = parent_params(floor=0.5)
    kernel, _ = ride_along_kernel(
        model=model, params=params, regime_name="parent", period=0
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    guard = kernel.core_programs()["main"].function
    assert isinstance(guard, _RegimeMassGuardedCore)
    core = guard.core
    assert isinstance(core, _EGMCore)
    reader = core.next_query
    assert reader is not None
    original_read = ScalarEGMQuery.read_prepared
    calls: list[
        tuple[
            Float1D, PreparedScalarEGMQuery, ScalarEGMQueryResult, ScalarEGMQueryResult
        ]
    ] = []

    # keyword-only-exempt: primary-argument=self
    def second_read_fault(
        self: ScalarEGMQuery,
        *,
        query: Float1D,
        grid: Float1D,
        value: Float1D,
        marginal: Float1D,
        data: EGMQueryData,
        prepared: PreparedScalarEGMQuery,
    ) -> ScalarEGMQueryResult:
        result = original_read(
            self,
            query=query,
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            prepared=prepared,
        )
        if self is reader:
            seeded = (
                result._replace(coverage_missing=jnp.ones_like(result.coverage_missing))
                if len(calls) == 1
                else result
            )
            calls.append((query, prepared, result, seeded))
            return seeded
        return result

    original_guard = _RegimeMassGuardedCore.__call__
    observed: list[BoundedOneRowOutput] = []

    # keyword-only-exempt: primary-argument=self
    def observe_guard(
        self: _RegimeMassGuardedCore,
        *,
        retains_regime_mass: BoolND,
        **arguments: PytreeValue,
    ) -> OneRowOutput:
        result = original_guard(
            self, retains_regime_mass=retains_regime_mass, **arguments
        )
        if self is guard:
            assert isinstance(result, BoundedResult)
            observed.append(cast("BoundedOneRowOutput", result))
        return result

    monkeypatch.setattr(ScalarEGMQuery, "read_prepared", second_read_fault)
    monkeypatch.setattr(_RegimeMassGuardedCore, "__call__", observe_guard)
    original_valuation = _EGMCore._value_reconstructed_action
    with patch.object(
        _EGMCore,
        "_value_reconstructed_action",
        autospec=True,
        side_effect=original_valuation,
    ) as valuation_calls:
        if active:
            with pytest.raises(
                InvalidValueFunctionError,
                match=r"query_coverage_missing.*saving.*period 1",
            ):
                model.solve(params=params, runtime_checks=False, log_level="off")
        else:
            solution = model.solve(params=params, runtime_checks=False, log_level="off")
            np.testing.assert_allclose(
                solution.values[0]["parent"],
                np.log([0.1375, 2.475]),
                rtol=100.0
                * np.finfo(np.asarray(solution.values[0]["parent"]).dtype).eps,
                atol=0.0,
            )
        steps = [
            call.kwargs["step"]
            for call in valuation_calls.call_args_list
            if call.args[0] is core
        ]

    _assert_query_records(
        model=model,
        kernel=kernel,
        active=active,
        current_nodes=current_nodes,
        steps=steps,
        calls=calls,
        observed=observed,
    )


def _assert_query_records(
    *,
    model: Model,
    kernel: _EGMPeriodKernel,
    active: bool,
    current_nodes: tuple[float, float],
    steps: Sequence[RetiredEGMResult],
    calls: Sequence[
        tuple[
            Float1D, PreparedScalarEGMQuery, ScalarEGMQueryResult, ScalarEGMQueryResult
        ]
    ],
    observed: Sequence[BoundedOneRowOutput],
) -> None:
    """Check actual target records, mask and the unchanged associated numeric tuple."""
    assert kernel.regime_name == "parent"
    assert kernel.continuation_target == "saving"
    assert kernel.source_period == 0
    assert kernel.own_bound_specs == ()
    authority = model._regimes["saving"].solution.query_authority
    assert authority is not None
    assert kernel.inherited_bound_specs == (authority.bound,)
    assert len(steps) == 1
    step = steps[0]
    assert isinstance(step, RetiredEGMResult)
    assert step.savings is not None
    np.testing.assert_array_equal(step.rescore_interior, [active, active])
    assert len(calls) == 2
    first_query, first_context, first, first_returned = calls[0]
    second_query, second_context, second, second_returned = calls[1]
    assert first_context is second_context
    assert first_returned is first
    np.testing.assert_array_equal(first.coverage_missing, [False, False, False])
    np.testing.assert_array_equal(second.coverage_missing, [False, False])
    np.testing.assert_array_equal(second_returned.coverage_missing, [True, True])
    assert second_returned.value is second.value
    assert second_returned.marginal is second.marginal
    assert second_returned.savings is second.savings
    assert second_returned.bound_status is second.bound_status
    tolerance = 100.0 * np.finfo(np.asarray(first_query).dtype).eps
    np.testing.assert_allclose(
        np.stack((step.consumption, step.savings)),
        [[0.15, 0.25], [0.45, 0.55]] if active else [[0.05, 0.3], [0.4, 0.6]],
        rtol=tolerance,
        atol=0.0,
    )
    np.testing.assert_allclose(first_query, [1.0, 1.5, 2.0], rtol=tolerance, atol=0.0)
    np.testing.assert_allclose(
        second_query,
        [1.25, 1.75] if active else [1.0, 1.0],
        rtol=tolerance,
        atol=0.0,
    )
    assert len(observed) == 1
    result = observed[0]
    assert len(result.bound_status) == 1
    assert result.bound_status[0] is first_context.status
    assert result.query_status is None
    assert not bool(result.bound_status[0].nonfinite_floor)
    assert not bool(result.bound_status[0].invalid_shifted_grid)
    value, carry, counts, coverage = result.output
    assert bool(coverage) is active
    expected = [0.61875, 1.71875] if active else [0.1375, 2.475]
    np.testing.assert_allclose(value, np.log(expected), rtol=tolerance, atol=0.0)
    np.testing.assert_allclose(
        carry.marginal_utility,
        [20.0 / 3.0, 4.0] if active else [20.0, 10.0 / 3.0],
        rtol=tolerance,
        atol=0.0,
    )
    np.testing.assert_array_equal(carry.endog_grid, jnp.asarray(current_nodes))
    assert carry.query_data_for_transport() is None
    if not active:
        np.testing.assert_array_equal(value, step.value)
        np.testing.assert_array_equal(carry.marginal_utility, step.marginal)
    np.testing.assert_array_equal(counts, [0, 0])
