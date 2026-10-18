"""Actual DC consumers recover conditional corners on a two-axis passive stencil."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import NotRequired, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import ContinuationPayload
from _lcm.egm import continuation as continuation_module
from _lcm.egm.carry import EGMCarry
from _lcm.egm.dcegm_query import DCEGMQuery, DCEGMQueryResult
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import (
    BoundedResult,
    FeasibilityKernelOutput,
    validate_query_input_status,
)
from _lcm.typing import EconFunctionKwargs
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExtremeValueTasteShocks,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
    post_decision_lower_bound,
)
from lcm.solvers import DCEGM
from lcm.transition import fixed_transition
from lcm.typing import (
    ActionName,
    Age,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    Float1D,
    FloatND,
    IntND,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_consumers import (
    _capture_with_checks_off,
    _run_installed,
)
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class Job:
    first: ScalarInt
    second: ScalarInt


def _utility(*, consumption: ContinuousAction, job: DiscreteAction) -> FloatND:
    return jnp.log(consumption) - job / 8.0


def _parent_utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _resources(
    *,
    liquid: ContinuousState,
    z: ContinuousState,
    w: ContinuousState,
    job: DiscreteAction,
    shift: float,
    period: ScalarInt,
    age: Age,
) -> FloatND:
    return (
        2.0 * liquid
        + 2.0
        + z / 4.0
        + w / 8.0
        + job / 4.0
        + shift
        + period / 8.0
        + (age - 40.0) / 8.0
    )


def _parent_resources(liquid: ContinuousState) -> FloatND:
    return liquid


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse(marginal_continuation: FloatND) -> ContinuousAction:
    return 1.0 / marginal_continuation


def _lower(
    *, z: ContinuousState, w: ContinuousState, floor: float, period: ScalarInt, age: Age
) -> FloatND:
    return floor + period / 8.0 + (age - 40.0) / 8.0 + z * (1.0 - z) + w / 8.0


def _source_lower(floor: float) -> FloatND:
    return jnp.asarray(floor)


def _next_liquid(*, savings: FloatND, job: DiscreteAction) -> ContinuousState:
    return 3.0 * savings + 4.0 + job / 2.0


def _child_liquid(savings: FloatND) -> ContinuousState:
    return 5.0 * savings - 1.25


def _child_z(z: ContinuousState) -> ContinuousState:
    return 0.25 + 0.5 * z


def _child_w(w: ContinuousState) -> ContinuousState:
    return 0.5 + 0.5 * w


def _terminal(
    *, liquid: ContinuousState, z: ContinuousState, w: ContinuousState
) -> FloatND:
    return liquid / 4.0 + z / 8.0 + w / 16.0


def _parent_savings_grid() -> IrregSpacedGrid:
    """Retain exact saving offsets while declaring the physical source origin."""
    grid = IrregSpacedGrid(points=(0.25, 0.375, 0.5))
    nodes = jnp.asarray(grid.to_jax(), dtype=jnp.asarray(0.0).dtype)
    offsets = nodes - nodes[0]
    np.testing.assert_array_equal(
        offsets, jnp.asarray((0.0, 0.125, 0.25), dtype=nodes.dtype)
    )
    np.testing.assert_array_equal(
        0.25 + offsets, jnp.asarray((0.25, 0.375, 0.5), dtype=nodes.dtype)
    )
    return grid


def _child_savings_grid() -> IrregSpacedGrid:
    """Retain exact child saving offsets with a covering raw declaration."""
    grid = IrregSpacedGrid(points=(0.5, 1.0, 1.5))
    nodes = jnp.asarray(grid.to_jax(), dtype=jnp.asarray(0.0).dtype)
    offsets = nodes - nodes[0]
    np.testing.assert_array_equal(
        offsets, jnp.asarray((0.0, 0.5, 1.0), dtype=nodes.dtype)
    )
    for floor in (3.0 / 8.0, 0.5, 5.0 / 8.0):
        np.testing.assert_array_equal(
            floor + offsets,
            jnp.asarray((floor, floor + 0.5, floor + 1.0), dtype=nodes.dtype),
        )
    return grid


def _model_and_params(*, ev1: bool, equality: bool) -> tuple[Model, UserParams]:
    """Keep a nonlinear floor and two retained action rows on a 2-by-3 passive grid."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    passive = {
        "z": IrregSpacedGrid(points=(0.0, 1.0)),
        "w": IrregSpacedGrid(points=(0.0, 1.0, 2.0)),
    }
    child = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=(0.0, 0.5, 1.5)), **passive},
        actions={
            "job": DiscreteGrid(category_class=Job),
            "consumption": IrregSpacedGrid(points=(0.025, 0.125, 0.5, 1.0, 4.0)),
        },
        functions={
            "utility": _utility,
            "resources": _resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=_lower)
        },
        state_transitions={
            "liquid": _next_liquid,
            "z": fixed_transition("z"),
            "w": fixed_transition("w"),
        },
        taste_shocks=ExtremeValueTasteShocks() if ev1 else None,
        solver=DCEGM(savings_grid=_child_savings_grid(), n_constrained_points=4),
    )
    parent = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=(0.375, 0.5)), **passive},
        actions={"consumption": IrregSpacedGrid(points=(0.025, 0.125, 0.25))},
        functions={
            "utility": _parent_utility,
            "resources": _parent_resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=_source_lower)
        },
        state_transitions={"liquid": _child_liquid, "z": _child_z, "w": _child_w},
        solver=DCEGM(
            savings_grid=_parent_savings_grid(),
            n_constrained_points=4,
        ),
    )
    done = Regime(
        states={"liquid": IrregSpacedGrid(points=(0.0, 16.0)), **passive},
        functions={"utility": _terminal},
    )
    model = Model(
        regimes={"parent": parent, "child": child, "done": done},
        ages=AgeGrid(start=40, inclusive_stop=42, step="Y"),
        edges={"parent": {"child": 40}, "child": {"done": 41}},
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
    )
    return model, {
        "parent": {
            "koopmans_aggregator": {"discount_factor": 0.75},
            "borrowing_lower": {"floor": 0.25},
        },
        "child": {
            "koopmans_aggregator": {"discount_factor": 0.5},
            "borrowing_lower": {"floor": 0.125},
            "resources": {"shift": -1.75 if equality else 0.125},
            **({"taste_shocks": {"scale": 0.25}} if ev1 else {}),
        },
        "done": {},
    }


class _ChoiceInputs(TypedDict):
    grid: FloatND
    value: FloatND
    marginal: FloatND
    data: EGMQueryData
    row_queries: FloatND
    row_gradients: FloatND
    passive_values: tuple[ScalarFloat, ...]
    passive_grids: tuple[Float1D, ...]
    action_names: tuple[ActionName, ...]
    action_values: tuple[FloatND | IntND, ...]
    pool: EconFunctionKwargs
    euler_gradient: ScalarFloat
    paired_marginal_read: NotRequired[bool]


class _AggregateInputs(TypedDict):
    carry: EGMCarry
    prepared_search_grid: FloatND
    prepared_valid_length: IntND
    has_taste_shocks: bool
    child_index: tuple[ScalarInt, ...]
    child_passive_values: tuple[ScalarFloat, ...]
    child_passive_grids: tuple[Float1D, ...]
    row_queries: FloatND
    row_gradients: FloatND
    paired_marginal_read: NotRequired[bool]
    n_outer_candidates: NotRequired[int]
    query: NotRequired[DCEGMQuery | None]
    query_pool: NotRequired[EconFunctionKwargs | None]
    query_euler_gradient: NotRequired[ScalarFloat | None]
    query_action_names: NotRequired[tuple[ActionName, ...]]
    query_action_values: NotRequired[tuple[FloatND | IntND, ...]]


def _with_corner_only_support(
    *, carry: ContinuationPayload, context: OracleContext
) -> OracleContext:
    """Check the real producer bank and remove ordinary support metadata only."""
    carries = context["next_regime_to_continuation"]
    assert isinstance(carry, EGMCarry)
    data = carry.query_data_for_transport()
    assert data is not None
    assert carry.endog_grid.shape[:-1] == (2, 3, 2)
    assert data.floor_used.shape == (2, 3)
    assert data.corner_continuation is not None
    assert data.corner_kind is not None
    assert data.liquid_corner_data is None
    assert data.corner_continuation.shape == (2, 3, 2, 1)
    expected_floor = np.broadcast_to(np.array([3.0 / 8.0, 0.5, 5.0 / 8.0]), (2, 3))
    np.testing.assert_array_equal(data.floor_used, expected_floor)
    z, w, job = np.meshgrid([0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 1.0], indexing="ij")
    expected_corner = (
        0.75 * expected_floor[..., None] + 1.0 + job / 8.0 + z / 8.0 + w / 16.0
    )
    np.testing.assert_array_almost_equal(
        data.corner_continuation[..., 0], expected_corner, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(
        data.corner_kind, np.full((2, 3, 2, 1), QueryNodeKind.FINITE)
    )
    assert np.count_nonzero(np.asarray(data.node_kind) == QueryNodeKind.FINITE) > 0
    absent_rows = replace(
        data,
        node_kind=jnp.full_like(data.node_kind, QueryNodeKind.PAD),
        link_valid=jnp.zeros_like(data.link_valid),
    )
    # Missing ordinary support leaves the real produced corner bank unchanged.
    changed = replace(carry, query_data=absent_rows)
    changed_data = changed.query_data_for_transport()
    assert changed_data is not None
    np.testing.assert_array_equal(
        changed_data.corner_continuation, data.corner_continuation
    )
    return {
        **context,
        "next_regime_to_continuation": cast(
            "Mapping[RegimeName, ContinuationPayload]",
            MappingProxyType({**carries, "child": changed}),
        ),
    }


def _observe_installed_consumer(
    *,
    kernel: _DCEGMPeriodKernel,
    context: OracleContext,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    FeasibilityKernelOutput,
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Observe unchanged conditional and aggregate results from the installed kernel."""
    conditional: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    aggregated: list[tuple[np.ndarray, np.ndarray]] = []
    original_choices = DCEGMQuery.read_choices
    original_aggregate = continuation_module._aggregate_child_choices

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_choices(
        facts: np.ndarray,
        values: np.ndarray,
        marginals: np.ndarray,
        savings: np.ndarray,
    ) -> None:
        conditional.append((facts, values, marginals, savings))

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_aggregate(facts: np.ndarray, pair: np.ndarray) -> None:
        aggregated.append((facts, pair))

    def observe_choices(
        self: DCEGMQuery, **kwargs: Unpack[_ChoiceInputs]
    ) -> DCEGMQueryResult:
        result = original_choices(self, **kwargs)
        assert isinstance(result, DCEGMQueryResult)
        pool = kwargs["pool"]
        assert kwargs["row_queries"].shape == (2, 3, 2)
        assert kwargs["row_gradients"].shape == (2, 3, 2)
        facts: list[FloatND | float] = []
        for name in (
            "liquid",
            "z",
            "w",
            "period",
            "age",
            "borrowing_lower__floor",
            "resources__shift",
            "koopmans_aggregator__discount_factor",
        ):
            numeric = pool[name]
            assert isinstance(numeric, jax.Array | float | int)
            facts.append(numeric)
        facts.append(kwargs["euler_gradient"])
        jax.debug.callback(
            record_choices,
            jnp.stack(facts),
            result.value,
            result.marginal,
            result.savings,
        )
        return result

    def observe_aggregate(
        **kwargs: Unpack[_AggregateInputs],
    ) -> (
        tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ):
        result = original_aggregate(**kwargs)
        if kwargs.get("query") is not None:
            pool = kwargs.get("query_pool")
            assert pool is not None
            assert isinstance(result, BoundedResult)
            facts: list[FloatND | float] = []
            for name in ("liquid", "z", "w"):
                numeric = pool[name]
                assert isinstance(numeric, jax.Array | float | int)
                facts.append(numeric)
            jax.debug.callback(
                record_aggregate, jnp.stack(facts), jnp.stack(result.output)
            )
        return result

    monkeypatch.setattr(DCEGMQuery, "read_choices", observe_choices)
    monkeypatch.setattr(
        continuation_module, "_aggregate_child_choices", observe_aggregate
    )
    output = _run_installed(kernel=kernel, context=context)
    jax.block_until_ready(output.value)
    jax.effects_barrier()
    assert isinstance(output, FeasibilityKernelOutput)
    return output, conditional, aggregated


def _assert_clean_installed_statuses(
    *, model: Model, output: FeasibilityKernelOutput
) -> None:
    """Accept every declared child and source construction summary.

    The parent's own source slot and its bound summaries exist exactly when it
    installs a query authority, which a literal floor without predicates does
    not.
    """
    slots = model._regimes["parent"].solution.query_input_authorities
    assert slots is not None
    assert slots[0][0][0:2] == ("child", 1)
    own_slot = ("parent", 0)
    owns_query = model._regimes["parent"].solution.query_authority is not None
    if owns_query:
        assert slots[0][-1][0:2] == own_slot
    else:
        assert own_slot not in {slot[0:2] for slot in slots[0]}
    assert output.query_status is not None
    np.testing.assert_array_equal(
        jnp.stack(tuple(jnp.stack(status) for status in output.query_status)),
        np.zeros((len(slots[0]), 3), dtype=bool),
    )
    validate_query_input_status(statuses=output.query_status, authorities=slots[0])
    if not owns_query:
        assert output.bound_status == ()
        return
    assert output.bound_status
    np.testing.assert_array_equal(
        jnp.stack(tuple(jnp.stack(status) for status in output.bound_status)),
        np.zeros((len(output.bound_status), 2), dtype=bool),
    )


@pytest.mark.parametrize("ev1", [False, True], ids=("hard-max", "ev1"))
@pytest.mark.parametrize("equality", [False, True], ids=("both-live", "first-at-floor"))
def test_actual_dc_consumer_recovers_two_axis_corners_before_choice_aggregation(
    *, ev1: bool, equality: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply actual target time, parameters and budgets before ranking choices."""
    model, params = _model_and_params(ev1=ev1, equality=equality)
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    assert authority.state_names == ("z", "w")
    assert authority.action_names == ("job",)
    assert authority.coordinate_kind == "resources"
    assert authority.corner_coordinate_kind == "compact_interval"
    assert authority.target_periods == (1,)
    assert authority.bound.lower_inputs == ("age", "floor", "period", "w", "z")
    assert authority.bound.dependency_identity == (
        (
            ("1",),
            (
                ((), _lower),
                (("age",), "age"),
                (("floor",), ("parameter", "child", "borrowing_lower", "floor")),
                (("period",), "period"),
                (("w",), "w"),
                (("z",), "z"),
            ),
        ),
    )
    assert dict(authority.evaluator_inputs)["lower_bound"] == (
        "age",
        "borrowing_lower__floor",
        "period",
        "w",
        "z",
    )
    source_params = context["flat_params"]["parent"]
    target_params = context["flat_params"]["child"]
    np.testing.assert_array_equal(source_params["borrowing_lower__floor"], 0.25)
    np.testing.assert_array_equal(target_params["borrowing_lower__floor"], 0.125)
    carries = context["next_regime_to_continuation"]
    carry = carries["child"]
    context = _with_corner_only_support(carry=carry, context=context)
    output, conditional, aggregated = _observe_installed_consumer(
        kernel=kernel, context=context, monkeypatch=monkeypatch
    )
    selected = [
        item for item in conditional if np.array_equal(item[0][:3], [0.0, 0.25, 0.5])
    ]
    final = [
        pair for facts, pair in aggregated if np.array_equal(facts, [0.0, 0.25, 0.5])
    ]
    assert selected, "The installed consumer did not reach the interior actual target."
    assert final, "The installed factory did not aggregate the conditional target rows."
    shift = -1.75 if equality else 0.125
    expected_facts = [0.0, 0.25, 0.5, 1.0, 41.0, 0.125, shift, 0.5, 5.0]
    consumption = (
        np.array([0.0, 0.25]) if equality else np.array([15.0 / 8.0, 17.0 / 8.0])
    )
    # Bilinear weights are (3/8, 3/8, 1/8, 1/8), not a query-floor read of W.
    continuation = np.array([89.0 / 64.0, 97.0 / 64.0])
    expected_values = (
        np.array([-np.inf, np.log(0.25) - 0.125 + 0.5 * continuation[1]])
        if equality
        else np.log(consumption) - np.array([0.0, 0.125]) + 0.5 * continuation
    )
    expected_marginals = np.array([0.0, 40.0]) if equality else 10.0 / consumption
    for facts, values, marginals, savings in selected:
        assert all(
            channel.dtype == np.dtype(jnp.asarray(1.0).dtype)
            for channel in (facts, values, marginals, savings)
        )
        np.testing.assert_array_equal(facts, expected_facts)
        np.testing.assert_array_equal(savings, [5.0 / 8.0, 5.0 / 8.0])
        np.testing.assert_array_almost_equal(
            values, expected_values, decimal=DECIMAL_PRECISION
        )
        np.testing.assert_array_almost_equal(
            marginals, expected_marginals, decimal=DECIMAL_PRECISION
        )
    if ev1:
        weights = np.exp((expected_values - np.max(expected_values)) / 0.25)
        probabilities = weights / weights.sum()
        expected_value = np.max(expected_values) + 0.25 * np.log(weights.sum())
        expected_marginal = probabilities @ expected_marginals
    else:
        owner = int(np.argmax(expected_values))
        expected_value = expected_values[owner]
        expected_marginal = expected_marginals[owner]
    for pair in final:
        np.testing.assert_array_almost_equal(
            pair, [expected_value, expected_marginal], decimal=DECIMAL_PRECISION
        )
    _assert_clean_installed_statuses(model=model, output=output)
