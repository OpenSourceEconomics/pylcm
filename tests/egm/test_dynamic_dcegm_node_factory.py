"""Actual feature-child stochastic folds retain node inputs, values and statuses."""

import functools
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import continuation as continuation_module
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _ChildCarryReader, _ChildCarryResult
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import (
    BoundedResult,
    FeasibilityKernelOutput,
    validate_query_input_status,
)
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    StochasticTransition,
    categorical,
    post_decision_lower_bound,
)
from lcm.solver_api import EGM_CONTINUATION
from lcm.solvers import DCEGM, STOCHASTIC_NODE_AXIS, RFCEnvelope
from lcm.transition import fixed_transition
from lcm.typing import (
    Age,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarFloat,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_consumers import (
    _capture_with_checks_off,
)
from tests.egm.test_dynamic_dcegm_passive_consumers import (
    RegimeId,
    _AggregateInputs,
    _inverse,
    _parent_resources,
    _parent_utility,
    _savings,
)
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=True)
class Node:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


def _probabilities(
    weights: tuple[float, float, float],
) -> Callable[[DiscreteState], FloatND]:
    def probabilities(kind: DiscreteState) -> FloatND:
        return jnp.asarray([weights, weights, weights])[kind]

    return probabilities


def _draw_law(*, savings: FloatND, next_kind: DiscreteState) -> ContinuousState:
    return 5.0 * savings - 1.25 + next_kind / 8.0


def _terminal_law(savings: FloatND) -> ContinuousState:
    return 3.0 * savings + 4.0


def _terminal(liquid: ContinuousState) -> FloatND:
    return liquid / 4.0


def _lower(
    *, kind: DiscreteState, floor: float, period: ScalarInt, age: Age
) -> FloatND:
    return floor + kind / 8.0 + period / 16.0 + (age - 40.0) / 16.0


def _resources(
    *,
    liquid: ContinuousState,
    kind: DiscreteState,
    shift: float,
    dead_shift: float,
    period: ScalarInt,
    age: Age,
) -> FloatND:
    return (
        2.0 * liquid
        + shift
        + kind / 4.0
        - dead_shift * (kind == Node.high)
        + period / 16.0
        + (age - 40.0) / 16.0
    )


def _model(
    *, case: Literal["finite", "zero-dead", "positive-dead"]
) -> tuple[Model, UserParams]:
    weights = (0.25, 0.75, 0.0) if case == "zero-dead" else (0.25, 0.5, 0.25)
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    actions = {
        "consumption": IrregSpacedGrid(points=np.array([0.025, 0.125, 0.5, 1.0, 4.0]))
    }
    model = Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "liquid": IrregSpacedGrid(points=np.array([1.0, 1.25])),
                    "kind": DiscreteGrid(category_class=Node),
                },
                actions=actions,
                functions={
                    "utility": _parent_utility,
                    "resources": _parent_resources,
                    "savings": _savings,
                    "inverse_marginal_utility": _inverse,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.25)
                },
                state_transitions={
                    "liquid": _draw_law,
                    "kind": StochasticTransition(func=_probabilities(weights)),
                },
                solver=DCEGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.25, 0.75, 1.25])),
                    n_constrained_points=4,
                    envelope=RFCEnvelope(),
                ),
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.0, 3.0, 6.0])),
                    "kind": DiscreteGrid(category_class=Node),
                },
                actions=actions,
                functions={
                    "utility": _parent_utility,
                    "resources": _resources,
                    "savings": _savings,
                    "inverse_marginal_utility": _inverse,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=_lower)
                },
                state_transitions={
                    "liquid": _terminal_law,
                    "kind": fixed_transition("kind"),
                },
                solver=DCEGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.0, 3.0, 6.0])),
                    n_constrained_points=4,
                ),
            ),
            "done": Regime(
                states={"liquid": IrregSpacedGrid(points=np.array([0.0, 32.0]))},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=42, step="Y"),
        edges={"parent": {"child": 40}, "child": {"done": 41}},
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(axis_widths={STOCHASTIC_NODE_AXIS: 2}),
    )
    return model, {
        "parent": {"koopmans_aggregator": {"discount_factor": 0.75}},
        "child": {
            "koopmans_aggregator": {"discount_factor": 0.5},
            "borrowing_lower": {"floor": 0.375},
            "resources": {
                "shift": 0.875,
                "dead_shift": 0.0 if case == "finite" else 1.25,
            },
        },
        "done": {},
    }


def _observe_node_factory(
    *,
    kernel: _DCEGMPeriodKernel,
    context: OracleContext,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    FeasibilityKernelOutput,
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    """Record unchanged actual per-node and blocked-reader returns."""
    nodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    aggregate = continuation_module._aggregate_child_choices
    read_child = _ChildCarryReader.__call__

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_node(facts: np.ndarray, pair: np.ndarray, flags: np.ndarray) -> None:
        nodes.append((facts, pair, flags))

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_fold(
        facts: np.ndarray, pair: np.ndarray, weights: np.ndarray, flags: np.ndarray
    ) -> None:
        folds.append((facts, pair, weights, flags))

    def observe_node(
        **kwargs: Unpack[_AggregateInputs],
    ) -> (
        tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ):
        result = aggregate(**kwargs)
        if kwargs.get("query") is not None:
            pool = kwargs.get("query_pool")
            assert pool is not None
            assert isinstance(result, BoundedResult)
            facts = jnp.stack(
                [
                    cast("ScalarFloat", pool[name])
                    for name in (
                        "liquid",
                        "kind",
                        "period",
                        "age",
                        "borrowing_lower__floor",
                        "resources__shift",
                        "resources__dead_shift",
                        "koopmans_aggregator__discount_factor",
                    )
                ]
            )
            flags = jnp.stack(
                tuple(jnp.stack(flag) for flag in result.query_status or ())
            )
            jax.debug.callback(record_node, facts, jnp.stack(result.output), flags)
        return result

    # keyword-only-exempt: library-callback=_ChildCarryReader.__call__
    def observe_fold(
        self: _ChildCarryReader, savings_value: ScalarFloat
    ) -> _ChildCarryResult:
        result = read_child(self, savings_value)
        if self.read.query is not None:
            assert self.read.stochastic_state_names == ("kind",)
            assert self.stochastic_node_width == 2
            assert tuple(values.size for values in self.stochastic_node_values) == (3,)
            assert isinstance(result, BoundedResult)
            ordinary = cast("BoundedResult[tuple[ScalarFloat, ScalarFloat]]", result)
            facts = jnp.stack(
                [savings_value, cast("ScalarFloat", self.combo_pool["kind"])]
            )
            flags = jnp.stack(
                tuple(jnp.stack(flag) for flag in ordinary.query_status or ())
            )
            jax.debug.callback(
                record_fold,
                facts,
                jnp.stack(ordinary.output),
                self.weight_vecs[0],
                flags,
            )
        return result

    monkeypatch.setattr(continuation_module, "_aggregate_child_choices", observe_node)
    monkeypatch.setattr(_ChildCarryReader, "__call__", observe_fold)
    program = kernel.core_programs()["main"]
    (node_axis,) = program.requirements.reduced_axes
    assert node_axis.name == STOCHASTIC_NODE_AXIS
    assert node_axis.coordinate_names == ("kind",)
    assert node_axis.coordinate_extents == (3,)
    output = kernel(
        compiled_cores={
            "main": functools.partial(program.function, **{node_axis.width_keyword: 2})
        },
        **context,
        logger=logging.getLogger(__name__),
    )
    jax.block_until_ready(output.value)
    jax.effects_barrier()
    assert isinstance(output, FeasibilityKernelOutput)
    source_carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(source_carry, EGMCarry)
    assert source_carry.query_data_for_transport() is None
    if output.bound_status:
        np.testing.assert_array_equal(
            jnp.stack(tuple(jnp.stack(flag) for flag in output.bound_status)),
            np.zeros((len(output.bound_status), 2), dtype=bool),
        )
    return output, nodes, folds


@pytest.mark.parametrize("case", ["finite", "zero-dead", "positive-dead"])
def test_actual_feature_factory_folds_three_distinct_nodes_in_width_two(
    *,
    case: Literal["finite", "zero-dead", "positive-dead"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve node-specific target inputs and unnormalized economic contributions."""
    model, params = _model(case=case)
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    data = child.query_data_for_transport()
    assert data is not None
    assert child.endog_grid.shape[0] == 3
    assert data.liquid_corner_data is None
    np.testing.assert_array_equal(data.floor_used, [0.5, 0.625, 0.75])
    assert data.corner_continuation is not None
    np.testing.assert_allclose(
        data.corner_continuation[:, 0],
        [1.375, 1.46875, 1.5625],
        rtol=10**-DECIMAL_PRECISION,
        atol=0,
    )
    np.testing.assert_array_equal(data.corner_kind, np.ones((3, 1), dtype=np.int32))
    absent = replace(
        data,
        node_kind=jnp.full_like(data.node_kind, QueryNodeKind.PAD),
        link_valid=jnp.zeros_like(data.link_valid),
    )
    assert absent.corner_continuation is data.corner_continuation
    changed_context: OracleContext = {
        **context,
        "next_regime_to_continuation": {
            **context["next_regime_to_continuation"],
            "child": replace(child, query_data=absent),
        },
    }
    output, nodes, folds = _observe_node_factory(
        kernel=kernel, context=changed_context, monkeypatch=monkeypatch
    )
    selected = [
        (facts, pair, flags)
        for facts, pair, flags in nodes
        if facts[0] == facts[1] / 8.0
    ]
    observed_nodes = {int(facts[1]) for facts, _, _ in selected}
    required_nodes = {0, 1} if case == "zero-dead" else {0, 1, 2}
    assert required_nodes <= observed_nodes <= {0, 1, 2}
    expected_values: list[float] = []
    expected_marginals: list[float] = []
    for node in range(3):
        dead = case != "finite" and node == 2
        consumption = 0.5 + 0.375 * node
        expected_values.append(
            -np.inf
            if dead
            else float(np.log(consumption) + 0.5 * (1.375 + 0.09375 * node))
        )
        expected_marginals.append(0.0 if dead else 10.0 / consumption)
    for facts, pair, flags in selected:
        node = int(facts[1])
        np.testing.assert_array_equal(
            facts[2:], [1.0, 41.0, 0.375, 0.875, 0.0 if case == "finite" else 1.25, 0.5]
        )
        np.testing.assert_allclose(
            pair,
            [expected_values[node], expected_marginals[node]],
            rtol=10**-DECIMAL_PRECISION,
            atol=0,
        )
        np.testing.assert_array_equal(flags, np.zeros((1, 3), dtype=bool))
    weights = np.asarray(
        [0.25, 0.75, 0.0] if case == "zero-dead" else [0.25, 0.5, 0.25]
    )
    expected_value = (
        -np.inf
        if case == "positive-dead"
        else sum(weights[n] * expected_values[n] for n in range(3) if weights[n] != 0.0)
    )
    expected_marginal = sum(weights[n] * expected_marginals[n] for n in range(3))
    selected_folds = [
        (pair, actual_weights, flags)
        for facts, pair, actual_weights, flags in folds
        if np.array_equal(facts, [0.25, 0.0])
    ]
    assert selected_folds
    for pair, actual_weights, flags in selected_folds:
        np.testing.assert_array_equal(actual_weights, weights)
        np.testing.assert_allclose(
            pair,
            [expected_value, expected_marginal],
            rtol=10**-DECIMAL_PRECISION,
            atol=0,
        )
        np.testing.assert_array_equal(flags, np.zeros((1, 3), dtype=bool))
    assert isinstance(output, FeasibilityKernelOutput)
    slots = model._regimes["parent"].solution.query_input_authorities
    assert slots is not None
    assert tuple((name, period) for name, period, _ in slots[0]) == (("child", 1),)
    assert output.query_status is not None
    validate_query_input_status(statuses=output.query_status, authorities=slots[0])
    np.testing.assert_array_equal(
        jnp.stack(tuple(jnp.stack(flag) for flag in output.query_status)),
        np.zeros((1, 3), dtype=bool),
    )
