"""Retain finite passive-corner contributions before their factor product underflows."""

import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import continuation as continuation_module
from _lcm.egm.carry import EGMCarry
from _lcm.egm.dcegm_query import DCEGMQuery, _passive_weights
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import BoundedResult, FeasibilityKernelOutput
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
)
from lcm.solvers import DCEGM
from lcm.transition import fixed_transition
from lcm.typing import ContinuousState, FloatND, ScalarFloat, UserParams
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_consumers import (
    _capture_with_checks_off,
    _run_installed,
)
from tests.egm.test_dynamic_dcegm_passive_consumers import (
    RegimeId,
    _AggregateInputs,
    _assert_clean_installed_statuses,
    _inverse,
    _parent_resources,
    _parent_utility,
    _savings,
    _source_lower,
)
from tests.solution._nbegm_direct_oracle import OracleContext
from tests.solution.test_egm_child_resources_age_factory import _continuation_plan

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _resources(liquid: ContinuousState) -> FloatND:
    return liquid + 1.0


def _liquid_law(savings: FloatND) -> ContinuousState:
    return savings


def _terminal(
    *, liquid: ContinuousState, z: ContinuousState, w: ContinuousState, scale: float
) -> FloatND:
    return liquid + scale * z * w


def _case_numbers() -> tuple[ScalarFloat, ScalarFloat, ScalarFloat]:
    """Check normal input factors and the independently calculated final value."""
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    assert dtype in (np.dtype(np.float32), np.dtype(np.float64))
    exponent = 80 if dtype == np.dtype(np.float32) else 600
    epsilon = jnp.asarray(math.ldexp(1.0, -exponent), dtype=dtype)
    scale = jnp.asarray(math.ldexp(1.0, exponent), dtype=dtype)
    exact = Fraction(1, 2**exponent) ** 2 * Fraction(2**exponent, 2)
    expected = jnp.asarray(float(exact), dtype=dtype)
    tiny = np.finfo(dtype).tiny
    for number in (epsilon, scale, expected):
        assert number.shape == ()
        assert np.dtype(number.dtype) == dtype
        assert np.isfinite(number)
        assert number >= tiny
    np.testing.assert_array_equal(expected, math.ldexp(1.0, -exponent - 1))
    np.testing.assert_array_equal(epsilon * scale, 1.0)
    return epsilon, scale, expected


def _model_and_params(*, epsilon: float, scale: float) -> tuple[Model, UserParams]:
    """Build two passive axes and an affine terminal continuation in liquid."""
    passive = {
        "z": IrregSpacedGrid(points=np.array([0.0, 1.0])),
        "w": IrregSpacedGrid(points=np.array([0.0, 1.0, 2.0])),
    }
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )

    def child_z(z: ContinuousState) -> ContinuousState:
        return epsilon + z * (1.0 - epsilon)

    def child_w(w: ContinuousState) -> ContinuousState:
        return epsilon + w * (1.0 - epsilon)

    child = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0])), **passive},
        actions={
            "consumption": IrregSpacedGrid(
                points=np.array([0.025, 0.125, 0.5, 1.0, 4.0])
            )
        },
        functions={
            "utility": _parent_utility,
            "resources": _resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=_source_lower)
        },
        state_transitions={
            "liquid": _liquid_law,
            "z": fixed_transition("z"),
            "w": fixed_transition("w"),
        },
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0])),
            n_constrained_points=4,
        ),
    )
    parent = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=np.array([0.125, 0.25])), **passive},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.025, 0.125, 0.25]))},
        functions={
            "utility": _parent_utility,
            "resources": _parent_resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=_source_lower)
        },
        state_transitions={"liquid": _liquid_law, "z": child_z, "w": child_w},
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.125, 0.25])),
            n_constrained_points=4,
        ),
    )
    model = Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0])),
                    **passive,
                },
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=42, step="Y"),
        edges={"parent": {"child": 40}, "child": {"done": 41}},
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
    )
    return model, {
        "parent": {
            "koopmans_aggregator": {"discount_factor": 0.75},
            "borrowing_lower": {"floor": 0.0},
        },
        "child": {
            "koopmans_aggregator": {"discount_factor": 0.5},
            "borrowing_lower": {"floor": 0.0},
        },
        "done": {"utility": {"scale": scale}},
    }


def _installed_query(*, model: Model) -> DCEGMQuery:
    """Read the actual target-bound factory from the installed parent program."""
    plan = _continuation_plan(model=model, regime="parent", period=0)
    query = plan.child_reads["child"].query
    assert isinstance(query, DCEGMQuery)
    authority = query.model.authority
    assert authority.state_names == ("z", "w")
    assert authority.action_names == ()
    assert authority.coordinate_kind == "resources"
    assert authority.corner_coordinate_kind == "compact_interval"
    assert authority.target_periods == (1,)
    assert dict(authority.evaluator_inputs)["lower_bound"] == (
        "borrowing_lower__floor",
    )
    assert query.input_names == frozenset(
        {"borrowing_lower__floor", "liquid", "koopmans_aggregator__discount_factor"}
    )
    return query


def _corner_bank(scale: ScalarFloat) -> FloatND:
    return scale * jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]])[..., None]


def _assert_stencil_activity(epsilon: ScalarFloat) -> None:
    """Keep two normal upper factors active despite their tiny joint product."""
    _, active, negative = _passive_weights(
        values=(epsilon, epsilon),
        grids=(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0, 2.0])),
    )
    np.testing.assert_array_equal(active, [[True, True, False], [True, True, False]])
    np.testing.assert_array_equal(negative, np.zeros((2, 3), dtype=bool))


def test_canonical_dc_query_retains_normal_corner_terms_from_tiny_joint_weights() -> (
    None
):
    """Retain beta times epsilon when both passive upper factors are epsilon."""
    epsilon, scale, expected = _case_numbers()
    model, _ = _model_and_params(epsilon=float(epsilon), scale=float(scale))
    query = _installed_query(model=model)
    capacity = query.model.authority.row_capacity
    shape = (2, 3, capacity)
    data = EGMQueryData(
        savings=jnp.zeros(shape),
        node_kind=jnp.full(shape, QueryNodeKind.PAD, dtype=jnp.int32),
        link_valid=jnp.zeros((2, 3, capacity - 1), dtype=jnp.bool_),
        owner_id=jnp.zeros(shape, dtype=jnp.int32),
        side_id=jnp.zeros(shape, dtype=jnp.int32),
        floor_used=jnp.zeros((2, 3)),
        corner_continuation=_corner_bank(scale),
        corner_kind=jnp.full((2, 3, 1), QueryNodeKind.FINITE, dtype=jnp.int32),
        corner_interval_id=jnp.zeros((2, 3, 1), dtype=jnp.int32),
        corner_side_id=jnp.zeros((2, 3, 1), dtype=jnp.int32),
    )
    _assert_stencil_activity(epsilon)
    result = query(
        grid=jnp.broadcast_to(jnp.arange(1.0, capacity + 1.0), shape),
        value=jnp.zeros(shape),
        marginal=jnp.zeros(shape),
        data=data,
        row_queries=jnp.ones((2, 3)),
        row_gradients=jnp.ones((2, 3)),
        passive_values=(epsilon, epsilon),
        passive_grids=(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0, 2.0])),
        pool={
            "liquid": jnp.asarray(0.0),
            "z": epsilon,
            "w": epsilon,
            "period": jnp.asarray(1),
            "age": jnp.asarray(41.0),
            "borrowing_lower__floor": jnp.asarray(0.0),
            "koopmans_aggregator__discount_factor": jnp.asarray(0.5),
        },
        resource_gradient=jnp.asarray(1.0),
    )
    assert result.value.dtype == expected.dtype
    np.testing.assert_allclose(
        result.value, expected, rtol=10**-DECIMAL_PRECISION, atol=0
    )
    np.testing.assert_array_equal([result.marginal, result.savings], [1.0, 0.0])
    np.testing.assert_array_equal(
        [*result.bound_status, result.coverage_missing], [False, False, False]
    )


def test_actual_dc_factory_retains_computed_tiny_passive_corner(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain the produced cross-corner through the actual factory."""
    epsilon, scale, expected = _case_numbers()
    model, params = _model_and_params(epsilon=float(epsilon), scale=float(scale))
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
    assert child.endog_grid.shape[:-1] == (2, 3)
    data = child.query_data_for_transport()
    assert data is not None
    np.testing.assert_array_equal(data.floor_used, np.zeros((2, 3)))
    np.testing.assert_array_equal(data.corner_continuation, _corner_bank(scale))
    np.testing.assert_array_equal(data.corner_kind, np.ones((2, 3, 1), dtype=np.int32))
    assert np.any(np.asarray(data.node_kind) == QueryNodeKind.FINITE)
    missing = replace(
        data,
        node_kind=jnp.full_like(data.node_kind, QueryNodeKind.PAD),
        link_valid=jnp.zeros_like(data.link_valid),
    )
    assert missing.corner_continuation is data.corner_continuation
    changed_context: OracleContext = {
        **context,
        "next_regime_to_continuation": {
            **context["next_regime_to_continuation"],
            "child": replace(child, query_data=missing),
        },
    }
    observed: list[tuple[np.ndarray, np.ndarray]] = []
    original = continuation_module._aggregate_child_choices

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record(facts: np.ndarray, pair: np.ndarray) -> None:
        observed.append((facts, pair))

    def observe(
        **kwargs: Unpack[_AggregateInputs],
    ) -> (
        tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ):
        result = original(**kwargs)
        pool = kwargs.get("query_pool")
        if kwargs.get("query") is not None:
            assert pool is not None
            assert isinstance(result, BoundedResult)
            assert kwargs["row_queries"].shape == (2, 3)
            facts: list[FloatND | float] = []
            for name in (
                "liquid",
                "z",
                "w",
                "period",
                "age",
                "borrowing_lower__floor",
                "koopmans_aggregator__discount_factor",
            ):
                numeric = pool[name]
                assert isinstance(numeric, jax.Array | float | int)
                facts.append(numeric)
            gradient = kwargs.get("query_euler_gradient")
            assert gradient is not None
            facts.append(gradient)
            jax.debug.callback(record, jnp.stack(facts), jnp.stack(result.output))
        return result

    monkeypatch.setattr(continuation_module, "_aggregate_child_choices", observe)
    _assert_stencil_activity(epsilon)
    output = _run_installed(kernel=kernel, context=changed_context)
    jax.block_until_ready(output.value)
    jax.effects_barrier()
    selected = [
        pair
        for facts, pair in observed
        if np.array_equal(
            facts,
            [0.0, float(epsilon), float(epsilon), 1.0, 41.0, 0.0, 0.5, 1.0],
        )
    ]
    assert selected, (
        "The installed factory did not query both tiny passive coordinates."
    )
    for pair in selected:
        assert pair.dtype == np.dtype(expected.dtype)
        np.testing.assert_allclose(
            pair[0], expected, rtol=10**-DECIMAL_PRECISION, atol=0
        )
        np.testing.assert_array_equal(pair[1], 1.0)
    assert isinstance(output, FeasibilityKernelOutput)
    _assert_clean_installed_statuses(model=model, output=output)
