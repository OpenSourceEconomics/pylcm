"""Conditional DC predicates govern actual child reads before choice ranking."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypedDict

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import validate_query_input_status
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExtremeValueTasteShocks,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
)
from lcm.regime import (
    ActionEntry,
    ConstraintEntry,
    FunctionEntry,
    StateEntry,
    StateTransitionEntry,
)
from lcm.solvers import DCEGM, GridSearch
from lcm.transition import fixed_transition
from lcm.typing import (
    ActionName,
    Age,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    FunctionName,
    ScalarFloat,
    ScalarInt,
    StateName,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_consumers import _capture_with_checks_off
from tests.egm.test_dynamic_dcegm_passive_consumers import (
    Job,
    RegimeId,
    _inverse,
    _observe_installed_consumer,
    _parent_resources,
    _savings,
    _source_lower,
)
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


class _ParentFields(TypedDict):
    """The five common declaration fields of the parent regime."""

    states: Mapping[StateName, StateEntry]
    actions: Mapping[ActionName, ActionEntry]
    functions: Mapping[FunctionName, FunctionEntry]
    constraints: Mapping[FunctionName, ConstraintEntry]
    state_transitions: Mapping[StateName, StateTransitionEntry]


def _child_utility(*, consumption: ContinuousAction, job: DiscreteAction) -> FloatND:
    return jnp.log(consumption / 2.0) + 10.0 * job


def _parent_utility(
    *, consumption: ContinuousAction, route: DiscreteAction, z: ContinuousState
) -> FloatND:
    return jnp.log(consumption / 0.25) + 3.0 * route + 0.0 * z


def _resources(
    *, liquid: ContinuousState, shift: float, period: ScalarInt, age: Age
) -> FloatND:
    return 2.0 * liquid + 2.0 + shift + (period - 1.0) + (age - 41.0)


def _threshold(*, job: DiscreteAction, z: ContinuousState, cutoff: float) -> BoolND:
    return (job == 0) | (z >= cutoff)


def _island(*, job: DiscreteAction, z: ContinuousState) -> BoolND:
    return (job == 0) | ((z > 0.4) & (z < 0.6))


def _next_liquid(savings: FloatND) -> ContinuousState:
    return savings


def _child_liquid(savings: FloatND) -> ContinuousState:
    return 4.0 * savings - 1.0


def _child_z(
    *, route: DiscreteAction, query_point: float | ScalarFloat
) -> ContinuousState:
    return jnp.where(route == 0, query_point, 0.0)


def _child_w(w: ContinuousState) -> ContinuousState:
    return 0.5 + 0.5 * w


def _terminal(
    *, liquid: ContinuousState, z: ContinuousState, w: ContinuousState
) -> FloatND:
    return liquid / 2.0 + 0.0 * z + 0.0 * w


def _model_and_params(
    *,
    predicate: Literal["none", "threshold", "island"],
    query_point: float | ScalarFloat,
    literal: bool = False,
    generic_parent: bool = False,
    ev1: bool = False,
) -> tuple[Model, UserParams]:
    """Keep exact first Euler tuples on a retained two-by-three passive stencil."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    passive = {
        "z": IrregSpacedGrid(points=np.array([0.0, 1.0])),
        "w": IrregSpacedGrid(points=np.array([0.0, 1.0, 2.0])),
    }
    child = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=np.array([0.0, 0.5, 1.5])), **passive},
        actions={
            "job": DiscreteGrid(category_class=Job),
            "consumption": IrregSpacedGrid(
                points=np.array([0.025, 0.125, 0.5, 1.0, 4.0])
            ),
        },
        functions={
            "utility": _child_utility,
            "resources": _resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={
            "borrowing": post_decision_lower_bound(
                margin=margin,
                lower=0.0 if literal else _source_lower,
            ),
            **({"eligibility": _threshold} if predicate == "threshold" else {}),
            **({"eligibility": _island} if predicate == "island" else {}),
        },
        state_transitions={
            "liquid": _next_liquid,
            "z": fixed_transition("z"),
            "w": fixed_transition("w"),
        },
        taste_shocks=ExtremeValueTasteShocks() if ev1 else None,
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 1.5])),
            n_constrained_points=4,
        ),
    )
    parent_fields: _ParentFields = {
        "states": {"liquid": IrregSpacedGrid(points=np.array([0.375, 0.5])), **passive},
        "actions": {
            "route": DiscreteGrid(category_class=Job),
            "consumption": IrregSpacedGrid(points=np.array([0.025, 0.125, 0.25])),
        },
        "functions": {
            "utility": _parent_utility,
            "resources": _parent_resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        "constraints": {
            "borrowing": post_decision_lower_bound(margin=margin, lower=0.25),
        },
        "state_transitions": {
            "liquid": _child_liquid,
            "z": _child_z,
            "w": _child_w,
        },
    }
    generic_fields: _ParentFields = {
        **parent_fields,
        "functions": {
            name: function
            for name, function in parent_fields["functions"].items()
            if name != "inverse_marginal_utility"
        },
    }
    parent = (
        Regime(**generic_fields, solver=GridSearch())
        if generic_parent
        else ConsumptionSavingsRegime(
            **parent_fields,
            liquid=margin,
            solver=DCEGM(
                savings_grid=IrregSpacedGrid(points=np.array([0.25, 0.375, 0.5])),
                n_constrained_points=4,
            ),
        )
    )
    model = Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.0, 16.0])),
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
            "koopmans_aggregator": {"discount_factor": 1.0},
            "next_z": {"query_point": query_point},
        },
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "resources": {"shift": 0.0},
            **({} if literal else {"borrowing_lower": {"floor": 0.0}}),
            **({"eligibility": {"cutoff": 0.6}} if predicate == "threshold" else {}),
            **({"taste_shocks": {"scale": 0.25}} if ev1 else {}),
        },
        "done": {},
    }


@pytest.mark.parametrize("literal", [True, False], ids=("literal", "feature"))
def test_no_predicate_dc_control_has_exact_published_endpoint_values(
    *, literal: bool
) -> None:
    """Both child rows survive and the parent selects its bonus-three route."""
    model, params = _model_and_params(
        predicate="none",
        query_point=0.5,
        literal=literal,
    )
    result = model.solve(params=params, log_level="off")
    np.testing.assert_array_almost_equal(
        result.values[1]["child"][0],
        np.full((2, 3), 10.0),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.values[0]["parent"][1],
        np.full((2, 3), 13.0),
        decimal=DECIMAL_PRECISION,
    )


def _assert_processed_cutoff(
    *, context: OracleContext, cutoff: FloatND, point: FloatND, case: str
) -> None:
    """Check the processed cutoff dtype, exact value and adjacent query points."""
    processed_cutoff = jnp.asarray(
        context["flat_params"]["child"]["eligibility__cutoff"]
    )
    assert processed_cutoff.dtype == cutoff.dtype
    np.testing.assert_array_equal(processed_cutoff, cutoff)
    if case == "below":
        np.testing.assert_array_equal(point, jnp.nextafter(processed_cutoff, -jnp.inf))
        assert bool(point < processed_cutoff)
    elif case == "above":
        np.testing.assert_array_equal(point, jnp.nextafter(processed_cutoff, jnp.inf))
        assert bool(point > processed_cutoff)
    elif case == "equal":
        np.testing.assert_array_equal(point, processed_cutoff)


@pytest.mark.parametrize(
    "case",
    ["interior", "below", "equal", "above", "island", "ev1-single"],
)
def test_installed_dc_predicate_precedes_choice_and_parent_aggregation(
    *, case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank actual eligible conditional rows without interpolating their masks."""
    cutoff = jnp.asarray(0.6, dtype=jnp.asarray(0.0).dtype)
    point = {
        "interior": jnp.asarray(0.5, dtype=cutoff.dtype),
        "below": jnp.nextafter(cutoff, -jnp.inf),
        "equal": cutoff,
        "above": jnp.nextafter(cutoff, jnp.inf),
        "island": jnp.asarray(0.5, dtype=cutoff.dtype),
        "ev1-single": jnp.asarray(0.5, dtype=cutoff.dtype),
    }[case]
    island = case == "island"
    admitted = case in ("equal", "above", "island")
    model, params = _model_and_params(
        predicate="island" if island else "threshold",
        query_point=point,
        ev1=case == "ev1-single",
    )
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    if not island:
        _assert_processed_cutoff(context=context, cutoff=cutoff, point=point, case=case)
    carry = context["next_regime_to_continuation"]["child"]
    assert isinstance(carry, EGMCarry)
    data = carry.query_data_for_transport()
    assert data is not None
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    assert authority.state_names == ("z", "w")
    assert authority.action_names == ("job",)
    assert authority.target_periods == (1,)
    assert data.node_kind.shape[:3] == (2, 3, 2)
    assert np.all(
        np.any(
            np.asarray(data.node_kind)[:, :, 1] == QueryNodeKind.FINITE,
            axis=-1,
        )
    )
    np.testing.assert_array_equal(data.floor_used, 0.0)
    assert data.corner_continuation is not None
    np.testing.assert_array_equal(data.corner_continuation, 0.0)
    if island:
        # Both stored z nodes exclude job 1; its conditional source stays live.
        np.testing.assert_array_almost_equal(
            context["next_regime_to_V_arr"]["child"][0],
            np.zeros((2, 3)),
            decimal=DECIMAL_PRECISION,
        )
    else:
        np.testing.assert_array_almost_equal(
            context["next_regime_to_V_arr"]["child"][0],
            np.broadcast_to(np.asarray([0.0, 10.0])[:, None], (2, 3)),
            decimal=DECIMAL_PRECISION,
        )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    output, conditional, aggregate = _observe_installed_consumer(
        kernel=kernel,
        context=context,
        monkeypatch=monkeypatch,
    )
    selected = [
        item for item in conditional if np.array_equal(item[0][:3], [0.0, point, 0.5])
    ]
    pairs = [
        pair for facts, pair in aggregate if np.array_equal(facts, [0.0, point, 0.5])
    ]
    assert selected, "The real installed actual target was not reached."
    assert pairs, "The real installed actual target was not reached."
    for facts, values, marginals, savings in selected:
        np.testing.assert_array_equal(
            facts, [0.0, point, 0.5, 1.0, 41.0, 0.0, 0.0, 1.0, 4.0]
        )
        np.testing.assert_array_equal(savings, [0.0, 0.0])
        np.testing.assert_array_almost_equal(
            values,
            [0.0, 10.0 if admitted else -np.inf],
            decimal=DECIMAL_PRECISION,
        )
        np.testing.assert_array_equal(marginals, [4.0, 4.0 if admitted else 0.0])
    for pair in pairs:
        np.testing.assert_array_almost_equal(
            pair,
            [10.0 if admitted else 0.0, 4.0],
            decimal=DECIMAL_PRECISION,
        )
    np.testing.assert_array_almost_equal(
        output.value[1],
        np.full((2, 3), 10.0 if admitted else 3.0),
        decimal=DECIMAL_PRECISION,
    )
    authority = model._regimes["parent"].solution.query_input_authorities
    assert authority is not None
    assert output.query_status is not None
    assert any(slot[:2] == ("child", 1) for slot in authority[0])
    for status in output.query_status:
        np.testing.assert_array_equal(jnp.stack(status), [False, False, False])
    validate_query_input_status(statuses=output.query_status, authorities=authority[0])
    for status in output.bound_status:
        np.testing.assert_array_equal(jnp.stack(status), [False, False])


def test_generic_parent_keeps_node_masked_child_value() -> None:
    """The ordinary value reader interpolates endpoint zero and ten to five."""
    model, params = _model_and_params(
        predicate="threshold",
        query_point=0.5,
        generic_parent=True,
    )
    result = model.solve(params=params, log_level="off")
    np.testing.assert_array_almost_equal(
        result.values[1]["child"][0],
        np.broadcast_to(np.asarray([0.0, 10.0])[:, None], (2, 3)),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.values[0]["parent"][1],
        np.full((2, 3), 5.0),
        decimal=DECIMAL_PRECISION,
    )
