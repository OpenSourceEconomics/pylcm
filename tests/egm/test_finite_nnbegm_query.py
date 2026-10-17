"""Finite NB reads keep native tuples and nodewise outer ownership."""

from dataclasses import replace
from types import MappingProxyType
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import (
    EGMRowTopologySpec,
    FiniteOuterQuerySpec,
    build_query_model_authority,
)
from _lcm.egm import nbegm_step
from _lcm.egm.nbegm_query import NBEGMQuery, build_nbegm_query
from _lcm.egm.nbegm_query_model import NBEGMQueryModel
from _lcm.egm.query_data import EGMQueryData
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.solution.feasibility import BoundInputStatus
from _lcm.solution.nbegm import build_nbegm_interval_recipe
from _lcm.solution.negm import (
    _with_no_adjustment_outer_function,
    _without_outer_post_decision,
)
from _lcm.typing import EconFunction, EconFunctionKwargs, EconFunctionsMapping
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_nbegm_conditional_query_producer import _OwnedQueryStep
from tests.solution._crra_preferences import crra_preferences

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


class _FiniteResult(Protocol):
    value: FloatND
    marginal: FloatND
    coverage_missing: BoolND
    bound_status: BoundInputStatus


def _resources(*, liquid: ContinuousState) -> FloatND:
    return 2.0 * liquid


def _floor(*, stock: ContinuousState, floor_scale: FloatND) -> FloatND:
    return floor_scale * stock * (1.0 - stock)


def _utility(*, consumption: ContinuousAction, bonus: FloatND) -> FloatND:
    return consumption + bonus


def _aggregator(*, utility: FloatND, CE: FloatND, discount_factor: FloatND) -> FloatND:
    return utility + discount_factor * CE


def _new_stock(*, stock: ContinuousState, investment: ContinuousAction) -> FloatND:
    return stock + investment


def _query(*, actions: bool) -> NBEGMQuery:
    """Declare native NB interval identity independently of the numerical bank."""
    functions = cast(
        "EconFunctionsMapping",
        MappingProxyType(
            {
                "resources": _resources,
                "floor": _floor,
                "utility": _utility,
                "koopmans_aggregator": _aggregator,
                "new_stock": _new_stock,
            }
        ),
    )
    bound = PostDecisionBoundSpec(
        post_decision="savings",
        liquid_state="liquid",
        action="consumption",
        resources_target="resources",
        kind="dag",
        literal_value=None,
        lower_target="floor",
        lower_inputs=("stock", "floor_scale"),
        dependency_identity=(
            ((), _floor),
            (("stock",), "stock"),
            (("floor_scale",), "floor_scale"),
        ),
        declaration_names=("borrowing",),
    )
    recipe = build_nbegm_interval_recipe(
        declarations=functions,
        functions=functions,
        liquid_name="liquid",
        state_names=("stock", "liquid"),
        action_names=("job",) if actions else (),
    )
    assert recipe.source_identity == ()
    authority = build_query_model_authority(
        bound=bound,
        state_names=("stock",),
        action_names=("job",) if actions else (),
        coordinate_kind="liquid",
        marginal_convention="ordinary_linear",
        corner_coordinate_kind="compact_interval",
        row_capacity=3,
        interval_ids=(0,),
        interval_side_ids=(0,),
        evaluator_targets=(
            ("resources", "resources"),
            ("utility", "utility"),
            ("koopmans_aggregator", "koopmans_aggregator"),
            ("outer_post_decision", "new_stock"),
        ),
        functions=functions,
        functions_by_period=MappingProxyType({0: functions}),
        target_periods=(0,),
        row_topology=EGMRowTopologySpec(
            kind="nb_budget_interval_run",
            owner_capacity=1,
        ),
        interval_source_identity=recipe.source_identity,
        outer_mode="finite_max",
        outer_capacity=2,
        finite_outer=FiniteOuterQuerySpec(
            outer_state="stock",
            outer_action="investment",
            outer_post_decision="new_stock",
            keeper_target=None,
            cost_target=None,
            outer_nodes=(1.0,),
        ),
    )
    pools = {
        "keeper": _with_no_adjustment_outer_function(
            functions=functions,
            durable_state="stock",
            outer_post_decision="new_stock",
            no_adjustment_func=None,
        ),
        "adjuster": _without_outer_post_decision(
            functions=functions,
            outer_post_decision="new_stock",
        ),
    }
    assert pools["keeper"]["new_stock"] is not functions["new_stock"]
    assert "new_stock" not in pools["adjuster"]
    inners = {}
    for role, role_pool in pools.items():
        inner_authority = build_query_model_authority(
            bound=bound,
            state_names=authority.state_names,
            action_names=authority.action_names,
            coordinate_kind="liquid",
            marginal_convention="ordinary_linear",
            corner_coordinate_kind="compact_interval",
            row_capacity=3,
            interval_ids=(0,),
            interval_side_ids=(0,),
            # O is supplied by the outer operator, not recomputed as an action.
            evaluator_targets=tuple(
                (purpose, target)
                for purpose, target in authority.evaluator_targets
                if purpose != "outer_post_decision"
            ),
            functions=role_pool,
            functions_by_period=MappingProxyType({0: role_pool}),
            target_periods=(0,),
            row_topology=authority.row_topology,
            interval_source_identity=(),
        )
        role_recipe = build_nbegm_interval_recipe(
            declarations=role_pool,
            functions=role_pool,
            liquid_name="liquid",
            state_names=("stock", "liquid"),
            action_names=authority.action_names,
        )
        assert role_recipe.source_identity == recipe.source_identity
        inners[role] = NBEGMQueryModel(
            authority=inner_authority,
            functions=role_pool,
            koopmans_aggregator=cast("EconFunction", _aggregator),
            offsets=jnp.asarray((0.0, 1.0 / 16.0, 0.5, 4.0)),
            intervals=role_recipe,
            intervals_by_period=MappingProxyType({0: role_recipe}),
            passive_state_names=("stock",),
            functions_by_period=MappingProxyType({0: role_pool}),
            retain_nested_feasibility=True,
        )
    parent = replace(
        inners["keeper"],
        authority=authority,
        functions=functions,
        functions_by_period=MappingProxyType({0: functions}),
        finite_inner_models=MappingProxyType(inners),
    )
    query = build_nbegm_query(
        model=parent,
        functions=functions,
        koopmans_aggregator=cast("EconFunction", _aggregator),
        target_period=0,
    )
    assert "investment" not in query.input_names
    assert "new_stock" not in query.input_names
    return query


def _owned(*, saving: float) -> EGMQueryData:
    """Emit actual NB source owner/link/saving/interval metadata at liquid one."""
    savings = jnp.asarray((0.0, saving, 4.0))
    # Geometry uses a subset of the declared offsets with identical endpoints;
    # interval/owner identity never claims an Euler-node index in the full grid.
    assert bool(jnp.all(jnp.isin(savings, jnp.asarray((0.0, 1.0 / 16.0, 0.5, 4.0)))))
    marginal = jnp.full((1, 3), 1.0 / (2.0 - saving))
    result = cast(
        "_OwnedQueryStep",
        nbegm_step.nbegm_per_interval_continuation_step_savings(
            liquid_grid=jnp.asarray((0.75, 1.0, 1.25)),
            savings_grid=savings,
            cont_value=marginal * savings[None, :],
            cont_marginal=marginal,
            discount_factor=jnp.asarray(1.0),
            preferences=crra_preferences(crra=1.0),
            coh_slopes=jnp.asarray((2.0,)),
            coh_intercepts=jnp.asarray((0.0,)),
            breakpoints=jnp.asarray(()),
            arithmetic="certified",
            retain_query=True,
            return_owner=True,
        ),
    )
    data = result.query_data
    assert isinstance(data, EGMQueryData)
    np.testing.assert_array_almost_equal(
        data.savings[1],
        saving,
        decimal=DECIMAL_PRECISION,
    )
    assert data.corner_continuation is not None
    np.testing.assert_array_equal(data.corner_continuation, (0.0,))
    return data


def _bank(
    *,
    values: FloatND,
    marginals: FloatND,
    savings: FloatND,
) -> tuple[FloatND, FloatND, FloatND, EGMQueryData]:
    """Pair declared economic row channels with genuine NB source geometry."""
    assert values.shape == marginals.shape == savings.shape
    # Native emitter establishes legal source provenance. Value and paired native
    # marginal channels below are independent represented-row algebra witnesses.
    sources = {
        float(item): _owned(saving=float(item))
        for item in np.unique(np.asarray(savings))
    }
    records = [sources[float(item)] for item in np.asarray(savings).flat]
    data = jax.tree.map(
        lambda *leaves: jnp.stack(leaves).reshape((*values.shape, *leaves[0].shape)),
        *records,
    )
    # b(C)=scale*C*(1-C) is zero at both contributing source passive nodes.
    data = replace(data, floor_used=jnp.asarray((0.0, 0.0)))
    grid = jnp.broadcast_to(jnp.asarray((0.75, 1.0, 1.25)), (*values.shape, 3))
    return (
        grid,
        jnp.broadcast_to(values[..., None], grid.shape),
        jnp.broadcast_to(
            marginals[..., None],
            grid.shape,
        ),
        data,
    )


def _read(
    *,
    query: NBEGMQuery,
    bank: tuple[FloatND, FloatND, FloatND, EGMQueryData],
    passive: float,
    floor_scale: float,
    bonus: float,
    gradient: float,
) -> _FiniteResult:
    grid, value, marginal, data = bank
    actions = query.model.authority.action_names
    query_shape = grid.shape[:-2]
    result = cast(
        "_FiniteResult",
        query.read_choices(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=jnp.full(query_shape, 1.0),
            row_gradients=jnp.full(query_shape, gradient),
            passive_values=(jnp.asarray(passive),),
            passive_grids=(jnp.asarray((0.0, 1.0)),),
            action_names=actions,
            action_values=(jnp.asarray((0, 1), dtype=jnp.int32),) if actions else (),
            pool=cast(
                "EconFunctionKwargs",
                {
                    "liquid": jnp.asarray(1.0),
                    "stock": jnp.asarray(passive),
                    "floor_scale": jnp.asarray(floor_scale),
                    "bonus": jnp.asarray(bonus),
                    "discount_factor": jnp.asarray(0.0),
                },
            ),
            euler_gradient=jnp.asarray(gradient),
            collect_continuation_read_counts=False,
        ),
    )
    assert not bool(jnp.any(result.coverage_missing))
    assert not bool(jnp.any(result.bound_status.nonfinite_floor))
    assert not bool(jnp.any(result.bound_status.invalid_shifted_grid))
    return result


def test_off_grid_floor_masks_owned_savings_before_discrete_choice() -> None:
    """An actual between-node floor rejects the high-valued illegal saving."""
    query = _query(actions=True)
    values = jnp.asarray((((4.0, 4.0), (3.0, 3.0)),) * 2)
    saves = jnp.asarray((((1.0 / 16.0,) * 2, (0.5,) * 2),) * 2)
    result = _read(
        query=query,
        bank=_bank(values=values, marginals=jnp.full_like(values, 7.0), savings=saves),
        passive=0.5,
        floor_scale=0.5,
        bonus=-10.0,
        gradient=1.0,
    )
    np.testing.assert_array_equal(result.value, (-65.0 / 8.0, 3.0))
    np.testing.assert_array_equal(result.marginal, (2.0, 7.0))
    assert int(jnp.argmax(result.value)) == 1


@pytest.mark.parametrize(
    "ordinary_keeper",
    [False, True],
    ids=["corner-wins", "keeper-row-tie"],
)
def test_finite_corner_uses_native_budget_gradient_once(
    *,
    ordinary_keeper: bool,
) -> None:
    """A corner and a tied keeper retain distinct native-liquid marginal units."""
    query = _query(actions=False)
    values = jnp.full((2, 2), 7.0 / 4.0)
    saving = 0.5 if ordinary_keeper else 1.0 / 16.0
    saves = jnp.asarray(((saving, 1.0 / 16.0),) * 2)
    result = _read(
        query=query,
        bank=_bank(values=values, marginals=jnp.full_like(values, 7.0), savings=saves),
        passive=0.5,
        floor_scale=1.0,
        bonus=0.0,
        gradient=3.0,
    )
    np.testing.assert_array_equal(result.value, 7.0 / 4.0)
    np.testing.assert_array_equal(result.marginal, 21.0 if ordinary_keeper else 6.0)


@pytest.mark.parametrize(
    ("passive", "value", "marginal"),
    [(0.5, 5.0, 5.0), (1.5, 7.0, 11.0)],
    ids=["interpolation", "signed-extrapolation"],
)
def test_outer_winners_precede_passive_blending(
    *,
    passive: float,
    value: float,
    marginal: float,
) -> None:
    """Each passive node supplies its own winning outer tuple and paired derivative."""
    query = _query(actions=False)
    values = jnp.asarray(((4.0, 0.0), (0.0, 6.0)))
    marginals = jnp.asarray(((2.0, 13.0), (17.0, 8.0)))
    result = _read(
        query=query,
        bank=_bank(
            values=values,
            marginals=marginals,
            savings=jnp.full_like(values, 0.5),
        ),
        passive=passive,
        floor_scale=0.0,
        bonus=-10.0,
        gradient=1.0,
    )
    np.testing.assert_array_equal(result.value, value)
    np.testing.assert_array_equal(result.marginal, marginal)
