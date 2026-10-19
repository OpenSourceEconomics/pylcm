"""Finite policy floors authenticate each demanded and realized period's DAG."""

import inspect
from collections.abc import Hashable

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.preferences import concatenate_regime_function
from _lcm.engine import NNBEGMPolicyRead
from _lcm.regime_building.phases import phase_variation_paths
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ByAge,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    Phased,
    Regime,
    Transition,
    categorical,
    fixed_transition,
    outer_unchanged,
    post_decision_lower_bound,
)
from lcm.solvers import NBEGM, NNBEGM, FiniteOuterGrid, GridSearch
from lcm.typing import ContinuousState, FloatND, ScalarInt, UserFunction
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution import test_nested_dynamic_borrowing as original

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    alive: ScalarInt
    done: ScalarInt


def _early_coefficient() -> FloatND:
    return jnp.asarray(1.0 / 4.0)


def _late_coefficient() -> FloatND:
    return jnp.asarray(1.0 / 8.0)


def _build_coefficient(age: float) -> UserFunction:
    return _early_coefficient if age < 2.0 else _late_coefficient


def _coefficient_signature(age: float) -> float:
    return age


def _ordinary_coefficient(*, age: FloatND) -> FloatND:
    return jnp.where(age < 2.0, 1.0 / 4.0, 1.0 / 8.0)


def _floor(*, illiquid: ContinuousState, floor_coefficient: FloatND) -> FloatND:
    return illiquid * floor_coefficient


def _model(*, specialized: bool, subset: bool) -> Model:
    """Demand alive at ages1,2 while its realized upstream route may visit only2."""
    coefficient = (
        AgeSpecializedFunction(
            build=_build_coefficient, signature=_coefficient_signature
        )
        if specialized
        else _ordinary_coefficient
    )
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    alive = NestedConsumptionSavingsRegime(
        states={"liquid": original._LIQUID, "illiquid": original._STOCK},
        actions={
            "consumption": original._CONSUMPTION,
            "investment": original._INVESTMENT,
        },
        state_transitions={
            "liquid": original.next_liquid,
            "illiquid": original.next_illiquid,
        },
        functions={
            "utility": original.utility,
            "service": original.service,
            "savings": original.savings,
            "new_illiquid": original.new_illiquid,
            "inverse_marginal_utility": original.inverse_marginal_utility,
            "chosen_floor_helper": original.chosen_floor_helper,
            "resources": original.resources,
            "floor_coefficient": coefficient,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=_floor)
        },
        solver=NNBEGM(
            inner=NBEGM(savings_grid=original._SAVINGS),
            outer_search=FiniteOuterGrid(grid=original._OUTER),
        ),
        liquid=margin,
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="investment",
            post_decision_state="new_illiquid",
            no_adjustment=outer_unchanged,
        ),
    )
    # Identity across the phase maps is part of the finite NN replay contract.
    alive_to_done = Transition(
        targets={"done": (1, 2)}, law=ByAge(cases={(1, 2): "done"})
    )
    perceived_source = Transition(
        targets={"alive": (0, 1)}, law=ByAge(cases={(0, 1): "alive"})
    )
    realized_source = (
        Transition(
            targets={"source": (0,), "alive": (1,)},
            law=ByAge(cases={0: "source", 1: "alive"}),
        )
        if subset
        else perceived_source
    )
    return Model(
        regimes={
            "source": Regime(
                states={"liquid": original._LIQUID, "illiquid": original._STOCK},
                state_transitions={
                    "liquid": fixed_transition("liquid"),
                    "illiquid": fixed_transition("illiquid"),
                },
                functions={"utility": original.terminal},
                solver=GridSearch(),
            ),
            "alive": alive,
            "done": Regime(
                states={"liquid": original._LIQUID, "illiquid": original._STOCK},
                functions={"utility": original.terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        edges=Phased(
            solve={"source": perceived_source, "alive": alive_to_done},
            simulate={"source": realized_source, "alive": alive_to_done},
        ),
        initial_nodes={(0, 1): "source"},
        regime_id_class=_RegimeId,
    )


def _evaluator_identity_pairs(
    value: Hashable,
) -> tuple[tuple[Hashable, Hashable], ...]:
    """Narrow canonical period/dependency identity tuples before dictionary reads."""
    assert isinstance(value, tuple)
    pairs: list[tuple[Hashable, Hashable]] = []
    for pair in value:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        key, identity = pair
        assert isinstance(key, Hashable)
        assert isinstance(identity, Hashable)
        pairs.append((key, identity))
    return tuple(pairs)


def _assert_actual_period_floor_authority(
    *,
    model: Model,
    specialized: bool,
    subset: bool,
) -> None:
    """Qualify the real canonical callbacks, periods, domains and floor values."""
    alive = model._regimes["alive"]
    assert alive.active_periods == (1, 2)
    expected = alive.solution.query_authority
    assert expected is not None
    assert expected.target_periods == (1, 2)
    assert expected.bound.kind == "dag"
    assert expected.finite_outer is not None
    assert expected.coordinate_kind == "liquid"
    assert expected.row_topology is not None
    assert expected.row_topology.kind == "nb_budget_interval_run"
    assert expected.finite_outer.outer_nodes == (0.0, 1.0, 2.0)
    assert expected.bound.lower_target is not None
    assert (
        phase_variation_paths(
            user_regime=model.user_regimes["alive"], law=model.graph.laws["alive"]
        )
        == ()
    )
    periods = (2,) if subset else (1, 2)
    actual_pools = alive.simulation.feasibility_pools_by_period
    assert tuple(actual_pools) == periods
    read = alive.simulation.egm_policy_read
    assert isinstance(read, NNBEGMPolicyRead)
    assert read.finite_query_authority is expected
    assert not read.replay_policy_is_nested
    assert read.policy_applicable
    assert read.policy_required
    assert read.consumer_route == "nnbegm_finite"
    contexts = dict(
        _evaluator_identity_pairs(dict(expected.evaluator_identity)["lower_bound"])
    )
    early = dict(_evaluator_identity_pairs(contexts[1]))["floor_coefficient"]
    late = dict(_evaluator_identity_pairs(contexts[2]))["floor_coefficient"]
    if specialized:
        assert early is _early_coefficient
        assert late is _late_coefficient
        assert early is not late
    else:
        assert early is late is _ordinary_coefficient
    assert dict(_evaluator_identity_pairs(contexts[None]))["floor_coefficient"] is early
    representative = late if subset else early
    assert alive.simulation.functions["floor_coefficient"] is representative
    for period in periods:
        assert actual_pools[period][0]["floor_coefficient"] is (
            early if period == 1 else late
        )
    _assert_actual_period_floor_values(model=model)


def _assert_actual_period_floor_values(*, model: Model) -> None:
    """The actual phase pools evaluate each age's original physical floor."""
    alive = model._regimes["alive"]
    expected = alive.solution.query_authority
    assert expected is not None
    assert expected.bound.lower_target is not None
    actual_pools = alive.simulation.feasibility_pools_by_period
    for period, coefficient in ((1, 1.0 / 4.0), (2, 1.0 / 8.0)):
        functions = alive.solution.feasibility_pools_by_period[period][0]
        floor = concatenate_regime_function(
            functions=functions, target=expected.bound.lower_target
        )
        operands = {
            "illiquid": jnp.asarray(1.0),
            "age": jnp.asarray(float(period)),
            "period": jnp.int32(period),
        }
        accepted = inspect.signature(floor).parameters
        assert set(accepted) <= operands.keys()
        actual = floor(**{name: operands[name] for name in accepted})
        np.testing.assert_array_equal(actual, coefficient)
        if period in actual_pools:
            simulated_floor = concatenate_regime_function(
                functions=actual_pools[period][0], target=expected.bound.lower_target
            )
            actual_simulated = simulated_floor(
                **{
                    name: operands[name]
                    for name in inspect.signature(simulated_floor).parameters
                }
            )
            np.testing.assert_array_equal(actual_simulated, coefficient)
        # Keep the original savings offset grid and physical stock domain.
        assert bool(actual >= 0.0)
        assert bool(actual < original._SAVINGS.to_jax()[-1])
    np.testing.assert_array_equal(
        alive.solution.grids["liquid"].to_jax(), original._LIQUID.to_jax()
    )
    np.testing.assert_array_equal(
        alive.solution.grids["illiquid"].to_jax(), original._STOCK.to_jax()
    )


@pytest.mark.parametrize(
    ("specialized", "subset"),
    [(True, True), (True, False), (False, True)],
    ids=["specialized-subset", "specialized-same-periods", "ordinary-subset"],
)
def test_finite_policy_authenticates_actual_period_floors(
    *,
    specialized: bool,
    subset: bool,
) -> None:
    """Each simulated age authenticates its own canonical floor callback."""
    model = _model(specialized=specialized, subset=subset)
    _assert_actual_period_floor_authority(
        model=model, specialized=specialized, subset=subset
    )
    result = model.solve(params={"discount_factor": 0.0}, log_level="debug")
    # At liquid0/stock1 the exact keeper/O2 maxima are log2 then log(5/2).
    # Beta0 isolates the actual current floor while retaining genuine publication.
    np.testing.assert_array_almost_equal(
        [result.values[period]["alive"][0, 1] for period in (1, 2)],
        np.log((2.0, 2.5)),
        decimal=DECIMAL_PRECISION,
    )
