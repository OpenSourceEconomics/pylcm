"""Test that validation vmaps over action combos like solve/simulate do.

When a model function wraps inputs into arrays and performs operations requiring
consistent first-dimension lengths (e.g. segment_sum), passing batched action arrays
(shape-(N,)) alongside MappingLeaf params (shape-(1,)) causes a shape mismatch.
Solve and simulate avoid this by vmapping over individual state/action combos.
Validation must do the same.
"""

import jax
import jax.numpy as jnp

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.params import MappingLeaf, as_leaf
from lcm.typing import ContinuousAction, ContinuousState, FloatND


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def _aggregate_with_ids(
    consumption: ContinuousAction,
    invariant_array: MappingLeaf,
) -> FloatND:
    """Aggregate consumption using segment ids from a MappingLeaf.

    This mimics the ttsim pattern: wraps inputs into arrays and does operations
    that require consistent first-dimension lengths.  Works with scalar inputs
    (as in solve/simulate) but breaks if consumption is shape-(N,) while
    invariant_array contains shape-(1,) arrays.
    """
    data = jnp.array([consumption])  # shape-(1,) when scalar
    ids = invariant_array.data["ids"]  # shape-(1,) always
    result = jax.ops.segment_sum(data, ids, num_segments=1)
    return result.squeeze()


def _utility(
    consumption: ContinuousAction,
    wealth: ContinuousState,
    invariant_array: MappingLeaf,
) -> FloatND:
    aggregated = _aggregate_with_ids(consumption, invariant_array)
    return jnp.log(consumption + 1) + 0.01 * wealth + 0.0 * aggregated


def _next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
) -> ContinuousState:
    return wealth - consumption


def _borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> FloatND:
    return consumption <= wealth


def _next_regime(period: int) -> FloatND:
    return jnp.where(period >= 2, RegimeId.dead, RegimeId.alive)


def test_validation_vmaps_over_action_combos():
    """solve_and_simulate succeeds when a MappingLeaf param requires scalar actions."""
    n_periods = 4

    alive = Regime(
        functions={"utility": _utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age, n=n_periods: age < n - 1,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age, n=n_periods: age >= n - 1,
    )

    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=n_periods - 1, step="Y"),
        regime_id_class=RegimeId,
    )

    params = {
        "discount_factor": 0.95,
        "alive": {
            "utility": {
                "invariant_array": as_leaf({"ids": jnp.array([0])}),
            },
        },
    }

    result = model.solve_and_simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([5.0, 7.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime": jnp.array([RegimeId.alive] * 2),
        },
        log_level="off",
    )

    df = result.to_dataframe()
    assert len(df) > 0
    assert "consumption" in df.columns
