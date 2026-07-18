"""A target-dependent deterministic `next_<state>` read ONLY by a constraint.

The target-conflict guard rejects a `next_<state>` whose deterministic law differs
across target bundles when the within-period decision reads it -- otherwise the
merged law bound into the decision disagrees with the per-target simulate update.
The guard must inspect the RAW constraint graph: `_get_feasibility` concatenates
each constraint with the transition law, resolving `next_<state>` *into* the
compiled feasibility callable and erasing it from that callable's ancestry. A guard
that inspects the compiled `feasibility` therefore misses a conflict reached only
through a constraint (never through utility). This pins the raw-graph check.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, Model, Regime, categorical
from lcm.typing import DiscreteAction, DiscreteState, FloatND, Period, ScalarInt


@categorical(ordered=True)
class Move:
    stay: ScalarInt
    switch: ScalarInt


@categorical(ordered=True)
class Stock:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    live: ScalarInt
    last: ScalarInt


def _utility(stock: DiscreteState, move: DiscreteAction) -> FloatND:
    # Reads the CURRENT stock but not next_stock: the conflict is constraint-only.
    return 0.0 * stock + 0.0 * move


def _grow(move: DiscreteAction) -> DiscreteState:
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _shrink(move: DiscreteAction) -> DiscreteState:
    return jnp.where(move == Move.stay, Stock.bad, Stock.good)


def _capacity(next_stock: DiscreteState) -> FloatND:
    return next_stock == Stock.good


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


def _last_regime() -> Regime:
    return Regime(
        transition=None,
        state_transitions={},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _utility},
    ).replace(active=lambda age: age >= 2)


def test_target_dependent_transition_read_only_by_constraint_is_rejected():
    """`next_stock` differs across targets (`_grow` vs `_shrink`) and is read only by
    the `capacity` constraint. The decision would bind one target's law while the
    simulate state-update uses the right one -- a silent disagreement. Reject it.
    """
    live = Regime(
        transition=_next_regime,
        state_transitions={"stock": {"live": _grow, "last": _shrink}},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _utility},
        constraints={"capacity": _capacity},
    ).replace(active=lambda age: age < 2)

    with pytest.raises(ValueError, match="target-dependent deterministic state law"):
        Model(
            regimes={"live": live, "last": _last_regime()},
            ages=AgeGrid(exact_values=(0, 1, 2)),
            regime_id_class=RegimeId,
            description="constraint-only transition conflict",
        )
