"""A per-target state law that REUSES one callable is still a target-dependent conflict.

Provenance must identify the declaration cell, not the raw callable object. A
per-target dict binds a TARGET-QUALIFIED parameter branch per cell, so two cells that
reuse the SAME parameterized callable can still bind different per-target parameter
values -- they are different laws. If within-period utility or feasibility reads
`next_<state>`, the engine merges the target bundles into one node, so it must reject
this as a conflict. A COARSE (bare) law, by contrast, binds ONE shared parameter branch
across all its target cells and must remain mergeable even though it too is realized
into per-target renamed wrappers.

Raw-callable-identity provenance told these two cases apart only by whether the user
happened to reuse the object; the fix keys provenance on the template location of the
params (`names_key`: bare for coarse, target-qualified for per-target), so callable
reuse no longer hides the conflict.
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


def _next_stock(
    stock: DiscreteState, move: DiscreteAction, preferred_move: int
) -> DiscreteState:
    return jnp.where(
        (move == preferred_move) & (stock == Stock.bad), Stock.good, Stock.bad
    )


def _utility(next_stock: DiscreteState, move: DiscreteAction) -> FloatND:
    return jnp.where(next_stock == Stock.good, 1.0, 0.0) + 0.0 * move


def _terminal_utility(stock: DiscreteState, move: DiscreteAction) -> FloatND:
    return 0.0 * stock + 0.0 * move


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


def _last_regime() -> Regime:
    return Regime(
        transition=None,
        state_transitions={},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _terminal_utility},
    ).replace(active=lambda age: age >= 2)


def _model(live: Regime) -> None:
    Model(
        regimes={"live": live, "last": _last_regime()},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="per-target reused callable",
    )


def test_per_target_dict_reusing_one_parameterized_callable_is_rejected():
    """Same callable in two per-target cells, read by utility -> conflict rejected."""
    live = Regime(
        transition=_next_regime,
        # Per-target dict REUSING one callable: cells bind different per-target
        # parameter branches, so they are different laws despite the shared object.
        state_transitions={"stock": {"live": _next_stock, "last": _next_stock}},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _utility},
    ).replace(active=lambda age: age < 2)
    with pytest.raises(ValueError, match="target-dependent deterministic state law"):
        _model(live)


def test_coarse_parameterized_law_read_by_utility_is_accepted():
    """The same callable as a BARE coarse law shares one parameter branch -> merges."""
    live = Regime(
        transition=_next_regime,
        state_transitions={"stock": _next_stock},  # coarse: one shared branch
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _utility},
    ).replace(active=lambda age: age < 2)
    # Must not raise: a coarse law is target-independent even when read by utility.
    _model(live)
