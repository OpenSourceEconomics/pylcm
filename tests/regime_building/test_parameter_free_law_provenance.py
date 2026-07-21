"""Provenance must separate PARAMETER-FREE per-target laws too.

Two per-target `next_<state>` laws built with `dags.rename_arguments` off one base,
renaming to STATE names, have no free scalar parameter. `_rename_params_to_qnames`
then adds no engine wrapper, so a `__wrapped__`-peeling provenance rule would strip
the USER wrapper and merge two genuinely different laws -- binding one target's law
into the decision while realization uses the other. Explicit engine provenance
(`LAW_SOURCE_ATTR`, stamped only on engine-created wrappers) keeps them distinct:
the unstamped param-free cells are compared by their own object identity, which
already differs. The merged-decision hazard must therefore still be rejected.
"""

import jax.numpy as jnp
import pytest
from dags.signature import rename_arguments

from lcm import AgeGrid, DiscreteGrid, Model, Regime, categorical, fixed_transition
from lcm.typing import DiscreteAction, DiscreteState, FloatND, Period, ScalarInt


@categorical(ordered=True)
class Move:
    stay: ScalarInt
    switch: ScalarInt


@categorical(ordered=True)
class Stock:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=True)
class Sig:
    lo: ScalarInt
    hi: ScalarInt


@categorical(ordered=False)
class RegimeId:
    live: ScalarInt
    last: ScalarInt


def _base(signal: DiscreteState, move: DiscreteAction) -> DiscreteState:
    return jnp.where((move == Move.stay) & (signal == Sig.hi), Stock.good, Stock.bad)


def _utility(
    next_stock: DiscreteState, stock: DiscreteState, move: DiscreteAction
) -> FloatND:
    return 0.0 * next_stock + 0.0 * stock + 0.0 * move


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


def test_parameter_free_rename_wrappers_read_by_utility_still_rejected():
    """The renamed arguments are STATES, so the laws are parameter-free and receive
    no engine wrapper. They must still be recognized as two different per-target laws
    and rejected when read by utility.
    """
    law_live = rename_arguments(_base, mapper={"signal": "signal_a"})
    law_last = rename_arguments(_base, mapper={"signal": "signal_b"})
    grid = DiscreteGrid(Sig)

    live = Regime(
        transition=_next_regime,
        state_transitions={
            "stock": {"live": law_live, "last": law_last},
            "signal_a": fixed_transition("signal_a"),
            "signal_b": fixed_transition("signal_b"),
        },
        states={
            "stock": DiscreteGrid(Stock),
            "signal_a": grid,
            "signal_b": grid,
        },
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _utility},
    ).replace(active=lambda age: age < 2)

    with pytest.raises(ValueError, match="target-dependent deterministic state law"):
        Model(
            regimes={"live": live, "last": _last_regime()},
            ages=AgeGrid(exact_values=(0, 1, 2)),
            regime_id_class=RegimeId,
            description="parameter-free law provenance",
        )
