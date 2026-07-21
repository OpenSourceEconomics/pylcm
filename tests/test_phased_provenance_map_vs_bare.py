"""Phase-local provenance: a within-period `next_<state>` read merges the COARSE side.

A `Phased` state transition is coarse (a bare law) in one phase and per-target in the
other (map-vs-bare), or per-target in both with asymmetric params. A within-period read
of `next_<state>` must be judged PER PHASE:

- the phase where the law is coarse (one bare law broadcast to every target) is
  unambiguous → the cells MERGE → the read is allowed;
- the phase where the law is a genuine per-target dict is target-dependent → the cells
  CONFLICT → the read is rejected.

The judgment is keyed off each phase's OWN declaration shape
(`processing._phase_coarse_state_law_names`), NOT the phase-union params template — so a
param-free or coarse side no longer inherits a false conflict from the OTHER phase's
params (round-8 F1). A genuinely per-target parameterized law reused across targets
still conflicts (round-6).

These are the in-engine regressions for round-8 F1; the phase is isolated with a
`Phased` utility so the other phase's (legitimate) per-target read does not mask the
result.
"""

from typing import Any, cast

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, Model, Phased, categorical
from lcm.regime import Regime
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


def _belief_free(stock: DiscreteState, move: DiscreteAction) -> DiscreteState:
    return jnp.where(move == Move.stay, stock, Stock.good)


def _belief_param(
    stock: DiscreteState, move: DiscreteAction, belief_bias: DiscreteState
) -> DiscreteState:
    return jnp.where(move == Move.stay, stock, belief_bias)


def _truth_param(
    stock: DiscreteState, move: DiscreteAction, truth_bias: DiscreteState
) -> DiscreteState:
    return jnp.where(move == Move.stay, stock, truth_bias)


def _truth_free(stock: DiscreteState, move: DiscreteAction) -> DiscreteState:
    return jnp.where(move == Move.stay, stock, Stock.bad)


def _util_reads(next_stock: DiscreteState, move: DiscreteAction) -> FloatND:
    return 0.0 * next_stock + 0.0 * move


def _util_plain(stock: DiscreteState, move: DiscreteAction) -> FloatND:
    return 0.0 * stock + 0.0 * move


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


def _make_model(*, stock_law: object, utility: object) -> Model:
    live = Regime(
        transition=_next_regime,
        state_transitions={"stock": cast("Any", stock_law)},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": cast("Any", utility)},
    ).replace(active=lambda age: age < 2)
    return Model(
        regimes={"live": live, "last": _last_regime()},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
    )


def test_both_per_target_param_free_solve_merges() -> None:
    """Round-8 F1: a param-free solve law reused per target MERGES (no false conflict).

    The simulate side is a parameterized per-target law; reading `next_stock` only in
    the solve phase isolates the solve merge from the legitimate simulate conflict.
    """
    _make_model(
        stock_law=Phased(
            solve={"live": _belief_free, "last": _belief_free},
            simulate={"live": _truth_param, "last": _truth_param},
        ),
        utility=Phased(solve=_util_reads, simulate=_util_plain),
    )


def test_map_vs_bare_parameterized_coarse_solve_merges() -> None:
    """A PARAMETERIZED coarse solve law (map-vs-bare) merges via its bare location.

    Its params bind per target (phase-union template), but the provenance stamp is at
    the shared bare `next_stock`, so a solve-phase read does not falsely conflict.
    """
    _make_model(
        stock_law=Phased(
            solve=_belief_param,
            simulate={"live": _truth_param, "last": _truth_param},
        ),
        utility=Phased(solve=_util_reads, simulate=_util_plain),
    )


def test_mirror_map_vs_bare_coarse_simulate_merges() -> None:
    """The mirror: a coarse simulate law merges for a simulate-phase read."""
    _make_model(
        stock_law=Phased(
            solve={"live": _belief_param, "last": _belief_param},
            simulate=_truth_free,
        ),
        utility=Phased(solve=_util_plain, simulate=_util_reads),
    )


def test_per_target_parameterized_reused_still_conflicts() -> None:
    """Round-6 preserved: a per-target parameterized law reused per target CONFLICTS.

    Reusing one parameterized callable across per-target cells binds a distinct param
    per target, so a within-period read is genuinely target-dependent and rejected.
    """
    with pytest.raises(ValueError, match="target-dependent"):
        _make_model(
            stock_law=Phased(
                solve={"live": _belief_param, "last": _belief_param},
                simulate={"live": _truth_param, "last": _truth_param},
            ),
            utility=Phased(solve=_util_reads, simulate=_util_plain),
        )
