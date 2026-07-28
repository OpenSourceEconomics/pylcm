"""The unproduced-`next_<state>` read guard fires in a COLLECTIVE terminal regime too.

`_fail_if_unproduced_next_state_is_read` converts a cryptic late missing-argument
failure into an early one naming the offending state. Its test is
`read_names & next_state_names`, so a builder that never receives `next_state_names`
intersects against the empty set and the guard silently cannot fire.

`get_Q_and_F_terminal_collective` used to be exactly that builder: it had no
`next_state_names` parameter, while its singleton twin `get_Q_and_F_terminal` did and
both call sites in `processing.py` threaded it. A collective terminal regime whose
per-stakeholder `utility_<s>` read a `next_<state>` therefore failed later, inside
`func_with_only_kwargs`, with `Expected arguments: [...], missing: {'next_stock'}`.

This is the same shape as the branch-2 audit's F1 (the collective twin dropping
arguments the singleton threads), so the test asserts the property for BOTH builders
in one parametrization: whichever twin is asked, the guard must name the state.

Reachability is not incidental -- `_declared_next_state_names` lists `next_<name>` for
every gridded state, and the terminal builders pass no `deterministic_transitions`, so
a terminal read of `next_<state>` is unproduced by construction.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, Model, Regime, categorical
from lcm.typing import DiscreteAction, FloatND, ScalarInt


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


def _reads_next_stock(
    next_stock: FloatND, move: DiscreteAction, stock: FloatND
) -> FloatND:
    """Terminal utility reading a `next_<state>` no terminal flow produces.

    `move` and `stock` enter with zero weight only so that every declared state and
    action is used somewhere: otherwise `validate_model_inputs` rejects the model first
    and the test would assert on the wrong guard.
    """
    return 1.0 * next_stock + 0.0 * move + 0.0 * stock


def _plain(stock: FloatND, move: DiscreteAction) -> FloatND:
    return 0.0 * stock + 0.0 * move


def _next_stock(move: DiscreteAction) -> FloatND:
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _next_regime() -> ScalarInt:
    return RegimeId.last


@pytest.mark.parametrize("collective", [False, True], ids=["singleton", "collective"])
def test_terminal_unproduced_next_state_read_is_named_early(collective):
    extra = {"stakeholders": ("f", "m")} if collective else {}
    live_functions = (
        {"utility_f": _plain, "utility_m": _plain}
        if collective
        else {"utility": _plain}
    )
    terminal_functions = (
        {"utility_f": _reads_next_stock, "utility_m": _reads_next_stock}
        if collective
        else {"utility": _reads_next_stock}
    )
    live = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        state_transitions={"stock": _next_stock},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions=live_functions,
        **extra,
    )
    last = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions=terminal_functions,
        **extra,
    )
    # The guard runs while `Model` builds the solution phase, so it is `Model(...)`
    # that must raise -- earlier than `solve`, which is the whole point of it.
    with pytest.raises(
        ValueError, match=r"reads the next value of state\(s\).*next_stock"
    ):
        Model(
            regimes={"live": live, "last": last},
            ages=AgeGrid(exact_values=(0, 1)),
            regime_id_class=RegimeId,
            description="terminal unproduced-next-state read guard",
        )
