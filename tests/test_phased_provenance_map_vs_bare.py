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

Round-10 F1 narrows one map-vs-bare shape: a PARAMETERIZED bare (coarse) side opposite a
per-target dict is rejected at construction — the phase-union template would replicate
its parameter per target with only the first leaf live (a dead-leaf trap). A PARAMETER-
FREE coarse side (nothing to replicate) and a parameterized coarse law spelled coarse in
BOTH phases (one shared leaf) both remain valid. See
`user_regime_validation._phased_per_target_shape_mismatch`.

These are the in-engine regressions for round-8 F1 and round-10 F1; the phase is
isolated with a `Phased` utility so the other phase's (legitimate) per-target read does
not mask the result.
"""

from typing import Any, cast

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, Model, Phased, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import DiscreteAction, DiscreteState, FloatND, Period, ScalarInt

_PARAM_COARSE_REJECT = "bare .coarse. law with a free parameter"


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


def _helper(stock: DiscreteState) -> DiscreteState:
    """A named regime function (a DAG node), NOT a free parameter."""
    return stock


def _belief_reads_helper(
    stock: DiscreteState, move: DiscreteAction, helper: DiscreteState
) -> DiscreteState:
    """A bare (coarse) solve law whose only non-state/action arg is the fn `helper`."""
    return jnp.where(move == Move.stay, stock, helper)


def _make_model(
    *,
    stock_law: object,
    utility: object,
    extra_functions: dict[str, object] | None = None,
) -> Model:
    functions: dict[str, Any] = {"utility": cast("Any", utility)}
    if extra_functions is not None:
        functions.update(cast("dict[str, Any]", extra_functions))
    live = Regime(
        transition=_next_regime,
        state_transitions={"stock": cast("Any", stock_law)},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions=functions,
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


def test_map_vs_bare_param_free_coarse_solve_merges() -> None:
    """A PARAMETER-FREE coarse solve law (map-vs-bare) merges via object identity.

    With no parameter there is nothing to replicate: the bare law broadcast to every
    target is identity-identical, so a solve-phase read of `next_stock` does not
    falsely conflict. The simulate side stays a parameterized per-target dict.
    """
    _make_model(
        stock_law=Phased(
            solve=_belief_free,
            simulate={"live": _truth_param, "last": _truth_param},
        ),
        utility=Phased(solve=_util_reads, simulate=_util_plain),
    )


def test_map_vs_bare_parameterized_coarse_solve_rejected() -> None:
    """Round-10 F1: a PARAMETERIZED coarse solve law (map-vs-bare) is rejected.

    The phase-union template would replicate `belief_bias` into one leaf per target
    while the coarse side binds a single law, so all but the first leaf are dead. The
    validator rejects the shape at construction and names the remedies.
    """
    with pytest.raises(RegimeInitializationError, match=_PARAM_COARSE_REJECT):
        _make_model(
            stock_law=Phased(
                solve=_belief_param,
                simulate={"live": _truth_param, "last": _truth_param},
            ),
            utility=Phased(solve=_util_reads, simulate=_util_plain),
        )


def test_mirror_map_vs_bare_parameterized_coarse_simulate_rejected() -> None:
    """The mirror: a PARAMETERIZED coarse simulate law (map-vs-bare) is rejected."""
    with pytest.raises(RegimeInitializationError, match=_PARAM_COARSE_REJECT):
        _make_model(
            stock_law=Phased(
                solve={"live": _belief_param, "last": _belief_param},
                simulate=_truth_param,
            ),
            utility=Phased(solve=_util_plain, simulate=_util_reads),
        )


def test_mirror_map_vs_bare_param_free_coarse_simulate_merges() -> None:
    """The mirror: a PARAMETER-FREE coarse simulate law merges for a simulate read."""
    _make_model(
        stock_law=Phased(
            solve={"live": _belief_param, "last": _belief_param},
            simulate=_truth_free,
        ),
        utility=Phased(solve=_util_plain, simulate=_util_reads),
    )


def test_parameterized_coarse_in_both_phases_is_accepted() -> None:
    """A parameterized coarse law spelled coarse in BOTH phases is one shared leaf.

    This is the supported spelling the map-vs-bare rejection points a parameterized
    coarse law toward; a within-period read merges in both phases.
    """
    _make_model(
        stock_law=Phased(solve=_belief_param, simulate=_truth_param),
        utility=Phased(solve=_util_reads, simulate=_util_reads),
    )


def test_map_vs_bare_bare_law_reading_named_helper_is_accepted() -> None:
    """Round-11 F2: a param-FREE bare (coarse) law that reads a named regime FUNCTION
    (a DAG node) is accepted, not rejected as parameterized-coarse.

    `_belief_reads_helper` reads `helper`, a phase-invariant regime function, and has
    NO free parameter. `_law_has_free_parameter` must count `helper` among the regime's
    own names (states | actions | functions | next_*), so there is nothing to replicate
    per target and the map-vs-bare shape merges. Before the fix, `functions` was omitted
    from that name set, so `helper` was misclassified as a free parameter and the build
    was falsely rejected with `_PARAM_COARSE_REJECT`.
    """
    _make_model(
        stock_law=Phased(
            solve=_belief_reads_helper,
            simulate={"live": _truth_param, "last": _truth_param},
        ),
        utility=Phased(solve=_util_reads, simulate=_util_plain),
        extra_functions={"helper": _helper},
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
