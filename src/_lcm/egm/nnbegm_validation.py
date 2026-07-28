"""Build-time validation of the NNBEGM nesting contract.

A regime configured with `solver=NNBEGM(inner=NBEGM(...), ...)` must declare a
real outer margin, distinct from the margin the inner NB-EGM consumes. Every
violation raises `ModelInitializationError` at `Model` construction, naming the
offending feature **and** the correct alternative solver, so no rejection path
silently degrades to a different algorithm. The inner 1-D NB-EGM contract is
checked separately, against the outer-margin-bound inner view the kernel builds;
this module checks only the *outer*/nesting contract:

- the outer margin exists: `outer_action` is a continuous action of the regime
  and `outer_post_decision` is a function the regime declares (or a state's
  `next_<state>` law) — a regime with no outer margin is a pure 1-D
  consumption-savings problem and must use `NBEGM`,
- the two margins are distinct: the inner NB-EGM claims the regime's first
  continuous action as its consumption margin, so the outer action must not be
  that one. The post-decision half is guarded at `NNBEGM` construction.

**Why there is no coupling rule here.** NEGM rejects a model whose outer margin
enters the inner Euler-state transition or multiplies consumption in utility,
because its outer max runs over a *frozen* inner inversion lifted onto a common
cash-on-hand axis by a credited-cost translation. NNBEGM does not lift: the
outer sweep binds `outer_post_decision` as a flat param and re-runs the entire
inner NB-EGM solve once per outer-grid node, so every inner function — the
budget, the Euler-state law, utility — sees the node as a constant and the inner
inversion is exact conditional on it. A Cobb-Douglas composite flow
`q = c^phi * s'^(1-phi)` and an Euler law that reads the durable stock are both
in scope. The approximation NNBEGM does make is the finite outer candidate set
(the grid plus the keeper), which is a property of the declared `outer_grid`
rather than of the model's function structure. For the same reason NEGM's
outer-cost contract and its durable-last carry layout do not carry over: there
is no credited-cost lift, and the outer envelope is a pointwise fold of carry
rows that already share the liquid grid.
"""

from collections.abc import Mapping
from typing import cast

from _lcm.egm.validation import (
    _continuous_non_process_names,
    _resolve_solve_functions,
    _solve_grids,
)
from _lcm.solution.nnbegm import NNBEGM
from _lcm.typing import FunctionName, RegimeName
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import ActionName, UserFunction


def validate_nnbegm_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Validate the NNBEGM contract for every regime with an `NNBEGM` solver.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.

    Raises:
        ModelInitializationError: If any regime with `solver=NNBEGM(...)`
            violates the NNBEGM model contract.

    """
    for regime_name, user_regime in user_regimes.items():
        if isinstance(user_regime.solver, NNBEGM):
            _validate_nnbegm_regime(
                regime_name=regime_name,
                user_regime=user_regime,
            )


def _validate_nnbegm_regime(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
) -> None:
    """Run all NNBEGM contract checks for a single regime, in order."""
    solver = cast("NNBEGM", user_regime.solver)
    functions = _resolve_solve_functions(user_regime=user_regime)

    _fail_if_outer_margin_absent(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )
    _fail_if_outer_action_is_the_inner_consumption_margin(
        regime_name=regime_name,
        user_regime=user_regime,
        solver=solver,
    )


def _fail_if_outer_margin_absent(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: NNBEGM,
) -> None:
    """The outer durable margin must be a real action and post-decision function.

    A regime that declares no outer continuous margin (the outer action is not
    among its continuous actions, or the outer post-decision is neither a
    declared function nor a state's law of motion) is a pure 1-D
    consumption-savings problem; NNBEGM would silently run as plain NB-EGM, so
    it is rejected with the `NBEGM` pointer.
    """
    continuous_actions = _continuous_action_names(user_regime=user_regime)
    if solver.outer_action not in continuous_actions:
        msg = (
            f"NNBEGM.outer_action '{solver.outer_action}' is not a continuous "
            f"action of regime '{regime_name}'. NNBEGM nests an outer "
            "continuous margin; this regime declares none (continuous actions: "
            f"{list(continuous_actions)}) — use `NBEGM` for a pure 1-D "
            "consumption-savings regime."
        )
        raise ModelInitializationError(msg)
    transition_names = {f"next_{name}" for name in user_regime.states}
    if (
        solver.outer_post_decision not in functions
        and solver.outer_post_decision not in transition_names
    ):
        msg = (
            f"NNBEGM.outer_post_decision '{solver.outer_post_decision}' is "
            f"neither a declared function of regime '{regime_name}' nor the "
            "transition of one of its states. The outer post-decision (the "
            "next-period durable stock) must be a regime function or the "
            "durable state's `next_<state>` law that the inner budget and the "
            "child-state index read; declare it, or use `NBEGM` for a pure 1-D "
            "consumption-savings regime."
        )
        raise ModelInitializationError(msg)


def _fail_if_outer_action_is_the_inner_consumption_margin(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    solver: NNBEGM,
) -> None:
    """The outer action must not be the action the inner NB-EGM consumes.

    `NBEGM` takes the regime's first continuous action as its consumption
    margin. With the outer action in that slot the inner solve would invert the
    Euler equation on the durable move and the outer sweep would search over
    consumption — the two margins swapped, with no error anywhere downstream.
    """
    continuous_actions = _continuous_action_names(user_regime=user_regime)
    inner_consumption_action = continuous_actions[0]
    if solver.outer_action == inner_consumption_action:
        msg = (
            f"NNBEGM.outer_action '{solver.outer_action}' of regime "
            f"'{regime_name}' coincides with the inner NB-EGM consumption "
            "action. The inner solver claims the regime's first continuous "
            f"action ('{inner_consumption_action}') as its consumption margin, "
            "so the outer durable/illiquid margin must be a different action — "
            "declare the consumption action first in `actions`."
        )
        raise ModelInitializationError(msg)


def _continuous_action_names(*, user_regime: UserRegime) -> tuple[ActionName, ...]:
    """Return the regime's continuous action names in declaration order.

    Declaration order is the order the canonical state-action space presents
    them in, so the first entry is the one the inner NB-EGM claims as its
    consumption margin.
    """
    return tuple(
        _continuous_non_process_names(grids=_solve_grids(slot=user_regime.actions))
    )
