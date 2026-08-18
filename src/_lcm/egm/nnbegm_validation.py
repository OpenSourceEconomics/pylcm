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
  that one. The post-decision half is guarded at `NNBEGM` construction,
- the outer state's law of motion does not read the inner margin: the solver
  evaluates that law to find what the next period carries, so a law reading the
  inner consumption action or post-decision would make the carried stock vary
  along the inner savings axis. Chained laws are followed, since the coupling
  can sit one sibling away.

**Which couplings are in scope.** NEGM rejects a model whose outer margin enters
the inner Euler-state transition or multiplies consumption in utility, because
its outer max runs over a *frozen* inner inversion lifted onto a common
cash-on-hand axis by a credited-cost translation. NNBEGM does not lift: the
outer sweep binds `outer_post_decision` as a flat param and re-runs the entire
inner NB-EGM solve once per outer-grid node, so every inner function — the
budget, the Euler-state law, utility — sees the node as a constant and the inner
inversion is exact conditional on it. A Cobb-Douglas composite flow
`q = c^phi * s'^(1-phi)` and an Euler law that reads the durable stock are both
in scope.

The re-solve absorbs the inner-reads-outer direction only. The reverse — the
outer state's law reading a value the inner solve *produces* — is not absorbed
by anything, which is why it is checked above. The other approximation NNBEGM
makes is in the outer candidate set, a property of the declared `outer_search`
rather than of the model's function structure: `FiniteOuterGrid` fixes it to the
grid plus the keeper, while `AdaptiveOuterMesh` refines it per node. For
the same reason NEGM's outer-cost contract and its durable-last carry layout do
not carry over: there is no credited-cost lift, and the outer envelope is a
pointwise fold of carry rows that already share the liquid grid.
"""

from collections.abc import Mapping
from typing import cast

from _lcm.egm.negm_validation import _ancestors_through_sibling_laws
from _lcm.egm.validation import (
    _continuous_non_process_names,
    _resolve_solve_functions,
    _solve_grids,
    _transition_variants,
    _without,
)
from _lcm.solution.nnbegm import NNBEGM, get_nnbegm_inner_spec
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
            validate_nnbegm_regime(
                regime_name=regime_name,
                user_regime=user_regime,
            )


def validate_nnbegm_regime(
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
    _fail_if_outer_law_reads_the_inner_margin(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )


def _fail_if_outer_law_reads_the_inner_margin(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: NNBEGM,
) -> None:
    """Reject an outer-state law of motion that reaches into the inner margin.

    The solver evaluates the durable's declared law as written, so what that law
    reads decides whether the outer margin stays a plain search. Reading the
    outer post-decision, the durable state, other states, or params is the
    ordinary case — a depreciating `next_z = (1 - delta) s'` is exactly that.
    Reading the inner consumption action or the inner post-decision is not: the
    stock carried forward would then vary along the inner savings axis, so
    conditioning on an outer node would no longer leave a one-dimensional
    problem and the outer max would not range over independent solves.

    This is the one coupling direction the re-solve-per-node design does not
    absorb. An inner function reading the outer margin is in scope, because the
    node is a bound constant when the inner solve runs; the outer law reading an
    inner value is not, because that value is what the inner solve produces.

    The outer post-decision is made opaque first, so a law reading the chosen
    stock is not charged with reading the outer action it is computed from.

    Args:
        regime_name: Name of the regime being validated.
        user_regime: The regime whose outer-state law is inspected.
        functions: Mapping of function names to the regime's solve functions.
        solver: The regime's `NNBEGM` solver config.

    Raises:
        ModelInitializationError: If the law reads the inner consumption action
            or the inner post-decision.

    """
    law = user_regime.state_transitions.get(solver.outer_state)
    if law is None:
        return
    inner_spec = get_nnbegm_inner_spec(inner=solver.inner)
    inner_margin_names = {
        _continuous_action_names(user_regime=user_regime)[0],
        inner_spec.post_decision_function,
    }
    opaque_functions = _without(functions=functions, names={solver.outer_post_decision})
    sibling_laws = {
        f"next_{state_name}": value
        for state_name, value in user_regime.state_transitions.items()
        if state_name != solver.outer_state
    }
    for label, transition_func in _transition_variants(value=law):
        ancestors = _ancestors_through_sibling_laws(
            functions=opaque_functions,
            target_func=transition_func,
            sibling_laws=sibling_laws,
        )
        coupled = ancestors & inner_margin_names
        if coupled:
            msg = (
                f"In regime '{regime_name}', the transition of the outer state "
                f"'{solver.outer_state}'{label} reads {sorted(coupled)!r}, which "
                "belongs to the inner margin. NNBEGM evaluates that law to find "
                "what the next period carries, so the stock carried forward "
                "would vary along the inner savings axis and conditioning on an "
                "outer node would no longer leave a one-dimensional problem. "
                f"Let the law read the outer post-decision "
                f"'{solver.outer_post_decision}', states, or params; if the two "
                "margins genuinely interact, the model needs a 2-D EGM "
                "foundation (G2EGM / multidim-RFC), not a nested outer search."
            )
            raise ModelInitializationError(msg)


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
    if solver.outer_post_decision not in functions:
        msg = (
            f"NNBEGM.outer_post_decision '{solver.outer_post_decision}' is not "
            f"a declared function of regime '{regime_name}'. The outer "
            "post-decision names this period's chosen level of the durable "
            f"state '{solver.outer_state}' — an ordinary function of this "
            "period's states and actions that the inner budget, the "
            "child-state index, and the durable's own law of motion all read. "
            "Declare it, or use `NBEGM` for a pure 1-D consumption-savings "
            "regime."
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
