"""Model-stage validation of the N-NB-EGM nesting contract.

`NestedConsumptionSavingsRegime` establishes the two structural margins before
this module runs: both states and actions exist with the required continuous
kind, all liquid and outer role names are distinct, and `NNBEGM` has been bound
to those declarations.  This module therefore owns only the remaining dynamic
coupling check that cannot be discharged by the three margin-validation tiers.

The outer sweep binds one outer post-decision node and re-runs the full inner
NBEGM solve.  Inner functions may consequently read the outer margin as a
constant.  The reverse direction is not separable: if the outer state's law
reads the inner consumption action or liquid post-decision, the next outer stock
varies along the inner savings solve and the candidates are no longer
independent one-dimensional problems.  Chained sibling laws are followed so an
indirect coupling is rejected too.
"""

from collections.abc import Mapping
from typing import cast

from _lcm.egm.negm_validation import _ancestors_through_sibling_laws
from _lcm.egm.validation import (
    _resolve_solve_functions,
    _transition_variants,
    _without,
)
from _lcm.solution.nnbegm import NNBEGM, _BoundNNBEGM
from _lcm.typing import FunctionName, RegimeName
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import UserFunction


def validate_nnbegm_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Validate the remaining model contract for every NNBEGM regime."""
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
    """Reject an outer transition coupled to the inner solved margin."""
    solver = cast("_BoundNNBEGM", user_regime.solver)
    _fail_if_removed_outer_action_is_reachable(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=_resolve_solve_functions(user_regime=user_regime),
        solver=solver,
    )
    _fail_if_outer_law_reads_the_inner_margin(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=_resolve_solve_functions(user_regime=user_regime),
        solver=solver,
    )


def _fail_if_removed_outer_action_is_reachable(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: _BoundNNBEGM,
) -> None:
    """Reject solve functions that still require the enumerated outer action."""
    opaque_functions = _without(functions=functions, names={solver.outer_post_decision})
    sibling_laws = {
        f"next_{state_name}": value
        for state_name, value in user_regime.state_transitions.items()
    }
    targets: list[tuple[str, UserFunction, Mapping[str, object]]] = [
        (f"function {name!r}", func, sibling_laws)
        for name, func in opaque_functions.items()
    ]
    for state_name, law in user_regime.state_transitions.items():
        siblings = {
            name: value
            for name, value in sibling_laws.items()
            if name != f"next_{state_name}"
        }
        targets.extend(
            (f"transition of state {state_name!r}{label}", variant, siblings)
            for label, variant in _transition_variants(value=law)
        )
    for label, target, siblings in targets:
        ancestors = _ancestors_through_sibling_laws(
            functions=opaque_functions,
            target_func=target,
            sibling_laws=siblings,
        )
        if solver.outer_action not in ancestors:
            continue
        msg = (
            f"In regime {regime_name!r}, {label} reads the outer action "
            f"{solver.outer_action!r}. NNBEGM removes that action while its "
            f"inner solves bind {solver.outer_post_decision!r} to one finite "
            "outer candidate. Depend on the bound post-decision value instead."
        )
        raise ModelInitializationError(msg)


def _fail_if_outer_law_reads_the_inner_margin(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: _BoundNNBEGM,
) -> None:
    """Reject an outer-state law that reaches into the inner solved margin."""
    law = user_regime.state_transitions.get(solver.outer_state)
    if law is None:
        return

    inner_margin_names = {
        solver.inner.continuous_action,
        solver.inner.post_decision_function,
    }
    opaque_functions = _without(
        functions=functions,
        names={solver.outer_post_decision},
    )
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
