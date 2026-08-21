"""The conventional accounting forms of a one-asset consumption-saving regime.

`cash_on_hand_with_subsidy` is a declaration rather than a description of one.
NB-EGM's case-piece kernels do not call a regime's budget node: they form
`liquid + subsidy` themselves, and the only way to establish that an arbitrary
callable computes the same thing is to hand it the very function the kernels
implement — sampling at finitely many points cannot do it, because a global
rescaling agrees at every sample and still moves every state's value. So a
regime on that route names this one, the solver accepts it by identity, and the
question never arises. A budget those kernels cannot form belongs on a
`lcm.piecewise_affine` schedule with a `post_decision_function`, which composes
it from the DAG, or under `GridSearch`.

The two liquid laws carry no such restriction: NB-EGM reads whatever law a
regime declares, and these are exported for the ordinary case rather than
required. `liquid_law_from_savings` states the law through a post-decision
savings node, which is what NB-EGM inverts the Euler equation against;
`liquid_law_from_resources` states the same arithmetic as a displacement of
cash-on-hand, for a `GridSearch` regime that declares no savings node.

All three are ordinary functions and stay executable, so a model declaring them
solves identically under either solver and the agreement tests keep their
meaning.
"""

from types import FunctionType

from lcm.typing import ContinuousAction, ContinuousState, FloatND

# Marker naming a declaration, readable through the wrappers the DAG applies.
FIXED_FORM_ATTRIBUTE = "lcm_fixed_form"


def _declared(func: FunctionType) -> FunctionType:
    """Stamp a form with its own name so the solver can recognize it."""
    setattr(func, FIXED_FORM_ATTRIBUTE, func.__name__)
    return func


@_declared
def liquid_law_from_savings(
    savings: FloatND, return_liquid: FloatND, income: FloatND
) -> ContinuousState:
    """Return `(1 + return_liquid) * savings + income`.

    The savings form of the liquid law, for a regime declaring a
    post-decision savings node.

    Args:
        savings: End-of-period liquid savings.
        return_liquid: Net return on liquid savings.
        income: Income received at the start of next period.

    Returns:
        Next period's liquid state.

    """
    return (1.0 + return_liquid) * savings + income


@_declared
def liquid_law_from_resources(
    resources: FloatND,
    consumption: ContinuousAction,
    return_liquid: FloatND,
    income: FloatND,
) -> ContinuousState:
    """Return `(1 + return_liquid) * (resources - consumption) + income`.

    The displacement form of the same law, for a regime that reaches savings
    through its budget and consumption rather than declaring a savings node.

    Args:
        resources: Cash on hand this period.
        consumption: Consumption chosen this period.
        return_liquid: Net return on liquid savings.
        income: Income received at the start of next period.

    Returns:
        Next period's liquid state.

    """
    return (1.0 + return_liquid) * (resources - consumption) + income


@_declared
def cash_on_hand_with_subsidy(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Return `liquid + subsidy`.

    The budget node the case-piece kernels apply: the liquid state plus whichever
    branch of the case split the state falls in.

    Args:
        liquid: Liquid wealth entering the period.
        subsidy: The case-contingent transfer into market resources.

    Returns:
        Cash on hand available for consumption and savings.

    """
    return liquid + subsidy
