"""The fixed economic forms NB-EGM's single-liquid kernels solve.

Those kernels do not call a regime's declared budget: they apply one hardcoded
affine law and one hardcoded cash-on-hand, reading the coefficients under fixed
qualified parameter names. A regime taking that route therefore has to declare
the same forms, and the only way to establish that is to hand it the very
functions the kernels implement — sampling an arbitrary callable at finitely many
points cannot do it, because a global rescaling agrees at every sample and still
moves every state's value.

So these are the declarations, not descriptions of them. A model on the
single-liquid route names one of the laws below; the solver accepts it by
identity, and the question of whether the declared budget equals the solved one
never arises. They are ordinary functions and stay executable, so the same model
solves identically under `GridSearch` and the agreement tests keep their meaning.

A regime whose budget these forms cannot express does not belong on this route.
Declare a `lcm.piecewise_affine` schedule with a `post_decision_function`, which
composes the budget from the DAG, or solve the regime with `GridSearch`.
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
