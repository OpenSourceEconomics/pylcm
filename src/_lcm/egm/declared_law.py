"""Compose a regime's own law of motion as a function of post-decision savings.

The endogenous grid method needs exactly two things from the budget constraint:
where a given level of savings lands next period, and how that landing point
moves when savings move. Both are properties of the law the modeller declared,
so a solver that rebuilds them from an assumed functional form solves a model
its user did not write.

This module reads them instead. The post-decision function is removed from the
DAG, which turns the savings node into an external input, so the regime's law
becomes a plain function of savings and params — differentiable, and composed
from the same callables the engine already holds.

The removal is not an optimization. Leaving the post-decision function in place
would let the DAG compute savings internally from the state and action leaves,
which runs and is silently wrong.
"""

import inspect
from collections.abc import Callable
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.regime_building.next_state import get_next_state_function_for_solution
from _lcm.typing import (
    EconFunctionsMapping,
    FloatND,
    RegimeName,
    StateName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import Float1D, ScalarFloat


def build_declared_liquid_law(
    *,
    transitions: TransitionFunctionsMapping,
    functions: EconFunctionsMapping,
    post_decision_name: str,
    target: RegimeName,
    target_state: StateName,
) -> Callable[..., tuple[Float1D, Float1D]]:
    """Build the declared law of motion as a function of savings.

    Args:
        transitions: The regime's transitions, keyed by target regime name.
        functions: The regime's auxiliary functions. The post-decision function
            is removed from the copy handed to the DAG.
        post_decision_name: Name of the function computing post-decision savings.
        target: The regime active next period, whose law is composed.
        target_state: The target's own name for its single continuous state.

    Returns:
        A callable taking `savings_grid` and the regime's flat params, returning
        the landing points on that grid and their derivative with respect to
        savings. The landing points are the full tabulation on the grid passed,
        so a consumer needing the corner takes the first entry, and one needing
        to invert the law can read the tabulation backwards.

    """
    law_name: TransitionFunctionName = f"next_{target_state}"
    functions_without_post = MappingProxyType(
        {name: func for name, func in functions.items() if name != post_decision_name}
    )
    next_state_func = get_next_state_function_for_solution(
        transitions=MappingProxyType({law_name: transitions[target][law_name]}),
        functions=functions_without_post,
    )
    # A caller holding the regime's whole flat param pool should not have to know
    # which subset this particular law reads, so the law selects its own.
    wanted = frozenset(inspect.signature(next_state_func).parameters) - {
        post_decision_name
    }

    def law(
        *, savings_grid: Float1D, **params: FloatND | float
    ) -> tuple[Float1D, Float1D]:
        # The DAG's own signature takes arrays, so a scalar param declared as a
        # plain Python float is lifted before it enters.
        array_params = {
            name: jnp.asarray(value) for name, value in params.items() if name in wanted
        }

        def landing(savings: ScalarFloat) -> FloatND:
            # The builder is annotated with the simulation shape, which nests by
            # target regime; built for the solution it returns one flat mapping
            # keyed by law name, so the single index yields the array.
            return next_state_func(**{post_decision_name: savings}, **array_params)[  # ty: ignore[invalid-return-type]
                law_name
            ]

        return jax.vmap(jax.value_and_grad(landing))(savings_grid)

    return law


def fail_if_declared_law_is_not_increasing(
    *,
    next_liquid: Float1D,
    regime_name: RegimeName,
    target: RegimeName,
) -> None:
    """Check the landing points ascend with savings.

    The endogenous grid is read back onto the regular grid by interpolation,
    whose abscissae must be sorted and distinct. A law that falls leaves them
    unsorted; a law that is flat over a band leaves them tied, so the savings
    level reaching a given landing point is not unique. Neither is detected by
    the interpolation — it returns quietly wrong numbers rather than failing.
    The conventional law is strictly increasing for any positive gross return;
    a declared law need not be, and a flat band is an ordinary model rather
    than a pathology (a means test clawing a transfer back one-for-one over a
    range of savings produces exactly one).
    """
    steps = jnp.diff(next_liquid)
    if bool(jnp.all(steps > 0.0)):
        return
    flat_only = bool(jnp.all(steps >= 0.0))
    diagnosis = (
        "is flat over a range of savings, so several savings levels reach the "
        "same landing point and the level behind a given one is not unique"
        if flat_only
        else "falls as savings rise over part of the grid"
    )
    msg = (
        f"The law of motion regime '{regime_name}' declares toward '{target}' "
        f"{diagnosis}. The endogenous grid method reads its solution back by "
        f"interpolation, which needs the landing points to ascend strictly, so "
        f"this law is outside what the method solves. Declare a law strictly "
        f"increasing in savings, or use GridSearch, which maximizes over the "
        f"action grid and needs no such ordering."
    )
    raise RegimeInitializationError(msg)
