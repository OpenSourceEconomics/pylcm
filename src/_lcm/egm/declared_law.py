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
) -> Callable[..., tuple[Float1D, Float1D, ScalarFloat]]:
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
        the landing points on that grid, their derivative with respect to
        savings, and the landing point at zero savings.

    """
    law_name: TransitionFunctionName = f"next_{target_state}"
    functions_without_post = MappingProxyType(
        {name: func for name, func in functions.items() if name != post_decision_name}
    )
    next_state_func = get_next_state_function_for_solution(
        transitions=MappingProxyType({law_name: transitions[target][law_name]}),
        functions=functions_without_post,
    )

    def law(
        *, savings_grid: Float1D, **params: FloatND | float
    ) -> tuple[Float1D, Float1D, ScalarFloat]:
        # The DAG's own signature takes arrays, so a scalar param declared as a
        # plain Python float is lifted before it enters.
        array_params = {name: jnp.asarray(value) for name, value in params.items()}

        def landing(savings: ScalarFloat) -> FloatND:
            # The builder is annotated with the simulation shape, which nests by
            # target regime; built for the solution it returns one flat mapping
            # keyed by law name, so the single index yields the array.
            return next_state_func(**{post_decision_name: savings}, **array_params)[  # ty: ignore[invalid-return-type]
                law_name
            ]

        next_liquid, marginal_return = jax.vmap(jax.value_and_grad(landing))(
            savings_grid
        )
        at_zero = landing(jnp.zeros((), dtype=savings_grid.dtype))
        return next_liquid, marginal_return, at_zero

    return law


def fail_if_declared_law_is_not_increasing(
    *,
    next_liquid: Float1D,
    regime_name: RegimeName,
    target: RegimeName,
) -> None:
    """Check the landing points ascend with savings.

    The endogenous grid is read back onto the regular grid by interpolation,
    whose abscissae must be sorted. A law that does not increase in savings
    leaves them unsorted, and the interpolation does not check — it returns
    quietly wrong numbers rather than failing. The conventional law satisfies
    this for any positive gross return; a declared law need not.
    """
    if not bool(jnp.all(jnp.diff(next_liquid) > 0.0)):
        msg = (
            f"The law of motion regime '{regime_name}' declares toward "
            f"'{target}' does not strictly increase in post-decision savings. "
            f"The endogenous grid method reads its solution back by "
            f"interpolation, which requires the landing points to ascend, so a "
            f"law that can fall as savings rise is outside what the method "
            f"solves. Declare a law increasing in savings, or use GridSearch, "
            f"which maximizes over the action grid and needs no such ordering."
        )
        raise RegimeInitializationError(msg)
