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

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.regime_building.next_state import get_next_state_function_for_solution
from _lcm.typing import (
    EconFunctionsMapping,
    FloatND,
    NextStateSimulationFunction,
    RegimeName,
    StateName,
    StateOrActionName,
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
    variable_names: frozenset[StateOrActionName],
) -> Callable[..., tuple[Float1D, Float1D]]:
    """Build the declared law of motion as a function of savings.

    Args:
        transitions: The regime's transitions, keyed by target regime name.
        functions: The regime's auxiliary functions. The post-decision function
            is removed from the copy handed to the DAG.
        post_decision_name: Name of the function computing post-decision savings.
        target: The regime active next period, whose law is composed.
        target_state: The target's own name for its single continuous state.
        variable_names: The regime's state and action names. A composed law that
            still reads one of them after the post-decision function is removed
            has reached it by some other route, which is refused here.

    Raises:
        RegimeInitializationError: If the composed law reads a state or action
            rather than the post-decision node.

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
    _fail_if_law_reaches_past_the_post_decision(
        reads=wanted,
        variable_names=variable_names,
        post_decision_name=post_decision_name,
        regime_target=target,
        law_name=law_name,
    )

    return _DeclaredLiquidLaw(
        next_state_func=next_state_func,
        post_decision_name=post_decision_name,
        law_name=law_name,
        wanted=wanted,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _DeclaredLiquidLaw:
    """A regime's composed law of motion as a function of post-decision savings.

    Called with `savings_grid` and the regime's flat params, it returns the
    landing points on that grid and their derivative with respect to savings.
    """

    next_state_func: NextStateSimulationFunction
    """The composed next-state DAG with the post-decision function removed."""

    post_decision_name: str
    """Name of the (removed) post-decision function, now a DAG input."""

    law_name: TransitionFunctionName
    """Key of the target state's law in the DAG's output mapping."""

    wanted: frozenset[str]
    """The params the law reads, selected from the caller's whole pool."""

    def __call__(
        self, *, savings_grid: Float1D, **params: FloatND | float
    ) -> tuple[Float1D, Float1D]:
        # The DAG's own signature takes arrays, so a scalar param declared as a
        # plain Python float is lifted before it enters.
        array_params = {
            name: jnp.asarray(value)
            for name, value in params.items()
            if name in self.wanted
        }
        landing = functools.partial(
            _landing_point,
            next_state_func=self.next_state_func,
            post_decision_name=self.post_decision_name,
            law_name=self.law_name,
            array_params=array_params,
        )
        return jax.vmap(jax.value_and_grad(landing))(savings_grid)


# keyword-only-exempt: library-callback=jax.value_and_grad
def _landing_point(
    savings: ScalarFloat,
    *,
    next_state_func: NextStateSimulationFunction,
    post_decision_name: str,
    law_name: TransitionFunctionName,
    array_params: dict[str, FloatND],
) -> FloatND:
    """Where one savings level lands next period under the composed law."""
    # The builder is annotated with the simulation shape, which nests by
    # target regime; built for the solution it returns one flat mapping
    # keyed by law name, so the single index yields the array.
    return next_state_func(**{post_decision_name: savings}, **array_params)[  # ty: ignore[invalid-return-type]
        law_name
    ]


def _fail_if_law_reaches_past_the_post_decision(
    *,
    reads: frozenset[str],
    variable_names: frozenset[StateOrActionName],
    post_decision_name: str,
    regime_target: RegimeName,
    law_name: TransitionFunctionName,
) -> None:
    """Require the composed law to read savings and params, nothing else.

    With the post-decision function removed from the DAG its name becomes an
    input, so whatever else the composed law still asks for it reached by some
    other route. A state or action among those means the law is not a function
    of savings at all, and neither the landing point nor its derivative with
    respect to savings is the quantity the Euler inversion needs.
    """
    reached = sorted(reads & variable_names)
    if not reached:
        return
    msg = (
        f"The law '{law_name}' toward regime '{regime_target}' reads "
        f"{reached} directly rather than reaching them through the "
        f"post-decision function '{post_decision_name}'. The endogenous grid "
        f"method inverts the Euler equation on the savings grid, so the law "
        f"must be a function of '{post_decision_name}' and parameters alone. "
        f"Write the law in terms of '{post_decision_name}', or use GridSearch, "
        f"which maximizes over the action grid and needs no such form."
    )
    raise RegimeInitializationError(msg)


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
