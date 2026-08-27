"""Run the solver preconditions that need real parameter values.

A solver's scope condition is sometimes a property of the *evaluated* model
rather than of its structure: whether the budget is affine in the liquid state
between declared breakpoints, whether a carried state's law is constant within
an interval. Deciding either requires the tax schedules, tables, and
coefficients that only exist once the user supplies params, so the check cannot
run in `Solver.validate`, which the engine calls while building kernels at
`Model` construction.

Such a solver publishes its checks as `SolutionKernels.param_checks`. `Model`
calls `check_solver_params` for every solve because the evaluated functions can
change with every parameter draw; a precondition established for one draw says
nothing about another.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp

from _lcm.engine import Regime
from _lcm.regime_building.collective import PARETO_OBJECTIVE_ENTRY, ParetoWeights
from _lcm.typing import FlatParams, FlatRegimeParams, RegimeName
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidParamsError
from lcm.typing import FloatND


def check_solver_params(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> None:
    """Run every regime solver's parameter-dependent preconditions.

    Each check receives the regime's *complete* parameter vector — the params
    supplied to `solve` overlaid on the ones fixed at model construction. A
    precondition is a statement about the model's functions, and a schedule the
    author declared fixed is still what the budget reads; handing over only the
    free params would leave the fixed ones to be synthesized, which is the
    situation running at solve time exists to avoid.

    Args:
        regimes: Immutable mapping of regime names to regimes.
        flat_params: Immutable mapping of regime names to flat parameter mappings.

    Raises:
        Exception: Whatever a failing check raises — each solver owns its own
            error type and message.

    """
    complete = MappingProxyType(
        {
            name: MappingProxyType(
                {
                    **regime.resolved_fixed_params,
                    **flat_params.get(name, MappingProxyType({})),
                }
            )
            for name, regime in regimes.items()
        }
    )
    for regime in regimes.values():
        for check in regime.solution.param_checks:
            check(flat_params=complete)


def check_pareto_weights(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
) -> None:
    """Check every collective regime's Pareto weights against its parameters.

    What a Pareto weighting is — one finite, non-negative weight per
    stakeholder, with a strictly positive total — is a property of the *values*
    when a weight is declared as a function of a parameter or a state. A
    declaration that is admissible for one draw is not for the next, so this
    runs on every solve, on the regime's own grid, at every age the regime is
    active.

    Args:
        regimes: Immutable mapping of regime names to regimes.
        flat_params: Immutable mapping of regime names to flat parameter
            mappings.
        ages: The model's age grid, read for the age of each active period.

    Raises:
        InvalidParamsError: On the first inadmissible weighting, naming the
            regime, the stakeholder, and the parameter entry it comes from.

    """
    for regime_name, regime in regimes.items():
        weights = regime.solution.pareto_weights
        if weights is None or not weights.arg_names:
            continue
        regime_params = MappingProxyType(
            {
                **regime.solution.resolved_fixed_params,
                **flat_params.get(regime_name, MappingProxyType({})),
            }
        )
        _check_one_regimes_weights(
            regime_name=regime_name,
            regime=regime,
            regime_params=regime_params,
            ages=ages,
        )


def _check_one_regimes_weights(
    *,
    regime_name: RegimeName,
    regime: Regime,
    regime_params: FlatRegimeParams,
    ages: AgeGrid,
) -> None:
    """Evaluate one regime's declared weights over its grid and judge them.

    Raises:
        InvalidParamsError: On the first inadmissible weighting.

    """
    weights = cast("ParetoWeights", regime.solution.pareto_weights)
    states = regime.solution.state_action_space(regime_params=regime_params).states
    read_states = [name for name in weights.arg_names if name in states]
    mesh = (
        dict(
            zip(
                read_states,
                jnp.meshgrid(*(states[name] for name in read_states), indexing="ij"),
                strict=True,
            )
        )
        if read_states
        else {}
    )
    for period in regime.active_periods:
        supplied = {
            **mesh,
            **regime_params,
            "period": jnp.int32(period),
            "age": jnp.asarray(ages.period_to_age(period)),
        }
        declared = weights.declared(
            **{name: supplied[name] for name in weights.arg_names}
        )
        _fail_if_not_a_pareto_weighting(
            regime_name=regime_name, declared=declared, period=period
        )


def _fail_if_not_a_pareto_weighting(
    *,
    regime_name: RegimeName,
    declared: Mapping[str, FloatND],
    period: int,
) -> None:
    """Judge one period's declared weights.

    Raises:
        InvalidParamsError: If a weight is non-finite or negative anywhere, or
            if the total is not strictly positive somewhere.

    """
    where = f"regime {regime_name!r} at period {period}"
    for name, weight in sorted(declared.items()):
        arr = jnp.asarray(weight)
        if not bool(jnp.all(jnp.isfinite(arr))):
            msg = (
                f"The Pareto weight of stakeholder {name!r} in {where} is not "
                f"finite everywhere on the regime's grid (min {float(jnp.min(arr))}, "
                f"max {float(jnp.max(arr))}). Check the "
                f"'{PARETO_OBJECTIVE_ENTRY}' entry of this regime's params."
            )
            raise InvalidParamsError(msg)
        if bool(jnp.any(arr < 0)):
            msg = (
                f"The Pareto weight of stakeholder {name!r} in {where} is "
                f"negative somewhere on the regime's grid (min "
                f"{float(jnp.min(arr))}). A Pareto weighting is non-negative; "
                f"check the '{PARETO_OBJECTIVE_ENTRY}' entry of this regime's "
                "params."
            )
            raise InvalidParamsError(msg)
    total = sum(jnp.asarray(weight) for weight in declared.values())
    if not bool(jnp.all(jnp.asarray(total) > 0)):
        msg = (
            f"The Pareto weights in {where} do not sum to a positive total "
            "everywhere on the regime's grid, so the household scalarization "
            "is identically zero there and its argmax is undefined. Check the "
            f"'{PARETO_OBJECTIVE_ENTRY}' entry of this regime's params."
        )
        raise InvalidParamsError(msg)
