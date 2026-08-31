"""Shared scaffolding for the NBEGM toy models.

Every NBEGM toy is a small lifecycle model solved twice — once by `GridSearch`
(the dense brute oracle) and once by a NBEGM variant — and compared. The pieces
that are identical across the toys live here:

- `RegimeId` — the alive/dead regime categorical.
- `crra_utility` / `utility` / `bequest` — the CRRA utility trio.
- `prob_stay_alive` / `prob_die` — the deterministic (0/1) survival transition.
- `feasible` — the borrowing constraint `consumption <= resources`.
- `next_liquid` / `savings` / `next_liquid_from_savings` — the liquid law of
  motion in cash-on-hand form (brute) and post-decision savings form (NBEGM).
- `resolve_solver` — the `"brute"` / `"nbegm"` variant dispatch.
- `make_alive_dead_model` — the two-regime (alive, dead) model assembler.

A toy re-exports the names it uses (module-level import from here), keeps its
own budget DAG (`resources` and friends), and toys with genuinely different
regime structure keep their own assembly.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp

from _lcm.grids.base import Grid
from _lcm.grids.continuous import ContinuousGrid
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
    liquid_law_from_resources,
    liquid_law_from_savings,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.regime import Regime
from lcm.solvers import NBEGM, GridSearch, OneMarginSolver
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def crra_utility(*, consumption: FloatND, crra: float | FloatND) -> FloatND:
    """CRRA utility, log at `crra == 1`.

    The inactive power branch's exponent/denominator is clamped at `crra == 1` so
    `jax.grad` through the `where` (the EGM marginal-utility path) stays finite;
    the unguarded `1/(1 - crra)` is infinite there and its zero-weighted gradient
    contribution turns into NaN.
    """
    one_minus_crra = jnp.where(crra == 1.0, 1.0, 1.0 - crra)
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption**one_minus_crra / one_minus_crra,
    )


def utility(*, consumption: ContinuousAction, crra: float) -> FloatND:
    """CRRA consumption utility."""
    return crra_utility(consumption=consumption, crra=crra)


def bequest(*, liquid: ContinuousState, crra: float) -> FloatND:
    """Terminal value: consume remaining liquid wealth."""
    return crra_utility(consumption=liquid, crra=crra)


def feasible(*, resources: FloatND, consumption: ContinuousAction) -> BoolND:
    """Borrowing constraint: consumption cannot exceed cash-on-hand."""
    return consumption <= resources


# The single-liquid route takes pylcm's own laws, so the toys declare those
# rather than a local spelling of the same arithmetic.
next_liquid = liquid_law_from_resources
next_liquid_from_savings = liquid_law_from_savings


def savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Post-decision savings: cash-on-hand net of consumption."""
    return resources - consumption


def prob_stay_alive(*, age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of staying alive next period."""
    return jnp.where(age + 1 < final_age_alive, 1.0, 0.0)


def prob_die(*, age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of dying next period."""
    return jnp.where(age + 1 >= final_age_alive, 1.0, 0.0)


def resolve_solver(
    *, variant: str, savings_grid: ContinuousGrid, **nbegm_kwargs: object
) -> OneMarginSolver | GridSearch:
    """Dispatch the toy's alive-regime solver from the variant name.

    - `"brute"` — `GridSearch`, the dense-grid oracle.
    - `"nbegm"` — numerical `NBEGM` configuration over `savings_grid`,
      forwarding extra numerical constructor arguments (block sizes, envelope
      settings, and so on). DAG role names belong to the regime's
      `LiquidMargin` and are not accepted here.

    Any other name raises `ValueError`.
    """
    if variant == "brute":
        return GridSearch()
    if variant == "nbegm":
        role_names = {
            "budget_target",
            "continuous_state",
            "continuous_action",
            "post_decision_function",
        }
        stale = sorted(role_names & nbegm_kwargs.keys())
        if stale:
            msg = (
                "NBEGM DAG role names moved to LiquidMargin; remove solver "
                f"arguments {stale}."
            )
            raise TypeError(msg)
        return NBEGM(
            savings_grid=savings_grid,
            # Forwards whatever numerical configuration the caller names, so the
            # values arrive as `object` rather than each parameter's own type.
            **nbegm_kwargs,  # ty: ignore[invalid-argument-type]
        )
    msg = f"unknown variant {variant!r}; use 'brute' or 'nbegm'."
    raise ValueError(msg)


def make_alive_dead_model(
    *,
    n_periods: int,
    n_liquid: int,
    liquid_max: float,
    n_consumption: int,
    alive_functions: Mapping[str, Callable[..., object]],
    liquid_law: Callable[..., object],
    alive_solver: OneMarginSolver | GridSearch,
    constraints: Mapping[str, Callable[..., object]],
    extra_actions: Mapping[str, Grid] | None = None,
    extra_states: Mapping[str, Grid] | None = None,
    extra_state_transitions: Mapping[str, Any] | None = None,
    survival_transition: Mapping[str, Any] | None = None,
    model_states: Mapping[str, Grid] | None = None,
    liquid_grid: Grid | None = None,
    dead_functions: Mapping[str, Callable[..., object]] | None = None,
    fixed_params: Mapping[str, Any] | None = None,
    liquid_state: str = "liquid",
    liquid_action: str = "consumption",
    liquid_resources: str = "resources",
    liquid_post_decision: str = "savings",
) -> Model:
    """Assemble the two-regime (alive, dead) toy around a toy-specific budget DAG.

    The alive regime consumes on a dense grid, carries the liquid state (plus any
    `extra_states`), evolves liquid by `liquid_law` toward both targets, and dies
    deterministically via the shared survival transition. The dead regime is
    terminal and values remaining wealth as a CRRA bequest.

    Args:
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        liquid_max: Upper bound of the liquid and consumption grids.
        n_consumption: Consumption-action grid size (brute only).
        alive_functions: The alive regime's function pool (must define `utility`
            and the budget node).
        liquid_law: Law of motion for `liquid`, applied toward both targets.
        alive_solver: Solver for the alive regime (from `resolve_solver`).
        constraints: Constraint pool for the alive regime (typically
            `{"feasible": feasible}` for brute, empty for NBEGM's savings form).
        extra_actions: Additional action grids beyond `consumption`.
        extra_states: Additional state grids beyond `liquid` (ride-along
            co-states, stochastic processes).
        extra_state_transitions: Transition entries for the extra states.
        survival_transition: Regime transition for the alive regime.
        model_states: States broadcast at model level.
        liquid_grid: Grid for the `liquid` state in both regimes. Defaults to a
            `LinSpacedGrid` spanning `[0.1, liquid_max]` with `n_liquid` points.
        dead_functions: Override for the dead regime's function pool (defaults
            to the CRRA bequest); e.g. a constant utility for a zero-marginal
            (flat) continuation.
        fixed_params: Parameters fixed at model construction rather than supplied
            to `solve`.

    Returns:
        The assembled `Model`.

    """
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    if liquid_grid is None:
        liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)
    alive_actions = {
        liquid_action: LinSpacedGrid(
            start=0.1, stop=liquid_max, n_points=n_consumption
        ),
        **(dict(extra_actions) if extra_actions else {}),
    }
    alive_states = {
        liquid_state: liquid_grid,
        **(dict(extra_states) if extra_states else {}),
    }
    alive_state_transitions = {
        liquid_state: {"alive": liquid_law, "dead": liquid_law},
        **(dict(extra_state_transitions) if extra_state_transitions else {}),
    }
    alive_transition = (
        dict(survival_transition)
        if survival_transition is not None
        else {
            "alive": MarkovTransition(prob_stay_alive),
            "dead": MarkovTransition(prob_die),
        }
    )
    alive_active = lambda age, fa=final_age: age < fa  # noqa: E731
    # Built per branch rather than from one shared mapping: the two regime
    # classes narrow `solver` differently, and a `**kwargs` mapping erases the
    # argument types the narrowing is expressed in.
    if isinstance(alive_solver, NBEGM):
        alive = ConsumptionSavingsRegime(
            actions=alive_actions,
            states=alive_states,
            state_transitions=alive_state_transitions,
            constraints=dict(constraints),
            transition=alive_transition,
            functions=dict(alive_functions),
            active=alive_active,
            solver=alive_solver,
            liquid=LiquidMargin(
                state=liquid_state,
                action=liquid_action,
                resources=liquid_resources,
                post_decision_state=liquid_post_decision,
            ),
        )
    else:
        alive = Regime(
            actions=alive_actions,
            states=alive_states,
            state_transitions=alive_state_transitions,
            constraints=dict(constraints),
            transition=alive_transition,
            functions=dict(alive_functions),
            active=alive_active,
            solver=alive_solver,
        )
    # The default survival law dies deterministically into the final age, so the
    # absorbing regime is needed only there. A caller-supplied law may put mass on
    # `dead` at any transition, and mass sent to a target that is inactive when it
    # is reached goes unrepresented, leaving the continuation short of unit mass
    # and the solved value NaN — so `dead` is active from the first age it can be
    # entered.
    first_dead_age = 1 if survival_transition is not None else final_age
    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid},
        functions=dict(dead_functions)
        if dead_functions is not None
        else {"utility": bequest},
        # `first_dead_age`, not `final_age`: with a survival transition the dead
        # regime must be active from the first age it can be entered, or the
        # mass sent to it is dropped and the survivors renormalized.
        active=lambda age, fa=first_dead_age: age >= fa,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
        states=dict(model_states) if model_states else {},
        fixed_params=dict(fixed_params) if fixed_params else {},
    )
