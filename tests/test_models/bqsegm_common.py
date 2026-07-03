"""Shared scaffolding for the BQSEGM toy models.

Every BQSEGM toy is a small lifecycle model solved twice — once by `GridSearch`
(the dense brute oracle) and once by a BQSEGM variant — and compared. The pieces
that are identical across the toys live here:

- `RegimeId` — the alive/dead regime categorical.
- `crra_utility` / `utility` / `bequest` — the CRRA utility trio.
- `prob_stay_alive` / `prob_die` — the deterministic (0/1) survival transition.
- `feasible` — the borrowing constraint `consumption <= coh`.
- `next_liquid` / `savings` / `next_liquid_from_savings` — the liquid law of
  motion in cash-on-hand form (brute) and post-decision savings form (BQSEGM).
- `resolve_solver` — the `"brute"` / `"bqsegm"` variant dispatch.
- `make_alive_dead_model` — the two-regime (alive, dead) model assembler.

A toy re-exports the names it uses (module-level import from here), keeps its
own budget DAG (`coh` and friends), and toys with genuinely different regime
structure keep their own assembly.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp

from _lcm.grids.base import Grid
from _lcm.grids.continuous import ContinuousGrid
from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, categorical
from lcm.regime import Regime
from lcm.solvers import BQSEGM, GridSearch, Solver
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def crra_utility(consumption: FloatND, crra: float | FloatND) -> FloatND:
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


def utility(consumption: ContinuousAction, crra: float) -> FloatND:
    """CRRA consumption utility."""
    return crra_utility(consumption, crra)


def bequest(liquid: ContinuousState, crra: float) -> FloatND:
    """Terminal value: consume remaining liquid wealth."""
    return crra_utility(liquid, crra)


def feasible(coh: FloatND, consumption: ContinuousAction) -> BoolND:
    """Borrowing constraint: consumption cannot exceed cash-on-hand."""
    return consumption <= coh


def next_liquid(
    coh: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law of motion: saved cash earns the liquid return, plus income."""
    return (1.0 + return_liquid) * (coh - consumption) + income


def savings(coh: FloatND, consumption: ContinuousAction) -> FloatND:
    """Post-decision savings: cash-on-hand net of consumption."""
    return coh - consumption


def next_liquid_from_savings(
    savings: FloatND, return_liquid: float, income: float
) -> ContinuousState:
    """Liquid law of motion in savings form: savings earn the return, plus income."""
    return (1.0 + return_liquid) * savings + income


def prob_stay_alive(age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of staying alive next period."""
    return jnp.where(age + 1 < final_age_alive, 1.0, 0.0)


def prob_die(age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of dying next period."""
    return jnp.where(age + 1 >= final_age_alive, 1.0, 0.0)


def resolve_solver(
    variant: str, *, savings_grid: ContinuousGrid, **bqsegm_kwargs: object
) -> Solver:
    """Dispatch the toy's alive-regime solver from the variant name.

    - `"brute"` — `GridSearch`, the dense-grid oracle.
    - `"bqsegm"` — `BQSEGM` over `savings_grid`, forwarding any extra
      constructor arguments (`budget_target`, `post_decision_function`, block
      sizes, …).

    Any other name raises `ValueError`.
    """
    if variant == "brute":
        return GridSearch()
    if variant == "bqsegm":
        return BQSEGM(savings_grid=savings_grid, **bqsegm_kwargs)  # ty: ignore[invalid-argument-type]
    msg = f"unknown variant {variant!r}; use 'brute' or 'bqsegm'."
    raise ValueError(msg)


def make_alive_dead_model(
    *,
    n_periods: int,
    n_liquid: int,
    liquid_max: float,
    n_consumption: int,
    alive_functions: Mapping[str, Callable[..., object]],
    liquid_law: Callable[..., object],
    alive_solver: Solver,
    constraints: Mapping[str, Callable[..., object]],
    extra_actions: Mapping[str, Grid] | None = None,
    extra_states: Mapping[str, Grid] | None = None,
    extra_state_transitions: Mapping[str, Any] | None = None,
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
            `{"feasible": feasible}` for brute, empty for BQSEGM's savings form).
        extra_actions: Additional action grids beyond `consumption`.
        extra_states: Additional state grids beyond `liquid` (ride-along
            co-states, stochastic processes).
        extra_state_transitions: Transition entries for the extra states.

    Returns:
        The assembled `Model`.

    """
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)
    alive = Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=0.1, stop=liquid_max, n_points=n_consumption
            ),
            **(dict(extra_actions) if extra_actions else {}),
        },
        states={
            "liquid": liquid_grid,
            **(dict(extra_states) if extra_states else {}),
        },
        state_transitions={
            "liquid": {"alive": liquid_law, "dead": liquid_law},
            **(dict(extra_state_transitions) if extra_state_transitions else {}),
        },
        constraints=dict(constraints),
        transition={
            "alive": MarkovTransition(prob_stay_alive),
            "dead": MarkovTransition(prob_die),
        },
        functions=dict(alive_functions),
        active=lambda age, fa=final_age: age < fa,
        solver=alive_solver,
    )
    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid},
        functions={"utility": bequest},
        active=lambda age, fa=final_age: age >= fa,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )
