"""An age-specialized grid belongs in `states` directly, never inside `Phased`.

`AgeSpecializedGrid` is a build-time marker: pylcm resolves it to each period's
concrete grid before anything reads it, and it looks for markers to resolve only
among the top-level `states` values. A marker nested inside a `Phased` variant
is therefore never resolved, so the phase grammar refuses it on either side
rather than letting an unresolved marker reach the kernels.
"""

import re

import jax.numpy as jnp
import pytest

from lcm import AgeSpecializedGrid, LinSpacedGrid, Phased, Regime
from lcm.exceptions import RegimeInitializationError
from lcm.transition import MarkovTransition
from lcm.typing import ContinuousState, FloatND

_TENURE = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_TENURE_CEILING_EARLY = 1.0
_TENURE_CEILING_LATE = 2.0


def test_age_specialized_grid_as_a_phased_states_solve_variant_is_rejected():
    """A marker on the solve side of a `Phased` state is refused at construction.

    Nothing about a carried state's solve imputation can resolve a grid marker,
    so admitting one would hand an unresolved marker to the solve DAG.
    """
    with pytest.raises(
        RegimeInitializationError,
        match=re.escape(
            "states['tenure']: `AgeSpecializedGrid` is not supported inside "
            "`Phased` (here: solve)."
        ),
    ):
        _build_regime(
            state_spec=Phased(
                solve=AgeSpecializedGrid(build=_tenure_grid, signature=_tenure_ceiling),
                simulate=_TENURE,
            )
        )


def test_age_specialized_grid_as_a_phased_states_simulate_variant_is_rejected():
    """A marker on the simulate side of a `Phased` state is refused by name.

    A carried state's grid is its simulate-phase domain, and an age-specialized
    one is not supported there. The refusal says so, rather than describing the
    declaration as a function derived in both phases.
    """
    with pytest.raises(
        RegimeInitializationError,
        match=re.escape(
            "states['tenure']: `AgeSpecializedGrid` is not supported inside "
            "`Phased` (here: simulate)."
        ),
    ):
        _build_regime(
            state_spec=Phased(
                solve=_impute_tenure,
                simulate=AgeSpecializedGrid(
                    build=_tenure_grid, signature=_tenure_ceiling
                ),
            )
        )


def _build_regime(*, state_spec: Phased) -> Regime:
    """Build a one-state regime whose `tenure` is declared as given."""
    return Regime(
        transition={"exit": MarkovTransition(_prob_one)},
        states={"tenure": state_spec},
        state_transitions={"tenure": _next_tenure},
        functions={"utility": _utility},
    )


def _tenure_ceiling(age: float) -> float:
    """The highest tenure the grid reaches at this age."""
    return _TENURE_CEILING_EARLY if age <= 1 else _TENURE_CEILING_LATE


def _tenure_grid(age: float) -> LinSpacedGrid:
    """The tenure grid: zero to the age's ceiling, on two nodes."""
    return LinSpacedGrid(start=0.0, stop=_tenure_ceiling(age), n_points=2)


def _prob_one(age: FloatND) -> FloatND:
    """Regime transition taken with certainty."""
    return jnp.ones_like(age, dtype=float)


def _impute_tenure() -> FloatND:
    """Tenure while the regime is solved: a constant, never a grid axis."""
    return jnp.asarray(0.0)


def _next_tenure(tenure: ContinuousState) -> FloatND:
    """Tenure grows by one unit per period."""
    return tenure + 1.0


def _utility(tenure: ContinuousState) -> FloatND:
    """The regime pays out its tenure."""
    return tenure
