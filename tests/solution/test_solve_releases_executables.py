"""A finished solve leaves no compiled executable behind.

Every executable a solve compiles is reachable only through the solve's own planned
cores, so dropping the result must let the garbage collector reclaim all of them. A
retained executable pins its compiled code and its memory mappings, and a session
that solves many models then exhausts the process's mapping budget long before its
memory budget.
"""

import gc

import jax

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.solvers import GridSearch
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_N_PERIODS = 2


def _live_compiled_executables() -> int:
    """Count the compiled executables the collector can still reach."""
    gc.collect()
    return sum(isinstance(obj, jax.stages.Compiled) for obj in gc.get_objects())


def _model() -> Model:
    final_age_alive = START_AGE + _N_PERIODS - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_a_dropped_solution_releases_every_compiled_executable() -> None:
    before = _live_compiled_executables()

    solution = _model().solve(params=get_params(n_periods=_N_PERIODS), log_level="off")
    del solution

    assert _live_compiled_executables() == before
