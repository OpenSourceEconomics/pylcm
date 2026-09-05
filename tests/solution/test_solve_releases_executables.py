"""A finished solve leaves no compiled executable behind.

Every executable a solve compiles is reachable only through the solve's own planned
cores, so dropping the result must let the garbage collector reclaim all of them. A
retained executable pins its compiled code and its memory mappings, and a session
that solves many models then exhausts the process's mapping budget long before its
memory budget.
"""

import gc
import weakref

import jax

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.solvers import AdaptiveOuterMesh, GridSearch
from tests.test_models import n_nbegm_toy
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
    return sum(_is_compiled_executable(obj) for obj in gc.get_objects())


def _is_compiled_executable(obj: object) -> bool:
    """Return whether `obj` is a compiled executable.

    A dead weak proxy raises `ReferenceError` on any attribute access, including
    the one `isinstance` performs; it is not an executable.
    """
    try:
        return isinstance(obj, jax.stages.Compiled)
    except ReferenceError:
        return False


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


def _adaptive_model() -> Model:
    """Build the smallest two-period adaptive-outer-mesh NNBEGM toy."""
    return n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        illiquid_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=2),
        outer_search=AdaptiveOuterMesh(
            initial_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=3),
            max_nodes=9,
            max_refinement_rounds=1,
            value_atol=1e-2,
            value_rtol=1e-2,
            golden_iterations=4,
            fail_closed=False,
        ),
    )


def _adaptive_params() -> dict[str, float]:
    """Return the parameters the adaptive-outer-mesh toy is solved at."""
    return {"discount_factor": 0.95}


def test_a_dropped_solution_releases_every_compiled_executable() -> None:
    before = _live_compiled_executables()

    solution = _model().solve(params=get_params(n_periods=_N_PERIODS), log_level="off")
    del solution

    assert _live_compiled_executables() == before


def test_counting_live_executables_tolerates_a_dead_weak_proxy() -> None:
    class _Referent:
        pass

    referent = _Referent()
    proxy = weakref.proxy(referent)
    del referent

    assert _live_compiled_executables() >= 0
    assert proxy is not None


def test_a_dropped_adaptive_nested_solution_releases_every_compiled_executable() -> (
    None
):
    """The adaptive outer mesh's host driver keeps no executable alive after solving."""
    warm = _adaptive_model()
    warm.solve(params=_adaptive_params(), log_level="off")
    del warm
    before = _live_compiled_executables()

    solution = _adaptive_model().solve(params=_adaptive_params(), log_level="off")
    del solution

    assert _live_compiled_executables() == before
