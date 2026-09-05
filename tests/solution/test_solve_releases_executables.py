"""A finished solve leaves no compiled executable behind.

Every executable a solve compiles is reachable only through the solve's own planned
cores, so dropping the result must let the garbage collector reclaim all of them. A
retained executable pins its compiled code and its memory mappings, and a session
that solves many models then exhausts the process's mapping budget long before its
memory budget.
"""

import gc
import inspect
import types
import weakref

import jax

from _lcm.solution import negm
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
_NEGM_SOURCE_FILE = negm.__file__
# A code object records the filename the compiler saw, which is not guaranteed to be
# the module's own `__file__` string: the beartype claw rebinds each module attribute
# to a wrapper whose own code object reports a synthetic name, so the pin has to reach
# through it. Assert the two agree at import, so a probe pointed at `_NEGM_SOURCE_FILE`
# cannot report zero merely by naming a file no code object claims — which would let
# the regression assertion below hold vacuously.
assert inspect.unwrap(negm._durable_values_at).__code__.co_filename == _NEGM_SOURCE_FILE


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


def _negm_model() -> Model:
    """Build the smallest two-period NEGM toy."""
    return n_nbegm_toy.build_model(
        variant="negm",
        n_periods=2,
        illiquid_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=2),
    )


def _live_nested_functions(*, source_file: str) -> int:
    """Count the per-call nested functions of one source file still reachable.

    A function defined inside another function carries `<locals>` in its qualified
    name; one that is still alive after the call that created it has been pinned by
    something outside that call, together with everything it closed over. Membership
    is decided by the code object's filename, so a wrapper another library defined
    and `functools.wraps` relabelled with the inspected module's name is not counted,
    and the count is complete over every function object the collector tracks.
    """
    gc.collect()
    return sum(
        1
        for obj in gc.get_objects()
        if isinstance(obj, types.FunctionType)
        and obj.__code__.co_filename == source_file
        and "<locals>" in obj.__qualname__
    )


def test_a_dropped_negm_solution_releases_every_compiled_executable() -> None:
    """A finished NEGM solve keeps no compiled executable alive."""
    warm = _negm_model()
    warm.solve(params={"discount_factor": 0.95}, log_level="off")
    del warm
    before = _live_compiled_executables()

    solution = _negm_model().solve(params={"discount_factor": 0.95}, log_level="off")
    del solution

    assert _live_compiled_executables() == before


def test_a_dropped_negm_model_and_solution_leave_no_nested_function_behind() -> None:
    """Building and solving a NEGM model creates no nested function that outlives it."""
    warm = _negm_model()
    warm.solve(params={"discount_factor": 0.95}, log_level="off")
    del warm
    before = _live_nested_functions(source_file=_NEGM_SOURCE_FILE)

    solution = _negm_model().solve(params={"discount_factor": 0.95}, log_level="off")
    del solution

    assert _live_nested_functions(source_file=_NEGM_SOURCE_FILE) == before


def test_the_nested_function_probe_counts_a_live_nested_function() -> None:
    """The probe reports a nested function of the source file it is pointed at."""
    baseline = _live_nested_functions(source_file=__file__)

    def _pinned_nested_function() -> None:
        """Stay alive for the rest of this test so the probe has one to find."""

    _pinned_nested_function()

    assert _live_nested_functions(source_file=__file__) > baseline
