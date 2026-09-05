"""Building and solving a model creates no nested function that outlives it.

pylcm runs under a beartype claw that decorates every function definition, including
one executed inside a call, and beartype memoizes each decorated function object for
the rest of the process. A function defined inside another function per build, per
solve, or per trace would therefore be pinned together with everything it closed over:
the grids, compiled kernels, and arrays of a model nobody references any more. The
engine keeps its per-call state in module-level functions and frozen dataclasses
instead, so a dropped model and solution leave nothing behind.
"""

import gc
import types
from collections.abc import Callable
from pathlib import Path

import pytest

import _lcm
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.solvers import AdaptiveOuterMesh, GridSearch
from lcm.typing import UserParams
from tests.test_models import n_nbegm_toy
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_ENGINE_SOURCE_ROOT = Path(_lcm.__file__).resolve().parent
# Sources that still define a function per call, and why:
# - the candidate certificate pins the reducer builders' bodies verbatim, nested
#   definitions included, so their shape changes with that certificate or not at all;
# - the remaining three are converted together with the execution work that rewrites
#   the same call sites.
_EXEMPT_SOURCES = frozenset(
    _ENGINE_SOURCE_ROOT / relative
    for relative in (
        Path("regime_building/max_Q_over_a.py"),
        Path("regime_building/collective.py"),
        Path("solution/negm.py"),
        Path("solution/nnbegm.py"),
        Path("regime_building/processing.py"),
    )
)
_TOY_PARAMS: UserParams = {"discount_factor": 0.95}


def _live_nested_functions(*, source_root: Path) -> int:
    """Count the per-call nested functions under `source_root` still reachable.

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
        and "<locals>" in obj.__qualname__
        # An annotated function also owns a deferred annotation thunk; it lives
        # and dies with its function, so counting it would double every survivor.
        and not obj.__qualname__.endswith(".__annotate__")
        and Path(obj.__code__.co_filename).is_relative_to(source_root)
        and Path(obj.__code__.co_filename) not in _EXEMPT_SOURCES
    )


def _active_in_the_first_period_only(age: int) -> bool:
    return age <= START_AGE


def _grid_search_model() -> Model:
    """Build the smallest one-period grid-search model."""
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=_active_in_the_first_period_only,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=START_AGE + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _toy_model(*, variant: str, outer_search: AdaptiveOuterMesh | None = None) -> Model:
    """Build the smallest two-period nested-margin toy of one variant."""
    return n_nbegm_toy.build_model(
        variant=variant,
        n_periods=2,
        illiquid_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=2),
        outer_search=outer_search,
    )


def _adaptive_outer_search() -> AdaptiveOuterMesh:
    return AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=3),
        max_nodes=9,
        max_refinement_rounds=1,
        value_atol=1e-2,
        value_rtol=1e-2,
        golden_iterations=4,
        fail_closed=False,
    )


_FAMILIES: dict[str, tuple[Callable[[], Model], Callable[[], UserParams]]] = {
    "grid_search": (_grid_search_model, lambda: get_params(n_periods=2)),
    "negm": (lambda: _toy_model(variant="negm"), lambda: _TOY_PARAMS),
    "n_nbegm_finite": (lambda: _toy_model(variant="n_nbegm"), lambda: _TOY_PARAMS),
    "n_nbegm_adaptive": (
        lambda: _toy_model(variant="n_nbegm", outer_search=_adaptive_outer_search()),
        lambda: _TOY_PARAMS,
    ),
}


def _build_solve_and_drop(family: str) -> None:
    build_model, build_params = _FAMILIES[family]
    model = build_model()
    solution = model.solve(params=build_params(), log_level="off")
    del solution, model


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_a_dropped_model_and_solution_leave_no_nested_engine_function_behind(
    family: str,
) -> None:
    """A second build-and-solve of a family pins no nested engine function."""
    assert _ENGINE_SOURCE_ROOT.is_relative_to(Path(__file__).resolve().parents[1])
    _build_solve_and_drop(family)
    before = _live_nested_functions(source_root=_ENGINE_SOURCE_ROOT)

    _build_solve_and_drop(family)

    assert _live_nested_functions(source_root=_ENGINE_SOURCE_ROOT) == before


def test_the_nested_function_probe_counts_a_live_nested_function() -> None:
    """The probe reports a nested function of the source tree it is pointed at."""
    source_root = Path(__file__).resolve().parent
    baseline = _live_nested_functions(source_root=source_root)

    def _pinned_nested_function() -> None:
        """Stay alive for the rest of this test so the probe has one to find."""

    assert _live_nested_functions(source_root=source_root) == baseline + 1
    _pinned_nested_function()
