"""Nested solver builds see only the periods in their resolved group."""

from dataclasses import dataclass, field
from functools import cache

import pytest

from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.periodization import (
    restrict_solver_build_context_to_period_group,
)
from lcm.solvers import GridSearch
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid_from_savings,
    savings,
    utility,
)


def _resources(liquid: ContinuousState) -> FloatND:
    """Use liquid wealth directly as resources."""
    return liquid


@dataclass(frozen=True, kw_only=True)
class _ContextRecordingGridSearch(GridSearch):
    """Record a real finalized solver context before delegating its build."""

    contexts: list[SolverBuildContext] = field(default_factory=list)

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Capture the build context and retain ordinary GridSearch construction."""
        self.contexts.append(context)
        return super().build_period_kernels(context=context)


@cache
def _context() -> SolverBuildContext:
    solver = _ContextRecordingGridSearch()
    make_alive_dead_model(
        n_periods=4,
        n_liquid=4,
        liquid_max=4.0,
        n_consumption=4,
        alive_functions={
            "utility": utility,
            "resources": _resources,
            "savings": savings,
        },
        liquid_law=next_liquid_from_savings,
        alive_solver=solver,
        constraints={"feasible": feasible},
    )
    return solver.contexts[0]


def test_period_group_restriction_changes_only_the_current_regime() -> None:
    """Target-regime lifecycle metadata survives a source-group restriction."""
    context = _context()

    restricted = restrict_solver_build_context_to_period_group(
        context=context,
        periods=(0, 2),
    )

    assert restricted.regimes_to_active_periods["alive"] == (0, 2)
    assert restricted.regimes_to_active_periods["dead"] == (3,)
    assert context.regimes_to_active_periods["alive"] == (0, 1, 2)


def test_full_period_group_reuses_the_original_context() -> None:
    """An ordinary non-nested build pays no mapping-copy cost."""
    context = _context()

    restricted = restrict_solver_build_context_to_period_group(
        context=context,
        periods=(0, 1, 2),
    )

    assert restricted is context


@pytest.mark.parametrize(
    "periods",
    [
        (),
        (1, 1),
        (2, 0),
        (0, 3),
    ],
)
def test_period_group_restriction_rejects_nonpartitions(
    periods: tuple[int, ...],
) -> None:
    """A nested caller cannot silently duplicate, reorder, or invent periods."""
    with pytest.raises(ValueError, match="nonempty ordered subset"):
        restrict_solver_build_context_to_period_group(
            context=_context(),
            periods=periods,
        )
