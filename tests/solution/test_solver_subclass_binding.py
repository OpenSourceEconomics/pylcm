"""A regime binds the solver instance it was given, not a stock replacement.

`OneMarginSolver` and `TwoMarginSolver` invite subclassing, so binding a regime's
margins onto a solver has to preserve the object the user constructed: its type,
the fields its subclass added, and the methods its subclass overrides.
"""

from dataclasses import dataclass, field
from typing import cast

from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.dcegm import _BoundDCEGM
from lcm.solvers import DCEGM
from tests.test_models import dcegm_paper_twin

_SAVINGS_GRID = dcegm_paper_twin.DCEGM_SOLVER.savings_grid


@dataclass(frozen=True, kw_only=True)
class _AnnotatedDCEGM(DCEGM):
    """A DC-EGM subclass carrying its own configuration and behaviour."""

    annotation: str = field(default="mine")

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Delegate, so the override is observable without changing results."""
        return super().build_period_kernels(context=context)


def _bound_solver(solver: DCEGM) -> _BoundDCEGM:
    """Bind `solver` onto an otherwise ordinary consumption-savings regime."""
    regime = dcegm_paper_twin._working_life(solver="dcegm")
    return cast("_BoundDCEGM", regime.replace(solver=solver).solver)


def test_binding_preserves_the_solver_subclass_type() -> None:
    """A bound subclass is still an instance of the subclass."""
    bound = _bound_solver(_AnnotatedDCEGM(savings_grid=_SAVINGS_GRID))

    assert isinstance(bound, _AnnotatedDCEGM)


def test_binding_preserves_a_field_the_subclass_added() -> None:
    """A field declared only by the subclass survives binding."""
    bound = cast(
        "_AnnotatedDCEGM",
        _bound_solver(_AnnotatedDCEGM(savings_grid=_SAVINGS_GRID, annotation="kept")),
    )

    assert bound.annotation == "kept"


def test_binding_preserves_a_method_the_subclass_overrides() -> None:
    """The bound solver still dispatches to the subclass's override."""
    bound = _bound_solver(_AnnotatedDCEGM(savings_grid=_SAVINGS_GRID))

    assert type(bound).build_period_kernels is _AnnotatedDCEGM.build_period_kernels


def test_binding_resolves_the_regimes_roles_onto_the_subclass() -> None:
    """Binding still fills the role the regime declares."""
    bound = _bound_solver(_AnnotatedDCEGM(savings_grid=_SAVINGS_GRID))

    assert bound.continuous_state == "wealth"


def test_binding_a_plain_solver_is_unaffected() -> None:
    """A stock solver binds exactly as before, whatever a subclass would do."""
    bound = _bound_solver(DCEGM(savings_grid=_SAVINGS_GRID))

    assert type(bound).__name__ == "_BoundDCEGM"
