"""Which axis names a model accepts in `ExecutionConfig.axis_widths`.

The legal set is what the model's own core programs declare — over every phase,
so a name is legal as soon as either the solve or the forward simulation
declares it — and an axis another solver declares is refused on a model whose
programs never mention it.
"""

import functools
from collections.abc import Mapping

import pytest

from _lcm.execution.core_program import (
    ReducedAxis,
    TiledOutputAxis,
    core_program_graph,
)
from _lcm.execution.workspace_planning import workspace_width_candidates
from _lcm.simulation.programs import SUBJECT_AXIS
from lcm import ExecutionConfig, Model
from lcm.exceptions import ExecutionPlanningError
from lcm.solvers import (
    ACTION_PRODUCT_AXIS,
    CELL_AXIS,
    ENVELOPE_CELL_AXIS,
    EULER_POINT_AXIS,
    OUTER_CANDIDATE_AXIS,
    SAVINGS_POINT_AXIS,
    STOCHASTIC_NODE_AXIS,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution.test_dcegm_axis_width_policy import _model as _dcegm_model
from tests.test_models import n_nbegm_toy

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

# The axis names each in-tree solver family declares, as a reader can check
# against the constants next to each solver's program. The scan test below pins
# this table to what the programs actually declare.
_DECLARED_AXIS_NAMES = {
    "grid_search": (ACTION_PRODUCT_AXIS,),
    "dcegm": (
        CELL_AXIS,
        ENVELOPE_CELL_AXIS,
        EULER_POINT_AXIS,
        SAVINGS_POINT_AXIS,
        STOCHASTIC_NODE_AXIS,
    ),
    "negm": (CELL_AXIS, ENVELOPE_CELL_AXIS, OUTER_CANDIDATE_AXIS, SAVINGS_POINT_AXIS),
    "nnbegm": (),
}
_HOST_AXIS_NAMES = {"nnbegm": (OUTER_CANDIDATE_AXIS,)}

# Forward simulation declares the same two axes whatever the regime is solved
# with: it tiles the subject population, and its decision streams the same
# action product the solve's own classification streams.
_DECLARED_SIMULATION_AXIS_NAMES = (ACTION_PRODUCT_AXIS, SUBJECT_AXIS)


def _legal_axis_names(*, family: str) -> list[str]:
    """Return every axis name one family's model accepts a width for."""
    return sorted(
        {
            *_DECLARED_AXIS_NAMES[family],
            *_HOST_AXIS_NAMES.get(family, ()),
            *_DECLARED_SIMULATION_AXIS_NAMES,
        }
    )


def _build(*, family: str, execution_config: ExecutionConfig) -> Model:
    """Build one family's model under `execution_config`."""
    if family == "dcegm":
        return _dcegm_model(execution_config=execution_config)
    variant = {"grid_search": "brute", "negm": "negm", "nnbegm": "n_nbegm"}[family]
    return n_nbegm_toy.build_model(
        variant=variant, n_periods=2, execution_config=execution_config
    )


@functools.cache
def _reference(*, family: str) -> Model:
    """Build one family's model with nothing declared."""
    return _build(family=family, execution_config=ExecutionConfig())


def _declared_axes(*, model: Model) -> Mapping[str, ReducedAxis | TiledOutputAxis]:
    """Return every axis the model's solve programs declare, by name."""
    found: dict[str, ReducedAxis | TiledOutputAxis] = {}
    for regime in model._regimes.values():
        for kernel in regime.solution.period_kernels.values():
            for program in core_program_graph(kernel=kernel).values():
                found.update({axis.name: axis for axis in program.requirements.axes})
    return found


def _declared_simulation_axis_names(*, model: Model) -> tuple[str, ...]:
    """Return every axis the model's simulation programs declare, by name."""
    return tuple(
        sorted(
            {
                name
                for regime in model._regimes.values()
                for name in regime.simulation.programs.declared_axis_names
            }
        )
    )


@pytest.mark.parametrize("family", sorted(_DECLARED_AXIS_NAMES))
def test_the_declared_names_are_what_the_programs_declare(*, family: str) -> None:
    """The table above is what the family's own core programs declare."""
    names = {
        name
        for regime in _reference(family=family)._regimes.values()
        for kernel in regime.solution.period_kernels.values()
        for program in core_program_graph(kernel=kernel).values()
        for name in program.requirements.axis_names
    }
    assert sorted(names) == sorted(
        (*_DECLARED_AXIS_NAMES[family], *_HOST_AXIS_NAMES.get(family, ()))
    )


@pytest.mark.parametrize(
    ("family", "axis"),
    [
        (family, axis)
        for family in sorted(_DECLARED_AXIS_NAMES)
        for axis in sorted(_DECLARED_AXIS_NAMES[family])
    ],
)
def test_a_declared_axis_width_reaches_the_plan(*, family: str, axis: str) -> None:
    """Naming an axis the model declares fixes the width the planner selects.

    The width asked for is the axis's whole extent, which the plan never picks
    on its own — an unbudgeted plan streams at the largest power of two strictly
    below the extent — so a width that was carried but never read would fail.
    """
    extent = _declared_axes(model=_reference(family=family))[axis].extent
    model = _build(
        family=family, execution_config=ExecutionConfig(axis_widths={axis: extent})
    )

    (candidate,) = workspace_width_candidates(
        axes=(_declared_axes(model=model)[axis],),
        fixed_widths=model._execution.axis_widths,
    )

    assert candidate[axis] == extent


@pytest.mark.parametrize("family", sorted(_DECLARED_AXIS_NAMES))
def test_the_simulation_names_are_what_the_simulation_programs_declare(
    *, family: str
) -> None:
    """The simulation table above is what every regime's own programs declare."""
    assert _declared_simulation_axis_names(model=_reference(family=family)) == tuple(
        sorted(_DECLARED_SIMULATION_AXIS_NAMES)
    )


@pytest.mark.parametrize("family", sorted(_DECLARED_AXIS_NAMES))
@pytest.mark.parametrize("width", [1, 3])
def test_a_subject_width_is_an_accepted_execution_axis(
    *, family: str, width: int
) -> None:
    """A model accepts a width for the subject axis its simulation declares."""
    model = _build(
        family=family,
        execution_config=ExecutionConfig(axis_widths={SUBJECT_AXIS: width}),
    )

    assert model._execution.axis_widths[SUBJECT_AXIS] == width


@pytest.mark.parametrize("family", sorted(_DECLARED_AXIS_NAMES))
def test_a_name_no_program_declares_is_refused(*, family: str) -> None:
    """The refusal lists exactly the axis names this model's programs declare."""
    with pytest.raises(ExecutionPlanningError) as refusal:
        _build(
            family=family,
            execution_config=ExecutionConfig(axis_widths={"not_an_axis": 4}),
        )

    assert f"declared axes are {_legal_axis_names(family=family)!r}." in str(
        refusal.value
    )


@pytest.mark.parametrize("family", sorted(_DECLARED_AXIS_NAMES))
def test_a_refusal_names_the_subject_axis_among_the_declared_ones(
    *, family: str
) -> None:
    """A near-miss on the subject axis is told the name the programs do declare."""
    with pytest.raises(ExecutionPlanningError, match=f"'{SUBJECT_AXIS}'"):
        _build(
            family=family,
            execution_config=ExecutionConfig(axis_widths={"subjects": 4}),
        )


def test_an_axis_another_solver_declares_is_refused() -> None:
    """A DC-EGM axis name is not legal on a model that runs no DC-EGM regime."""
    with pytest.raises(
        ExecutionPlanningError, match=f"axis_widths names {CELL_AXIS!r}"
    ):
        _build(
            family="grid_search",
            execution_config=ExecutionConfig(axis_widths={CELL_AXIS: 4}),
        )
