"""One declared regime-period of a built model compiles without a solve.

Diagnosing whether a core's graph can be lowered, compiled and admitted at a
chosen set of widths otherwise costs a whole backward induction, and a solve
that dies on memory cannot say whether the graph was the problem or the width
search was. Compiling one declared position at fixed widths separates them, and
compiles nothing else.
"""

import re
from types import MappingProxyType, SimpleNamespace

import jax
import pytest

from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    ReducedAxis,
    TiledOutputAxis,
)
from _lcm.execution.workspace_planning import bootstrap_width, bootstrap_widths
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.period_replay import _project_axis_widths
from lcm import Model
from tests.test_models.deterministic.discrete import get_model, get_params


def _reduced_axis() -> ReducedAxis:
    """A reduced axis wide enough that its bootstrap width is below its extent."""
    return ReducedAxis(
        name="action_product",
        coordinate_names=("consumption", "labor"),
        coordinate_extents=(64, 8),
        canonical_order="c",
        reduction=HARD_MAX_REDUCTION,
        width_keyword="_lcm_action_product_width",
    )


def _tiled_axis() -> TiledOutputAxis:
    """A tiled output axis whose extent leaves room above the fixed width below."""
    return TiledOutputAxis(
        name="cell",
        state_names=("wealth",),
        extent=4096,
        width_keyword="_lcm_cell_width",
    )


_N_PERIODS = 3

# Every axis the toy's cores declare then takes its bootstrap width, which is
# what the unbudgeted route lowers; the tests below pin both the fallback and
# the refusal of a width that binds nothing.
_NO_WIDTHS: dict[str, int] = {}


@pytest.fixture
def model() -> Model:
    """The fully discrete toy, built but never solved."""
    return get_model(n_periods=_N_PERIODS)


def test_a_declared_position_compiles_every_core_its_kernel_publishes(
    model: Model,
) -> None:
    """The result names one compilation per core of that regime-period."""
    compilations = model._compile_period_cores(
        params=get_params(n_periods=_N_PERIODS),
        regime_name="working_life",
        period=1,
        axis_widths=_NO_WIDTHS,
    )
    assert [compiled.core_name for compiled in compilations] == ["main"]


def test_a_compilation_carries_the_age_its_period_sits_at(model: Model) -> None:
    """The age is what a declared position is read against in a solve log."""
    (compiled,) = model._compile_period_cores(
        params=get_params(n_periods=_N_PERIODS),
        regime_name="working_life",
        period=1,
        axis_widths=_NO_WIDTHS,
    )
    assert compiled.age == float(model.ages.values[1])


def test_a_compilation_reports_a_positive_compile_wall(model: Model) -> None:
    """A wall of zero would mean the backend compilation was never timed."""
    (compiled,) = model._compile_period_cores(
        params=get_params(n_periods=_N_PERIODS),
        regime_name="working_life",
        period=1,
        axis_widths=_NO_WIDTHS,
    )
    assert compiled.compile_seconds > 0.0


def test_a_compilation_reports_a_positive_lowering_wall(model: Model) -> None:
    """Lowering and compiling are reported apart, so one cost is attributable."""
    (compiled,) = model._compile_period_cores(
        params=get_params(n_periods=_N_PERIODS),
        regime_name="working_life",
        period=1,
        axis_widths=_NO_WIDTHS,
    )
    assert compiled.lowering_seconds > 0.0


def test_compiling_a_declared_position_dispatches_nothing(
    *, model: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An abstract compilation must never become a runtime claim."""

    def refuse(self: object, *args: object, **kwargs: object) -> object:  # noqa: ARG001
        pytest.fail("A compile-only call reached numerical dispatch.")

    monkeypatch.setattr(jax.stages.Compiled, "__call__", refuse)
    compilations = model._compile_period_cores(
        params=get_params(n_periods=_N_PERIODS),
        regime_name="working_life",
        period=1,
        axis_widths=_NO_WIDTHS,
    )
    assert len(compilations) == 1


def test_an_unknown_regime_is_refused_by_name(model: Model) -> None:
    """A position naming a regime the model does not declare is a typo."""
    with pytest.raises(ValueError, match="no_such_regime"):
        model._compile_period_cores(
            params=get_params(n_periods=_N_PERIODS),
            regime_name="no_such_regime",
            period=1,
            axis_widths=_NO_WIDTHS,
        )


def test_a_period_the_regime_is_inactive_in_is_refused(model: Model) -> None:
    """A regime has no kernel outside its activity window."""
    with pytest.raises(ValueError, match=re.escape("period 9")):
        model._compile_period_cores(
            params=get_params(n_periods=_N_PERIODS),
            regime_name="working_life",
            period=9,
            axis_widths=_NO_WIDTHS,
        )


def test_a_width_that_binds_no_axis_is_refused(model: Model) -> None:
    """A width naming an axis no core declares is a typo, not a no-op."""
    with pytest.raises(ValueError, match="no execution axis"):
        model._compile_period_cores(
            params=get_params(n_periods=_N_PERIODS),
            regime_name="working_life",
            period=1,
            axis_widths={"no_such_axis": 32},
        )


def test_an_omitted_axis_takes_its_bootstrap_width() -> None:
    """An axis the caller leaves out is bound where the unbudgeted route binds it."""
    axes = (_reduced_axis(), _tiled_axis())
    program = SimpleNamespace(
        name="main",
        requirements=CoreExecutionRequirements(
            reduced_axes=(axes[0],), tiled_axes=(axes[1],)
        ),
    )
    projected = _project_axis_widths(program=program, axis_widths={})
    assert dict(projected) == dict(
        bootstrap_widths(axes=axes, fixed_widths=MappingProxyType({}))
    )


def test_a_named_axis_overrides_its_bootstrap_width() -> None:
    """A fixed width is bound as given while its neighbours stay at bootstrap."""
    reduced, tiled = _reduced_axis(), _tiled_axis()
    program = SimpleNamespace(
        name="main",
        requirements=CoreExecutionRequirements(
            reduced_axes=(reduced,), tiled_axes=(tiled,)
        ),
    )
    projected = _project_axis_widths(program=program, axis_widths={"cell": 8})
    assert dict(projected) == {
        "action_product": bootstrap_width(extent=reduced.extent),
        "cell": 8,
    }


def test_a_fixed_width_reaches_the_compiled_core(model: Model) -> None:
    """The width a caller fixes is the width that core is compiled at."""
    (compiled,) = model._compile_period_cores(
        params=get_params(n_periods=_N_PERIODS),
        regime_name="working_life",
        period=1,
        axis_widths={"cell": 2},
    )
    assert compiled.tile_widths["cell"] == 2
