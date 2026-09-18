"""Outcome-space grids carry no execution or hardware-placement settings."""

import functools
from collections.abc import Callable

import pytest

from _lcm.grids.base import Grid
from _lcm.solution import fingerprint
from lcm import (
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    NormalIIDProcess,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
)
from lcm.typing import ScalarInt


@categorical(ordered=False)
class _Kind:
    a: ScalarInt
    b: ScalarInt


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            functools.partial(DiscreteGrid, category_class=_Kind), id="discrete"
        ),
        pytest.param(
            functools.partial(LinSpacedGrid, start=0.0, stop=1.0, n_points=3),
            id="continuous",
        ),
        pytest.param(
            functools.partial(LogSpacedGrid, start=1.0, stop=2.0, n_points=3),
            id="log",
        ),
        pytest.param(functools.partial(IrregSpacedGrid, n_points=3), id="runtime"),
        pytest.param(
            functools.partial(
                PiecewiseLinSpacedGrid,
                start=0.0,
                stop=1.0,
                breakpoints=(),
                points_per_segment=(3,),
            ),
            id="piecewise-linear",
        ),
        pytest.param(
            functools.partial(
                PiecewiseLogSpacedGrid,
                start=1.0,
                stop=2.0,
                breakpoints=(),
                points_per_segment=(3,),
            ),
            id="piecewise-log",
        ),
        pytest.param(functools.partial(NormalIIDProcess, n_points=3), id="process"),
    ],
)
@pytest.mark.parametrize(
    ("field", "value"), [("batch_size", 1), ("distributed", False)]
)
def test_grids_refuse_execution_fields(
    *, factory: Callable[..., Grid], field: str, value: int | bool
) -> None:
    """Even inert hardware settings are absent from grid constructors."""
    with pytest.raises(TypeError, match=field):
        factory(**{field: value})


@pytest.mark.parametrize(
    "name",
    ["_GRID_EXECUTION_FIELDS", "_BUILTIN_EXECUTION_FIELDS_BY_TYPE", "_exclude_field"],
)
def test_every_declared_grid_and_solver_field_enters_the_fingerprint(
    *, name: str
) -> None:
    """Fingerprinting needs no execution-field exclusions once those fields are gone."""
    assert not hasattr(fingerprint, name)
