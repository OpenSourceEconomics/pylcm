"""Original grid construction stages admitted by the existing entry owner.

Declarations already exist when this boundary is entered. Their actual buffers
must be retained by the owner before the first stage. This module neither creates
a Grid surrogate nor changes the coordinate arithmetic or endpoint ownership.
"""

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.grids import (
    DiscreteGrid,
    Grid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)
from _lcm.processes.base import _ContinuousStochasticProcess
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import Float1D, Int1D

if TYPE_CHECKING:
    from _lcm.materialization import DatedGenerationMaterializer
else:
    # The owner protocol imports the grid declarations during runtime initialization.
    type DatedGenerationMaterializer = object


def admitted_grid_extent(*, spec: Grid) -> int:
    """Read original built-in dimensions without calling a numerical producer."""
    if type(spec) is DiscreteGrid:
        return len(spec.categories)
    if isinstance(spec, PiecewiseLinSpacedGrid | PiecewiseLogSpacedGrid) and type(
        spec
    ) in (PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid):
        counts = tuple(int(np.asarray(n)) for n in spec.points_per_segment)
        count = sum(counts)
        if count > np.iinfo(np.int32).max:
            raise ExecutionPlanningError(
                "Admitted piecewise grid extent exceeds the original int32 axis."
            )
        return count
    if isinstance(spec, LinSpacedGrid | LogSpacedGrid | IrregSpacedGrid) and type(
        spec
    ) in (LinSpacedGrid, LogSpacedGrid, IrregSpacedGrid):
        return int(np.asarray(spec.n_points))
    if isinstance(spec, _ContinuousStochasticProcess):
        # Process dimensions are plain declaration metadata; .params is not.
        return int(spec.n_points)
    raise ExecutionPlanningError(
        f"Cold dated grid {type(spec).__qualname__} has no admitted original "
        "construction route."
    )


def _grid_linspace(*, start: jax.Array, stop: jax.Array, n_points: int) -> jax.Array:
    """Original single linspace stage with the same represented endpoints."""
    return jnp.linspace(start, stop, n_points)


def _grid_log(*, value: jax.Array) -> jax.Array:
    """Original separately rounded logarithm stage."""
    return jnp.log(value)


def _grid_logspace(*, start: jax.Array, stop: jax.Array, n_points: int) -> jax.Array:
    """Original logspace call, without absorbing its eager logarithms or scatters."""
    return jnp.logspace(start, stop, n_points, base=jnp.e)


def _grid_set_endpoint(*, grid: jax.Array, value: jax.Array, index: int) -> jax.Array:
    """Original separate endpoint update."""
    return grid.at[index].set(value)


def _grid_index(*, array: jax.Array, index: int) -> jax.Array:
    """Original piecewise scalar indexing operation."""
    return array[index]


def _grid_concatenate(*, segments: tuple[jax.Array, ...]) -> jax.Array:
    """Original piecewise concatenation."""
    return jnp.concatenate(segments)


def _grid_placeholder(*, n_points: int) -> jax.Array:
    """Original NaN support placeholder; it is not a resolved support."""
    return jnp.full(n_points, jnp.nan)


def _uniform_points(
    *,
    owner: DatedGenerationMaterializer,
    start: jax.Array,
    stop: jax.Array,
    n_points: int,
    logarithmic: bool,
) -> jax.Array:
    if not logarithmic:
        return owner.produce(
            function=partial(_grid_linspace, n_points=n_points),
            arguments={"start": start, "stop": stop},
        )
    start_linear = owner.produce(function=_grid_log, arguments={"value": start})
    stop_linear = owner.produce(function=_grid_log, arguments={"value": stop})
    grid = owner.produce(
        function=partial(_grid_logspace, n_points=n_points),
        arguments={"start": start_linear, "stop": stop_linear},
    )
    grid = owner.produce(
        function=partial(_grid_set_endpoint, index=0),
        arguments={"grid": grid, "value": start},
    )
    return owner.produce(
        function=partial(_grid_set_endpoint, index=-1),
        arguments={"grid": grid, "value": stop},
    )


def materialize_original_grid(
    *, owner: DatedGenerationMaterializer, spec: Grid, placeholder: bool
) -> Float1D | Int1D:
    """Run only original built-in factories after their source roots are retained.

    Continuous process supports use the owner's existing process resolver, not
    this module. Custom Grid factories remain unsupported before invocation.
    """
    count = admitted_grid_extent(spec=spec)
    if type(spec) is DiscreteGrid:
        return owner(
            value=np.asarray(spec.codes, dtype=np.int32),
            dtype=np.dtype(np.int32),
            name="dated_grid_codes",
        )
    if type(spec) is IrregSpacedGrid:
        if spec.pass_points_at_runtime:
            if not placeholder:
                # Preserve its original diagnostic without a numerical allocation.
                return spec.to_jax()
            return owner.produce(
                function=partial(_grid_placeholder, n_points=count), arguments={}
            )
        points = spec.to_jax()
        return owner(value=points, dtype=points.dtype, name="dated_grid_points")
    if isinstance(spec, LinSpacedGrid | LogSpacedGrid) and type(spec) in (
        LinSpacedGrid,
        LogSpacedGrid,
    ):
        return _uniform_points(
            owner=owner,
            start=spec.start,
            stop=spec.stop,
            n_points=count,
            logarithmic=type(spec) is LogSpacedGrid,
        )
    if isinstance(spec, PiecewiseLinSpacedGrid | PiecewiseLogSpacedGrid) and type(
        spec
    ) in (PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid):
        segments = []
        for index, n in enumerate(spec.points_per_segment):
            start = owner.produce(
                function=partial(_grid_index, index=index),
                arguments={"array": spec._segment_starts},  # noqa: SLF001 - original stored geometry
            )
            stop = owner.produce(
                function=partial(_grid_index, index=index),
                arguments={"array": spec._segment_stops},  # noqa: SLF001 - original stored geometry
            )
            segments.append(
                _uniform_points(
                    owner=owner,
                    start=start,
                    stop=stop,
                    n_points=int(np.asarray(n)),
                    logarithmic=type(spec) is PiecewiseLogSpacedGrid,
                )
            )
        return owner.produce(
            function=_grid_concatenate, arguments={"segments": tuple(segments)}
        )
    raise ExecutionPlanningError(
        "Continuous processes require the existing admitted process support owner."
    )
