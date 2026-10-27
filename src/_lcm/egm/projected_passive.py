"""Original deterministic passive interpolation of already aggregated U rows."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from _lcm.continuation import ProjectedProducerDate
from _lcm.egm.projected_child import (
    read_projected_coordinate,
    require_projected_carry,
)
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.solution.feasibility import (
    BoundedResult,
    QueryInputStatus,
    reduce_continuation_read_counts,
    reduce_query_input_status,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import Float1D, RegimeName, ScalarFloat


@dataclass(frozen=True, kw_only=True, eq=False)
class ProjectedPassiveReader:
    """The actual child row axis and its original economic interpolation rule."""

    date: ProjectedProducerDate
    target: RegimeName
    grid: Float1D

    def read(
        self,
        *,
        carry: ProjectedEGMCarry,
        coordinate: ScalarFloat,
        jacobian: ScalarFloat,
        passive: ScalarFloat,
        collect_read_counts: bool = False,
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        """Read original rows before blending; never differentiate stored values."""
        from _lcm.egm.continuation import (  # noqa: PLC0415 - preserve original declaration import order
            _blend_passive_axes,
        )
        from _lcm.egm.dcegm_query import _passive_weights  # noqa: PLC0415

        if type(carry) is not ProjectedEGMCarry or self.date.guard_program is not None:
            raise UnsupportedOperationError(
                "Passive projected reads require original exact U rows."
            )
        require_projected_carry(
            date=self.date,
            target=self.target,
            carry=carry,
            passive_shape=(self.grid.shape[0],),
        )
        _, active, _ = _passive_weights(values=(passive,), grids=(self.grid,))
        rows = jax.vmap(
            lambda row: read_projected_coordinate(
                authority=self.date.authority,
                carry=row,
                guard_program=None,
                query=coordinate,
                jacobian=jacobian,
                collect_read_counts=collect_read_counts,
            )
        )(carry)
        values, marginals = rows.output
        value, marginal = _blend_passive_axes(
            value_at_child=values,
            marginal_arrays=(marginals,),
            child_passive_values=(passive,),
            child_passive_grids=(self.grid,),
            n_outer_candidates=0,
        )
        statuses = reduce_query_input_status(statuses=rows.query_status, active=active)
        if statuses is None or len(statuses) != 1:
            raise TypeError("Passive projected read lost its original status slot.")
        status = statuses[0]
        finite = (
            jnp.isfinite(passive)
            & jnp.isfinite(coordinate)
            & jnp.isfinite(jacobian)
            & (jnp.isfinite(value) | (jnp.isneginf(value) & (marginal == 0)))
            & jnp.isfinite(marginal)
        )
        return BoundedResult(
            output=(value, marginal),
            bound_status=(),
            query_status=(
                QueryInputStatus(
                    status.nonfinite_floor,
                    status.invalid_shifted_grid,
                    status.coverage_missing | ~finite | ~jnp.all(carry.valid),
                ),
            ),
            continuation_read_counts=reduce_continuation_read_counts(
                counts=rows.continuation_read_counts, active=active
            ),
        )
