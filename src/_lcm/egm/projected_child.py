"""Model-bound pre-shock child reads with explicit economic chain factors."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from _lcm.continuation import ProjectedContinuationSpec, ProjectedProducerDate
from _lcm.egm.projected_continuation import (
    ProjectedChildAuthority,
    ProjectedEGMCarry,
    ProjectedGuardProgram,
    read_projected_row,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationReadCounts,
    QueryInputStatus,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.solver_api import ContinuationCapabilities
from lcm.typing import RegimeName, ScalarFloat


@dataclass(frozen=True, kw_only=True, eq=False)
class ProjectedChildReader:
    """One actual projected child, bound to its source next-state law.

    The child has already integrated its current shock and its choices. This
    read performs neither operation again. The parent's actual next-state law
    supplies the pre-shock coordinate and its Jacobian; the child marginal is
    the published economic channel rather than AD of an interpolated value.
    """

    authority: ProjectedChildAuthority
    """Exact producer date/key/domain/capacity from the installed declaration."""
    carry: ProjectedEGMCarry
    """Original native input arrays acquired under the producer's artifact key."""
    next_liquid: Callable[[ScalarFloat], ScalarFloat]
    """Actual source next-state law with original parameters/context bound."""
    collect_read_counts: bool = False
    """Whether this actual paired read reports its independently eligible lane."""

    guard_program: ProjectedGuardProgram | None = None
    """Original producer-bound callback program; guard data stays numerical."""

    @classmethod
    def bind(
        cls,
        *,
        spec: ProjectedContinuationSpec,
        target: RegimeName,
        period: int,
        carry: ProjectedEGMCarry,
        next_liquid: Callable[[ScalarFloat], ScalarFloat],
        collect_read_counts: bool = False,
    ) -> ProjectedChildReader:
        """Authenticate the actual template and date before installing this read."""
        return cls.bind_date(
            date=spec.date_at(period=period),
            target=target,
            carry=carry,
            next_liquid=next_liquid,
            collect_read_counts=collect_read_counts,
        )

    @classmethod
    def bind_date(
        cls,
        *,
        date: ProjectedProducerDate,
        target: RegimeName,
        carry: ProjectedEGMCarry,
        next_liquid: Callable[[ScalarFloat], ScalarFloat],
        collect_read_counts: bool = False,
    ) -> ProjectedChildReader:
        """Authenticate a prepared date before any aggregate spec is assembled."""
        authority = require_projected_carry(date=date, target=target, carry=carry)
        return cls(
            authority=authority,
            carry=carry,
            next_liquid=next_liquid,
            collect_read_counts=collect_read_counts,
            guard_program=date.guard_program,
        )

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Expose the actual physical liquid marginal, without relabelling it."""
        return ContinuationCapabilities(
            value=True,
            marginal_states=frozenset({self.authority.liquid_state}),
            exact_candidate_identity=False,
            discontinuities=False,
        )

    def __call__(
        self, savings: ScalarFloat
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        """Read mandatory integrity before the parent applies any discount."""
        query = self.next_liquid(savings)
        jacobian = jax.grad(self.next_liquid)(savings)
        return self.read_coordinate(query=query, jacobian=jacobian)

    def read_coordinate(
        self, *, query: ScalarFloat, jacobian: ScalarFloat
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        """Compose a declared coordinate direction with the guarded economic row."""
        return read_projected_coordinate(
            authority=self.authority,
            carry=self.carry,
            guard_program=self.guard_program,
            query=query,
            jacobian=jacobian,
            collect_read_counts=self.collect_read_counts,
        )


def require_projected_carry(
    *,
    date: ProjectedProducerDate,
    target: RegimeName,
    carry: ProjectedEGMCarry,
    passive_shape: tuple[int, ...] = (),
) -> ProjectedChildAuthority:
    """Authenticate the actual dated numerical schema for a coordinate reader."""
    authority = date.authority
    if target != authority.producer:
        raise TypeError("Projected child target differs from its actual producer.")
    template = date.template
    if not isinstance(template, ProjectedEGMCarry):
        raise TypeError("Projected child requires its actual typed template.")
    if carry.grid.shape[:-1] != passive_shape:
        raise UnsupportedOperationError(
            "Projected child passive/stochastic row selection is not installed."
        )
    if carry.guard_required != (date.guard_program is not None):
        raise TypeError(
            "Projected child guard program disagrees with its numeric payload."
        )
    if jax.tree.structure(carry) != jax.tree.structure(template):
        raise TypeError("Projected child guard differs from its actual template.")
    for actual, expected in zip(
        jax.tree.leaves(carry), jax.tree.leaves(template), strict=True
    ):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise TypeError("Projected child input differs from its producer schema.")
    return authority


def read_projected_coordinate(
    *,
    authority: ProjectedChildAuthority,
    carry: ProjectedEGMCarry,
    guard_program: ProjectedGuardProgram | None,
    query: ScalarFloat,
    jacobian: ScalarFloat,
    collect_read_counts: bool = False,
) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
    """Read stored economics and the same producer guard at an actual coordinate."""
    row = carry
    read = read_projected_row(
        query=query,
        grid=row.grid,
        value=row.value,
        economic_marginal=row.economic_marginal,
        owner_id=row.owner_id,
        side_id=row.side_id,
        node_kind=row.node_kind,
        link_valid=row.link_valid,
        valid_length=row.valid_length,
        domain_lower=row.domain_lower,
        domain_upper=row.domain_upper,
        valid=row.valid,
        value_slope=row.value_slope_channel,
    )
    if row.guard_required:
        if guard_program is None:
            raise TypeError(
                "Balanced child read lacks its original source guard program."
            )
        guarded = guard_program.validate(carry=row, query=query)
    else:
        if guard_program is not None:
            raise TypeError("Ordinary projected carry cannot borrow a balanced guard.")
        guarded = jnp.ones((), dtype=jnp.bool_)
    if row.guard_required:
        from _lcm.egm.current_iid_balanced import (  # noqa: PLC0415
            BalancedProjectedEGMCarry,
        )
        from _lcm.egm.current_iid_case_u_balanced import (  # noqa: PLC0415
            BalancedCaseUCarry,
        )

        if (
            not isinstance(row, (BalancedProjectedEGMCarry, BalancedCaseUCarry))
            or "period" not in row.guard.pool
        ):
            raise TypeError("Balanced child guard lacks its actual producer clock.")
        guarded = guarded & (row.guard.pool["period"] == authority.period)
    valid = read.valid & guarded & jnp.isfinite(query) & jnp.isfinite(jacobian)
    return BoundedResult(
        output=(read.value, read.economic_marginal * jacobian),
        bound_status=(),
        query_status=(
            QueryInputStatus(
                nonfinite_floor=jnp.zeros((), dtype=jnp.bool_),
                invalid_shifted_grid=jnp.zeros((), dtype=jnp.bool_),
                coverage_missing=~valid,
            ),
        ),
        continuation_read_counts=(
            ContinuationReadCounts(
                below_first_live=(query < row.domain_lower).astype(jnp.int32),
                eligible=jnp.asarray(1, dtype=jnp.int32),
            )
            if collect_read_counts
            else None
        ),
    )
