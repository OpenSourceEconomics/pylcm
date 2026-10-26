"""Shared dated producer declarations for native pre-shock continuations."""

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType

import jax.numpy as jnp

from _lcm.continuation import ProjectedContinuationSpec
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.grids import Grid
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.shock_registry import ShockBinding
from lcm.exceptions import UnsupportedOperationError
from lcm.regime import Regime as UserRegime
from lcm.shocks import ShockGroupName
from lcm.typing import Float1D, Phase, RegimeName, StateName


def _projected_template(*, grid: Float1D, capacity: int) -> ProjectedEGMCarry:
    """Materialize the original shape template on the genuinely supplied domain."""
    prefix = jnp.linspace(grid[0], grid[-1], capacity, dtype=grid.dtype)
    return ProjectedEGMCarry(
        grid=prefix,
        value=jnp.zeros_like(prefix),
        economic_marginal=jnp.zeros_like(prefix),
        owner_id=jnp.zeros((capacity,), dtype=jnp.int32),
        side_id=jnp.zeros((capacity,), dtype=jnp.int32),
        node_kind=jnp.full((capacity,), QueryNodeKind.FINITE, dtype=jnp.int32),
        link_valid=jnp.ones((capacity - 1,), dtype=jnp.bool_),
        valid_length=jnp.asarray(capacity, dtype=jnp.int32),
        domain_lower=grid[0],
        domain_upper=grid[-1],
        valid=jnp.ones((), dtype=jnp.bool_),
    )


def _materialize_projected_template(
    *,
    grid: Float1D,
    capacity: int,
    materializer: DatedGenerationMaterializer | None,
) -> ProjectedEGMCarry:
    """Keep the numerical source identical with explicit input/root custody."""
    if materializer is None:
        return _projected_template(grid=grid, capacity=capacity)
    return materializer.produce(
        function=partial(_projected_template, capacity=capacity),
        arguments={"grid": grid},
    )


def _nb_projection_capacity(
    *,
    state_grid: Grid,
    target: RegimeName,
    period: int,
    liquid_support_by_date: Mapping[tuple[RegimeName, int], Float1D] | None,
    materializer: DatedGenerationMaterializer | None,
) -> int:
    """Read NB's actual original support shape without an unadmitted grid read."""
    if materializer is None:
        return int(state_grid.to_jax().shape[0])
    if liquid_support_by_date is None:
        raise TypeError("NB projection requires actual dated support.")
    return int(liquid_support_by_date[target, period].shape[0])


def projected_producer_declarations(  # noqa: C901, PLR0912 - original solver-specific dated shapes and profile refusal
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    bindings: Mapping[tuple[RegimeName, Phase, ShockGroupName], ShockBinding],
    state_grids: Mapping[RegimeName, Mapping[StateName, Grid]],
    active_periods: Mapping[RegimeName, tuple[int, ...]],
    liquid_support_by_date: Mapping[tuple[RegimeName, int], Float1D] | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> Mapping[RegimeName, ProjectedContinuationSpec]:
    """Declare the actual target schema before any parent's core is constructed.

    The same declaration object is reused by the genuine producer. The existing
    declare-continuation-reads pass checks that identity once all producers
    exist; no provisional DCEGMQuery or borrowing authority is constructed.
    Numerical template placeholders provide shapes, not coverage evidence.
    """
    from _lcm.solution.dcegm import _BoundDCEGM  # noqa: PLC0415
    from _lcm.solution.nbegm import _BoundNBEGM  # noqa: PLC0415

    if materializer is not None and liquid_support_by_date is None:
        raise TypeError(
            "Admitted projected declarations require actual call-owned liquid support."
        )
    targets = frozenset(owner for owner, phase, _ in bindings if phase == "solve")
    declarations = {}
    for target in targets:
        user_regime = user_regimes[target]
        solver = user_regime.solver
        if not isinstance(solver, (_BoundDCEGM, _BoundNBEGM)):
            continue
        if isinstance(solver, _BoundDCEGM) and solver.accuracy == "certified":
            raise UnsupportedOperationError(
                "Certified current-IID EGM publication requires its complete "
                "profile implementation."
            )
        if isinstance(solver, _BoundNBEGM):
            if not active_periods[target]:
                continue
            capacity = _nb_projection_capacity(
                state_grid=state_grids[target][solver.continuous_state],
                target=target,
                period=active_periods[target][0],
                liquid_support_by_date=liquid_support_by_date,
                materializer=materializer,
            )
        else:
            capacity = solver.current_iid_projection_capacity
        if liquid_support_by_date is None:
            grid = jnp.asarray(
                state_grids[target][solver.continuous_state].to_jax(),
                dtype=canonical_float_dtype(),
            )
            template = _projected_template(grid=grid, capacity=capacity)
            periods = active_periods[target]
            if not periods:
                continue
            templates = dict.fromkeys(periods, template)
        else:
            periods = active_periods[target]
            if not periods:
                continue
            templates = {}
        if liquid_support_by_date is not None:
            missing = {(target, period) for period in periods} - set(
                liquid_support_by_date
            )
            if missing:
                raise TypeError(
                    "Projected declaration lacks actual producer-date support: "
                    f"{sorted(missing)!r}."
                )
            templates = {
                period: _materialize_projected_template(
                    grid=liquid_support_by_date[target, period],
                    capacity=capacity,
                    materializer=materializer,
                )
                for period in periods
            }
            template = templates[periods[0]]
        declarations[target] = ProjectedContinuationSpec(
            template=template,
            producer=target,
            liquid_state=solver.continuous_state,
            templates_by_period=templates,
        )
    return MappingProxyType(declarations)
