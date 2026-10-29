"""Bounded pre-shock projection of one canonical owned resources candidate row."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.egm.current_iid_balanced import (
    BalancedProjectedEGMCarry,
    BalancedResourcesGuard,
    BalancedResourcesProgram,
    ResourcesRegionCatalog,
    _ordered_points,
    build_resources_catalog,
)
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluator
from _lcm.egm.current_iid_pullback import pullback_resource_boundary
from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from _lcm.egm.current_iid_topology import (
    CommonResourcesPartition,
    pullback_native_partition,
    require_resources_topology,
)
from _lcm.egm.projected_continuation import (
    ProjectedChildAuthority,
    ProjectedEGMCarry,
    ProjectedRowRead,
)
from _lcm.egm.query_data import QueryNodeKind
from _lcm.probability import is_represented_zero
from _lcm.typing import EconFunctionKwargs
from lcm.shocks import ShockName
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class ProjectedResourcesReference:
    """Invocation-local economics bound to the actual native union partition."""

    evaluator: CurrentResourcesEvaluator
    """Original owned row and the exact structurally certified resources map."""
    physical: Mapping[ShockName, FloatND | IntND]
    """Current quadrature realizations, temporary rather than durable carry axes."""
    factors: FloatND
    """Original factored joint weights, retaining represented tiny probabilities."""
    law_valid: ScalarBool
    """Mandatory admitted-law and support summary, before economic masking."""
    partition: CommonResourcesPartition
    """Actual native boundary pullbacks in the common liquid coordinate."""

    def __call__(self, liquid: ScalarFloat) -> ProjectedRowRead:
        return self.evaluator.reference(
            liquid=liquid,
            physical=self.physical,
            factors=self.factors,
            law_valid=self.law_valid,
            partition=self.partition.points,
            partition_length=self.partition.length,
        )

    def with_selection(
        self, liquid: ScalarFloat
    ) -> tuple[ProjectedRowRead, OwnedResourcesRead]:
        """Return the selected node tuples used for these same economic channels."""
        return self.evaluator.reference_with_selection(
            liquid=liquid,
            physical=self.physical,
            factors=self.factors,
            law_valid=self.law_valid,
            partition=self.partition.points,
            partition_length=self.partition.length,
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class FiniteResourcesDemands:
    """Fixture-local exact samples and selection evidence, not a native artifact.

    The result retains the actual invocation context rather than a roster hash.
    Node tuples remain separate from the carry and are meaningful only under
    `live`, in original quadrature order. Production parent-demand preparation,
    native evidence storage and replay custody are not installed by this object.
    """

    authority: ProjectedChildAuthority
    """Explicit fixture producer date/key; not production authentication."""
    reference: ProjectedResourcesReference
    """The same original row, physical law frame, weights and bound context."""
    demanded_queries: Float1D
    """Original unpadded dynamic bank; every coordinate is required."""
    samples: ProjectedEGMCarry
    """Exact records with no smooth links or full-domain coverage claim."""
    selected: OwnedResourcesRead
    """Original selected tuples, indexed by sample record then quadrature row."""
    live: BoolND
    """Represented-live quadrature mask; inactive rows are not corner selections."""


def publish_finite_resources_demands(
    *,
    authority: ProjectedChildAuthority,
    evaluator: CurrentResourcesEvaluator,
    physical: Mapping[ShockName, FloatND | IntND],
    factors: FloatND,
    law_valid: ScalarBool,
    liquid_grid: Float1D,
    demanded_queries: Float1D,
) -> FiniteResourcesDemands:
    """Evaluate an explicit private fixture demand bank with unresolved holes.

    Every demand is active; padding is not accepted. Law/source corruption,
    out-of-domain demand or missing required coordinates invalidates the bank.
    The smooth producer and production parent-demand route remain refused.
    """
    live = jnp.all(~is_represented_zero(factors), axis=0)
    reference = ProjectedResourcesReference(
        evaluator=evaluator,
        physical=physical,
        factors=factors,
        law_valid=law_valid,
        partition=pullback_native_partition(
            row=evaluator.row,
            resources=evaluator.resources,
            physical=physical,
            live=live,
            liquid_grid=liquid_grid,
            demanded_queries=demanded_queries,
            capacity=authority.capacity,
        ),
    )
    samples, selected = assemble_exact_resources_samples(
        reference=reference, capacity=authority.capacity
    )
    return FiniteResourcesDemands(
        authority=authority,
        reference=reference,
        demanded_queries=demanded_queries,
        samples=samples,
        selected=selected,
        live=live,
    )


def publish_projected_resources(
    *, reference: ProjectedResourcesReference, capacity: int
) -> ProjectedEGMCarry:
    """Require complete source-selector topology before retaining a child row."""
    del reference, capacity
    require_resources_topology()


def _assemble_resources_samples(
    *, reference: ProjectedResourcesReference, capacity: int
) -> ProjectedEGMCarry:
    """Assemble non-admitted exact snapshots for inspection, without smooth links.

    Each record is an actual represented-coordinate query. Its validity covers
    that exact query only. Predecessor queries are not analytical limits, and
    partition indices are not proofs of original source-candidate ownership.
    This operation does not publish a child continuation or run refinement.
    """
    samples, _ = assemble_exact_resources_samples(
        reference=reference, capacity=capacity
    )
    return samples


def assemble_exact_resources_samples(
    *, reference: ProjectedResourcesReference, capacity: int
) -> tuple[ProjectedEGMCarry, OwnedResourcesRead]:
    """Return only native numeric samples and their same-evaluation node roster.

    The caller supplies a model-closed finite demand set or keeps the result
    fixture-local. Unresolved open cells remain unsupported; this assembly
    neither proves topology nor publishes a Python reference object.
    """
    partition = reference.partition
    required = partition.length
    positions = jnp.arange(capacity)
    active = positions < required
    source = jnp.clip(positions, 0, partition.points.size - 1)
    lower = partition.points[0]
    upper = partition.points[jnp.maximum(partition.length - 1, 0)]
    grid = jnp.where(active, partition.points[source], lower)
    sample, selected = jax.vmap(reference.with_selection)(grid)
    known = jnp.isfinite(sample.economic_marginal) & (
        jnp.isfinite(sample.value)
        | (jnp.isneginf(sample.value) & (sample.economic_marginal == 0.0))
    )
    kinds = jnp.where(
        active & known,
        jnp.where(
            jnp.isneginf(sample.value),
            QueryNodeKind.ECONOMIC_MINUS_INF,
            QueryNodeKind.FINITE,
        ),
        QueryNodeKind.PAD,
    ).astype(jnp.int32)
    samples = ProjectedEGMCarry(
        grid=grid,
        value=jnp.where(active, sample.value, 0.0),
        economic_marginal=jnp.where(active, sample.economic_marginal, 0.0),
        owner_id=jnp.where(active, sample.owner_id, -1),
        side_id=jnp.where(active, sample.side_id, 0),
        node_kind=kinds,
        link_valid=jnp.zeros((capacity - 1,), dtype=jnp.bool_),
        valid_length=jnp.minimum(required, capacity),
        domain_lower=lower,
        domain_upper=upper,
        valid=(
            partition.valid
            & (required <= capacity)
            & jnp.all(~active | (sample.valid & known))
        ),
    )
    return samples, selected


def publish_balanced_resources(
    *,
    reference: ProjectedResourcesReference,
    program: BalancedResourcesProgram,
    pool: EconFunctionKwargs,
    capacity: int,
) -> tuple[BalancedProjectedEGMCarry, ScalarInt]:
    """Publish sampled-accurate Hermite-read links outside uncertainty holes.

    Original declared/demanded coordinates and resource exact records consume
    the existing capacities. There is no silent truncation or capacity growth.

    Returns:
        Tuple of the published row and the least record count its partition and
        refinement needed, which exceeds `capacity` only when capacity refused
        one of them.

    """
    from _lcm.egm.current_iid_projection import refine_projected_row  # noqa: PLC0415

    if not isinstance(reference, ProjectedResourcesReference):
        raise TypeError("Balanced publication requires the original native reference.")
    evaluator = reference.evaluator
    active = jnp.arange(reference.partition.points.size) < reference.partition.length
    required = jnp.where(
        active, reference.partition.points, reference.partition.points[0]
    )
    live = jnp.all(~is_represented_zero(reference.factors), axis=0)

    def resources_nodes(
        inputs: tuple[Mapping[ShockName, ScalarFloat | ScalarInt], ScalarBool],
    ) -> Float1D:
        draw, active_draw = inputs
        return jax.lax.cond(
            active_draw,
            lambda: jax.vmap(lambda liquid: evaluator.resources(liquid, draw))(
                required
            ),
            lambda: jnp.full_like(required, evaluator.row.resources_domain[0]),
        )

    mapped = jax.lax.map(resources_nodes, (reference.physical, live))
    mapped_active = live[:, None] & active[None, :]
    # Inactive law rows are not evaluated as economic proof obligations.
    required_resources = jnp.where(
        mapped_active, mapped, evaluator.row.resources_domain[0]
    ).reshape(-1)
    scale, _, utility_valid = program.utility.coefficients(
        pool=pool, dtype=required.dtype
    )
    catalog = build_resources_catalog(
        evaluator=evaluator, scale=scale, required=required_resources, capacity=capacity
    )
    guard_pool = {
        name: value
        for name, value in pool.items()
        if name in {*program.resources_names, *program.utility.names, "period", "age"}
        and name != program.utility.action_name
    }
    guard = BalancedResourcesGuard.from_native(
        row=evaluator.row,
        pool=MappingProxyType(guard_pool),
        physical=reference.physical,
        factors=reference.factors,
        law_valid=reference.law_valid & utility_valid,
        resources_slope=evaluator.resources_slope,
        catalog=catalog,
    )
    partition = _pullback_balanced_partition(
        reference=reference, catalog=catalog, live=live, capacity=capacity
    )
    exact_reference = replace(reference, partition=partition)
    initial, _ = assemble_exact_resources_samples(
        reference=exact_reference, capacity=capacity
    )
    # This scalar is only the numeric row reader's connectivity address.
    # Links come from the resource catalog, never equality of this address;
    # the original-node guard checks ownership ex post only.
    initial = replace(initial, owner_id=jnp.zeros_like(initial.owner_id))

    # keyword-only-exempt: library-callback=jax.vmap
    def supported_link(left: ScalarFloat, right: ScalarFloat) -> ScalarBool:
        def supported_draw(
            inputs: tuple[Mapping[ShockName, ScalarFloat | ScalarInt], ScalarBool],
        ) -> ScalarBool:
            draw, active_draw = inputs

            def active_support() -> ScalarBool:
                a = evaluator.resources(left, draw)
                b = evaluator.resources(right, draw)
                search = jnp.where(
                    jnp.arange(capacity) < catalog.length, catalog.points, jnp.inf
                )
                index = jnp.clip(
                    jnp.searchsorted(search, a, side="right") - 1, 0, capacity - 2
                )
                # The complete image stays inside one admitted resource cell;
                # its endpoints retain separate exact-node precedence.
                return (
                    (a >= catalog.points[index])
                    & (b <= catalog.points[index + 1])
                    & catalog.smooth[index]
                    & (a <= b)
                )

            return jax.lax.cond(
                active_draw, active_support, lambda: jnp.ones((), dtype=jnp.bool_)
            )

        supported = jax.lax.map(supported_draw, (reference.physical, live))
        return (left < right) & jnp.all(~live | supported)

    links = jax.vmap(supported_link)(initial.grid[:-1], initial.grid[1:])
    initial = replace(
        initial,
        link_valid=(jnp.arange(capacity - 1) < partition.length - 1) & links,
        valid=initial.valid & guard.law_valid,
    )

    # Probes read the reference only; `program.validate_guard` is ex post.
    def probe_reference(liquid: ScalarFloat) -> ProjectedRowRead:
        return exact_reference(liquid)._replace(owner_id=jnp.zeros((), dtype=jnp.int32))

    refined, refined_needed, slopes = refine_projected_row(
        initial=initial,
        reference=probe_reference,
        accuracy="balanced",
        value_slope=lambda liquid: program.value_slope(guard=guard, query=liquid),
    )
    if slopes is None:
        raise TypeError("Case-R refinement lost its value slopes.")
    # Exact-only carries stay valid: every interior read requires its own link.
    return BalancedProjectedEGMCarry(
        **{
            field.name: getattr(refined, field.name)
            for field in fields(ProjectedEGMCarry)
        },
        value_slope=slopes,
        guard=guard,
    ), jnp.maximum(partition.records_needed, refined_needed)


def _pullback_balanced_partition(
    *,
    reference: ProjectedResourcesReference,
    catalog: ResourcesRegionCatalog,
    live: BoolND,
    capacity: int,
) -> CommonResourcesPartition:
    evaluator = reference.evaluator
    active = jnp.arange(reference.partition.points.size) < reference.partition.length
    lower = reference.partition.points[0]
    upper = reference.partition.points[jnp.maximum(reference.partition.length - 1, 0)]

    def one_draw(
        inputs: tuple[Mapping[ShockName, ScalarFloat | ScalarInt], ScalarBool],
    ) -> tuple[Float1D, ScalarBool]:
        draw, live_draw = inputs

        def one_cut(cut: ScalarFloat) -> tuple[Float1D, ScalarBool]:
            def at_liquid(liquid: ScalarFloat) -> ScalarFloat:
                return evaluator.resources(liquid, draw)

            ge = pullback_resource_boundary(
                resources=at_liquid, boundary=cut, lower=lower, upper=upper
            )
            gt = pullback_resource_boundary(
                resources=at_liquid, boundary=cut, lower=lower, upper=upper, strict=True
            )
            points = jnp.stack(
                (
                    jnp.nextafter(ge.point, lower),
                    ge.point,
                    jnp.nextafter(gt.point, lower),
                    gt.point,
                )
            )
            keep = jnp.stack((ge.interior, ge.interior, gt.interior, gt.interior))
            return jnp.where(keep, points, jnp.inf), ge.valid & gt.valid

        def active_draw() -> tuple[Float1D, ScalarBool]:
            points, valid = jax.lax.map(
                one_cut,
                jnp.where(
                    jnp.arange(capacity) < catalog.length,
                    catalog.points,
                    catalog.points[0],
                ),
            )
            retained = jnp.arange(capacity) < catalog.length
            return jnp.where(retained[:, None], points, jnp.inf).reshape(-1), jnp.all(
                ~retained | valid
            )

        return jax.lax.cond(
            live_draw,
            active_draw,
            lambda: (
                jnp.full((4 * capacity,), jnp.inf, dtype=lower.dtype),
                jnp.ones((), dtype=jnp.bool_),
            ),
        )

    cuts, pull_valid = jax.lax.map(one_draw, (reference.physical, live))
    raw = jnp.concatenate((reference.partition.points, cuts.reshape(-1)))
    raw_active = jnp.concatenate((active, jnp.isfinite(cuts.reshape(-1))))
    points, length, retained = _ordered_points(
        raw=raw, active=raw_active, capacity=capacity
    )
    return CommonResourcesPartition(
        points=points,
        length=length,
        valid=reference.partition.valid
        & retained
        & catalog.valid
        & jnp.all(pull_valid),
        # An exhausted merge reports the least count it proves, not the total.
        records_needed=jnp.maximum(
            reference.partition.records_needed,
            jnp.where(retained, length, capacity + 1),
        ),
    )
