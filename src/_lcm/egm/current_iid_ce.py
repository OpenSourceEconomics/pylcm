"""Private economic CE over the original joint lottery.

Accumulate N=sum(w*g(v,p)) and Z=sum(w) before applying h(N/Z,p).
The source channel retains both w*Dg and Dw*g, including null events, and
normalizes as (DN-(N/Z)*DZ)/Z. Only declared g/h and coordinate/law callbacks
are differentiated; Dv is the separately published economic child marginal.
The original continuation class validates savings-independent lottery weights
and CE parameters, so the savings channel differentiates child coordinates.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from _lcm.certainty_equivalent import PowerMean, QuasiArithmeticMean, _args_for
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import (
    ContinuationPlan,
    _ChildCarryReader,
    _get_child_carry_reader,
)
from _lcm.egm.current_iid_ce_nodes import (
    _bind_ce_parameters,
    economic_target_schedule,
    needed_economic_nodes,
    read_economic_node_record,
)
from _lcm.egm.current_iid_economic import _require_scalar_child
from _lcm.egm.current_iid_future import _SourceNodeRead
from _lcm.egm.current_iid_power_bank import power_mean_continuation
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.probability import probabilities_form_distribution
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationReadCounts,
    QueryInputStatus,
    merge_continuation_read_counts,
    reduce_continuation_read_counts,
)
from _lcm.typing import EconFunctionKwargs
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import (
    BoolND,
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarBool,
    ScalarFloat,
)

if TYPE_CHECKING:
    import _lcm.egm.current_iid_case_u

type _CEPartials = tuple[
    ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool
]


@dataclass(frozen=True, kw_only=True, eq=False)
class EconomicCEContinuation:
    """Bind the actual CE and jointly reduce primal and economic directions.

    This private class requires finite needed transforms and inverse arithmetic.
    The stable PowerMean path retains a genuine owner-bound joint scratch bank.
    Zero-payoff limits, passive/choice queries and projected intersections still
    require their corresponding economic capabilities rather than a level JVP.
    """

    plan: ContinuationPlan
    """Actual source plan retaining its declared certainty-equivalent object."""
    pool: EconFunctionKwargs
    """Original source clock, context and flat parameter namespace."""
    direction: Mapping[ReferenceName, ScalarFloat]
    """Source physical and parameter directions, excluding selected savings."""
    children: MappingProxyType[RegimeName, EGMCarry | ProjectedEGMCarry]
    """Original typed child artifacts consumed by the ordinary reader factory."""
    dtype: DTypeLike
    """Actual point precision."""
    stochastic_node_width: int | None = None
    """Existing node dispatch width, shared with the candidate continuation."""
    collect_read_counts: bool = False
    """Existing optional paired-read inventory."""

    # Optional keeps a legal qualified class ForwardRef during cyclic decoration.
    power_mean_owner: Optional[  # noqa: UP045 - preserve the lazy class child
        "_lcm.egm.current_iid_case_u.CaseUPointSolver"  # noqa: UP037
    ] = None
    """Actual point owner for the stable joint layout; not production admission."""

    def __post_init__(self) -> None:
        self.declaration()
        if self.plan.query_targets:
            raise UnsupportedOperationError(
                "Nonlinear economic CE requires raw scalar child capabilities."
            )
        for read in self.plan.child_reads.values():
            _require_scalar_child(read=read)

    def declaration(self) -> QuasiArithmeticMean:
        """Keep the original declaration as the transform and inverse authority."""
        declared = self.plan.certainty_equivalent
        if not isinstance(declared, QuasiArithmeticMean) or type(declared) not in (
            QuasiArithmeticMean,
            PowerMean,
        ):
            raise UnsupportedOperationError(
                "The private economic CE requires its exact declared finite "
                "QuasiArithmeticMean or stable PowerMean; "
                "custom CE capabilities remain required."
            )
        return declared

    def __call__(
        self, savings: ScalarFloat
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat, ScalarFloat]]:
        return self.reduce(savings=savings, savings_channel=True, source_channel=True)

    def reduce(
        self, *, savings: ScalarFloat, savings_channel: bool, source_channel: bool
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat, ScalarFloat]]:
        """Reduce value plus the statically requested economic directions.

        An unrequested direction is neither traced nor validated; its slot is zero.
        """
        declared = self.declaration()
        if type(declared) is PowerMean:
            return power_mean_continuation(
                continuation=self,
                savings=savings,
                declared=declared,
                savings_channel=savings_channel,
                source_channel=source_channel,
            )
        params = _bind_ce_parameters(declared=declared, pool=self.pool)
        parameter_direction = {
            name: self.direction.get(flat, jnp.zeros_like(params[name]))
            for name, flat in declared.flat_param_names.items()
        }
        transform = _EconomicTransform(
            declared=declared,
            params=params,
            parameter_direction=parameter_direction,
            savings_channel=savings_channel,
            source_channel=source_channel,
        )
        names = tuple(self.direction)

        def probabilities(
            values: tuple[ScalarFloat, ...],
        ) -> Mapping[RegimeName, ScalarFloat]:
            return self.plan.compute_regime_transition_probs(
                **{**self.pool, **dict(zip(names, values, strict=True))}
            )

        probabilities_at_source, probability_directions = jax.jvp(
            probabilities,
            (tuple(self.pool[name] for name in names),),
            (tuple(self.direction[name] for name in names),),
        )
        targets = (*self.plan.stateful_targets, *self.plan.scalar_targets)
        valid = probabilities_form_distribution(
            probabilities=(probabilities_at_source[target] for target in targets),
            dtype=self.dtype,
        )
        for parameter in (
            *params.values(),
            *(
                (*parameter_direction.values(), *self.direction.values())
                if source_channel
                else ()
            ),
        ):
            valid &= jnp.all(jnp.isfinite(parameter))
        zero = jnp.zeros_like(savings)
        totals: _CEPartials = (zero, zero, zero, zero, zero, valid)
        counts: ContinuationReadCounts | None = None
        for target in targets:
            if target in self.plan.stateful_targets:
                result = self.target_partials(
                    target=target,
                    savings=savings,
                    transform=transform,
                    probability=probabilities_at_source[target],
                    probability_direction=probability_directions[target],
                )
            else:
                carry = self.children[target]
                if not isinstance(carry, EGMCarry) or carry.value.shape != (1,):
                    raise UnsupportedOperationError(
                        "A nonlinear scalar target requires its actual single-value "
                        "raw carry; other scalar artifact shapes remain required."
                    )
                result = BoundedResult(
                    output=_ce_block_partials(
                        values=carry.value,
                        savings_directions=jnp.zeros(1, dtype=self.dtype),
                        source_directions=jnp.zeros(1, dtype=self.dtype),
                        factors=jnp.asarray(probabilities_at_source[target]).reshape(
                            1, 1
                        ),
                        factor_directions=jnp.asarray(
                            probability_directions[target]
                        ).reshape(1, 1),
                        transform=transform,
                    ),
                    bound_status=(),
                    query_status=None,
                    continuation_read_counts=None,
                )
            totals = _merge_ce_partials(left=totals, right=result.output)
            counts = merge_continuation_read_counts(
                left=counts, right=result.continuation_read_counts
            )
        value, savings_direction, source_direction, valid = _invert_economic_ce(
            declared=declared,
            params=params,
            parameter_direction=parameter_direction,
            partials=totals,
            savings_channel=savings_channel,
            source_channel=source_channel,
        )
        inactive = jnp.zeros((), dtype=jnp.bool_)
        return BoundedResult(
            output=(value, savings_direction, source_direction),
            bound_status=(),
            query_status=(QueryInputStatus(inactive, inactive, ~valid),),
            continuation_read_counts=counts,
        )

    def savings(
        self, savings: ScalarFloat
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        """Supply the same economic CE to original Euler candidate inversion."""
        result = self.reduce(
            savings=savings, savings_channel=True, source_channel=False
        )
        value, marginal, _ = result.output
        return BoundedResult(
            output=(value, marginal),
            bound_status=(),
            query_status=result.query_status,
            continuation_read_counts=result.continuation_read_counts,
        )

    def source(
        self, savings: ScalarFloat
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        """Supply the source envelope after original owned candidate selection."""
        result = self.reduce(
            savings=savings, savings_channel=False, source_channel=True
        )
        value, _, marginal = result.output
        return BoundedResult(
            output=(value, marginal),
            bound_status=(),
            query_status=result.query_status,
            continuation_read_counts=result.continuation_read_counts,
        )

    def target_partials(
        self,
        *,
        target: RegimeName,
        savings: ScalarFloat,
        transform: _EconomicTransform,
        probability: ScalarFloat,
        probability_direction: ScalarFloat,
    ) -> BoundedResult[_CEPartials]:
        """Reduce one target's original nodes without inverting its partial mean."""
        reader = _get_child_carry_reader(
            read=self.plan.child_reads[target],
            carry=self.children[target],
            combo_pool=self.pool,
            post_decision_name=self.plan.post_decision_name,
            stochastic_node_width=self.stochastic_node_width,
            collect_continuation_read_counts=self.collect_read_counts,
        )
        if not isinstance(reader, _ChildCarryReader):
            raise UnsupportedOperationError(
                "Projected nonlinear economic CE capabilities remain required."
            )
        schedule = economic_target_schedule(
            reader=reader,
            direction=self.direction,
            stochastic_node_width=self.stochastic_node_width,
        )
        tangents = schedule.weight_directions
        blocks, active = schedule.blocks, schedule.active
        zero = jnp.zeros_like(savings)
        initial = BoundedResult(
            output=(
                zero,
                zero,
                zero,
                zero,
                zero,
                schedule.valid,
            ),
            bound_status=(),
            query_status=None,
            continuation_read_counts=ContinuationReadCounts(jnp.int32(0), jnp.int32(0))
            if self.collect_read_counts
            else None,
        )
        final, _ = jax.lax.scan(
            _CETargetFold(
                source=_SourceNodeRead(
                    reader=reader, direction=self.direction, savings=savings
                ),
                transform=transform,
                probability=probability,
                probability_direction=probability_direction,
                weight_directions=tangents,
            ),
            initial,
            (blocks, active),
        )
        return final


@dataclass(frozen=True, kw_only=True, eq=False)
class _EconomicTransform:
    """Apply the genuine transform with explicit value and parameter tangents."""

    declared: QuasiArithmeticMean
    """Original CE declaration retained by identity through the source plan."""
    params: Mapping[str, FloatND]
    """Actual flat parameter leaves at their supplied precision."""
    parameter_direction: Mapping[str, FloatND]
    """Explicit directions in that same declared CE parameter namespace."""
    savings_channel: bool
    """Whether the selected-savings direction is traced and validated."""
    source_channel: bool
    """Whether the source direction is traced and validated."""

    def __call__(
        self, inputs: tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]
    ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]:
        value, savings_direction, source_direction, needed = inputs

        def transform_value() -> tuple[
            ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool
        ]:
            # keyword-only-exempt: library-callback=jax.jvp
            def transform(
                value: ScalarFloat, params: Mapping[str, FloatND]
            ) -> ScalarFloat:
                return self.declared.transform(
                    value=value,
                    **_args_for(func=self.declared.transform, params=params),
                )

            saving = source = jnp.zeros_like(value)
            checked = [value]
            if self.savings_channel:
                transformed, saving = jax.jvp(
                    transform,
                    (value, self.params),
                    (savings_direction, jax.tree.map(jnp.zeros_like, self.params)),
                )
                checked += [savings_direction, saving]
            if self.source_channel:
                source_transformed, source = jax.jvp(
                    transform,
                    (value, self.params),
                    (source_direction, self.parameter_direction),
                )
                if not self.savings_channel:
                    transformed = source_transformed
                checked += [source_direction, source]
            if transformed.shape != value.shape:
                raise UnsupportedOperationError(
                    "Economic CE transform must be elementwise."
                )
            valid = jnp.all(jnp.isfinite(jnp.stack((*checked, transformed))))
            return transformed, saving, source, valid

        zero = jnp.zeros_like(value)
        return jax.lax.cond(
            needed,
            transform_value,
            lambda: (zero, zero, zero, jnp.ones((), dtype=jnp.bool_)),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _CETargetFold:
    """Accumulate transform partials over the original target/node schedule."""

    source: _SourceNodeRead
    """Original support/coordinate/row reader with separate economic tangents."""
    transform: _EconomicTransform
    """Declared transform applied before this target's node expectation."""
    probability: ScalarFloat
    """Original regime probability, retained as its own factor."""
    probability_direction: ScalarFloat
    """Source regime-weight derivative before any node or liveness product."""
    weight_directions: tuple[FloatND, ...]
    """Original node-factor directions on the canonical lottery axes."""

    # keyword-only-exempt: library-callback=jax.lax.scan
    def __call__(
        self, state: BoundedResult[_CEPartials], block: tuple[tuple[IntND, ...], BoolND]
    ) -> tuple[BoundedResult[_CEPartials], None]:
        indices, active = block
        active.shape[0]
        result = read_economic_node_record(
            source=self.source,
            indices=indices,
            active=active,
            probability=self.probability,
            probability_direction=self.probability_direction,
            weight_directions=self.weight_directions,
        )
        (
            values,
            economic,
            saving_coordinate,
            source_coordinate,
            valid,
            factors,
            tangents,
        ) = result.output
        saving = jnp.where(valid, economic * saving_coordinate, jnp.nan)
        source = jnp.where(valid, economic * source_coordinate, jnp.nan)
        partials = _ce_block_partials(
            values=values,
            savings_directions=saving,
            source_directions=source,
            factors=factors,
            factor_directions=tangents,
            transform=self.transform,
        )
        counts = state.continuation_read_counts
        if result.bound_status or result.query_status is not None:
            raise TypeError(
                "Economic CE node reads require original raw scalar capabilities."
            )
        if result.continuation_read_counts is not None:
            counts = merge_continuation_read_counts(
                left=counts,
                right=reduce_continuation_read_counts(
                    counts=result.continuation_read_counts,
                    active=_needed_ce_nodes(factors=factors, tangents=tangents),
                ),
            )
        return BoundedResult(
            output=_merge_ce_partials(left=state.output, right=partials),
            bound_status=(),
            query_status=None,
            continuation_read_counts=counts,
        ), None


def _needed_ce_nodes(*, factors: FloatND, tangents: FloatND) -> BoolND:
    return needed_economic_nodes(factors=factors, tangents=tangents)


def _ce_block_partials(
    *,
    values: FloatND,
    savings_directions: FloatND,
    source_directions: FloatND,
    factors: FloatND,
    factor_directions: FloatND,
    transform: _EconomicTransform,
) -> _CEPartials:
    transformed, saving, source, valid = jax.vmap(transform)(
        (
            values,
            savings_directions,
            source_directions,
            _needed_ce_nodes(factors=factors, tangents=factor_directions),
        )
    )
    ones = jnp.ones_like(values)
    numerator = jnp.sum(factored_weighted_term(factors=factors, values=transformed))
    mass = jnp.sum(factored_weighted_term(factors=factors, values=ones))
    saving_sum = source_sum = mass_source = jnp.zeros_like(mass)
    if transform.savings_channel:
        saving_sum = jnp.sum(factored_weighted_term(factors=factors, values=saving))
    if transform.source_channel:
        source_sum = jnp.sum(factored_weighted_term(factors=factors, values=source))
        for column in range(factors.shape[0]):
            differentiated = factors.at[column].set(factor_directions[column])
            source_sum += jnp.sum(
                factored_weighted_term(factors=differentiated, values=transformed)
            )
            mass_source += jnp.sum(
                factored_weighted_term(factors=differentiated, values=ones)
            )
    valid = jnp.all(valid) & jnp.all(
        jnp.isfinite(jnp.stack((numerator, mass, saving_sum, source_sum, mass_source)))
    )
    return numerator, mass, saving_sum, source_sum, mass_source, valid


def _merge_ce_partials(*, left: _CEPartials, right: _CEPartials) -> _CEPartials:
    numerator = left[0] + right[0]
    mass = left[1] + right[1]
    saving_sum = left[2] + right[2]
    source_sum = left[3] + right[3]
    mass_source = left[4] + right[4]
    valid = (
        left[5]
        & right[5]
        & jnp.all(
            jnp.isfinite(
                jnp.stack((numerator, mass, saving_sum, source_sum, mass_source))
            )
        )
    )
    return numerator, mass, saving_sum, source_sum, mass_source, valid


def _invert_economic_ce(
    *,
    declared: QuasiArithmeticMean,
    params: Mapping[str, FloatND],
    parameter_direction: Mapping[str, FloatND],
    partials: _CEPartials,
    savings_channel: bool,
    source_channel: bool,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]:
    numerator, mass, saving_sum, source_sum, mass_source, valid = partials
    normalized = numerator / mass

    # keyword-only-exempt: library-callback=jax.jvp
    def inverse(value: ScalarFloat, params: Mapping[str, FloatND]) -> ScalarFloat:
        return declared.inverse(
            value=value, **_args_for(func=declared.inverse, params=params)
        )

    saving = source = jnp.zeros_like(normalized)
    directions, outputs = [normalized], []
    if savings_channel:
        saving_direction = saving_sum / mass
        value, saving = jax.jvp(
            inverse,
            (normalized, params),
            (saving_direction, jax.tree.map(jnp.zeros_like, params)),
        )
        directions.append(saving_direction)
        outputs.append(saving)
    if source_channel:
        source_direction = (source_sum - normalized * mass_source) / mass
        source_value, source = jax.jvp(
            inverse, (normalized, params), (source_direction, parameter_direction)
        )
        if not savings_channel:
            value = source_value
        directions.append(source_direction)
        outputs.append(source)
    if value.shape != normalized.shape:
        raise UnsupportedOperationError("Economic CE inverse must be elementwise.")
    valid &= jnp.all(jnp.isfinite(jnp.stack(directions)))
    valid &= (mass > 0.0) & jnp.all(jnp.isfinite(jnp.stack((value, *outputs))))
    return value, saving, source, valid
