"""Transient original-node PowerMean bank for the genuine private point owner.

Complete geometry is source-bound here; production compiler/resident-budget
admission remains owned by the enclosing CoreProgram integration. Nothing in
this module grants Model admission or treats block width as total capacity.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from _lcm.certainty_equivalent import PowerMean
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _ChildCarryReader, _get_child_carry_reader
from _lcm.egm.current_iid_ce_nodes import (
    EconomicNodeRecord,
    EconomicTargetSchedule,
    PowerMeanJointLayout,
    PowerMeanTargetLayout,
    _bind_ce_parameters,
    economic_target_schedule,
    needed_economic_nodes,
    read_economic_node_record,
)
from _lcm.egm.current_iid_future import _SourceNodeRead
from _lcm.egm.current_iid_power_mean import (
    PowerMeanNodeBank,
    reduce_power_mean_economic,
)
from _lcm.probability import probabilities_form_distribution, scaled_exact_product
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationReadCounts,
    QueryInputStatus,
    merge_continuation_read_counts,
    reduce_continuation_read_counts,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarBool, ScalarFloat

if TYPE_CHECKING:
    import _lcm.egm.current_iid_ce


@dataclass(frozen=True, kw_only=True, eq=False)
class _PowerMeanRecordFold:
    source: _SourceNodeRead
    probability: ScalarFloat
    probability_direction: ScalarFloat
    weight_directions: tuple[FloatND, ...]

    # keyword-only-exempt: library-callback=jax.lax.scan
    def __call__(
        self, state: BoundedResult[ScalarBool], block: tuple[tuple[IntND, ...], BoolND]
    ) -> tuple[BoundedResult[ScalarBool], EconomicNodeRecord]:
        indices, active = block
        result = read_economic_node_record(
            source=self.source,
            indices=indices,
            active=active,
            probability=self.probability,
            probability_direction=self.probability_direction,
            weight_directions=self.weight_directions,
        )
        values, economic, saving, source, valid, factors, tangents = result.output
        if result.bound_status or result.query_status is not None:
            raise TypeError(
                "PowerMean node bank requires original raw scalar read slots."
            )
        needed = needed_economic_nodes(factors=factors, tangents=tangents)
        counts = state.continuation_read_counts
        if result.continuation_read_counts is not None:
            counts = merge_continuation_read_counts(
                left=counts,
                right=reduce_continuation_read_counts(
                    counts=result.continuation_read_counts, active=needed
                ),
            )
        return BoundedResult(
            output=state.output & jnp.all(~needed | valid),
            bound_status=(),
            query_status=None,
            continuation_read_counts=counts,
        ), (values, economic, saving, source, valid, factors, tangents)


def _event_product(factors: FloatND) -> tuple[FloatND, IntND]:
    """Scale each event separately; a bankwide common lift is insufficient."""
    return scaled_exact_product(factors=factors)


def _bank_coefficients(
    *,
    factors: FloatND,
    tangents: FloatND,
    differentiate: bool = True,
) -> tuple[FloatND, IntND, FloatND, IntND]:
    coefficients, shifts = jax.vmap(_event_product)(factors.T)
    if not differentiate:
        return (
            coefficients,
            shifts,
            jnp.zeros((factors.shape[0], *coefficients.shape), coefficients.dtype),
            jnp.zeros((factors.shape[0], *shifts.shape), shifts.dtype),
        )

    def differentiated(column: IntND) -> tuple[FloatND, IntND]:
        return jax.vmap(_event_product)(factors.at[column].set(tangents[column]).T)

    derivative_coefficients, derivative_shifts = jax.vmap(differentiated)(
        jnp.arange(factors.shape[0], dtype=jnp.int32)
    )
    return coefficients, shifts, derivative_coefficients, derivative_shifts


def _joint_layout(
    *,
    # Resolve the actual class after cyclic module decoration.
    continuation: (
        "_lcm.egm.current_iid_ce.EconomicCEContinuation"  # noqa: UP037
    ),
    declared: PowerMean,
) -> tuple[PowerMeanJointLayout, dict[RegimeName, EconomicTargetSchedule]]:
    owner = continuation.power_mean_owner
    if owner is None:
        raise UnsupportedOperationError(
            "Stable PowerMean lacks its actual point owner."
        )
    targets = (*continuation.plan.stateful_targets, *continuation.plan.scalar_targets)
    schedules: dict[RegimeName, EconomicTargetSchedule] = {}
    target_layouts: list[PowerMeanTargetLayout] = []
    offset = 0
    for target in targets:
        if target in continuation.plan.stateful_targets:
            reader = _get_child_carry_reader(
                read=continuation.plan.child_reads[target],
                carry=continuation.children[target],
                combo_pool=continuation.pool,
                post_decision_name=continuation.plan.post_decision_name,
                stochastic_node_width=continuation.stochastic_node_width,
                collect_continuation_read_counts=continuation.collect_read_counts,
            )
            if not isinstance(reader, _ChildCarryReader):
                raise UnsupportedOperationError(
                    "Stable projected-future economic capabilities remain required."
                )
            schedule = economic_target_schedule(
                reader=reader,
                direction=continuation.direction,
                stochastic_node_width=continuation.stochastic_node_width,
            )
            schedules[target] = schedule
            shapes = tuple(array.shape for array in reader.stochastic_node_values)
            expected_shapes = (
                *(array.shape for array in reader.read.stochastic_node_values),
                *((size,) for size in reader.read.local_draw_sizes),
            )
            if (
                shapes != expected_shapes
                or tuple(array.shape for array in reader.weight_vecs) != shapes
            ):
                raise TypeError(
                    "PowerMean support/weight shapes changed the declared joint roster."
                )
            if any(len(shape) != 1 for shape in shapes):
                raise UnsupportedOperationError(
                    "Stable PowerMean requires canonical one-dimensional "
                    "joint node axes."
                )
            n_nodes = schedule.n_nodes
            arity = 2 + len(reader.weight_vecs)
            read = reader.read
        else:
            carry = continuation.children[target]
            if not isinstance(carry, EGMCarry) or carry.value.shape != (1,):
                raise UnsupportedOperationError(
                    "Stable scalar targets require their actual single-value "
                    "raw artifact."
                )
            shapes, n_nodes, arity, read = (), 1, 1, None
        target_layouts.append(
            PowerMeanTargetLayout(
                target=target,
                read=read,
                axis_shapes=shapes,
                factor_arity=arity,
                start=offset,
                stop=offset + n_nodes,
            )
        )
        offset += n_nodes
    layout = PowerMeanJointLayout(
        owner=owner,
        plan=continuation.plan,
        declared=declared,
        targets=tuple(target_layouts),
    )
    return layout, schedules


def power_mean_continuation(
    *,
    # Resolve the actual class after cyclic module decoration.
    continuation: (
        "_lcm.egm.current_iid_ce.EconomicCEContinuation"  # noqa: UP037
    ),
    savings: ScalarFloat,
    declared: PowerMean,
    savings_channel: bool,
    source_channel: bool,
) -> BoundedResult[tuple[ScalarFloat, ScalarFloat, ScalarFloat]]:
    """Read the actual complete joint roster with its original point authority.

    Only the statically requested economic directions are traced and validated.
    """
    owner = continuation.power_mean_owner
    if owner is None:
        raise UnsupportedOperationError(
            "Stable PowerMean bank requires its actual private point owner; "
            "automatic enclosing CoreProgram budget admission remains required."
        )
    actual_params = _bind_ce_parameters(declared=declared, pool=continuation.pool)
    parameter_direction = {
        name: continuation.direction.get(flat, jnp.zeros_like(actual_params[name]))
        for name, flat in declared.flat_param_names.items()
    }
    names = tuple(continuation.direction)

    def probabilities(
        values: tuple[ScalarFloat, ...],
    ) -> Mapping[RegimeName, ScalarFloat]:
        return continuation.plan.compute_regime_transition_probs(
            **{**continuation.pool, **dict(zip(names, values, strict=True))}
        )

    probabilities_at_source, probability_directions = jax.jvp(
        probabilities,
        (tuple(continuation.pool[name] for name in names),),
        (tuple(continuation.direction[name] for name in names),),
    )
    targets = (*continuation.plan.stateful_targets, *continuation.plan.scalar_targets)
    valid = probabilities_form_distribution(
        probabilities=(probabilities_at_source[target] for target in targets),
        dtype=continuation.dtype,
    )
    for parameter in (
        *actual_params.values(),
        *(
            (
                *parameter_direction.values(),
                *continuation.direction.values(),
                *probability_directions.values(),
            )
            if source_channel
            else ()
        ),
    ):
        valid &= jnp.all(jnp.isfinite(parameter))
    layout, schedules = _joint_layout(continuation=continuation, declared=declared)
    banks: list[PowerMeanNodeBank] = []
    counts: ContinuationReadCounts | None = None
    for target_layout in layout.targets:
        target = target_layout.target
        probability = probabilities_at_source[target]
        probability_direction = probability_directions[target]
        if target in schedules:
            schedule = schedules[target]
            initial = BoundedResult(
                output=schedule.valid,
                bound_status=(),
                query_status=None,
                continuation_read_counts=ContinuationReadCounts(
                    jnp.int32(0), jnp.int32(0)
                )
                if continuation.collect_read_counts
                else None,
            )
            state, records = jax.lax.scan(
                _PowerMeanRecordFold(
                    source=_SourceNodeRead(
                        reader=schedule.reader,
                        direction=continuation.direction,
                        savings=savings,
                        source_channel=source_channel,
                    ),
                    probability=probability,
                    probability_direction=probability_direction,
                    weight_directions=schedule.weight_directions,
                ),
                initial,
                (schedule.blocks, schedule.active),
            )
            values, economic, saving, source, _, factors, tangents = records
            n = schedule.n_nodes
            values, economic, saving, source = (
                array.reshape(-1)[:n] for array in (values, economic, saving, source)
            )
            factors, tangents = (
                jnp.transpose(array, (1, 0, 2)).reshape(array.shape[1], -1)[:, :n]
                for array in (factors, tangents)
            )
            valid &= state.output
            counts = merge_continuation_read_counts(
                left=counts, right=state.continuation_read_counts
            )
        else:
            carry = continuation.children[target]
            if not isinstance(carry, EGMCarry):
                raise TypeError("Stable scalar artifact changed after layout binding.")
            values = carry.value
            economic = saving = source = jnp.zeros_like(values)
            factors = jnp.asarray(probability).reshape(1, 1)
            tangents = jnp.asarray(probability_direction).reshape(1, 1)
        if (
            values.shape != (target_layout.stop - target_layout.start,)
            or factors.shape[0] != target_layout.factor_arity
        ):
            raise TypeError(
                "Stable PowerMean bank differs from its actual original roster."
            )
        coefficient, shift, derivative_coefficient, derivative_shift = (
            _bank_coefficients(
                factors=factors, tangents=tangents, differentiate=source_channel
            )
        )
        pad = layout.factor_arity - target_layout.factor_arity
        banks.append(
            PowerMeanNodeBank(
                values,
                economic,
                saving,
                source,
                coefficient,
                shift,
                jnp.pad(derivative_coefficient, ((0, pad), (0, 0))),
                jnp.pad(derivative_shift, ((0, pad), (0, 0))),
                needed_economic_nodes(factors=factors, tangents=tangents),
                valid,
            )
        )
    bank = PowerMeanNodeBank(
        values=jnp.concatenate(tuple(part.values for part in banks), axis=0),
        economic=jnp.concatenate(tuple(part.economic for part in banks), axis=0),
        savings_coordinates=jnp.concatenate(
            tuple(part.savings_coordinates for part in banks), axis=0
        ),
        source_coordinates=jnp.concatenate(
            tuple(part.source_coordinates for part in banks), axis=0
        ),
        coefficients=jnp.concatenate(
            tuple(part.coefficients for part in banks), axis=0
        ),
        shifts=jnp.concatenate(tuple(part.shifts for part in banks), axis=0),
        differentiated_coefficients=jnp.concatenate(
            tuple(part.differentiated_coefficients for part in banks), axis=1
        ),
        differentiated_shifts=jnp.concatenate(
            tuple(part.differentiated_shifts for part in banks), axis=1
        ),
        needed=jnp.concatenate(tuple(part.needed for part in banks), axis=0),
        valid=valid,
    )
    if bank.values.shape != (layout.n_events,):
        raise TypeError("Stable PowerMean lost a mandatory original joint event.")
    value, saving, source, valid = reduce_power_mean_economic(
        declared=declared,
        params=actual_params,
        parameter_direction=parameter_direction,
        bank=bank,
        savings_channel=savings_channel,
        source_channel=source_channel,
    )
    inactive = jnp.zeros((), dtype=jnp.bool_)
    return BoundedResult(
        output=(value, saving, source),
        bound_status=(),
        query_status=(QueryInputStatus(inactive, inactive, ~valid),),
        continuation_read_counts=counts,
    )
