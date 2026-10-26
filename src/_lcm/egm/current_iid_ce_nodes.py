"""Original economic node records shared by finite and stable CE sinks."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from _lcm.certainty_equivalent import PowerMean, QuasiArithmeticMean
from _lcm.egm.continuation import (
    ContinuationPlan,
    _block_node_indices,
    _ChildCarryReader,
    _ChildRead,
    _node_index_mesh,
    _resolve_child_node_operands,
)
from _lcm.egm.current_iid_future import _future_law_valid, _SourceNodeRead
from _lcm.probability import is_represented_zero
from _lcm.solution.feasibility import BoundedResult
from _lcm.typing import EconFunctionKwargs
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

type EconomicNodeRecord = tuple[
    FloatND, FloatND, FloatND, FloatND, BoolND, FloatND, FloatND
]


@dataclass(frozen=True, kw_only=True, eq=False)
class PowerMeanTargetLayout:
    """One actual original target's unpadded joint-axis extent."""

    target: RegimeName
    read: _ChildRead | None
    axis_shapes: tuple[tuple[int, ...], ...]
    factor_arity: int
    start: int
    stop: int


@dataclass(frozen=True, kw_only=True, eq=False)
class PowerMeanJointLayout:
    """Exact transient geometry bound to the actual point owner and plan.

    This records required storage, not a budget or production admission.
    The actual enclosing CoreProgram profile/resident-budget gate remains a
    separate required installation before model-level allocation/dispatch.
    """

    # Resolve the actual owner class after cyclic module decoration.
    owner: "_lcm.egm.current_iid_case_u.CaseUPointSolver"  # noqa: UP037
    plan: ContinuationPlan
    declared: PowerMean
    targets: tuple[PowerMeanTargetLayout, ...]

    def __post_init__(self) -> None:
        if not self.targets:
            raise TypeError("PowerMean joint layout requires actual lottery targets.")
        if self.owner.pieces.continuation_plan is not self.plan:
            raise TypeError("PowerMean layout does not own the original source plan.")
        if self.plan.certainty_equivalent is not self.declared:
            raise TypeError("PowerMean layout has a different CE declaration owner.")
        if type(self.declared) is not PowerMean:
            raise TypeError("PowerMean layout requires the exact genuine declaration.")
        targets = (*self.plan.stateful_targets, *self.plan.scalar_targets)
        if tuple(target.target for target in self.targets) != targets:
            raise TypeError("PowerMean joint layout changed the original target order.")
        offset = 0
        for target in self.targets:
            if target.start != offset or target.stop <= target.start:
                raise TypeError(
                    "PowerMean joint layout has a missing/overlapping event."
                )
            if (
                target.read is not None
                and self.plan.child_reads[target.target] is not target.read
            ):
                raise TypeError(
                    "PowerMean target layout changed the original read owner."
                )
            offset = target.stop

    @property
    def n_events(self) -> int:
        return self.targets[-1].stop

    @property
    def factor_arity(self) -> int:
        return max(target.factor_arity for target in self.targets)

    @property
    def scratch_shapes(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Actual complete bank owners; block/core temporaries are additional."""
        n = self.n_events
        return (
            ("values", (n,)),
            ("economic", (n,)),
            ("savings_coordinates", (n,)),
            ("source_coordinates", (n,)),
            ("coefficients", (n,)),
            ("shifts", (n,)),
            ("differentiated_coefficients", (self.factor_arity, n)),
            ("differentiated_shifts", (self.factor_arity, n)),
            ("needed", (n,)),
            ("unit_variance_pair_row", (n,)),
        )

    def bank_templates(self, *, dtype: DTypeLike) -> Mapping[str, jax.ShapeDtypeStruct]:
        """Expose complete bank geometry to the actual producer materializer.

        These are abstract owners, not allocated scratch or a budget grant.
        Original read/block/context and overlapping assembly temporaries must
        also be charged by that actual enclosing program's compiler profile.
        """
        return {
            name: jax.ShapeDtypeStruct(
                shape,
                jnp.bool_
                if name == "needed"
                else jnp.int32
                if name.endswith("shifts")
                else dtype,
            )
            for name, shape in self.scratch_shapes
        }


@dataclass(frozen=True, kw_only=True, eq=False)
class EconomicTargetSchedule:
    """Original support/weight schedule, reused by both concrete CE sinks."""

    reader: _ChildCarryReader
    weight_directions: tuple[FloatND, ...]
    blocks: tuple[IntND, ...]
    active: BoolND
    n_nodes: int
    valid: ScalarBool


def economic_target_schedule(
    *,
    reader: _ChildCarryReader,
    direction: Mapping[ReferenceName, ScalarFloat],
    stochastic_node_width: int | None,
) -> EconomicTargetSchedule:
    names = tuple(direction)

    def weights(values: tuple[ScalarFloat, ...]) -> tuple[FloatND, ...]:
        pool = {**reader.combo_pool, **dict(zip(names, values, strict=True))}
        return _resolve_child_node_operands(read=reader.read, combo_pool=pool)[1]

    _, tangents = jax.jvp(
        weights,
        (tuple(reader.combo_pool[name] for name in names),),
        (tuple(direction[name] for name in names),),
    )
    indices = (
        _node_index_mesh(stochastic_node_values=reader.stochastic_node_values)
        if reader.stochastic_node_values
        else ()
    )
    n_nodes = indices[0].shape[0] if indices else 1
    width = (
        n_nodes
        if stochastic_node_width is None
        else min(stochastic_node_width, n_nodes)
    )
    blocks = (
        _block_node_indices(flat_node_indices=indices, width=width) if indices else ()
    )
    pad = (-(-n_nodes // width)) * width - n_nodes
    active = (jnp.arange(n_nodes + pad) < n_nodes).reshape(-1, width)
    return EconomicTargetSchedule(
        reader=reader,
        weight_directions=tangents,
        blocks=blocks,
        active=active,
        n_nodes=n_nodes,
        valid=_future_law_valid(reader=reader, weight_directions=tangents),
    )


def needed_economic_nodes(*, factors: FloatND, tangents: FloatND) -> BoolND:
    needed = jnp.all(~is_represented_zero(factors), axis=0)
    for column in range(factors.shape[0]):
        others = jnp.concatenate((factors[:column], factors[column + 1 :]), axis=0)
        needed |= ~is_represented_zero(tangents[column]) & jnp.all(
            ~is_represented_zero(others), axis=0
        )
    return needed


def read_economic_node_record(
    *,
    source: _SourceNodeRead,
    indices: tuple[IntND, ...],
    active: BoolND,
    probability: ScalarFloat,
    probability_direction: ScalarFloat,
    weight_directions: tuple[FloatND, ...],
) -> BoundedResult[EconomicNodeRecord]:
    """Use the original row/coordinate read once for either CE reduction sink."""
    width = active.shape[0]
    result = jax.vmap(source.economic_components, axis_size=width)(indices)
    values, economic, savings_coordinate, source_coordinate, valid = result.output
    factors = jnp.stack(
        (
            jnp.full((width,), probability),
            *(
                vec[index]
                for vec, index in zip(source.reader.weight_vecs, indices, strict=True)
            ),
            active.astype(values.dtype),
        )
    )
    tangents = jnp.stack(
        (
            jnp.full((width,), probability_direction),
            *(
                vec[index]
                for vec, index in zip(weight_directions, indices, strict=True)
            ),
            jnp.zeros(width, dtype=values.dtype),
        )
    )
    return BoundedResult(
        output=(
            values,
            economic,
            savings_coordinate,
            source_coordinate,
            valid,
            factors,
            tangents,
        ),
        bound_status=result.bound_status,
        query_status=result.query_status,
        continuation_read_counts=result.continuation_read_counts,
    )


def _bind_ce_parameters(
    *, declared: QuasiArithmeticMean, pool: EconFunctionKwargs
) -> Mapping[str, FloatND]:
    """Bind actual floating CE parameter leaves without changing their precision."""
    params: dict[str, FloatND] = {}
    for name, flat in declared.flat_param_names.items():
        value = pool[flat]
        if not isinstance(value, jax.Array | float | int):
            raise TypeError(f"Economic CE parameter {flat!r} is not a numeric leaf.")
        numeric = jnp.asarray(value)
        if not jnp.issubdtype(numeric.dtype, jnp.floating):
            raise UnsupportedOperationError(
                f"Economic CE parameter {flat!r} needs a floating tangent capability."
            )
        params[name] = numeric
    return params
