"""Current-node expectation after canonical owned-resource candidate selection."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.egm.current_iid_resource_query import (
    OwnedResourcesRead,
    read_resources_candidate,
    read_resources_candidate_details,
)
from _lcm.egm.projected_continuation import ProjectedRowRead
from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.probability import is_represented_zero
from _lcm.typing import PytreeChild
from _lcm.zero_safe import factored_weighted_term
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

_MIN_PARTITION_POINTS = 2


class CurrentResourcesEvaluation(NamedTuple):
    """Integrated economic channels and mandatory original-row integrity."""

    value: ScalarFloat
    economic_marginal: ScalarFloat
    empty: ScalarBool
    valid: ScalarBool

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], CurrentResourcesEvaluation],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_current_resources_evaluation, (None, tuple(self))


@dataclass(frozen=True, kw_only=True, eq=False)
class CurrentResourcesEvaluator:
    """Invocation-local reader of one common native candidate row.

    Current physical nodes and factor banks are explicit call operands. Only
    the resources map consumes a current physical draw. Utility and the owned
    row were admitted using the complete opaque-resource dependency proof.
    Selected node tuples are transient proof operands, not durable carry axes.
    """

    row: OwnedResourcesRow
    """Canonical candidate row constructed once on the common resources domain."""
    resources: Callable[[ScalarFloat, Mapping[ShockName, ScalarFloat]], ScalarFloat]
    """Actual structurally certified resources function, with context bound."""
    resources_slope: ScalarFloat
    """Exact positive affine coefficient in the physical pre-shock liquid state."""
    utility_of_action: Callable[[ScalarFloat], ScalarFloat]
    """Admitted local utility, with opaque resources and context already bound."""
    owner_capacity: int
    """Original exact-envelope run authority, not a fabricated projection owner."""

    def node(
        self, *, liquid: ScalarFloat, physical: Mapping[ShockName, ScalarFloat]
    ) -> OwnedResourcesRead:
        """Select policy/value/saving through the same actual resources coordinate."""
        return read_resources_candidate(
            row=self.row,
            resources=self.resources(liquid, physical),
            utility_of_action=self.utility_of_action,
            owner_capacity=self.owner_capacity,
        )

    def integrate(
        self,
        *,
        liquid: ScalarFloat,
        physical: Mapping[ShockName, FloatND | IntND],
        factors: FloatND,
        law_valid: ScalarBool,
    ) -> CurrentResourcesEvaluation:
        """Return integrated channels from the original node selection."""
        result, _ = self.integrate_with_selection(
            liquid=liquid, physical=physical, factors=factors, law_valid=law_valid
        )
        return result

    def integrate_with_selection(
        self,
        *,
        liquid: ScalarFloat,
        physical: Mapping[ShockName, FloatND | IntND],
        factors: FloatND,
        law_valid: ScalarBool,
    ) -> tuple[CurrentResourcesEvaluation, OwnedResourcesRead]:
        """Retain the same original selection used by the economic channels."""
        result, nodes, _, _ = self.integrate_with_roster(
            liquid=liquid, physical=physical, factors=factors, law_valid=law_valid
        )
        return result, nodes

    def integrate_with_roster(
        self,
        *,
        liquid: ScalarFloat,
        physical: Mapping[ShockName, FloatND | IntND],
        factors: FloatND,
        law_valid: ScalarBool,
    ) -> tuple[CurrentResourcesEvaluation, OwnedResourcesRead, BoolND, IntND]:
        """Sum selected levels and explicit economic marginal chain factors.

        The admitted first producer has no liquid-dependent law or weights.
        A future producer with those dependencies must add their full economic
        derivative channels; it cannot infer them from final-output AD.
        """
        live = jnp.all(~is_represented_zero(factors), axis=0)

        def select(
            inputs: tuple[Mapping[ShockName, ScalarFloat], ScalarBool],
        ) -> tuple[OwnedResourcesRead, ScalarBool, ScalarInt]:
            draw, active = inputs
            zero = jnp.zeros_like(liquid)
            no = jnp.zeros((), dtype=jnp.bool_)
            neutral = OwnedResourcesRead(
                value=zero,
                marginal=zero,
                policy=zero,
                savings=zero,
                native_owner=jnp.asarray(-1, dtype=jnp.int32),
                side_id=jnp.asarray(0, dtype=jnp.int32),
                corner=no,
                feasible=no,
                valid=~no,
            )
            return jax.lax.cond(
                active,
                lambda: read_resources_candidate_details(
                    row=self.row,
                    resources=self.resources(liquid, draw),
                    utility_of_action=self.utility_of_action,
                    owner_capacity=self.owner_capacity,
                ),
                lambda: (neutral, no, jnp.asarray(-1, dtype=jnp.int32)),
            )

        # A real scalar conditional is retained per row. vmap(cond) would
        # evaluate poisoned optional callbacks in both branches before select.
        nodes, row_legal, piece = jax.lax.map(select, (physical, live))
        values = factored_weighted_term(factors=factors, values=nodes.value)
        marginal = factored_weighted_term(
            factors=factors, values=nodes.marginal * self.resources_slope
        )
        empty = jnp.any(live & ~nodes.feasible)
        economic_minus_inf = jnp.any(live & jnp.isneginf(nodes.value))
        value = jnp.sum(values)
        raw_marginal = jnp.sum(marginal)
        economic_marginal = jnp.where(empty | economic_minus_inf, 0.0, raw_marginal)
        value_terms_valid = jnp.all(
            ~live
            | (jnp.isfinite(nodes.value) & jnp.isfinite(values))
            | (jnp.isneginf(nodes.value) & jnp.isneginf(values))
        )
        integrated_valid = (
            value_terms_valid
            & (jnp.isfinite(value) | (economic_minus_inf & jnp.isneginf(value)))
            & jnp.all(~live | jnp.isfinite(marginal))
            & jnp.isfinite(raw_marginal)
            & jnp.isfinite(economic_marginal)
        )
        result = CurrentResourcesEvaluation(
            value=value,
            economic_marginal=economic_marginal,
            empty=empty,
            valid=(
                law_valid
                & ~self.row.construction_invalid
                & jnp.all(~live | nodes.valid)
                & jnp.isfinite(self.resources_slope)
                & (self.resources_slope > 0.0)
                & integrated_valid
            ),
        )
        return result, nodes, row_legal, piece

    def reference(
        self,
        *,
        liquid: ScalarFloat,
        physical: Mapping[ShockName, FloatND | IntND],
        factors: FloatND,
        law_valid: ScalarBool,
        partition: Float1D,
        partition_length: ScalarInt,
    ) -> ProjectedRowRead:
        """Return exact economics with the common partition address."""
        result, _ = self.reference_with_selection(
            liquid=liquid,
            physical=physical,
            factors=factors,
            law_valid=law_valid,
            partition=partition,
            partition_length=partition_length,
        )
        return result

    def reference_with_selection(
        self,
        *,
        liquid: ScalarFloat,
        physical: Mapping[ShockName, FloatND | IntND],
        factors: FloatND,
        law_valid: ScalarBool,
        partition: Float1D,
        partition_length: ScalarInt,
    ) -> tuple[ProjectedRowRead, OwnedResourcesRead]:
        """Retain actual selected tuples from the same integration evaluation.

        Partition identities index represented query cells only. They do not
        authenticate the selected source-candidate roster or an incoming or
        outgoing limit. Publication requires separate complete topology;
        original envelope identity remains on the replay row.
        """
        result, nodes = self.integrate_with_selection(
            liquid=liquid, physical=physical, factors=factors, law_valid=law_valid
        )
        search = jnp.where(
            jnp.arange(partition.size) < partition_length, partition, jnp.inf
        )
        owner = jnp.maximum(jnp.searchsorted(search, liquid, side="right") - 1, 0)
        read = ProjectedRowRead(
            value=result.value,
            economic_marginal=result.economic_marginal,
            owner_id=jnp.minimum(owner, partition_length - 2).astype(jnp.int32),
            side_id=jnp.asarray(0, dtype=jnp.int32),
            valid=result.valid & (partition_length >= _MIN_PARTITION_POINTS),
        )
        return read, nodes


def _flatten_current_resources_evaluation(
    record: CurrentResourcesEvaluation,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(record), None


def _flatten_current_resources_evaluation_with_keys(
    record: CurrentResourcesEvaluation,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(CurrentResourcesEvaluation._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_current_resources_evaluation(
    metadata: None,
    children: Iterable[PytreeChild],
) -> CurrentResourcesEvaluation:
    """Restore native numeric, abstract or role leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(CurrentResourcesEvaluation._fields):
        raise TypeError(
            "Native economic evaluation reconstruction requires its exact fields."
        )
    return cast(
        "CurrentResourcesEvaluation", tuple.__new__(CurrentResourcesEvaluation, leaves)
    )


jax.tree_util.register_pytree_with_keys(
    CurrentResourcesEvaluation,
    _flatten_current_resources_evaluation_with_keys,
    _unflatten_current_resources_evaluation,
    _flatten_current_resources_evaluation,
)
