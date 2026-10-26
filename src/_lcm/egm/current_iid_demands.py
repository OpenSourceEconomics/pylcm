"""Model-owned exact parent demands for a scalar projected child.

These declarations describe the actual Euler consumer. They do not certify
smooth links or permit arbitrary queries of the resulting exact-only bank.
"""

import inspect
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from _lcm.continuation import ProjectedContinuationSpec
from _lcm.egm.continuation import _ChildEulerState
from _lcm.egm.step_core import _EgmKernelPieces, bind_cell_floor
from _lcm.typing import EconFunctionKwargs
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import Float1D, RegimeName, ScalarBool


@dataclass(frozen=True, kw_only=True, eq=False)
class ParentResourcesDemand:
    """An original prepared Euler reader, with its exact producer/date identity."""

    source: RegimeName
    """Actual parent regime whose parameter namespace is evaluated."""
    period: int
    """Actual parent date, one before this producer date."""
    target: RegimeName
    """Producer named by the parent's original continuation plan."""
    spec: ProjectedContinuationSpec
    """The shared declaration object, authenticated before native installation."""
    pieces: _EgmKernelPieces
    """Original savings/floor and continuation-law declarations."""
    transition_targets: tuple[RegimeName, ...]
    """Original parent namespaces required by the continuation call pool."""

    def __post_init__(self) -> None:
        plan = self.pieces.continuation_plan
        read = plan.child_reads.get(self.target)
        if read is None:
            raise TypeError("Incoming parent omitted its projected child read.")
        try:
            parameters = inspect.signature(read.euler_state_func).parameters
        except (TypeError, ValueError) as error:
            raise UnsupportedOperationError(
                "Exact parent demands require an inspectable closed Euler law."
            ) from error
        if any(
            parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters.values()
        ):
            raise UnsupportedOperationError(
                "Exact parent demands require a nonvariadic closed Euler law."
            )
        if (
            self.pieces.combo_names
            or plan.risk_aversion_param_name is not None
            or read.stochastic_state_names
            or read.local_draw_names
            or read.euler_draw_names
            or read.passive_state_names
            or read.discrete_state_names
            or self.pieces.lower_func is not None
            or self.pieces._prepared_nested_floor is not None  # noqa: SLF001
            or self.pieces.euler_state_name in parameters
        ):
            raise UnsupportedOperationError(
                "Exact parent demands require scalar deterministic linear-CE "
                "consumers with a literal savings floor."
            )
        if (
            read.projected_date is not self.spec.date_at(period=self.period + 1)
            or self.spec.producer != self.target
        ):
            raise TypeError("Parent demand does not name its original producer/date.")

    def prepare(
        self, *, pool: EconFunctionKwargs, savings_nodes: Float1D
    ) -> tuple[Float1D, ScalarBool]:
        """Re-evaluate the original parent law on every actual savings/floor node."""
        from dataclasses import replace  # noqa: PLC0415

        pieces = bind_cell_floor(
            pieces=replace(self.pieces, savings_nodes=savings_nodes), pool=pool
        )
        read = pieces.continuation_plan.child_reads[self.target]
        law = _ChildEulerState(
            next_state_func=read.euler_state_func,
            combo_pool=pool,
            post_decision_name=pieces.continuation_plan.post_decision_name,
            next_state_key=read.next_state_key,
        )
        queries = jax.vmap(law)(pieces.savings_nodes)
        valid = (
            jnp.all(jnp.isfinite(pieces.savings_nodes))
            & jnp.all(pieces.savings_nodes[1:] > pieces.savings_nodes[:-1])
            & jnp.all(jnp.isfinite(queries))
        )
        if pieces.floor_status is not None:
            valid = (
                valid
                & ~pieces.floor_status.nonfinite_floor
                & ~pieces.floor_status.invalid_shifted_grid
            )
        return queries, valid


def prepare_parent_resources_demands(
    *,
    declarations: tuple[ParentResourcesDemand, ...],
    pools: tuple[EconFunctionKwargs, ...],
    savings: tuple[Float1D, ...],
    dtype: jnp.dtype,
) -> tuple[Float1D, ScalarBool]:
    """Close the exact dynamic demand population; neither padding nor probes count."""
    queries = []
    valid = jnp.ones((), dtype=jnp.bool_)
    for declaration, pool, nodes in zip(declarations, pools, savings, strict=True):
        query, source_valid = declaration.prepare(pool=pool, savings_nodes=nodes)
        queries.append(query)
        valid = valid & source_valid
    return (
        jnp.concatenate(queries) if queries else jnp.empty((0,), dtype=dtype),
        valid,
    )
