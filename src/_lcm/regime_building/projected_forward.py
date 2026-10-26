"""Original dated child reads over explicitly addressed forward operands."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.continuation import ProjectedContinuationSpec, ProjectedProducerDate
from _lcm.egm.projected_child import read_projected_coordinate, require_projected_carry
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.execution.core_program import ValueRead
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.feasibility import QueryInputStatus, validate_query_input_status
from _lcm.typing import QAndFArg, QualifiedName
from lcm.exceptions import ExecutionPlanningError, UnsupportedOperationError
from lcm.solver_api import ArtifactRef
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarFloat


@dataclass(frozen=True, kw_only=True, eq=False)
class ProjectedForwardRead:
    """One genuine producer date and its canonical flat numerical argument roster."""

    date: ProjectedProducerDate
    """The installed producer date and its model-owned guard program."""
    target: RegimeName
    """Original child regime whose retained artifact supplies every leaf."""
    argument_by_leaf: Mapping[tuple[str, ...], QualifiedName]
    """Canonical stored leaf address to the matching flat numerical argument."""
    native_arguments: tuple[QualifiedName, ...]
    """Argument order matching the original registered native leaf order."""

    @staticmethod
    def can_read(*, spec: ProjectedContinuationSpec, period: int) -> bool:
        """Keep other producer families on their existing construction route."""
        from _lcm.egm.current_iid_case_u_balanced import (  # noqa: PLC0415 - installed producer metadata
            BalancedCaseUCarry,
            SaturatedCaseUProgram,
        )

        date = spec.date_at(period=period)
        return (
            type(date.template) is BalancedCaseUCarry
            and type(date.guard_program) is SaturatedCaseUProgram
        )

    @classmethod
    def create(
        cls, *, spec: ProjectedContinuationSpec, target: RegimeName, period: int
    ) -> ProjectedForwardRead:
        from _lcm.egm.current_iid_case_u_balanced import (  # noqa: PLC0415
            BalancedCaseUCarry,
            SaturatedCaseUProgram,
        )

        date = spec.date_at(period=period)
        template = date.template
        if not isinstance(template, BalancedCaseUCarry) or not isinstance(
            date.guard_program, SaturatedCaseUProgram
        ):
            raise UnsupportedOperationError(
                "Ordinary projected forward requires its admitted saturated-U child."
            )
        require_projected_carry(date=date, target=target, carry=template)
        # BalancedCaseUCarry._paths follows the genuine registered tree in its
        # original flatten order, including actual nested guard leaves.
        paths = tuple(template.leaves())
        arguments = {
            path: f"_lcm_projected_forward__{target}__{index}"
            for index, path in enumerate(paths)
        }
        if any(
            left is not right
            for left, right in zip(
                template.leaves().values(), jax.tree.leaves(template), strict=True
            )
        ):
            raise TypeError("Projected forward lost its original native leaf order.")
        return cls(
            date=date,
            target=target,
            argument_by_leaf=MappingProxyType(arguments),
            native_arguments=tuple(arguments.values()),
        )

    def reads(
        self, *, source: RegimeName, period: int, core: str
    ) -> tuple[ValueRead, ...]:
        if self.date.authority.period != period + 1:
            raise TypeError("Projected forward cannot borrow another child date.")
        return continuation_leaf_reads(
            template=self.date.template,
            artifact_key=self.date.authority.artifact_key,
            target=self.target,
            source_regime=source,
            source_period=period,
            core_key=core,
            argument_by_leaf=self.argument_by_leaf,
        )

    def operands(
        self, *, carry: ProjectedEGMCarry
    ) -> Mapping[QualifiedName, FloatND | IntND | BoolND]:
        require_projected_carry(date=self.date, target=self.target, carry=carry)
        leaves = carry.leaves()
        result = {}
        for path, argument in self.argument_by_leaf.items():
            leaf = leaves[path]
            if not isinstance(leaf, jax.Array):
                raise TypeError(
                    "Projected forward requires actual retained numerical leaves."
                )
            result[argument] = leaf
        return MappingProxyType(result)

    def read(
        self, *, coordinate: ScalarFloat, arguments: Mapping[QualifiedName, QAndFArg]
    ) -> tuple[ScalarFloat, BoolND]:
        from _lcm.egm.current_iid_case_u_balanced import (  # noqa: PLC0415 - avoid numerical declaration import cycles
            BalancedCaseUCarry,
        )

        if coordinate.ndim != 0:
            raise UnsupportedOperationError(
                "Projected forward requires a scalar landing."
            )
        carry = jax.tree.unflatten(
            jax.tree.structure(self.date.template),
            tuple(arguments[name] for name in self.native_arguments),
        )
        if not isinstance(carry, BalancedCaseUCarry):
            raise TypeError("Projected forward received a foreign numerical tree.")
        require_projected_carry(date=self.date, target=self.target, carry=carry)
        result = read_projected_coordinate(
            authority=self.date.authority,
            carry=carry,
            guard_program=self.date.guard_program,
            query=coordinate,
            # This is the identity liquid direction for the stored paired read;
            # forward uses no savings derivative or AD of projected value.
            jacobian=jnp.ones_like(coordinate),
        )
        valid = jnp.ones((), dtype=jnp.bool_)
        for status in result.query_status or ():
            for flag in status:
                valid &= ~jnp.any(flag)
        return result.output[0], valid


_QUERY_FLAG_COUNT = len(QueryInputStatus._fields)


type ProjectedForwardInputs = Mapping[int, Mapping[RegimeName, ProjectedEGMCarry]]


def projected_forward_operands(
    *,
    readers: Mapping[RegimeName, ProjectedForwardRead],
    inputs: ProjectedForwardInputs,
) -> Mapping[QualifiedName, FloatND | IntND | BoolND]:
    """Borrow only authenticated stored leaves named by actual reader bindings."""
    result = {}
    for reader in readers.values():
        ref = ArtifactRef(
            period=reader.date.authority.period,
            regime=reader.target,
            key=reader.date.authority.artifact_key,
        )
        if ref.regime not in inputs.get(ref.period, {}):
            raise ExecutionPlanningError(
                "Projected forward requires its retained original dated continuation."
            )
        result.update(reader.operands(carry=inputs[ref.period][ref.regime]))
    return MappingProxyType(result)


def consume_projected_forward_outputs(
    *, outputs: tuple[QAndFArg, ...], readers: Mapping[RegimeName, ProjectedForwardRead]
) -> tuple[QAndFArg, ...]:
    """Check actual scalar paired-read summaries before action interpretation."""
    if not readers:
        return outputs
    if len(readers) != 1 or len(outputs) <= _QUERY_FLAG_COUNT:
        raise TypeError("Projected forward lacks its single original reader summary.")
    reader = next(iter(readers.values()))
    nonfinite, invalid_grid, missing = outputs[-_QUERY_FLAG_COUNT:]
    if (
        not isinstance(nonfinite, jax.Array)
        or not isinstance(invalid_grid, jax.Array)
        or not isinstance(missing, jax.Array)
    ):
        raise TypeError("Projected forward integrity requires original array flags.")
    if any(
        flag.shape != () or flag.dtype != jnp.bool_
        for flag in (nonfinite, invalid_grid, missing)
    ):
        raise TypeError("Projected forward integrity requires scalar Boolean flags.")
    validate_query_input_status(
        statuses=(
            QueryInputStatus(
                nonfinite_floor=nonfinite,
                invalid_shifted_grid=invalid_grid,
                coverage_missing=missing,
            ),
        ),
        authorities=(
            (reader.target, reader.date.authority.period, reader.date.authority),
        ),
    )
    return outputs[:-_QUERY_FLAG_COUNT]
