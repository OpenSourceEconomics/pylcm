"""Whole bounded handovers preserve structure and every tracing-relevant leaf."""

from collections.abc import Hashable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import InternalOutputSpec
from _lcm.execution.internal_outputs import (
    ResolvedProducer,
    assert_internal_inputs,
    assert_width_invariant_internal_outputs,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    ContinuationReadCounts,
    QueryInputStatus,
)
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND, ValueND

type _Payload = ValueND | tuple[ValueND]
type _Change = Literal[
    "shape",
    "dtype",
    "weak_type",
    "output_structure",
    "query_presence",
    "count_presence",
    "bound_slots",
]

_CHANGES: tuple[_Change, ...] = (
    "shape",
    "dtype",
    "weak_type",
    "output_structure",
    "query_presence",
    "count_presence",
    "bound_slots",
)


def _record(*, checked: bool) -> BoundedResult[_Payload]:
    weak = jnp.asarray(1.0)
    inactive = jnp.asarray(a=False)
    return BoundedResult(
        output=jnp.asarray(1.0, dtype=weak.dtype),
        bound_status=(BoundInputStatus(inactive, inactive),),
        query_status=(QueryInputStatus(inactive, inactive, inactive),),
        continuation_read_counts=(
            ContinuationReadCounts(jnp.int32(2), jnp.int32(3)) if checked else None
        ),
    )


def _abstract_leaf(leaf: ValueND) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, weak_type=leaf.weak_type)


def _template(*, record: BoundedResult[_Payload]) -> ShapeDtypePytree:
    return cast("ShapeDtypePytree", jax.tree.map(_abstract_leaf, record))


def _changed_record(
    *,
    record: BoundedResult[_Payload],
    change: _Change,
) -> BoundedResult[_Payload]:
    value = cast("FloatND", record.output)
    if change in ("shape", "dtype"):
        changed = value[None] if change == "shape" else value.astype(jnp.int32)
        return replace(record, output=changed)
    if change == "weak_type":
        weak = jnp.asarray(1.0)
        assert weak.dtype == value.dtype
        assert weak.shape == value.shape
        assert weak.weak_type
        assert not value.weak_type
        return replace(record, output=weak)
    if change == "output_structure":
        return replace(record, output=(value,))
    if change == "query_presence":
        return replace(record, query_status=None)
    if change == "count_presence":
        return replace(
            record,
            continuation_read_counts=(
                ContinuationReadCounts(jnp.int32(2), jnp.int32(3))
                if record.continuation_read_counts is None
                else None
            ),
        )
    return replace(record, bound_status=(*record.bound_status, *record.bound_status))


@pytest.mark.parametrize("checked", [False, True], ids=["ordinary", "checked"])
def test_whole_bounded_handover_matches_its_abstract_template(*, checked: bool) -> None:
    """A separately reconstructed record with matching structure and leaves is
    accepted.
    """
    template = _template(record=_record(checked=checked))
    actual = _record(checked=checked)
    assert (
        assert_internal_inputs(
            arguments={"upstream": actual},
            templates={"upstream": template},
            label="consumer",
        )
        is None
    )


@pytest.mark.parametrize("checked", [False, True], ids=["ordinary", "checked"])
@pytest.mark.parametrize("change", _CHANGES)
def test_whole_bounded_handover_refuses_changed_signature(
    *,
    checked: bool,
    change: _Change,
) -> None:
    """Structure, diagnostic presence and each leaf's tracing metadata stay strict."""
    original = _record(checked=checked)
    changed = _changed_record(record=original, change=change)
    with pytest.raises(ValueError, match="internal input 'upstream'"):
        assert_internal_inputs(
            arguments={"upstream": changed},
            templates={"upstream": _template(record=original)},
            label="consumer",
        )


def _identity(*, value: PytreeValue) -> PytreeValue:
    return value


def _producer(
    *,
    record: BoundedResult[_Payload],
    width: int,
) -> ResolvedProducer:
    return ResolvedProducer(
        name="producer",
        function=_identity,
        internal_input_templates=MappingProxyType({}),
        static_kwargs=MappingProxyType({"width": width}),
        internal_outputs=(InternalOutputSpec(label="result", path=()),),
        abstract_output=_template(record=record),
    )


@pytest.mark.parametrize("checked", [False, True], ids=["ordinary", "checked"])
def test_whole_bounded_output_is_invariant_across_widths(*, checked: bool) -> None:
    """Distinct records with the same whole-output signature admit either width."""
    candidates: Mapping[Hashable, ResolvedProducer] = {
        width: _producer(record=_record(checked=checked), width=width)
        for width in (2, 4)
    }
    assert assert_width_invariant_internal_outputs(candidates=candidates) is None


@pytest.mark.parametrize("checked", [False, True], ids=["ordinary", "checked"])
@pytest.mark.parametrize("change", _CHANGES)
def test_whole_bounded_output_refuses_width_dependent_signature(
    *,
    checked: bool,
    change: _Change,
) -> None:
    """A width cannot change whole-output structure, diagnostics or tracing metadata."""
    original = _record(checked=checked)
    candidates: Mapping[Hashable, ResolvedProducer] = {
        2: _producer(record=original, width=2),
        4: _producer(record=_changed_record(record=original, change=change), width=4),
    }
    with pytest.raises(
        ExecutionPlanningError, match="publishes internal output 'result'"
    ):
        assert_width_invariant_internal_outputs(candidates=candidates)
