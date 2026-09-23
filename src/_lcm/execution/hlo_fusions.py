"""Structural reading of an optimized HLO module's reduce fusions.

The GridSearch solve planner reads it to keep a continuation lookup fused into its
reduction, and the execution-settings tuner reads it to prune candidates untimed.
"""

import dataclasses
import re
from collections.abc import Mapping
from enum import Enum
from functools import cached_property
from types import MappingProxyType

_HLO_COMPUTATION = re.compile(r"^(?:ENTRY\s+)?%(?P<name>[^\s(]+)\s+\(.*\{\s*$")
_HLO_INSTRUCTION = re.compile(
    r"^\s*(?P<root>ROOT\s+)?%(?P<name>\S+) = (?:\(.*?\)|\S+) "
    r"(?P<opcode>[\w-]+)\((?P<operands>[^)]*)\)(?P<attributes>.*)$"
)
_HLO_OPERAND = re.compile(r"%([^\s,)]+)")
_HLO_CALLS = re.compile(r"\bcalls=%(?P<name>[^\s,]+)")
_HLO_TUPLE_INDEX = re.compile(r"\bindex=(?P<index>\d+)")
_HLO_VIEWS = frozenset({"bitcast", "copy"})


class ReduceFusionVerdict(Enum):
    """Where a compiled reduce fusion's gathered tables come from."""

    MATERIALISED_GATHER = "materialised_gather"
    """A gather reads its table from a separate gather fusion's output in memory."""
    FUSED_GATHER = "fused_gather"
    """The fusion gathers, but no table is another gather fusion's output."""
    NO_GATHER = "no_gather"
    """The fusion gathers nothing."""


def classify_reduce_fusions(hlo_text: str) -> Mapping[str, ReduceFusionVerdict]:
    """Classify each reduce fusion of an optimized HLO module by its gather tables.

    A reduce fusion is `MATERIALISED_GATHER` when one of its gathers reads its
    table from a fusion parameter that a separate gather fusion produced. One
    kernel then writes the gathered table to device memory and the reduction
    reads it back. When the compiler keeps that first gather inside the reduce
    fusion, the table is recomputed from its source and the fusion is
    `FUSED_GATHER`. Whether a gather result is reduced directly does not matter;
    only its use as a gather table does. The verdict reads the program's
    structure, never its shapes or widths.

    Args:
        hlo_text: One optimized module, as `jax.stages.Compiled.as_text()`
            prints it.

    Returns:
        The verdict of every input (reduce) fusion, keyed by instruction name.

    """
    module = _HloModule.parse(hlo_text)
    verdicts: dict[str, ReduceFusionVerdict] = {}
    for name, call in module.instructions.items():
        body = module.computations.get(_called(call), {})
        if (
            call.opcode != "fusion"
            or "kind=kInput" not in call.attributes
            or not any(i.opcode == "reduce" for i in body.values())
        ):
            continue
        gathers = [i for i in body.values() if i.opcode == "gather"]
        if not gathers:
            verdicts[name] = ReduceFusionVerdict.NO_GATHER
        elif any(
            module.reads_gathered_table(gather=gather, body=body, call=call)
            for gather in gathers
        ):
            verdicts[name] = ReduceFusionVerdict.MATERIALISED_GATHER
        else:
            verdicts[name] = ReduceFusionVerdict.FUSED_GATHER
    return MappingProxyType(verdicts)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _HloInstruction:
    """One instruction of an optimized HLO module, as far as the classifier reads it."""

    opcode: str
    """The HLO opcode."""
    operands: tuple[str, ...]
    """Operand instruction names, in order; empty for a `parameter`."""
    raw_operands: str
    """The text between the parentheses; a `parameter`'s number."""
    attributes: str
    """Everything after the operand list."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class _HloModule:
    """The computations of an optimized HLO module and their root instructions."""

    computations: Mapping[str, Mapping[str, _HloInstruction]]
    """Instructions of each computation, by name."""
    roots: Mapping[str, str]
    """Name of each computation's root instruction."""

    @cached_property
    def instructions(self) -> dict[str, _HloInstruction]:
        """Every instruction of the module, by name; names are module-unique."""
        return {
            name: instruction
            for body in self.computations.values()
            for name, instruction in body.items()
        }

    @classmethod
    def parse(cls, text: str) -> _HloModule:
        computations: dict[str, dict[str, _HloInstruction]] = {}
        roots: dict[str, str] = {}
        current: str | None = None
        for line in text.splitlines():
            header = _HLO_COMPUTATION.match(line)
            if header is not None:
                current = header["name"]
                computations[current] = {}
                continue
            match = _HLO_INSTRUCTION.match(line)
            if current is None or match is None:
                continue
            computations[current][match["name"]] = _HloInstruction(
                opcode=match["opcode"],
                operands=tuple(_HLO_OPERAND.findall(match["operands"])),
                raw_operands=match["operands"].strip(),
                attributes=match["attributes"],
            )
            if match["root"]:
                roots[current] = match["name"]
        return cls(computations=computations, roots=roots)

    def reads_gathered_table(
        self,
        *,
        gather: _HloInstruction,
        body: Mapping[str, _HloInstruction],
        call: _HloInstruction,
    ) -> bool:
        """Whether `gather`'s table enters the fusion as another gather's output."""
        table = body.get(_skip_views(name=gather.operands[0], instructions=body))
        if table is None or table.opcode != "parameter":
            return False
        source, index = self._source(name=call.operands[int(table.raw_operands)])
        producer = self.instructions.get(source)
        computation = None if producer is None else _called(producer)
        if producer is None or producer.opcode != "fusion" or computation is None:
            return False
        fused = self.computations[computation]
        result = fused[self.roots[computation]]
        if result.opcode == "tuple" and index is not None:
            result = fused[_skip_views(name=result.operands[index], instructions=fused)]
        return result.opcode == "gather"

    def _source(self, *, name: str) -> tuple[str, int | None]:
        """Follow tuple reads and views back to the producing instruction."""
        instructions = self.instructions
        index = None
        while (instruction := instructions.get(name)) is not None:
            if instruction.opcode == "get-tuple-element":
                found = _HLO_TUPLE_INDEX.search(instruction.attributes)
                index = None if found is None else int(found["index"])
            elif instruction.opcode not in _HLO_VIEWS:
                break
            name = instruction.operands[0]
        return name, index


def _called(instruction: _HloInstruction) -> str | None:
    found = _HLO_CALLS.search(instruction.attributes)
    return None if found is None else found["name"]


def _skip_views(*, name: str, instructions: Mapping[str, _HloInstruction]) -> str:
    """Follow bitcasts and copies back to the instruction they view."""
    while (instruction := instructions.get(name)) is not None and (
        instruction.opcode in _HLO_VIEWS
    ):
        name = instruction.operands[0]
    return name
