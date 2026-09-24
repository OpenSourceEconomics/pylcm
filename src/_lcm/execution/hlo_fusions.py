"""Structural reading of an optimized HLO module's reduce fusions.

The GridSearch solve planner reads it to keep a continuation lookup fused into its
reduction, and the execution-settings tuner reads it to prune candidates untimed.
"""

import dataclasses
import re
from collections.abc import Mapping
from enum import Enum
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
# Producers whose result comes from a computation or runtime the classifier does
# not follow; a gather table one of them produced has no proved origin.
_HLO_OPAQUE_ATTRIBUTES = re.compile(
    r"\b(?:body|condition|branch_computations|true_computation|false_computation|"
    r"called_computations)="
)
_HLO_OPAQUE_OPCODES = frozenset({"call", "custom-call", "async-done"})


class UnrecognisedHloError(ValueError):
    """Raised when an HLO module's text is not read completely enough to classify."""


class ReduceFusionVerdict(Enum):
    """Where a compiled reduce fusion's gathered tables come from."""

    MATERIALISED_GATHER = "materialised_gather"
    """A gather reads its table from a separate gather's output in memory."""
    FUSED_GATHER = "fused_gather"
    """The fusion gathers, and no table's proved origin is another gather."""
    NO_GATHER = "no_gather"
    """The fusion gathers nothing."""
    UNKNOWN = "unknown"
    """The fusion, or the origin of one of its tables, was not read completely."""


def classify_reduce_fusions(hlo_text: str) -> Mapping[str, ReduceFusionVerdict]:
    """Classify each reduce fusion of an optimized HLO module by its gather tables.

    A reduce fusion is `MATERIALISED_GATHER` when one of its gathers reads its
    table from a fusion parameter that a separate gather produced, through any
    chain of copies, bitcasts and tuple projections on either side of the fusion
    boundary. One kernel then writes the gathered table to device memory and the
    reduction reads it back. When the compiler keeps that first gather inside the
    reduce fusion, the table is recomputed from its source and the fusion is
    `FUSED_GATHER`. Whether a gather result is reduced directly does not matter;
    only its use as a gather table does. The verdict reads the program's
    structure, never its shapes or widths.

    A negative verdict (`FUSED_GATHER`, `NO_GATHER`) needs every computation it
    reads to have parsed completely and every table's origin to be proved. A
    fusion for which either fails is `UNKNOWN`, never assumed fused.

    Args:
        hlo_text: One optimized module, as `jax.stages.Compiled.as_text()`
            prints it.

    Returns:
        The verdict of every input (reduce) fusion, keyed by instruction name. A
        completely parsed module without reduce fusions maps to nothing.

    Raises:
        UnrecognisedHloError: The text has no recognised entry computation, or an
            instruction outside any recognised computation.

    """
    module = _HloModule.parse(hlo_text)
    verdicts: dict[str, ReduceFusionVerdict] = {}
    for caller, instructions in module.computations.items():
        for name, call in instructions.items():
            if call.opcode != "fusion" or "kind=kInput" not in call.attributes:
                continue
            verdict = module.classify(call=call, caller=caller)
            if verdict is not None:
                verdicts[name] = verdict
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
    incomplete: frozenset[str]
    """Computations with a line the parser did not recognise, or without a root."""

    @classmethod
    def parse(cls, text: str) -> _HloModule:
        computations: dict[str, dict[str, _HloInstruction]] = {}
        roots: dict[str, str] = {}
        incomplete: set[str] = set()
        has_entry = False
        current: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            header = _HLO_COMPUTATION.match(line)
            if header is not None:
                current = header["name"]
                has_entry = has_entry or stripped.startswith("ENTRY")
                computations[current] = {}
                continue
            if current is None:
                if stripped.startswith(("%", "ROOT")):
                    msg = f"HLO instruction outside any computation: {stripped!r}"
                    raise UnrecognisedHloError(msg)
                continue
            if stripped == "}":
                current = None
                continue
            match = _HLO_INSTRUCTION.match(line)
            if match is None:
                if stripped:
                    incomplete.add(current)
                continue
            computations[current][match["name"]] = _HloInstruction(
                opcode=match["opcode"],
                operands=tuple(_HLO_OPERAND.findall(match["operands"])),
                raw_operands=match["operands"].strip(),
                attributes=match["attributes"],
            )
            if match["root"]:
                roots[current] = match["name"]
        if not has_entry:
            msg = "The HLO text has no recognised entry computation."
            raise UnrecognisedHloError(msg)
        return cls(
            computations=computations,
            roots=roots,
            incomplete=frozenset(incomplete | (set(computations) - set(roots))),
        )

    def classify(
        self, *, call: _HloInstruction, caller: str
    ) -> ReduceFusionVerdict | None:
        """Classify one input fusion; `None` when it is proved not to reduce."""
        callee = self._readable_callee(call)
        if callee is None or caller in self.incomplete:
            return ReduceFusionVerdict.UNKNOWN
        body = self.computations[callee]
        if not any(i.opcode == "reduce" for i in body.values()):
            return None
        gathers = [i for i in body.values() if i.opcode == "gather"]
        if not gathers:
            return ReduceFusionVerdict.NO_GATHER
        origins = {
            self._table_origin(gather=gather, body=body, call=call, caller=caller)
            for gather in gathers
        }
        if ReduceFusionVerdict.MATERIALISED_GATHER in origins:
            return ReduceFusionVerdict.MATERIALISED_GATHER
        if ReduceFusionVerdict.UNKNOWN in origins:
            return ReduceFusionVerdict.UNKNOWN
        return ReduceFusionVerdict.FUSED_GATHER

    def _readable_callee(self, call: _HloInstruction) -> str | None:
        """The computation `call` runs, when it was parsed completely."""
        found = _HLO_CALLS.search(call.attributes)
        if found is None or found["name"] not in self.computations:
            return None
        return None if found["name"] in self.incomplete else found["name"]

    def _table_origin(
        self,
        *,
        gather: _HloInstruction,
        body: Mapping[str, _HloInstruction],
        call: _HloInstruction,
        caller: str,
    ) -> ReduceFusionVerdict:
        """Classify where `gather`'s table comes from.

        - `MATERIALISED_GATHER`: another gather outside the fusion produced it.
        - `FUSED_GATHER`: its origin is proved to be anything else.
        - `UNKNOWN`: the chain leads somewhere the classifier does not read.
        """
        if not gather.operands:
            return ReduceFusionVerdict.UNKNOWN
        table = body.get(_skip_views(name=gather.operands[0], instructions=body))
        if table is None:
            return ReduceFusionVerdict.UNKNOWN
        if table.opcode != "parameter":
            return ReduceFusionVerdict.FUSED_GATHER
        try:
            operand = call.operands[int(table.raw_operands)]
        except ValueError, IndexError:
            return ReduceFusionVerdict.UNKNOWN
        return self._origin(name=operand, computation=caller, index=None)

    def _origin(
        self, *, name: str, computation: str, index: int | None
    ) -> ReduceFusionVerdict:
        """Classify the producer `name` leads back to.

        Views and tuple projections are followed inside the caller and, through a
        producing fusion, inside that fusion's root, so a gathered table reached
        through a root copy or bitcast keeps its origin.
        """
        instruction, index = _producer(
            name=name, instructions=self.computations[computation], index=index
        )
        if instruction is None:
            return ReduceFusionVerdict.UNKNOWN
        if instruction.opcode == "fusion":
            callee = self._readable_callee(instruction)
            if callee is None:
                return ReduceFusionVerdict.UNKNOWN
            return self._origin(
                name=self.roots[callee], computation=callee, index=index
            )
        if instruction.opcode == "gather" and index is None:
            return ReduceFusionVerdict.MATERIALISED_GATHER
        opaque = (
            index is not None
            or instruction.opcode in _HLO_OPAQUE_OPCODES
            or _HLO_OPAQUE_ATTRIBUTES.search(instruction.attributes) is not None
        )
        return (
            ReduceFusionVerdict.UNKNOWN if opaque else ReduceFusionVerdict.FUSED_GATHER
        )


def _producer(
    *, name: str, instructions: Mapping[str, _HloInstruction], index: int | None
) -> tuple[_HloInstruction | None, int | None]:
    """Follow views and tuple projections from `name` to the producing instruction.

    Returns the producer and the tuple element still to be selected from it, or
    `None` as the producer when the chain leaves what was parsed.
    """
    instruction = instructions.get(name)
    while instruction is not None:
        if instruction.opcode == "get-tuple-element" and index is None:
            found = _HLO_TUPLE_INDEX.search(instruction.attributes)
            if found is None:
                return None, None
            index = int(found["index"])
            operand = _operand(instruction=instruction, position=0)
        elif instruction.opcode == "tuple" and index is not None:
            operand = _operand(instruction=instruction, position=index)
            index = None
        elif instruction.opcode in _HLO_VIEWS:
            operand = _operand(instruction=instruction, position=0)
        else:
            break
        instruction = None if operand is None else instructions.get(operand)
    return instruction, index


def _operand(*, instruction: _HloInstruction, position: int) -> str | None:
    """Name `instruction`'s operand at `position`, if it has one."""
    if position < len(instruction.operands):
        return instruction.operands[position]
    return None


def _skip_views(*, name: str, instructions: Mapping[str, _HloInstruction]) -> str:
    """Follow bitcasts and copies back to the instruction they view."""
    while (instruction := instructions.get(name)) is not None and (
        instruction.opcode in _HLO_VIEWS
    ):
        name = instruction.operands[0]
    return name
