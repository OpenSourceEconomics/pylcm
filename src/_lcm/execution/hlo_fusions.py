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
_HLO_REFERENCE = re.compile(r"%(?P<name>[^\s,}]+)")
_HLO_PASSED_CALLEE = re.compile(r"\b(?P<key>calls|to_apply|body)=%(?P<name>[^\s,}]+)")
# The attribute through which each caller passes its operands to a computation's
# parameters; a while loop passes its one operand as the body's loop state.
_HLO_PASSING_KEYS = MappingProxyType(
    {"fusion": "calls", "call": "to_apply", "while": "body"}
)
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
    fusion for which either fails is `UNKNOWN`, never assumed fused. A table
    reached through a tuple projection needs its origin proved outside the
    fusion. A table reaching a computation as a parameter is proved only at an
    entry parameter, or through the one fusion, call or while loop calling that
    computation; a while loop passes it on only when its body carries that
    table unchanged. An empty result needs every computation that can call a
    fusion to have parsed completely, so no reduce fusion goes unlisted.

    Args:
        hlo_text: One optimized module, as `jax.stages.Compiled.as_text()`
            prints it.

    Returns:
        The verdict of every input (reduce) fusion, keyed by instruction name. A
        completely parsed module without reduce fusions maps to nothing.

    Raises:
        UnrecognisedHloError: The text has no recognised entry computation, an
            instruction outside any recognised computation, or a line the parser
            does not read in a computation other than a fusion body.

    """
    module = _HloModule.parse(hlo_text)
    fusion_bodies = {
        found["name"]
        for instructions in module.computations.values()
        for instruction in instructions.values()
        if instruction.opcode == "fusion"
        and (found := _HLO_CALLS.search(instruction.attributes)) is not None
    }
    unlisted = sorted(module.incomplete - fusion_bodies)
    if unlisted:
        msg = f"Computations {unlisted!r} were not read completely."
        raise UnrecognisedHloError(msg)
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
    entry: str
    """Name of the entry computation."""
    callers: Mapping[str, tuple[tuple[str, _HloInstruction], ...]]
    """Instructions naming each computation in their attributes, with their caller."""

    @classmethod
    def parse(cls, text: str) -> _HloModule:
        computations: dict[str, dict[str, _HloInstruction]] = {}
        roots: dict[str, str] = {}
        incomplete: set[str] = set()
        entry: str | None = None
        current: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            header = _HLO_COMPUTATION.match(line)
            if header is not None:
                current = header["name"]
                entry = current if stripped.startswith("ENTRY") else entry
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
        if entry is None:
            msg = "The HLO text has no recognised entry computation."
            raise UnrecognisedHloError(msg)
        return cls(
            computations=computations,
            roots=roots,
            incomplete=frozenset(incomplete | (set(computations) - set(roots))),
            entry=entry,
            callers=_callers(computations),
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

        Views and tuple projections are followed inside the fusion; a projection
        still unresolved at a fusion parameter is resolved in the caller.
        """
        if not gather.operands:
            return ReduceFusionVerdict.UNKNOWN
        table, index = _producer(name=gather.operands[0], instructions=body, index=None)
        if table is None:
            return ReduceFusionVerdict.UNKNOWN
        if table.opcode != "parameter":
            return (
                ReduceFusionVerdict.UNKNOWN
                if index is not None
                else ReduceFusionVerdict.FUSED_GATHER
            )
        try:
            operand = call.operands[int(table.raw_operands)]
        except ValueError, IndexError:
            return ReduceFusionVerdict.UNKNOWN
        return self._origin(name=operand, computation=caller, index=index)

    def _origin(
        self, *, name: str, computation: str, index: int | None
    ) -> ReduceFusionVerdict:
        """Classify the producer `name` leads back to.

        Views and tuple projections are followed inside the caller and, through a
        producing fusion, inside that fusion's root, so a gathered table reached
        through a root copy or bitcast keeps its origin. A parameter is followed
        to the operand its caller passes.
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
        if instruction.opcode == "parameter":
            return self._parameter_origin(
                parameter=instruction, computation=computation, index=index
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

    def _parameter_origin(
        self, *, parameter: _HloInstruction, computation: str, index: int | None
    ) -> ReduceFusionVerdict:
        """Classify a parameter of `computation` by what its one caller passes.

        - An entry parameter is a proved origin unless a tuple element is still
          to be selected from it.
        - Through the one fusion or call calling `computation`, the parameter is
          the operand at its position.
        - Through the one while loop running `computation` as its body, the loop
          state is the loop's operand, when the body's root carries the selected
          element unchanged.
        - Anything else is `UNKNOWN`.
        """
        if computation == self.entry:
            return (
                ReduceFusionVerdict.FUSED_GATHER
                if index is None
                else ReduceFusionVerdict.UNKNOWN
            )
        passed = self._passed_operand(
            parameter=parameter, computation=computation, index=index
        )
        if passed is None:
            return ReduceFusionVerdict.UNKNOWN
        caller, operand = passed
        return self._origin(name=operand, computation=caller, index=index)

    def _passed_operand(
        self, *, parameter: _HloInstruction, computation: str, index: int | None
    ) -> tuple[str, str] | None:
        """The one caller of `computation` and the operand it passes to `parameter`.

        `None` when `computation` has other than one caller, either was not read
        completely, or a while body does not carry element `index` unchanged.
        """
        sites = self.callers.get(computation, ())
        if len(sites) != 1 or computation in self.incomplete:
            return None
        caller, site = sites[0]
        passed = {
            found["key"]: found["name"]
            for found in _HLO_PASSED_CALLEE.finditer(site.attributes)
        }
        key = _HLO_PASSING_KEYS.get(site.opcode)
        if (
            key is None
            or passed.get(key) != computation
            or caller in self.incomplete
            or not parameter.raw_operands.isdigit()
        ):
            return None
        position = int(parameter.raw_operands)
        if site.opcode == "while":
            carried, carried_index = _producer(
                name=self.roots[computation],
                instructions=self.computations[computation],
                index=index,
            )
            if position != 0 or carried is not parameter or carried_index != index:
                return None
        operand = _operand(instruction=site, position=position)
        return None if operand is None else (caller, operand)


def _callers(
    computations: Mapping[str, Mapping[str, _HloInstruction]],
) -> Mapping[str, tuple[tuple[str, _HloInstruction], ...]]:
    """Map each computation to the instructions naming it, with their caller."""
    callers: dict[str, list[tuple[str, _HloInstruction]]] = {}
    for caller, instructions in computations.items():
        for instruction in instructions.values():
            named = {
                found["name"]
                for found in _HLO_REFERENCE.finditer(instruction.attributes)
            }
            for callee in named & set(computations):
                callers.setdefault(callee, []).append((caller, instruction))
    return MappingProxyType({callee: tuple(sites) for callee, sites in callers.items()})


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
