"""Typed records for serialized analysis and mutable construction buffers."""
import hashlib
import json
from fractions import Fraction
from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

type JSONValue = bool | int | float | str | Sequence[JSONValue] | Mapping[str, JSONValue] | None


class Diagnostic(TypedDict, total=False):
    kind: str
    scope: str
    cell: str
    callback: str
    detail: str


class SemanticCheck(TypedDict, total=False):
    status: str
    errors: list[Diagnostic]
    unresolved: list[Diagnostic]
    checks: int
    source_sha256: str
    normalized_sha256: str
    guarantee: str
    dense_cartesian_rows_evaluated: int
    trusted_boundary: list[str]


class CheckLayer(TypedDict):
    status: str
    checks: int
    notes: list[str]
    witnesses: NotRequired[list[JSONValue]]


class DependencyEdge(TypedDict, total=False):
    source: str
    target: str
    kind: str
    phase: str
    offset: int
    updates: dict[str, JSONValue]
    status_updates: dict[str, JSONValue]
    weight_expression: JSONValue
    probability: JSONValue
    weight: JSONValue
    when: JSONValue
    source_path: str
    source_anchor: JSONValue


class Program(TypedDict, total=False):
    transitions: list[DependencyEdge]
    reads: list[DependencyEdge]


class PreparedAtom(TypedDict, total=False):
    id: str
    age: int
    age_index: int
    status: dict[str, JSONValue]
    channel: str
    original_regime: str
    declared_states: list[str]
    declared_actions: list[str]
    constant_states: dict[str, JSONValue]
    forced_actions: dict[str, JSONValue]
    live_states: set[str] | list[str]
    free_actions: list[str]
    primitive_reads: dict[str, list[str]]
    forcing: Mapping[str, JSONValue]
    role: dict[str, JSONValue]
    routing_owner: str | None
    terminal: bool
    allocation: str
    source_anchor: JSONValue
    retention_reasons: dict[str, list[str]]
    dropped_states: list[str]
    specialized_interfaces: dict[str, list[str]]
    _env: dict[str, JSONValue]
    _program: Program
    _direct: set[str]


class IntervalCandidate(TypedDict):
    start: int
    stop: int
    segment: Mapping[str, JSONValue]


def canonical(value: object) -> str:
    """Return the sorted, compact JSON text that identities are digested from."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    """Return the sha256 of the canonical JSON text."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()
