"""Typed records for serialized analysis and mutable construction buffers."""
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
    projection_family_complete_verified: bool
    trusted_boundary: list[str]


class CheckLayer(TypedDict):
    status: str
    checks: int
    notes: list[str]
    witnesses: NotRequired[list[JSONValue]]


class LinearObjective(TypedDict):
    constant: int
    terms: list[tuple[str, int]]


class PolynomialConstraint(TypedDict):
    terms: list[tuple[str, int]]
    relation: str
    rhs: int


class SignatureModel(TypedDict, total=False):
    schema: str
    variables: list[str]
    constraints: list[PolynomialConstraint]
    objectives: dict[str, LinearObjective]
    metadata: dict[str, JSONValue]
    instance_sha256: str
    cell_order: list[str]
    bins: list[JSONValue]
    pareto_complete_representation: bool
    full_plan_set_preserved: bool
    frontier_complete_verified: bool
    guard: str


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


class NumericalRow(TypedDict):
    reward: Fraction
    discount: Fraction
    terminal: Fraction | None
    kernel: dict[str, Fraction]


class NumericalReport(TypedDict, total=False):
    format_version: str
    status: str
    guarantee: str
    layers: dict[str, str]
    comparisons: list[dict[str, JSONValue]]
    mass_errors: dict[str, dict[str, str]]
    witnesses: list[dict[str, JSONValue]]
    positive_edges: list[dict[str, str]]
    trust_boundary: str
    contract: dict[str, JSONValue]
    maxima: dict[str, str]
    max_errors: dict[str, str]


class CandidateCheck(TypedDict, total=False):
    status: str
    checking_complete: bool


class Scenario(TypedDict, total=False):
    allocation_contract: Mapping[str, JSONValue]
    status: str
    error: str
    frontier: list[dict[str, JSONValue]]
    independent_check: CandidateCheck


class RenderPoint(TypedDict):
    ordinal: int
    prefix: str
    point: dict[str, JSONValue]
    graph: dict[str, JSONValue]
    cost_preferred: bool
    decision: dict[str, JSONValue]
    consequences: dict[str, JSONValue]


class RenderScenario(TypedDict):
    ordinal: int
    title: str
    conditional: bool
    scenario: dict[str, JSONValue]
    points: list[RenderPoint]


class BaselineBlock(TypedDict):
    component: int
    label_group_mask: int
    cells: list[str]
    cost: tuple[int, ...]
    option_by_cell: dict[str, str]


class DecisionSummary(TypedDict, total=False):
    operation: str
    scenario: int
    graph: str
    decision: str
    changed_interfaces: list[dict[str, JSONValue]]
    remaining_kinds: dict[str, int]
    examples: list[dict[str, JSONValue]]
    reasons: list[str]


class EndpointPreference(TypedDict, total=False):
    scenario: int
    recommended_graph: str | None
    allow_final_only_split: bool
    reason: str
    deferred_graph_ids: list[str]
