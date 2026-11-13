"""Read fibre facts from authenticated original declaration metadata."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from types import MappingProxyType
from typing import Literal

from .declarations import Normalizer, _select, fields, normalize_declarations, pointer
from .fibre_facts import (
    CoordinateName,
    CoordinateRead,
    FibreLink,
    FibreReadFacts,
    JointReadSet,
    get_joint_read_set,
)
from .graph_scope import AR1_TYPES, IID_TYPES
from .records import JSONValue
from .semantic_check import (
    SourceProof,
    _Abstract,
    _CallableRecords,
    _UNKNOWN,
    _check_capture,
    _codes,
    _constant,
    _original_transition_contexts,
    _pointer,
    _providers,
    _static,
    _value_key,
)

type CellId = str
type RelationId = str
type CallbackId = str
type Wire = dict[str, JSONValue]
type Phase = Literal["solve", "simulation"]


@dataclass(frozen=True)
class _Primitive:
    callback: CallbackId | None
    """Original callback, or a missing producer that must remain conservative."""
    phase: Phase
    """Primitive channel selecting the original dependency DAG."""
    relation: RelationId | None = None
    """Original source-target incidence supplying any next-state providers."""


@dataclass(frozen=True)
class TargetSupport:
    """Retain every target code not excluded by an original deterministic law."""

    possible_codes: frozenset[int]
    """Possible or unproved target codes; no probability renormalisation."""
    proved: bool
    """Whether an original deterministic update fixes the complete support."""


@dataclass(frozen=True, init=False)
class FibreSource:
    """Inspect original metadata without executing any economic callback.

    Primitive certificates keep ordinary coordinates, other actions, shocks and
    runtime parameters free. Source authentication is independent of producer
    read sets. Unsupported primitives conservatively retain all source states.
    """

    _cells: Mapping[CellId, Wire]
    """Owned normalized records of the original admitted cells."""
    _relations: Mapping[RelationId, Wire]
    """Original transitions and within-period computations with source pointers."""
    _domains: Mapping[str, Wire]
    """Unchanged original grid declarations indexed by their identities."""
    _callbacks: Mapping[CallbackId, Wire]
    """Original callback records; arbitrary producer read sets are not authority."""
    _roots: Mapping[CellId, tuple[_Primitive, ...]]
    """Direct consumers of every required primitive channel."""
    _constraints: Mapping[CellId, tuple[_Primitive, ...]]
    """Original feasible-set predicates in each primitive channel."""
    _contexts: Mapping[tuple[int, str], dict[str, str]]
    """Incidence-specific next-state providers, keyed by record identity and phase."""
    _captures: _CallableRecords
    """Validated immutable captured callables with local interpretation caches."""
    source_obligations: tuple[str, ...]
    """Original-source qualifications, kept separate from dependency facts."""

    def __init__(self, original: dict[str, JSONValue]) -> None:
        source = _mapping(original.get("graph_snapshot", original))
        if source.get("schema") != "constructed-pylcm-model-graph-1":
            raise ValueError("Fibre input must be original constructed-Model metadata")
        declarations = normalize_declarations(original)
        failures: list[str] = []
        qualifications: list[str] = []

        def fail(kind: str, **details: JSONValue) -> None:
            failures.append(f"{kind}: {details}")

        def unknown(kind: str, **details: JSONValue) -> None:
            qualifications.append(f"{kind}: {details}")

        _check_capture(original, declarations, fail, unknown)
        if failures:
            raise ValueError(
                "Original capture authentication failed: " + "; ".join(failures)
            )
        cells = {_text(row["id"]): row for row in _rows(declarations["cells"])}
        relations = {_text(row["id"]): row for row in _rows(declarations["relations"])}
        qualifications.extend(
            f"policy query owner is outside admitted fibre cells: {relation_id}"
            for relation_id, relation in relations.items()
            if relation.get("kind") == "policy_query"
        )
        domains = {
            name: _mapping(value)
            for name, value in _mapping(declarations["domains"]).items()
        }
        callbacks = {
            name: _mapping(value)
            for name, value in _mapping(declarations["callbacks"]).items()
        }
        normal = Normalizer(original, None)
        roots: dict[CellId, tuple[_Primitive, ...]] = {}
        constraints: dict[CellId, tuple[_Primitive, ...]] = {}
        for cell_id, cell in cells.items():
            raw = _mapping(_pointer(original, _text(cell["source_path"])))
            body = _mapping(fields(raw))
            base = (
                pointer(cell["source_path"], "fields")
                if raw.get("kind") in {"object", "dataclass"}
                else cell["source_path"]
            )
            direct: list[_Primitive] = []
            feasible: list[_Primitive] = []
            for consumer in _rows(cell["consumers"]):
                for phase in _phases(consumer.get("phase", "both")):
                    direct.append(_Primitive(_text(consumer["callback"]), phase))
            for phase in ("solve", "simulation"):
                phase_functions = _mapping(_mapping(cell["phase_functions"])[phase])
                outputs = (
                    "utility",
                    "discount_factor",
                    *_strings(body.get("output_only_functions", [])),
                )
                for output in outputs:
                    if output in phase_functions:
                        direct.append(_Primitive(_text(phase_functions[output]), phase))
                for slot in (
                    "koopmans_aggregator",
                    "certainty_equivalent",
                    "pareto_objective",
                ):
                    if body.get(slot) is not None:
                        selected = _select(
                            body[slot], phase, cell["period"], cell.get("age")
                        )
                        direct.append(
                            _Primitive(
                                normal.callback(selected, pointer(base, slot), phase),
                                phase,
                            )
                        )
                for slot in ("constraints", "value_constraints"):
                    for name, value in _mapping(body.get(slot, {})).items():
                        selected = _select(
                            value, phase, cell["period"], cell.get("age")
                        )
                        if isinstance(selected, dict) and str(
                            selected.get("type", "")
                        ).endswith("ValueDependentConstraint"):
                            selected = fields(selected).get("predicate")
                        primitive = _Primitive(
                            normal.callback(
                                selected, pointer(pointer(base, slot), name), phase
                            ),
                            phase,
                        )
                        direct.append(primitive)
                        feasible.append(primitive)
            for relation_id, relation in relations.items():
                if relation["source"] == cell_id:
                    direct.extend(
                        _Primitive(callback, phase, relation_id)
                        for callback in _strings(relation.get("callbacks", []))
                        for phase in _phases(relation.get("phase", "both"))
                    )
            roots[cell_id] = tuple(dict.fromkeys(direct))
            constraints[cell_id] = tuple(dict.fromkeys(feasible))
        callbacks.update(normal.callbacks)
        # Only typed library declarations have an opaque, authoritative input contract.
        for callback in callbacks.values():
            raw = (
                _mapping(callback.get("original", {}))
                if isinstance(callback.get("original"), Mapping)
                else {}
            )
            tag = raw.get("type")
            typed = isinstance(tag, str) and tag in (
                IID_TYPES
                | AR1_TYPES
                | {
                    "lcm.shocks.ShockGroup",
                    "lcm.decoupled.Decoupled",
                    "lcm.koopmans_aggregation.LinearAggregator",
                    "lcm.koopmans_aggregation.CESAggregator",
                    "_lcm.certainty_equivalent.LinearExpectation",
                    "_lcm.identity_transition._IdentityTransition",
                }
            )
            if typed and callback.get("status") != "unresolved":
                callback["dependency_contract"] = "pylcm-declared-inputs/v1"
                if callback.get("status") != "proved":
                    callback["status"] = "structural"
        captures = _CallableRecords(declarations.get("callable_records", {}))
        if captures.errors:
            raise ValueError(
                "Original callable capture changed: " + "; ".join(captures.errors)
            )
        contexts = _original_transition_contexts(
            relations=tuple(relations.values()), callbacks=callbacks
        )
        for name, value in (
            ("_cells", cells),
            ("_relations", relations),
            ("_domains", domains),
            ("_callbacks", callbacks),
            ("_roots", roots),
            ("_constraints", constraints),
            ("_contexts", contexts),
        ):
            object.__setattr__(self, name, MappingProxyType(value))
        object.__setattr__(self, "_captures", captures)
        object.__setattr__(self, "source_obligations", tuple(qualifications))

    def get_cell_ids(self) -> tuple[CellId, ...]:
        """Return admitted original cells in declaration order."""
        return tuple(self._cells)

    def get_relation_ids(self) -> tuple[RelationId, ...]:
        """Return all original dependency incidences, including value reads."""
        return tuple(self._relations)

    def get_cell_facts(self) -> tuple[FibreReadFacts, ...]:
        """Build unspecialised dependency facts for the original admitted cells.

        These give a conservative baseline before status refinement. Joint and
        ordinary updates of the same source/target/phase share one incidence.
        Imputations and gates are computations, not recursive value reads.
        Query-only owners stay source qualifications; their source states cannot
        be removed by assuming the missing target is irrelevant.
        """
        facts = []
        query_guard = any(
            relation.get("kind") == "policy_query"
            for relation in self._relations.values()
        )
        value_kinds = {"transition", "joint", "value_read", "fallback", "exit"}
        for cell_id, cell in self._cells.items():
            coordinates = frozenset(_mapping(cell["states"]))
            direct = (
                JointReadSet(coordinates, (), ())
                if query_guard
                else self.get_direct_reads(
                    cell_id=cell_id, statuses={}, removal_order=()
                )
            )
            groups: dict[tuple[CellId, Phase, str], list[RelationId]] = {}
            for relation_id, relation in self._relations.items():
                if (
                    relation["source"] != cell_id
                    or relation.get("kind") not in value_kinds
                ):
                    continue
                group = (
                    "transition"
                    if relation["kind"] in {"transition", "joint"}
                    else relation_id
                )
                for target in _strings(relation.get("targets", [])):
                    for phase in _phases(relation.get("phase", "both")):
                        groups.setdefault((target, phase, group), []).append(
                            relation_id
                        )
            links = []
            for (target, phase, _), relation_ids in groups.items():
                updates = []
                target_states = _mapping(self._cell(target)["states"])
                for coordinate in target_states:
                    producers = [
                        relation_id
                        for relation_id in relation_ids
                        if coordinate
                        in _mapping(
                            self._relations[relation_id].get("state_updates", {})
                        )
                    ]
                    if len(producers) > 1:
                        raise ValueError(
                            f"Multiple original update producers: {cell_id} -> {target}: {coordinate}"
                        )
                    reads = (
                        self.get_update_reads(
                            relation_id=producers[0],
                            target_coordinate=coordinate,
                            statuses={},
                            removal_order=(),
                        )
                        if producers
                        else JointReadSet(coordinates, (), ())
                    )
                    updates.append(CoordinateRead(coordinate=coordinate, reads=reads))
                links.append(
                    FibreLink(
                        target=target,
                        updates=tuple(updates),
                        phase=phase,
                        source_relations=tuple(relation_ids),
                    )
                )
            facts.append(
                FibreReadFacts(
                    identifier=cell_id,
                    coordinates=coordinates,
                    direct=direct,
                    links=tuple(links),
                )
            )
        return tuple(facts)

    def get_direct_reads(
        self,
        *,
        cell_id: CellId,
        statuses: Mapping[CoordinateName, int],
        removal_order: tuple[CoordinateName, ...],
    ) -> JointReadSet:
        """Certify all direct consumers across solve and both simulation channels."""
        cell = self._cell(cell_id)
        self._validate_statuses(cell=cell, statuses=statuses)
        return self._joint_reads(
            cell=cell,
            primitives=self._roots[cell_id],
            statuses=statuses,
            removal_order=removal_order,
        )

    def get_update_reads(
        self,
        *,
        relation_id: RelationId,
        target_coordinate: CoordinateName,
        statuses: Mapping[CoordinateName, int],
        removal_order: tuple[CoordinateName, ...],
    ) -> JointReadSet:
        """Read a target-coordinate update in its original source/phase context."""
        relation, cell = self._relation(relation_id)
        self._target_domain(relation=relation, target_coordinate=target_coordinate)
        self._validate_statuses(cell=cell, statuses=statuses)
        callback = _mapping(relation.get("state_updates", {})).get(target_coordinate)
        primitives = tuple(
            _Primitive(
                _text(callback) if callback is not None else None, phase, relation_id
            )
            for phase in _phases(relation.get("phase", "both"))
        )
        return self._joint_reads(
            cell=cell,
            primitives=primitives,
            statuses=statuses,
            removal_order=removal_order,
        )

    def get_target_support(
        self,
        *,
        relation_id: RelationId,
        target_coordinate: CoordinateName,
        statuses: Mapping[CoordinateName, int],
    ) -> TargetSupport:
        """Exclude target codes only from a proved original deterministic update."""
        relation, cell = self._relation(relation_id)
        domain = self._target_domain(
            relation=relation, target_coordinate=target_coordinate
        )
        self._validate_statuses(cell=cell, statuses=statuses)
        pairs = _code_values(domain)
        full = frozenset(code for code, _ in pairs)
        callback_id = _mapping(relation.get("state_updates", {})).get(target_coordinate)
        if callback_id is None:
            return TargetSupport(full, False)
        callback = self._callbacks[_text(callback_id)]
        raw = _mapping(callback.get("original", {}))
        resolved = self._captures.resolve(raw)
        if not isinstance(resolved, Mapping):
            return TargetSupport(full, False)
        identity = _identity_name(resolved)
        deterministic = identity is not None or (
            resolved.get("kind") == "callable" and not resolved.get("bound_self")
        )
        if not deterministic:
            return TargetSupport(full, False)
        support: set[int] = set()
        for phase in _phases(relation.get("phase", "both")):
            answer = self._evaluate(
                cell=cell,
                primitive=_Primitive(_text(callback_id), phase, relation_id),
                fixed_statuses=statuses,
            )
            if not answer.proved or answer.value is _UNKNOWN:
                return TargetSupport(full, False)
            matching = {
                code
                for code, value in pairs
                if _value_key(value) == _value_key(answer.value)
            }
            if not matching:
                raise ValueError(
                    f"Original deterministic update leaves target grid: {target_coordinate}"
                )
            support.update(matching)
        return TargetSupport(frozenset(support), True)

    def get_excluded_action_values(
        self,
        *,
        cell_id: CellId,
        action: CoordinateName,
        status_values: Mapping[CoordinateName, tuple[int, ...]],
    ) -> frozenset[int]:
        """Prove exclusion on the full status rectangle, including nonmember points."""
        cell = self._cell(cell_id)
        actions = _mapping(cell["actions"])
        if action not in actions:
            raise ValueError(f"Unknown action: {action}")
        domain = self._domains[_text(actions[action])]
        for name, codes in status_values.items():
            if not codes or len(codes) != len(set(codes)):
                raise ValueError(
                    f"Status rectangle must contain distinct values: {name}"
                )
            for code in codes:
                self._validate_statuses(cell=cell, statuses={name: code})
        names = tuple(status_values)
        excluded = []
        for code, value in _code_values(domain):
            everywhere = True
            for assignment in product(*(status_values[name] for name in names)):
                statuses = dict(zip(names, assignment, strict=True))
                for phase in ("solve", "simulation"):
                    constraints = (
                        primitive
                        for primitive in self._constraints[cell_id]
                        if primitive.phase == phase
                    )
                    if not any(
                        (
                            answer := self._evaluate(
                                cell=cell,
                                primitive=primitive,
                                fixed_statuses=statuses,
                                fixed_actions={action: value},
                            )
                        ).proved
                        and answer.value is False
                        for primitive in constraints
                    ):
                        everywhere = False
                        break
                if not everywhere:
                    break
            if everywhere:
                excluded.append(code)
        return frozenset(excluded)

    def _cell(self, cell_id: CellId) -> Wire:
        try:
            return self._cells[cell_id]
        except KeyError:
            raise ValueError(f"Unknown original cell: {cell_id}") from None

    def _relation(self, relation_id: RelationId) -> tuple[Wire, Wire]:
        try:
            relation = self._relations[relation_id]
        except KeyError:
            raise ValueError(f"Unknown original relation: {relation_id}") from None
        return relation, self._cell(_text(relation["source"]))

    def _target_domain(
        self, *, relation: Wire, target_coordinate: CoordinateName
    ) -> Wire:
        targets = _strings(relation.get("targets", []))
        domains = {
            _text(_mapping(self._cell(target)["states"])[target_coordinate])
            for target in targets
            if target_coordinate in _mapping(self._cell(target)["states"])
        }
        if len(domains) != 1 or any(
            target_coordinate not in _mapping(self._cell(target)["states"])
            for target in targets
        ):
            raise ValueError(
                f"Unknown or inconsistent target coordinate: {target_coordinate}"
            )
        return self._domains[next(iter(domains))]

    def _validate_statuses(
        self, *, cell: Wire, statuses: Mapping[CoordinateName, int]
    ) -> None:
        states = _mapping(cell["states"])
        for name, code in statuses.items():
            if name not in states or type(code) is not int:
                raise ValueError(f"Unknown status or noninteger code: {name}")
            if code not in {
                item[0] for item in _code_values(self._domains[_text(states[name])])
            }:
                raise ValueError(f"Unknown status code: {name}={code}")

    def _joint_reads(
        self,
        *,
        cell: Wire,
        primitives: tuple[_Primitive, ...],
        statuses: Mapping[CoordinateName, int],
        removal_order: tuple[CoordinateName, ...],
    ) -> JointReadSet:
        coordinates = frozenset(_mapping(cell["states"]))
        aliases = {
            _text(_mapping(value)["argument"]): name
            for name, value in _mapping(cell.get("category_codes", {})).items()
        }

        def read(
            *, fixed_statuses: Mapping[CoordinateName, int]
        ) -> frozenset[CoordinateName]:
            required: set[CoordinateName] = set()
            for primitive in primitives:
                answer = self._evaluate(
                    cell=cell, primitive=primitive, fixed_statuses=fixed_statuses
                )
                if not answer.proved:
                    return coordinates
                required.update(aliases.get(name, name) for name in answer.reads)
            return frozenset(required)

        return get_joint_read_set(
            coordinates=coordinates,
            statuses=statuses,
            removal_order=removal_order,
            read_primitives=read,
        )

    def _evaluate(
        self,
        *,
        cell: Wire,
        primitive: _Primitive,
        fixed_statuses: Mapping[CoordinateName, int],
        fixed_actions: Mapping[CoordinateName, JSONValue] | None = None,
    ) -> _Abstract:
        if primitive.callback is None:
            return _Abstract(proved=False)
        domains = dict(self._domains)
        static = _static(cell, domains)
        if isinstance(cell.get("age"), str):
            static["age"] = Fraction(_text(cell["age"]))
        providers = _providers(cell, domains, primitive.phase)
        for name, code in fixed_statuses.items():
            domain = domains[_text(_mapping(cell["states"])[name])]
            static[name] = dict(_code_values(domain))[code]
        static.update(fixed_actions or {})
        if primitive.phase == "solve":
            for name, identity in _mapping(cell["states"]).items():
                domain = domains[_text(identity)]
                if domain.get("solve_axis") is False:
                    static.pop(name, None)
                    imputation = domain.get("solve_imputation")
                    if isinstance(imputation, str):
                        providers[name] = imputation
        static = _codes(cell, domains, static)
        callback = self._callbacks[primitive.callback]
        raw = self._captures.resolve(callback.get("original", {}))
        identity = _identity_name(raw) if isinstance(raw, Mapping) else None
        if identity is not None:
            return (
                _constant(static[identity])
                if identity in static
                else _Abstract(frozenset({identity}))
            )
        if primitive.relation is not None:
            relation = self._relations[primitive.relation]
            providers.update(self._contexts.get((id(relation), primitive.phase), {}))
        return SourceProof(
            self._callbacks, providers, static, callable_records=self._captures
        ).callback(primitive.callback)


def _mapping(value: JSONValue) -> Wire:
    if not isinstance(value, Mapping):
        raise ValueError("Original metadata field must be a mapping")
    return dict(value)


def _text(value: JSONValue) -> str:
    if not isinstance(value, str):
        raise ValueError("Original metadata identifier must be text")
    return value


def _rows(value: JSONValue) -> tuple[Wire, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Original metadata rows must be a sequence")
    return tuple(_mapping(item) for item in value)


def _strings(value: JSONValue) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Original metadata identifiers must be a sequence")
    return tuple(_text(item) for item in value)


def _phases(value: JSONValue) -> tuple[Phase, ...]:
    if value == "both":
        return ("solve", "simulation")
    if value == "solve":
        return ("solve",)
    if value == "simulation":
        return ("simulation",)
    raise ValueError(f"Unknown primitive phase: {value}")


def _code_values(domain: Wire) -> tuple[tuple[int, JSONValue], ...]:
    if domain.get("kind") != "categorical":
        raise ValueError("Requested status or action must have a categorical grid")
    codes, values = domain.get("codes"), domain.get("values")
    if (
        not isinstance(codes, (list, tuple))
        or not isinstance(values, (list, tuple))
        or len(codes) != len(values)
        or not codes
    ):
        raise ValueError("Original categorical codes and values must be available")
    integer_codes = []
    for code in codes:
        if not isinstance(code, int) or isinstance(code, bool):
            raise ValueError("Original categorical codes must be integers")
        integer_codes.append(code)
    if len(set(integer_codes)) != len(integer_codes):
        raise ValueError("Original categorical codes must be distinct integers")
    return tuple(zip(integer_codes, values, strict=True))


def _identity_name(value: Mapping[str, JSONValue]) -> CoordinateName | None:
    from .graph_scope import identity_state

    named = identity_state(value)
    if named is not None:
        return named
    if value.get("type") == "_lcm.identity_transition._IdentityTransition":
        name = _mapping(value.get("fields", {})).get("state_name")
        return name if isinstance(name, str) else None
    return None
