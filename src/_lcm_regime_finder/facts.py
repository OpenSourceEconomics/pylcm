"""What each regime-period cell depends on, answered by one of three backends.

Every backend answers the same primitive question: which current state and
action coordinates one callback reads in one cell and phase, with the
characteristic callbacks in `stops` cut off and, optionally, some status
coordinates fixed to a code. Forced actions, constant states, identity carries
and characteristic images are the backend's other facts. Current-period reads,
update reads, the live-state closure and characteristic eligibility are derived
from these facts by the shared functions below, so every backend applies the
same `produces`, zero-support and incidence rules.

`metadata` interprets the admitted scalar expression language of the captured
declarations. It over-approximates the exact backends: it reads at least what
they read and forces at most what they force.
"""

import ast
from collections.abc import Callable, Iterable, Mapping
from typing import Literal, Protocol

from .expressions import AdmissionError, jsonable, partial
from .frontier import canonical

type Record = dict
type FactsBackendName = Literal["metadata", "dense", "implicit"]

FACTS_BACKENDS: tuple[FactsBackendName, ...] = ("metadata", "dense", "implicit")


class Facts(Protocol):
    """The facts every backend supplies."""

    @property
    def cells(self) -> Mapping[str, Record]:
        """The admitted cells by identifier."""
        ...

    @property
    def callbacks(self) -> Mapping[str, Record]:
        """The declared callbacks by identifier."""
        ...

    @property
    def domains(self) -> Mapping[str, Record]:
        """The declared domains by identifier."""
        ...

    @property
    def relations(self) -> list[Record]:
        """The declared relations, including zero-support ones."""
        ...

    @property
    def constants(self) -> Mapping[str, dict]:
        """Each cell's states with a single value."""
        ...

    @property
    def forced(self) -> Mapping[str, dict]:
        """Each cell's actions with a single admissible value."""
        ...

    def callback_reads(
        self,
        callback: str,
        cell: Record,
        phase: str,
        *,
        stops: frozenset[str] = frozenset(),
        relation: Record | None = None,
        statuses: Mapping[str, int] | None = None,
    ) -> frozenset[str]:
        """Return the cell coordinates `callback` reads in `phase`."""
        ...

    def identity_update(self, update: object, axis: str, target_axis: str) -> bool:
        """Return whether `update` carries `axis` unchanged into `target_axis`."""
        ...

    def characteristic_image(self, spec: Record) -> list | None:
        """Return the characteristic's value for each original value, or None."""
        ...


def phases(record: Mapping) -> tuple[str, ...]:
    """Return the phases a consumer or relation applies in."""
    phase = record.get("phase", "both")
    return ("solve", "simulation") if phase == "both" else (phase,)


def identity_stops(callbacks: Mapping[str, Record], callback_ids: Iterable[str]) -> frozenset[str]:
    """Return every callback sharing an original identity with `callback_ids`."""
    identities = {callbacks[cid].get("identity") for cid in callback_ids}
    return frozenset(k for k, cb in callbacks.items() if cb.get("identity") in identities)


def live_relations(facts: Facts, source: str | None = None) -> list[Record]:
    """Return the relations with possible support, optionally from one source."""
    return [
        r for r in facts.relations
        if r.get("support") != "zero" and (source is None or r["source"] == source)
    ]


def current_reads(
    facts: Facts,
    cell: Record,
    stops: frozenset[str] = frozenset(),
    statuses: Mapping[str, int] | None = None,
) -> set[str]:
    """Return what the cell's consumers and outgoing relation callbacks read.

    Probability, gate, exit and read weights are always consumers. Updates are
    propagated separately, only when the destination coordinate is live.
    """
    reads: set[str] = set()
    for consumer in cell.get("consumers", []):
        if consumer.get("callback"):
            for phase in phases(consumer):
                reads |= facts.callback_reads(
                    consumer["callback"], cell, phase, stops=stops, statuses=statuses
                )
    for relation in live_relations(facts, cell["id"]):
        for callback in relation.get("callbacks", []):
            for phase in phases(relation):
                reads |= facts.callback_reads(
                    callback, cell, phase, stops=stops, relation=relation, statuses=statuses
                )
    return reads


def update_reads(
    facts: Facts,
    relation: Record,
    target_axis: str,
    stops: frozenset[str] = frozenset(),
    statuses: Mapping[str, int] | None = None,
) -> set[str]:
    """Return the source coordinates the update of `target_axis` reads."""
    source = facts.cells[relation["source"]]
    everything = set(source["states"]) | set(source["actions"])
    update = relation.get("state_updates", {}).get(target_axis)
    if update is None:
        # Unknown reconstruction cannot justify removal of any source input.
        return everything
    if isinstance(update, str) and update in source["states"]:
        return {update}
    if isinstance(update, dict):
        if update.get("kind") == "identity":
            return {update.get("source", target_axis)}
        if update.get("kind") == "constant":
            return set()
        update = update.get("callback")
    if isinstance(update, str) and update in facts.callbacks:
        result: set[str] = set()
        for phase in phases(relation):
            result |= facts.callback_reads(
                update, source, phase, stops=stops, relation=relation, statuses=statuses
            )
        return result
    return everything


def live_states(
    facts: Facts, guard: Callable[[], None]
) -> tuple[dict[str, set[str]], dict[str, dict[str, list[str]]]]:
    """Return each cell's live states and why each is live.

    A state is live if a current consumer reads it, or if a live state of a
    possible successor is updated from it.
    """
    live: dict[str, set[str]] = {}
    reasons: dict[str, dict[str, list[str]]] = {}
    for cid, cell in facts.cells.items():
        live[cid] = set(cell["states"]) & current_reads(facts, cell) - set(facts.constants[cid])
        reasons[cid] = {name: ["current/phase/required-output consumer"] for name in live[cid]}
    changed = True
    while changed:
        guard()
        changed = False
        for relation in live_relations(facts):
            sid = relation["source"]
            source = facts.cells[sid]
            for tid in relation.get("targets", []):
                for target_axis in tuple(live[tid]):
                    if "produces" in relation and target_axis not in relation["produces"]:
                        continue
                    needed = update_reads(facts, relation, target_axis) & set(source["states"]) - set(facts.constants[sid])
                    new = needed - live[sid]
                    if new:
                        changed = True
                        live[sid].update(new)
                        for name in new:
                            reasons[sid][name] = [f"{relation['id']} -> {tid}/{target_axis}"]
    return live, reasons


def characteristic_cells(
    facts: Facts, live: Mapping[str, set[str]], spec: Record
) -> tuple[set[str], dict[str, set[str]]]:
    """Return the cells where a characteristic can replace its state, and carries.

    A cell is eligible if the state is live there only through the
    characteristic, and every successor either carries it unchanged into an
    eligible cell or reads it only through the characteristic. The second
    return maps each eligible cell to the eligible cells it carries it into.
    """
    axis, domain_id = spec["state"], spec["domain"]
    stops = identity_stops(facts.callbacks, [spec["callback"]])
    eligible = {
        cid for cid, cell in facts.cells.items()
        if cell["states"].get(axis) == domain_id and axis in live[cid]
        and axis not in current_reads(facts, cell, stops)
    }
    changed = True
    while changed:
        changed = False
        for relation in live_relations(facts):
            sid = relation["source"]
            if sid not in eligible:
                continue
            for tid in relation.get("targets", []):
                for target_axis in live[tid]:
                    if "produces" in relation and target_axis not in relation["produces"]:
                        continue
                    update = relation.get("state_updates", {}).get(target_axis)
                    if target_axis == axis and facts.identity_update(update, axis, target_axis):
                        safe = tid in eligible
                    else:
                        safe = axis not in update_reads(facts, relation, target_axis, stops)
                    if not safe:
                        eligible.discard(sid)
                        changed = True
                        break
                if sid not in eligible:
                    break
    requires: dict[str, set[str]] = {cid: set() for cid in eligible}
    for relation in live_relations(facts):
        sid = relation["source"]
        if sid not in eligible or ("produces" in relation and axis not in relation["produces"]):
            continue
        if facts.identity_update(relation.get("state_updates", {}).get(axis), axis, axis):
            requires[sid].update(
                t for t in relation.get("targets", []) if t in eligible and axis in live[t]
            )
    return eligible, requires


class MetadataFacts:
    """Facts from the admitted scalar expression language of the declarations.

    A callback outside that language reads every argument it is declared with;
    an unresolved callback reads every coordinate of its cell.
    """

    def __init__(self, declarations: Record, limits: Mapping, guard: Callable[[], None]) -> None:
        self.decl = declarations
        self.limits = limits
        self.guard = guard
        self.cells = {c["id"]: c for c in declarations["cells"]}
        if len(self.cells) != len(declarations["cells"]):
            raise ValueError("Duplicate exact admission cell")
        self.callbacks = declarations["callbacks"]
        self.domains = declarations["domains"]
        self.relations = declarations["relations"]
        self._transition_contexts = transition_provider_index(relations=self.relations, callbacks=self.callbacks)
        self._cache: dict = {}
        self._env_cache: dict = {}
        self.forced: dict[str, dict] = {}
        self.constants: dict[str, dict] = {}
        for c in self.cells.values():
            for kind in ("states", "actions"):
                if not isinstance(c[kind], dict):
                    raise ValueError(f'{c["id"]}: {kind} must map axis names to domains')
                for name, domain in c[kind].items():
                    if domain not in self.domains:
                        raise ValueError(f"Unknown domain {name}/{domain}")
            self.constants[c["id"]] = {
                name: self.domains[domain]["values"][0] for name, domain in c["states"].items()
                if self.domains[domain].get("size") == 1
                and len(self.domains[domain].get("values", [])) == 1}
            self.forced[c["id"]] = self.forced_actions(c)
            self._env_cache.clear()
        for relation in self.relations:
            if relation["source"] not in self.cells or any(t not in self.cells for t in relation.get("targets", [])):
                raise ValueError("Relation names an undeclared exact admission")

    def callback_reads(self, callback, cell, phase, *, stops=frozenset(), relation=None, statuses=None):
        """Return the coordinates `callback` reads; status values are not used."""
        overrides = self._transition_contexts.get((id(relation), phase), {}) if relation is not None else None
        return frozenset(self.expanded_reads(callback, cell, phase, stops, provider_overrides=overrides))

    @staticmethod
    def scalar_bindings(callback):
        return {k: v for k, v in callback.get("bindings", {}).items()
                if isinstance(v, (bool, int, float, str, list)) or v is None}

    def environment(self, cell, callback=None, phase="solve", overrides=None):
        """Bind provably known date/parameter DAG values, never native functions.

        A declaration may represent age as ``age(period)``. Treating that value
        as unknown would miss a sustained feasibility restriction. This finite
        fixed point evaluates only the admitted scalar expression language.
        """
        key = (cell["id"], phase)
        cached = self._env_cache.get(key) if not overrides else None
        if cached is not None:
            env = dict(cached)
        else:
            env = {k: cell[k] for k in ("age", "period") if cell.get(k) is not None}
            env.update(self.decl.get("parameters", {}))
            env.update(self.constants.get(cell["id"], {}))
            env.update(self.forced.get(cell["id"], {}))
            env.update(overrides or {})
            env = self.bind_codes(cell, env)
            providers = cell.get("phase_functions", {}).get(phase, cell.get("functions", {}))
            pending = dict(providers)
            for _ in range(len(pending) + 1):
                changed = False
                for name, cbid in list(pending.items()):
                    cb = self.callbacks[cbid]
                    if name in env:
                        pending.pop(name)
                        continue
                    if cb.get("status") != "proved" or cb.get("expression") is None:
                        continue
                    try:
                        value = partial(cb["expression"], self.scalar_bindings(cb) | env)
                    except (AdmissionError, TypeError, ValueError, ArithmeticError):
                        continue
                    if value.known:
                        env[name] = value.value
                        pending.pop(name)
                        changed = True
                for alias, name in cell.get("derived_code_providers", {}).items():
                    if name in env and alias not in env:
                        domain = self.domains[cell["derived_categoricals"][name]]
                        values, codes = domain.get("values", []), domain.get("codes", [])
                        matches = [i for i, v in enumerate(values) if v == env[name]]
                        if len(matches) == 1 and len(values) == len(codes):
                            env[alias] = codes[matches[0]]
                            changed = True
                if not changed:
                    break
            if not overrides and cell["id"] in self.forced:
                self._env_cache[key] = dict(env)
        return self.bind_codes(cell, (self.scalar_bindings(callback) if callback else {}) | env)

    @staticmethod
    def bind_codes(cell, env):
        env = dict(env)
        for axis, record in cell.get("category_codes", {}).items():
            if axis not in env:
                continue
            values, codes = record.get("values", []), record.get("codes", [])
            matches = [i for i, value in enumerate(values) if canonical(value) == canonical(env[axis])]
            if len(matches) == 1 and len(codes) == len(values):
                env[record["argument"]] = codes[matches[0]]
        return env

    def direct_reads(self, callback_id, cell, overrides=None, phase="solve"):
        cb = self.callbacks[callback_id]
        env = self.environment(cell, cb, phase, overrides)
        expression = cb.get("expression")
        if expression is not None:
            try:
                return set(partial(expression, env).reads)
            except (AdmissionError, TypeError, ValueError):
                pass
        return set(cb.get("reads", cb.get("arguments", []))) - set(env)

    def expanded_reads(self, callback_id, cell, phase="solve", stops=frozenset(), trail=frozenset(), *, provider_overrides=None):
        provider_overrides = provider_overrides or {}
        key = (callback_id, cell["id"], phase, tuple(sorted(stops)), tuple(sorted(provider_overrides.items())))
        if key in self._cache:
            return set(self._cache[key])
        if callback_id in stops:
            return set()
        if callback_id in trail:
            raise ValueError(f"Cyclic derived function DAG at {callback_id}")
        cb = self.callbacks[callback_id]
        if cb.get("status") == "unresolved":
            return set(cell["states"]) | set(cell["actions"])
        providers = cell.get("phase_functions", {}).get(phase, cell.get("functions", {})) | provider_overrides
        result = set()
        code_axes = {v["argument"]: k for k, v in cell.get("category_codes", {}).items()}
        for name in self.direct_reads(callback_id, cell, phase=phase):
            name = code_axes.get(name, name)
            name = cell.get("derived_code_providers", {}).get(name, name)
            if name in cell["states"] or name in cell["actions"]:
                result.add(name)
            elif name in providers:
                result.update(self.expanded_reads(providers[name], cell, phase, stops, trail | {callback_id}, provider_overrides=provider_overrides))
        self._cache[key] = frozenset(result)
        return result

    def identity_update(self, update, axis, target_axis):
        if update == axis:
            return True
        if isinstance(update, dict):
            if update.get("kind") == "identity":
                return update.get("source", target_axis) == axis
            update = update.get("callback")
        if isinstance(update, str) and update in self.callbacks:
            cb = self.callbacks[update]
            if cb.get("status") == "proved" and isinstance(cb.get("expression"), str):
                try:
                    node = ast.parse(cb["expression"], mode="eval").body
                    return isinstance(node, ast.Name) and node.id == axis
                except SyntaxError:
                    pass
        return False

    def forced_actions(self, cell):
        result = {}
        constraints = [x for x in cell.get("consumers", [])
                       if x.get("kind") in ("constraint", "feasibility", "feasible") and x.get("callback")]
        decision_phases = cell.get("decision_phases", ("solve", "simulation"))
        for name, domain_id in cell["actions"].items():
            self.guard()
            domain = self.domains[domain_id]
            values = domain.get("values")
            if not isinstance(values, list) or len(values) != domain.get("size") or len(values) > self.limits["metadata_categories"]:
                continue
            possible = []
            for value in values:
                self.guard()
                # A shared action interface must cover the UNION of the
                # phase-specific feasible sets. Rejection by the simulation
                # constraints does not remove a choice available to solve.
                phase_allowed = []
                for decision_phase in decision_phases:
                    allowed = True
                    for constraint in constraints:
                        applies = constraint.get("decision_phases")
                        if applies is None:
                            evaluation_phases = (decision_phase,) if decision_phase in phases(constraint) else ()
                        else:
                            evaluation_phases = phases(constraint) if decision_phase in applies else ()
                        if not evaluation_phases:
                            continue
                        cb = self.callbacks[constraint["callback"]]
                        if cb.get("expression") is None or cb.get("status") != "proved":
                            continue
                        for phase in evaluation_phases:
                            try:
                                v = partial(cb["expression"], self.environment(cell, cb, phase, {name: value}))
                            except (AdmissionError, TypeError, ValueError, ArithmeticError):
                                continue
                            if v.known and v.value is False:
                                allowed = False
                                break
                        if not allowed:
                            break
                    phase_allowed.append(allowed)
                if any(phase_allowed):
                    possible.append(value)
            if not possible:
                raise ValueError(f'{cell["id"]}: no admissible value of action {name}')
            if len(possible) == 1:
                result[name] = possible[0]
        return result

    def characteristic_image(self, spec):
        axis, domain_id = spec["state"], spec["domain"]
        domain = self.domains[domain_id]
        cb = self.callbacks[spec["callback"]]
        values = domain.get("values")
        if domain.get("kind") != "categorical" or cb.get("status") != "proved" or cb.get("expression") is None:
            return None
        if not isinstance(values, list) or len(values) != domain.get("size") or len(values) > self.limits["metadata_categories"]:
            return None
        mapping = []
        try:
            for index, value in enumerate(values):
                self.guard()
                env = self.scalar_bindings(cb)
                env[axis] = value
                code_arguments = {c.get("category_codes", {}).get(axis, {}).get("argument")
                                  for c in self.cells.values() if c["states"].get(axis) == domain_id}
                code_arguments.discard(None)
                if spec.get("code_argument"):
                    code_arguments.add(spec["code_argument"])
                for argument in code_arguments:
                    if len(domain.get("codes", [])) == len(values):
                        env[argument] = domain["codes"][index]
                observed = partial(cb["expression"], env)
                if not observed.known:
                    raise ValueError("Characteristic has unresolved original arguments")
                mapping.append(jsonable(observed.value))
        except (AdmissionError, TypeError, ValueError):
            return None
        return mapping


def transition_provider_index(*, relations, callbacks) -> dict[tuple[int, str], dict[str, str]]:
    """Bind next-state providers separately for every original edge and phase."""
    contexts: dict[tuple[str, tuple[str, ...], str], dict[str, str]] = {}
    index: dict[tuple[int, str], dict[str, str]] = {}
    for relation in relations:
        if relation.get("kind") not in {"transition", "joint"}:
            continue
        source, targets = relation.get("source"), relation.get("targets", [])
        if not isinstance(source, str) or not isinstance(targets, list) or any(not isinstance(target, str) for target in targets):
            raise ValueError("Transition context lacks original source/target identifiers")
        target_names = tuple(target for target in targets if isinstance(target, str))
        updates = relation.get("state_updates", {})
        if not isinstance(updates, Mapping):
            raise ValueError("Transition context lacks a state-update mapping")
        selected = relation.get("phase", "both")
        if not isinstance(selected, str) or selected not in ("both", "solve", "simulation"):
            raise ValueError("Transition context has an unsupported phase")
        for phase in ("solve", "simulation") if selected == "both" else (selected,):
            context = contexts.setdefault((source, target_names, phase), {})
            index[id(relation), phase] = context
            for axis, callback in updates.items():
                if not isinstance(callback, str) or callback not in callbacks:
                    continue
                name = "next_" + axis
                if name in context and context[name] != callback:
                    raise ValueError("Conflicting original next-state providers in one edge/phase")
                context[name] = callback
    return index
