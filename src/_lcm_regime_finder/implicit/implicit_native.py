"""Metadata-only adapter for the exact original point-table contract.

This uses the unchanged reference SourceProof and category decoder.
No user callback, Model construction, solver, simulation, or JAX is invoked.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import cast
from ..records import JSONValue
from ..fibre_source import FibreSource, Wire, _Primitive, _mapping, _strings, _text
from .implicit_core import Assignment
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json

from .implicit_core import (
    Eval,
    Failure,
    MDD,
    ReadCertificate,
    Support,
    UNKNOWN,
    joint_reads,
)
from .binding_purity import certify_binding_cache


class LazyStatic(Mapping[str, object]):
    """An extensionally identical static dictionary with deferred scalar loads."""

    def __init__(self, constants: Mapping[str, object]) -> None:
        self.entries: dict[str, tuple[bool, object]] = {
            name: (False, value) for name, value in constants.items()
        }

    def __getitem__(self, name: str) -> object:
        lazy, value = self.entries[name]
        return cast(Callable[[], object], value)() if lazy else value

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: object) -> bool:
        return name in self.entries

    def constant(self, name: str, value: object) -> None:
        self.entries[name] = (False, value)

    def deferred(self, name: str, operation: Callable[[], object]) -> None:
        self.entries[name] = (True, operation)

    def remove(self, name: str) -> None:
        self.entries.pop(name, None)


class CellCompiler:
    def __init__(
        self, *, source: FibreSource, cell_id: str, status_axes: Sequence[str]
    ) -> None:
        from _lcm_regime_finder.fibre_source import _mapping, _text

        self.source = source
        self.cell = source._cell(cell_id)
        self.states = {
            name: _text(identity)
            for name, identity in _mapping(self.cell["states"]).items()
        }
        self.actions = {
            name: _text(identity)
            for name, identity in _mapping(self.cell["actions"]).items()
        }
        declared = frozenset(status_axes)
        if len(declared) != len(status_axes) or not declared <= self.states.keys():
            raise ValueError("Status declaration is not an original coordinate subset")
        # This matches compile_primitive_fibres.py: order is original state order,
        # not arbitrary order in the external classification packet.
        self.axes = tuple(name for name in self.states if name in declared)
        domains = tuple(
            (name, tuple(source._category(source._domains[self.states[name]]).by_code))
            for name in self.axes
        )
        self.diagram = MDD(domains)
        self.coordinates = frozenset(self.states)
        self.eval_cache: dict[
            tuple[_Primitive, frozenset[str], tuple[tuple[str, int], ...]], int
        ] = {}
        self.cache_purity = certify_binding_cache(
            callbacks=source._callbacks,
            records=source._captures.records,
            resolve=source._captures.resolve,
        )

    def evaluate(
        self,
        *,
        primitive: _Primitive,
        fixed: frozenset[str],
        assignment: Assignment,
        fixed_actions: Mapping[str, int] | None = None,
    ) -> Eval:
        """Lazy value setup, then exactly the unchanged SourceProof interpreter.

        Its per-call steps, active stack, fallback ordering, exception handling,
        effective signatures and original law contexts are not altered.
        """
        from _lcm_regime_finder.fibre_source import _identity_name, _mapping, _text
        from _lcm_regime_finder.semantic_check import (
            SourceProof,
            _Abstract,
            _UNKNOWN,
            _constant,
            _providers,
            _static,
            _value_key,
        )

        source, cell = self.source, self.cell
        if primitive.callback is None:
            return Eval(frozenset(), UNKNOWN, False)
        domains = dict(source._domains)
        static = LazyStatic(_static(cell, domains))
        if isinstance(cell.get("age"), str):
            static.constant("age", Fraction(_text(cell["age"])))
        providers = _providers(cell, domains, primitive.phase)
        axes = {**_mapping(cell["states"]), **_mapping(cell["actions"])}
        fixed_codes: dict[str, Callable[[], int]] = {
            name: (lambda name=name: assignment.code(name)) for name in fixed
        }
        for name, code in (fixed_actions or {}).items():
            fixed_codes[name] = lambda code=code: code
        for name, get_code in fixed_codes.items():
            categories = source._category(domains[_text(axes[name])])
            static.deferred(
                name,
                lambda categories=categories, get_code=get_code: categories.by_code[
                    get_code()
                ],
            )
        if primitive.phase == "solve":
            for name, domain_id in self.states.items():
                domain = domains[_text(domain_id)]
                if domain.get("solve_axis") is False:
                    static.remove(name)
                    fixed_codes.pop(name, None)
                    if isinstance(domain.get("solve_imputation"), str):
                        providers[name] = _text(domain["solve_imputation"])
        for name, domain_id in axes.items():
            domain = domains[_text(domain_id)]
            categories = source._categories.get(_text(domain["identity"]))
            if categories is None or (
                primitive.phase == "solve" and domain.get("solve_axis") is False
            ):
                continue
            if name not in static and len(categories.by_code) == 1:
                static.constant(name, next(iter(categories.by_code.values())))
            if name not in static:
                continue
            argument = _text(
                _mapping(_mapping(cell.get("category_codes", {})).get(name, {})).get(
                    "argument", name + "_code"
                )
            )
            get_code = fixed_codes.get(name)
            if get_code is not None:
                static.deferred(argument, get_code)
            else:
                # This may demand a code after a deliberately colliding alias;
                # replay then follows the same original axis-loop order.
                matching = categories.by_value.get(
                    _value_key(static[name]), frozenset()
                )
                if len(matching) == 1:
                    static.constant(argument, next(iter(matching)))
        callback = source._callbacks[primitive.callback]
        raw = source._captures.resolve(callback.get("original", {}))
        named = _identity_name(raw) if isinstance(raw, Mapping) else None
        if named is not None:
            answer = (
                _constant(static[named])
                if named in static
                else _Abstract(frozenset({named}))
            )
        else:
            if primitive.relation is not None:
                relation = source._relations[primitive.relation]
                providers.update(
                    source._contexts.get((id(relation), primitive.phase), {})
                )
            answer = SourceProof(
                source._callbacks, providers, static, callable_records=source._captures
            ).callback(primitive.callback)
        return Eval(
            answer.reads,
            UNKNOWN if answer.value is _UNKNOWN else answer.value,
            answer.proved,
        )

    def primitive(
        self,
        *,
        primitive: _Primitive,
        fixed: frozenset[str],
        fixed_actions: Mapping[str, int] | None = None,
    ) -> int:
        key = (primitive, frozenset(fixed), tuple((fixed_actions or {}).items()))
        if key not in self.eval_cache:

            def run(assignment: Assignment) -> Eval | Failure:
                try:
                    return self.evaluate(
                        primitive=primitive,
                        fixed=fixed,
                        assignment=assignment,
                        fixed_actions=fixed_actions,
                    )
                except Exception as error:
                    return Failure(type(error).__name__, str(error))

            self.eval_cache[key] = self.diagram.demand(run)
        return self.eval_cache[key]

    def read_set(self, *, primitives: Sequence[_Primitive]) -> int:
        aliases = {
            name + "_code": name
            for name, identity in self.states.items()
            if self.source._domains[identity].get("kind") == "categorical"
        }
        aliases.update(
            {
                _text(_mapping(record)["argument"]): name
                for name, record in _mapping(
                    self.cell.get("category_codes", {})
                ).items()
            }
        )

        def reader(fixed: frozenset[str]) -> int:
            # Pair state preserves the reference reader's early return after the
            # first unproved primitive.  Later exceptions cannot override it.
            result = self.diagram.leaf((True, frozenset()))
            for primitive in primitives:
                current = self.primitive(primitive=primitive, fixed=fixed)

                def merge(acc: object, answer: object) -> object:
                    acc = cast(Failure | tuple[bool, frozenset[str]], acc)
                    answer = cast(Failure | Eval, answer)
                    if isinstance(acc, Failure) or not acc[0]:
                        return acc
                    if isinstance(answer, Failure):
                        return answer
                    if not answer.proved:
                        return (False, self.coordinates)
                    return (
                        True,
                        acc[1]
                        | frozenset(aliases.get(name, name) for name in answer.reads),
                    )

                result = self.diagram.apply(merge, result, current)
            return self.diagram.apply(
                lambda item: (
                    item
                    if isinstance(item, Failure)
                    else cast(tuple[bool, frozenset[str]], item)[1]
                ),
                result,
            )

        return joint_reads(
            self.diagram,
            coordinates=self.coordinates,
            removal_order=self.axes,
            reader=reader,
        )

    def direct(self) -> int:
        # The authoritative target here is compile_primitive_fibres.py.  Do not
        # inject build_source_screen's separate global policy-query fallback.
        return self.read_set(primitives=self.source._roots[_text(self.cell["id"])])

    def update(self, *, relation_id: str, coordinate: str) -> int:
        from _lcm_regime_finder.fibre_source import _Primitive, _phases

        relation, _ = self.source._relation(relation_id)
        self.source._target_domain(relation=relation, target_coordinate=coordinate)
        callback = _mapping(relation.get("state_updates", {})).get(coordinate)
        primitives = tuple(
            _Primitive(
                _text(callback) if callback is not None else None, phase, relation_id
            )
            for phase in _phases(relation.get("phase", "both"))
        )
        return self.read_set(primitives=primitives)

    def exclusions(self, *, action: str) -> int:
        """Point facts on the entire original hull, including unkept points."""
        source, cell = self.source, self.cell
        categories = source._category(source._domains[self.actions[action]])
        excluded = self.diagram.leaf(frozenset())
        for code in categories.by_code:
            both = self.diagram.leaf(True)
            for phase in ("solve", "simulation"):
                one = self.diagram.leaf(False)
                for primitive in source._constraints[_text(cell["id"])]:
                    if primitive.phase != phase:
                        continue
                    answer = self.primitive(
                        primitive=primitive,
                        fixed=frozenset(self.axes),
                        fixed_actions={action: code},
                    )

                    def disjoin(previous: object, value: object) -> object:
                        value = cast(Failure | Eval, value)
                        if isinstance(previous, Failure) or previous:
                            return previous
                        if isinstance(value, Failure):
                            return value
                        return value.proved and value.value is False

                    one = self.diagram.apply(disjoin, one, answer)

                def conjoin(previous: object, value: object) -> object:
                    if isinstance(previous, Failure) or not previous:
                        return previous
                    return value

                both = self.diagram.apply(conjoin, both, one)

            def add(previous: object, yes: object, code: int = code) -> object:
                if isinstance(previous, Failure):
                    return previous
                if isinstance(yes, Failure):
                    return yes
                return cast(frozenset[int], previous) | {code} if yes else previous

            excluded = self.diagram.apply(add, excluded, both)
        return excluded

    def support(self, *, relation_id: str, coordinate: str) -> int:
        from _lcm_regime_finder.fibre_source import _Primitive, _identity_name, _phases
        from _lcm_regime_finder.semantic_check import _value_key

        source = self.source
        relation, _ = source._relation(relation_id)
        domain = source._target_domain(relation=relation, target_coordinate=coordinate)
        categories = source._category(domain)
        full = Support(frozenset(categories.by_code), False)
        callback_id = _mapping(relation.get("state_updates", {})).get(coordinate)
        if callback_id is None:
            return self.diagram.leaf(full)
        raw = source._callbacks[_text(callback_id)].get("original", {})
        resolved = source._captures.resolve(raw)
        deterministic = isinstance(resolved, Mapping) and (
            _identity_name(resolved) is not None
            or (resolved.get("kind") == "callable" and not resolved.get("bound_self"))
        )
        if not deterministic:
            return self.diagram.leaf(full)
        result = self.diagram.leaf(Support(frozenset(), True))
        for phase in _phases(relation.get("phase", "both")):
            answer = self.primitive(
                primitive=_Primitive(_text(callback_id), phase, relation_id),
                fixed=frozenset(self.axes),
            )

            def append(previous: object, value: object) -> object:
                previous = cast(Failure | Support, previous)
                value = cast(Failure | Eval, value)
                if isinstance(previous, Failure) or not previous.proved:
                    return previous
                if isinstance(value, Failure):
                    return value
                if not value.proved or value.value == UNKNOWN:
                    return full
                matching = categories.by_value.get(_value_key(value.value), frozenset())
                if not matching:
                    return Failure(
                        "ValueError",
                        "Original deterministic update leaves target grid: "
                        + coordinate,
                    )
                return Support(previous.possible_codes | matching, True)

            result = self.diagram.apply(append, result, answer)
        return result

    def compile(
        self, *, status_axes: Mapping[str, Sequence[str]]
    ) -> dict[str, JSONValue]:
        # The baseline binding memo stores decoded values and logical size but
        # not expanded depth. A warm memo can alter its conservative depth64
        # outcome. Reordering is therefore permitted only after a source-only
        # expanded-binding-height proof. Otherwise retain the original traversal
        # order while constructing reduced diagrams, never a point table.
        ordered = not self.cache_purity["proved"]
        roots = (
            self.ordered_roots(status_axes=status_axes)
            if ordered
            else {"direct": self.direct()}
        )
        if not ordered:
            for action, identity in self.actions.items():
                if self.source._domains[identity].get("kind") == "categorical":
                    roots["exclude:" + action] = self.exclusions(action=action)
        metadata = []
        for rid, relation in self.source._relations.items():
            if (
                relation["source"] != self.cell["id"]
                or relation["kind"]
                not in {"transition", "joint", "value_read", "fallback", "exit"}
                or not relation["targets"]
            ):
                continue
            target = self.source._cells[_strings(relation["targets"])[0]]
            if not ordered:
                for coordinate in _mapping(target["states"]):
                    roots["read:" + rid + "::" + coordinate] = self.update(
                        relation_id=rid, coordinate=coordinate
                    )
                for coordinate in status_axes[_text(target["regime"])]:
                    roots["support:" + rid + "::" + coordinate] = self.support(
                        relation_id=rid, coordinate=coordinate
                    )
            metadata.append(
                {
                    key: relation[key]
                    for key in (
                        "id",
                        "source",
                        "targets",
                        "phase",
                        "kind",
                        "state_updates",
                    )
                    if key in relation
                }
            )
        # A failed reference call is not a fact. Do not publish a COMPLETE diagram.
        for name, root in roots.items():
            for _, value in self.diagram.cubes(root):
                if isinstance(value, Failure):
                    raise ValueError(
                        "Original query failed at " + name + ": " + repr(value)
                    )
        return {
            "cell_id": self.cell["id"],
            "regime": self.cell["regime"],
            "period": self.cell["period"],
            "forced": [],
            "relations": metadata,
            "source_reference": {
                "cell_id": self.cell["id"],
                "source_path": self.cell["source_path"],
            },
            "construction_mode": "original-order-stream"
            if ordered
            else "pure-demand-diagram",
            "binding_cache_purity": self.cache_purity,
            "diagram": cast(JSONValue, self.diagram.export(roots)),
            "construction_counts": {
                "analyzer_attempts": self.diagram.attempts,
                "demand_splits": self.diagram.demand_splits,
            },
        }

    def ordered_roots(
        self, *, status_axes: Mapping[str, Sequence[str]]
    ) -> dict[str, int]:
        """Exact operational fallback for cache-sensitive original records.

        Leaves are visited in itertools.product order, and original queries are
        issued in compile_primitive_fibres.py order. Completed subtrees are reduced
        bottom-up. Storage is final shared subgraphs plus a traversal stack; no
        rows or pair graph are retained. Every original point may be assessed.
        """
        source, cell = self.source, self.cell
        outgoing = [
            (rid, relation)
            for rid, relation in source._relations.items()
            if relation["source"] == cell["id"]
            and relation["kind"]
            in {"transition", "joint", "value_read", "fallback", "exit"}
            and relation["targets"]
        ]

        def certificate(value: object) -> ReadCertificate:
            from ..fibre_facts import JointReadSet

            assert isinstance(value, JointReadSet)
            return ReadCertificate(
                value.live_coordinates, value.removal_order, value.removed_statuses
            )

        def point(statuses: Mapping[str, int]) -> dict[str, int]:
            values = {
                "direct": certificate(
                    source.get_direct_reads(
                        cell_id=_text(cell["id"]),
                        statuses=statuses,
                        removal_order=self.axes,
                    )
                )
            }
            for action, domain in self.actions.items():
                if source._domains[domain].get("kind") == "categorical":
                    values["exclude:" + action] = source.get_excluded_action_values(
                        cell_id=_text(cell["id"]),
                        action=action,
                        status_values={
                            name: (code,) for name, code in statuses.items()
                        },
                    )
            for rid, relation in outgoing:
                target = source._cells[_strings(relation["targets"])[0]]
                for coordinate in _mapping(target["states"]):
                    values["read:" + rid + "::" + coordinate] = certificate(
                        source.get_update_reads(
                            relation_id=rid,
                            target_coordinate=coordinate,
                            statuses=statuses,
                            removal_order=self.axes,
                        )
                    )
                for coordinate in status_axes[_text(target["regime"])]:
                    support = source.get_target_support(
                        relation_id=rid, target_coordinate=coordinate, statuses=statuses
                    )
                    values["support:" + rid + "::" + coordinate] = Support(
                        support.possible_codes, support.proved
                    )
            return {name: self.diagram.leaf(value) for name, value in values.items()}

        def visit(index: int, statuses: dict[str, int]) -> dict[str, int]:
            if index == len(self.axes):
                return point(statuses)
            name, codes = self.diagram.domains[index]
            children = [visit(index + 1, {**statuses, name: code}) for code in codes]
            keys = tuple(children[0])
            if any(tuple(child) != keys for child in children):
                raise ValueError("Reference fact inventory changed within one cell")
            return {
                key: self.diagram.branch(index, [child[key] for child in children])
                for key in keys
            }

        return visit(0, {})


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_table(*, snapshot: Path, axes_file: Path) -> Wire:
    """Produce separate fact diagrams; independent replay is still required."""
    from _lcm_regime_finder.cli import load_problem
    from _lcm_regime_finder.fibre_source import FibreSource
    from _lcm_regime_finder.structural_store import final_input_check
    import _lcm_regime_finder

    package = Path(_lcm_regime_finder.__file__).parent
    guards = {path.name: sha(path) for path in sorted(package.glob("*.py"))}
    here = Path(__file__).resolve().parent
    pinned_sources = json.loads((here / "reference_sources.json").read_text())
    if any(guards.get(name) != digest for name, digest in pinned_sources.items()):
        raise ValueError(
            "Authoritative baseline source bytes changed; requalify instead of silently rebinding"
        )
    algorithm = {
        name: sha(here / name)
        for name in (
            "implicit_core.py",
            "implicit_native.py",
            "binding_purity.py",
            "reference_sources.json",
        )
    }
    snapshot_sha = sha(snapshot)
    axes_sha = sha(axes_file)
    original = load_problem(snapshot)
    source = FibreSource(original)
    status_axes = {
        name: _strings(value)
        for name, value in _mapping(json.loads(axes_file.read_text())).items()
    }
    regimes = {cell["regime"] for cell in source._cells.values()}
    if set(status_axes) != regimes:
        raise ValueError(
            "Status classification must cover every original regime exactly"
        )
    cells = [
        CellCompiler(
            source=source,
            cell_id=cell_id,
            status_axes=status_axes[_text(source._cells[cell_id]["regime"])],
        ).compile(status_axes=status_axes)
        for cell_id in source.get_cell_ids()
    ]
    final_input_check(original)
    if (
        sha(snapshot) != snapshot_sha
        or sha(axes_file) != axes_sha
        or any(sha(package / name) != digest for name, digest in guards.items())
        or any(sha(here / name) != digest for name, digest in algorithm.items())
    ):
        raise ValueError("Source transaction changed")
    result = {
        "schema": "exact-implicit-fibre-table/v1",
        "reference_contract": "compile_primitive_fibres.py@587478274ff55f1d873ae9d056bb9bf3e222b873",
        "status": "PRODUCED_UNVERIFIED",
        "snapshot_sha256": snapshot_sha,
        "classification_sha256": axes_sha,
        "reference_sources": guards,
        "algorithm_sources": algorithm,
        "metadata_by_reference": {
            "snapshot_sha256": snapshot_sha,
            "cells": list(source._cells),
            "relations": list(source._relations),
            "domains": list(source._domains),
            "scope": "All original declaration fields, owners, laws, ages, phases and structural payloads remain in the pinned unchanged source; no field is projected away.",
        },
        "source_obligations": list(source.source_obligations),
        "cells": cells,
        "qualification": "Exact producer representation; independent full-domain replay and finite-screen proof are still required. No native adoption or optimality claim.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "axes",
        type=Path,
        help="JSON mapping original regimes to status coordinate arrays",
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compile_table(snapshot=args.snapshot, axes_file=args.axes)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
