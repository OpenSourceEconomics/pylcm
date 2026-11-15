"""Portable exact finite decision diagrams; no pylcm or callback execution.

Nodes share calculations, not original fibre identities.  Every domain code is
still addressable.  Leaf reduction is typed equality, never numeric-code equality.
"""

from dataclasses import dataclass, fields
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import Literal, TypedDict, cast
from ..records import JSONValue
from fractions import Fraction
import json


@dataclass(frozen=True)
class Unknown:
    pass


UNKNOWN = Unknown()


@dataclass(frozen=True)
class Failure:
    exception: str
    message: str


@dataclass(frozen=True)
class Eval:
    reads: frozenset[str]
    value: object = UNKNOWN
    proved: bool = True


@dataclass(frozen=True)
class ReadCertificate:
    live_coordinates: frozenset[str]
    removal_order: tuple[str, ...]
    removed_statuses: tuple[str, ...]


@dataclass(frozen=True)
class Support:
    possible_codes: frozenset[int]
    proved: bool


def pack(value: object) -> JSONValue:
    """Lossless, closed-format leaf encoding, including tuple/list and bool/int."""
    if value is None or type(value) in (str, bool, int):
        return [type(value).__name__, cast(JSONValue, value)]
    if type(value) is float:
        return ["float", value.hex()]
    if isinstance(value, Fraction):
        return ["Fraction", str(value)]
    if isinstance(value, (tuple, list, set, frozenset)):
        children = [pack(item) for item in value]
        if isinstance(value, (set, frozenset)):
            children.sort(key=lambda x: json.dumps(x, sort_keys=True))
        return [type(value).__name__, children]
    if isinstance(value, dict):
        return ["dict", [[pack(k), pack(v)] for k, v in value.items()]]
    if type(value) in (Unknown, Failure, Eval, ReadCertificate, Support):
        return [
            type(value).__name__,
            [
                pack(getattr(value, f.name))
                for f in fields(
                    cast(Unknown | Failure | Eval | ReadCertificate | Support, value)
                )
            ],
        ]
    raise TypeError("Unsupported terminal payload type: " + type(value).__name__)


def unpack(data: JSONValue) -> object:
    """Decode internal terminals; published certificates use a separate verifier."""
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        raise ValueError("Malformed typed terminal")
    tag, value = data
    if not isinstance(tag, str):
        raise ValueError("Malformed terminal tag")
    if tag == "NoneType":
        return None
    if tag in ("str", "bool", "int"):
        if type(value).__name__ != tag:
            raise ValueError("Bad typed scalar")
        return value
    if tag in ("float", "Fraction"):
        if not isinstance(value, str):
            raise ValueError("Bad numeric terminal")
        return float.fromhex(value) if tag == "float" else Fraction(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("Bad terminal field sequence")
    if tag in ("tuple", "list", "set", "frozenset"):
        constructors: Mapping[str, Callable[..., object]] = {
            "tuple": tuple,
            "list": list,
            "set": set,
            "frozenset": frozenset,
        }
        return constructors[tag](unpack(item) for item in value)
    if tag == "dict":
        result: dict[Hashable, object] = {}
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Bad terminal mapping entry")
            key = unpack(pair[0])
            if not isinstance(key, Hashable):
                raise TypeError("Unhashable terminal key")
            result[key] = unpack(pair[1])
        return result
    decoded = tuple(unpack(item) for item in value)
    if tag == "Unknown" and not decoded:
        return UNKNOWN
    if tag == "Failure" and len(decoded) == 2:
        exception, message = decoded
        if isinstance(exception, str) and isinstance(message, str):
            return Failure(exception, message)
    if tag in {"Eval", "ReadCertificate"} and len(decoded) == 3:
        live, second, third = decoded
        if isinstance(live, frozenset) and all(isinstance(name, str) for name in live):
            names = cast(frozenset[str], live)
            if tag == "Eval" and type(third) is bool:
                return Eval(names, second, third)
            if (
                isinstance(second, tuple)
                and isinstance(third, tuple)
                and all(isinstance(name, str) for name in second + third)
            ):
                return ReadCertificate(
                    names, cast(tuple[str, ...], second), cast(tuple[str, ...], third)
                )
    if tag == "Support" and len(decoded) == 2:
        codes, proved = decoded
        if (
            isinstance(codes, frozenset)
            and type(proved) is bool
            and all(type(code) is int for code in codes)
        ):
            return Support(cast(frozenset[int], codes), proved)
    raise ValueError("Unknown terminal tag: " + tag)


def identity(value: object) -> str:
    return json.dumps(pack(value), sort_keys=True, separators=(",", ":"))


class Demand(BaseException):
    """Not caught by the reference analyzer's conservative Exception handlers."""

    def __init__(self, axis: str) -> None:
        self.axis = axis


class Assignment:
    def __init__(
        self, fixed: Mapping[str, int], domains: Mapping[str, tuple[int, ...]]
    ) -> None:
        self.fixed, self.domains = fixed, domains

    def code(self, axis: str) -> int:
        if axis not in self.domains:
            raise KeyError(axis)
        if axis not in self.fixed:
            raise Demand(axis)
        return self.fixed[axis]


type Branch = tuple[Literal["branch"], int, tuple[int, ...]]
type Node = tuple[Literal["leaf"], object] | Branch


class NodeWire(TypedDict, total=False):
    leaf: JSONValue
    axis: int
    children: list[int]


class DiagramWire(TypedDict):
    domains: list[list]
    nodes: list[NodeWire]
    roots: dict[str, int]


type SelectionState = tuple[frozenset[str], tuple[str, ...]]


class MDD:
    """Reduced ordered finite-domain diagrams with exact functional apply."""

    def __init__(self, domains: Sequence[tuple[str, Sequence[int]]]) -> None:
        self.domains = tuple((name, tuple(codes)) for name, codes in domains)
        if len({name for name, _ in self.domains}) != len(self.domains):
            raise ValueError("Duplicate status axis")
        for _, codes in self.domains:
            if (
                not codes
                or any(type(x) is not int for x in codes)
                or len(set(codes)) != len(codes)
            ):
                raise ValueError("Expected nonempty distinct integer original codes")
        self.positions = {name: i for i, (name, _) in enumerate(self.domains)}
        self.nodes: list[Node] = []
        self.intern: dict[tuple[Literal["leaf"], str] | Branch, int] = {}
        self.attempts = 0
        self.demand_splits = 0

    def leaf(self, value: object) -> int:
        key = ("leaf", identity(value))
        if key not in self.intern:
            self.intern[key] = len(self.nodes)
            self.nodes.append(("leaf", value))
        return self.intern[key]

    def top(self, root: int) -> int:
        node = self.nodes[root]
        return len(self.domains) if node[0] == "leaf" else node[1]

    def branch(self, axis: int, children: Sequence[int]) -> int:
        children = tuple(children)
        if len(children) != len(self.domains[axis][1]):
            raise ValueError("Domain coverage gap")
        if len(set(children)) == 1:
            return children[0]
        if any(self.top(child) <= axis for child in children):
            raise ValueError("Diagram variable order violation")
        key = ("branch", axis, children)
        if key not in self.intern:
            self.intern[key] = len(self.nodes)
            self.nodes.append(key)
        return self.intern[key]

    def cofactor(self, root: int, axis: int, index: int) -> int:
        top = self.top(root)
        if top == axis:
            node = self.nodes[root]
            assert node[0] == "branch"
            return node[2][index]
        if top > axis:
            return root
        raise ValueError("Cofactor requested below current root")

    def apply(self, operation: Callable[..., object], *roots: int) -> int:
        memo = {}

        def visit(items: tuple[int, ...]) -> int:
            if items in memo:
                return memo[items]
            axis = min(self.top(x) for x in items)
            if axis == len(self.domains):
                answer = self.leaf(operation(*(self.nodes[x][1] for x in items)))
            else:
                answer = self.branch(
                    axis,
                    [
                        visit(tuple(self.cofactor(x, axis, i) for x in items))
                        for i in range(len(self.domains[axis][1]))
                    ],
                )
            memo[items] = answer
            return answer

        return visit(tuple(roots))

    def select(self, axis: int) -> int:
        return self.branch(axis, [self.leaf(code) for code in self.domains[axis][1]])

    def case(self, axis: int, children: Sequence[int]) -> int:
        """Substitute children even when they test earlier variables."""
        children = tuple(children)
        codes = self.domains[axis][1]
        if len(children) != len(codes):
            raise ValueError("Missing branch")
        selector = self.select(axis)
        # Applying the selector together with all children enforces one common
        # original variable order.  Only the selected leaf influences the result.
        return self.apply(
            lambda code, *values: values[codes.index(code)], selector, *children
        )

    def bind(self, root: int, operation: Callable[[object], int]) -> int:
        memo = {}

        def visit(node_id: int) -> int:
            if node_id not in memo:
                node = self.nodes[node_id]
                memo[node_id] = (
                    operation(node[1])
                    if node[0] == "leaf"
                    else self.case(node[1], [visit(child) for child in node[2]])
                )
            return memo[node_id]

        return visit(root)

    def demand(self, operation: Callable[[Assignment], object]) -> int:
        """Restart a pure terminating operation only at an observed code demand.

        No inference from a representative is made: undemanded code values have
        never been inspected by this execution.  Native independent certification
        is still required before accepting a producer's coverage claim.
        """
        domains = dict(self.domains)

        def visit(fixed: dict[str, int]) -> int:
            self.attempts += 1
            try:
                return self.leaf(operation(Assignment(fixed, domains)))
            except Demand as requested:
                axis = requested.axis
                if axis in fixed or axis not in self.positions:
                    raise ValueError("Invalid or repeated demand")
                self.demand_splits += 1
                return self.case(
                    self.positions[axis],
                    [visit({**fixed, axis: code}) for code in domains[axis]],
                )

        return visit({})

    def evaluate(self, root: int, statuses: Mapping[str, int]) -> object:
        if set(statuses) != set(self.positions):
            raise ValueError("Address must specify every original status")
        for name, codes in self.domains:
            if type(statuses[name]) is not int or statuses[name] not in codes:
                raise ValueError("Unknown original status code")
        while self.nodes[root][0] != "leaf":
            node = self.nodes[root]
            assert node[0] == "branch"
            _, axis, children = node
            name, codes = self.domains[axis]
            root = children[codes.index(statuses[name])]
        return self.nodes[root][1]

    def cubes(self, root: int) -> Iterator[tuple[tuple[frozenset[int], ...], object]]:
        """Yield disjoint complete terminal boxes; preserve omitted axes in full."""
        whole = tuple(frozenset(codes) for _, codes in self.domains)

        def visit(
            node_id: int, box: tuple[frozenset[int], ...]
        ) -> Iterator[tuple[tuple[frozenset[int], ...], object]]:
            node = self.nodes[node_id]
            if node[0] == "leaf":
                yield box, node[1]
                return
            _, axis, children = node
            grouped = {}
            for code, child in zip(self.domains[axis][1], children):
                grouped.setdefault(child, set()).add(code)
            for child, codes in grouped.items():
                new = list(box)
                new[axis] = frozenset(codes)
                yield from visit(child, tuple(new))

        yield from visit(root, whole)

    def export(self, roots: Mapping[str, int]) -> DiagramWire:
        reachable = set()

        def visit(root: int) -> None:
            if root in reachable:
                return
            reachable.add(root)
            if self.nodes[root][0] == "branch":
                node = self.nodes[root]
                assert node[0] == "branch"
                for child in node[2]:
                    visit(child)

        for root in roots.values():
            visit(root)
        order = sorted(reachable)
        new = {old: i for i, old in enumerate(order)}
        nodes: list[NodeWire] = []
        for old in order:
            node = self.nodes[old]
            nodes.append(
                {"leaf": pack(node[1])}
                if node[0] == "leaf"
                else {"axis": node[1], "children": [new[x] for x in node[2]]}
            )
        return {
            "domains": [[name, list(codes)] for name, codes in self.domains],
            "nodes": nodes,
            "roots": {name: new[root] for name, root in roots.items()},
        }


def joint_reads(
    diagram: MDD,
    *,
    coordinates: frozenset[str],
    removal_order: Sequence[str],
    reader: Callable[[frozenset[str]], int],
) -> int:
    """Lift the exact ordered get_joint_read_set state machine over diagrams.

    reader(mask) returns a diagram of read sets with precisely mask statuses fixed
    together; ordinary states/actions/parameters remain free in its implementation.
    """
    statuses = frozenset(removal_order)
    if tuple(name for name, _ in diagram.domains) != tuple(removal_order):
        raise ValueError("Original declaration order differs")
    if not statuses <= coordinates:
        raise ValueError("Undeclared status")

    def intersect(value: object) -> object:
        return (
            value
            if isinstance(value, Failure)
            else cast(frozenset[str], value) & coordinates
        )

    cache = {}

    def reads(kept: frozenset[str]) -> int:
        mask = frozenset(kept) & statuses
        if mask not in cache:
            cache[mask] = diagram.apply(intersect, reader(mask))
        return cache[mask]

    initial = reads(frozenset())
    state = diagram.apply(
        lambda r: r if isinstance(r, Failure) else (statuses | r, ()), initial
    )
    for name in removal_order:

        def step(record: object, name: str = name) -> int:
            if isinstance(record, Failure):
                return diagram.leaf(record)
            kept, removed = cast(SelectionState, record)
            candidate = kept - {name}

            def update(r: object) -> object:
                if isinstance(r, Failure):
                    return r
                return (
                    (candidate, removed + (name,))
                    if cast(frozenset[str], r) <= candidate
                    else record
                )

            return diagram.apply(update, reads(candidate))

        state = diagram.bind(state, step)

    def finish(record: object) -> int:
        if isinstance(record, Failure):
            return diagram.leaf(record)
        kept, removed = cast(SelectionState, record)

        def result(r: object) -> object:
            if isinstance(r, Failure):
                return r
            live = (kept & statuses) | cast(frozenset[str], r)
            if not live <= kept:
                return Failure(
                    "ValueError",
                    "Primitive reader changed its joint dependency certificate",
                )
            return ReadCertificate(frozenset(live), tuple(removal_order), removed)

        return diagram.apply(result, reads(kept))

    return diagram.bind(state, finish)
