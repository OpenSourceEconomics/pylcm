"""Enumerate tiny existing-cell families without production search or cost helpers.

Projection policies are the offered globally coherent policies, not a claim that
all possible economic reductions were discovered. Every label partition and cut
mask is enumerated. Raw costs use per-block cardinality products and literal
nonzero-support source/target pairs. No source callback is executed.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import json
from math import prod


type JSON = None | bool | int | float | str | Sequence[JSON] | Mapping[str, JSON]
type CellId = str
type DomainId = str
type Label = str


@dataclass(frozen=True)
class Block:
    cells: tuple[CellId, ...]
    """Original cells covered exactly once by this block."""
    selected_options: tuple[tuple[CellId, str], ...]
    """Offered projection option for each original cell."""
    state_rows: int
    """Product of retained solve-state domain cardinalities."""
    action_rows: int
    """Product of retained action domain cardinalities."""
    evaluations: int
    """State-action evaluations before link charges."""
    link_charge: int
    """Literal outgoing-pair charge for this block."""
    directed_links: int
    """Distinct outgoing source/target pairs, without a price."""
    unpriced_link_rows: int
    """State-action entries times outgoing pairs, without a price."""
    storage: int
    """One state array for each covered original cell."""
    characteristics: int
    """Distinct declared characteristic maps, reported without a price."""
    dispatch: int
    """Additional original regime labels dispatched by the block."""
    source_contexts: tuple[str, ...]
    """Canonical unchanged original-cell context records."""


@dataclass(frozen=True)
class Layout:
    blocks: tuple[Block, ...]
    """A complete feasible partition under one offered policy."""
    policy: str
    """Identifier of the globally coherent offered projection policy."""

    @property
    def evaluations(self) -> int:
        return sum(block.evaluations for block in self.blocks)

    @property
    def cost(self) -> tuple[int, int, int]:
        return (
            sum(block.evaluations + block.link_charge for block in self.blocks),
            sum(block.storage for block in self.blocks),
            len(self.blocks),
        )


def _mapping(value: JSON) -> Mapping[str, JSON]:
    assert isinstance(value, Mapping)
    return value


def _rows(value: JSON) -> tuple[Mapping[str, JSON], ...]:
    assert isinstance(value, (list, tuple))
    return tuple(_mapping(item) for item in value)


def _name(value: JSON) -> str:
    assert isinstance(value, str)
    return value


def _sequence(value: JSON) -> Sequence[JSON]:
    assert isinstance(value, (list, tuple))
    return value


def _integer(value: JSON) -> int:
    assert type(value) is int
    return value


def _canonical(value: JSON) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _partitions(labels: tuple[Label, ...]) -> Iterator[tuple[tuple[Label, ...], ...]]:
    """Enumerate canonical assignments, without search recurrences or contraction."""
    for assignment in product(range(len(labels)), repeat=len(labels)):
        used = -1
        for index in assignment:
            if index > used + 1:
                break
            used = max(used, index)
        else:
            yield tuple(
                tuple(
                    label
                    for label, index in zip(labels, assignment, strict=True)
                    if index == group
                )
                for group in range(used + 1)
            )


def enumerate_layouts(
    *, analysis: Mapping[str, JSON], admission: Mapping[str, JSON]
) -> tuple[Layout, ...]:
    """Keep every admitted layout witness in the finite supplied existing family."""
    assert admission["schema"] == "lcm-regime-block-admission/v1"
    declaration = _mapping(analysis["declarations"])
    allocation = _mapping(
        analysis.get("allocation_scenario", declaration["allocation"])
    )
    assert allocation["kind"] == "per_block"
    boundary = allocation.get("boundary_unit", 1)
    assert type(boundary) is int and boundary >= 0
    cells = {_name(cell["id"]): cell for cell in _rows(declaration["cells"])}
    assert len(cells) == len(_rows(declaration["cells"]))
    domains = dict(_mapping(declaration["domains"])) | dict(
        _mapping(analysis.get("derived_domains", {}))
    )
    interfaces = _mapping(analysis["interfaces"])
    options = {
        cid: {
            _name(option["id"]): option
            for option in _rows(_mapping(interfaces[cid])["options"])
        }
        for cid in cells
    }
    block_spec = _mapping(admission["block"])
    defaults = _mapping(block_spec["missing_field_values"])
    equal_fields = block_spec["equal_fields"]
    assert isinstance(equal_fields, (tuple, list))
    context_fields = tuple(
        _name(field)
        for field in _sequence(
            _mapping(admission["source_dispatch"])["preserve_fields"]
        )
    )
    components: dict[str, dict[Label, list[CellId]]] = {}
    for cid, cell in cells.items():
        context = _canonical(
            [cell.get(_name(field), defaults[_name(field)]) for field in equal_fields]
        )
        components.setdefault(context, {}).setdefault(_name(cell["regime"]), []).append(
            cid
        )
    pairs = {
        (_name(relation["source"]), _name(target))
        for relation in _rows(declaration["relations"])
        if relation.get("support") != "zero"
        for target in _sequence(relation.get("targets", []))
    }

    def make_block(
        *, members: tuple[CellId, ...], policy: Mapping[str, JSON]
    ) -> Block | None:
        chosen = [options[cid][_name(policy[cid])] for cid in members]
        products = []
        for kind in ("states", "actions"):
            union: dict[str, DomainId] = {}
            for option in chosen:
                for axis, domain in _mapping(option[kind]).items():
                    identity = _name(domain)
                    if axis in union and union[axis] != identity:
                        return None
                    union[axis] = identity
            cardinalities = []
            for identity in union.values():
                domain = _mapping(domains[identity])
                if domain.get("solve_axis") is False:
                    continue
                size = domain["size"]
                assert type(size) is int and size > 0
                cardinalities.append(size)
            products.append(prod(cardinalities))
        states, actions = products
        outgoing = sum(
            source in members and target not in members for source, target in pairs
        )
        maps = {
            _canonical(projection)
            for option in chosen
            for projection in _mapping(option["state_projection"]).values()
            if _mapping(projection)["kind"] == "characteristic"
        }
        contexts = tuple(
            _canonical(
                {
                    "id": cid,
                    "regime": cells[cid]["regime"],
                    "period": cells[cid]["period"],
                    **{
                        _name(field): cells[cid].get(_name(field))
                        for field in context_fields
                    },
                }
            )
            for cid in members
        )
        return Block(
            cells=members,
            selected_options=tuple((cid, _name(policy[cid])) for cid in members),
            state_rows=states,
            action_rows=actions,
            evaluations=len(members) * states * actions,
            link_charge=outgoing * boundary * states * actions,
            directed_links=outgoing,
            unpriced_link_rows=outgoing * states * actions,
            storage=len(members) * states,
            characteristics=len(maps),
            dispatch=len({cells[cid]["regime"] for cid in members}) - 1,
            source_contexts=contexts,
        )

    def intervals(
        *, ids: tuple[CellId, ...], policy: Mapping[str, JSON]
    ) -> Iterator[tuple[Block, ...]]:
        dates = sorted({_integer(cells[cid]["period"]) for cid in ids})
        buckets = [
            tuple(sorted(cid for cid in ids if cells[cid]["period"] == date))
            for date in dates
        ]
        for mask in range(1 << (len(buckets) - 1)):
            blocks, members = [], ()
            for index, bucket in enumerate(buckets):
                members += bucket
                if index == len(buckets) - 1 or mask & (1 << index):
                    block = make_block(members=tuple(sorted(members)), policy=policy)
                    if block is None:
                        break
                    blocks.append(block)
                    members = ()
            else:
                yield tuple(blocks)

    result = []
    for record in _rows(analysis["projection_policies"]):
        policy = _mapping(record["option_by_cell"])
        assert set(policy) == set(cells)
        component_choices = []
        for component in components.values():
            choices = []
            for groups in _partitions(tuple(sorted(component))):
                interval_choices = [
                    tuple(
                        intervals(
                            ids=tuple(
                                cid for label in group for cid in component[label]
                            ),
                            policy=policy,
                        )
                    )
                    for group in groups
                ]
                choices.extend(
                    tuple(block for group in chosen for block in group)
                    for chosen in product(*interval_choices)
                )
            component_choices.append(choices)
        for selected in product(*component_choices):
            blocks = tuple(block for component in selected for block in component)
            assert sorted(cid for block in blocks for cid in block.cells) == sorted(
                cells
            )
            result.append(Layout(blocks=blocks, policy=_name(record["id"])))
    return tuple(result)


def pareto_layouts(layouts: Sequence[Layout]) -> tuple[Layout, ...]:
    """Retain every witness for each nondominated raw three-count vector."""
    return tuple(
        layout
        for layout in layouts
        if not any(
            rival.cost != layout.cost
            and all(
                left <= right
                for left, right in zip(rival.cost, layout.cost, strict=True)
            )
            for rival in layouts
        )
    )


def at_most_curve(
    *, layouts: Sequence[Layout], include_links: bool = True
) -> tuple[Layout, ...]:
    """Choose least work at each feasible regime bound, then regimes and storage."""
    curve = []
    for bound in range(1, max(len(layout.blocks) for layout in layouts) + 1):
        eligible = [layout for layout in layouts if len(layout.blocks) <= bound]
        if eligible:
            curve.append(
                min(
                    eligible,
                    key=lambda layout: (
                        layout.cost[0] if include_links else layout.evaluations,
                        len(layout.blocks),
                        layout.cost[1],
                        repr(layout),
                    ),
                )
            )
    return tuple(curve)
