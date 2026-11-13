"""Joint status certificates preserve every agreeing state assignment."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from itertools import permutations, product

import pytest

from _lcm_regime_finder.expressions import partial
from _lcm_regime_finder.fibre_facts import (
    CoordinateRead,
    FibreLink,
    FibreLiveness,
    FibreReadFacts,
    JointReadSet,
    PrimitiveReads,
    calculate_fibre_liveness,
    get_joint_read_set,
)
from _lcm_regime_finder.records import JSONValue


type Assignment = tuple[int, ...]
type TruthTable = Mapping[Assignment, int]


def _truth_reader(
    names: tuple[str, ...],
    table: TruthTable,
    calls: list[dict[str, int]],
) -> PrimitiveReads:
    def read(*, fixed_statuses: Mapping[str, int]) -> frozenset[str]:
        calls.append(dict(fixed_statuses))
        allowed = tuple(
            assignment
            for assignment in table
            if all(assignment[names.index(k)] == v for k, v in fixed_statuses.items())
        )
        reads = set()
        for index, name in enumerate(names):
            if name in fixed_statuses:
                continue
            if any(
                table[left] != table[right]
                and all(
                    a == b
                    for j, (a, b) in enumerate(zip(left, right, strict=True))
                    if j != index
                )
                for left in allowed
                for right in allowed
            ):
                reads.add(name)
        return frozenset(reads)

    return read


def _literal_status_certificate(
    names: tuple[str, ...],
    table: TruthTable,
    fibre: Assignment,
    order: tuple[str, ...],
) -> frozenset[str]:
    kept = set(names)
    for name in order:
        candidate = kept - {name}
        payoffs = {
            payoff
            for assignment, payoff in table.items()
            if all(
                assignment[names.index(k)] == fibre[names.index(k)] for k in candidate
            )
        }
        if payoffs == {table[fibre]}:
            kept = candidate
    return frozenset(kept)


def _expression_reader(
    expression: JSONValue, calls: list[dict[str, int]]
) -> PrimitiveReads:
    def read(*, fixed_statuses: Mapping[str, int]) -> frozenset[str]:
        calls.append(dict(fixed_statuses))
        return partial(expression, fixed_statuses).reads

    return read


@pytest.mark.parametrize("fibre", tuple(product((0, 1), repeat=3)))
@pytest.mark.parametrize("order", tuple(permutations(("a", "b", "c"))))
def test_majority_certificate_agrees_with_literal_finite_payoffs(
    fibre: Assignment, order: tuple[str, ...]
) -> None:
    names = ("a", "b", "c")
    table = {
        assignment: int(sum(assignment) >= 2)
        for assignment in product((0, 1), repeat=3)
    }
    statuses = dict(zip(names, fibre, strict=True))
    calls: list[dict[str, int]] = []
    result = get_joint_read_set(
        coordinates=frozenset(names),
        statuses=statuses,
        removal_order=order,
        read_primitives=_truth_reader(names, table, calls),
    )
    assert result.live_coordinates == _literal_status_certificate(
        names, table, fibre, order
    )
    assert len(result.live_coordinates) == 2
    assert result.removal_order == order
    assert result.removed_statuses == tuple(
        name for name in order if name not in result.live_coordinates
    )
    for assignment, payoff in table.items():
        if all(
            assignment[names.index(k)] == statuses[k] for k in result.live_coordinates
        ):
            assert payoff == table[fibre]
    kept = set(names)
    for removed in result.removed_statuses:
        kept.remove(removed)
        assert {k: statuses[k] for k in kept} in calls
    assert all(set(call) <= set(statuses) for call in calls)
    assert all(call[k] == statuses[k] for call in calls for k in call)
    assert statuses == dict(zip(names, fibre, strict=True))


def test_majority_zero_and_one_fibres_have_distinct_kept_values() -> None:
    names = ("a", "b", "c")
    table = {
        assignment: int(sum(assignment) >= 2)
        for assignment in product((0, 1), repeat=3)
    }
    certificates = []
    for fibre in ((0, 0, 0), (1, 1, 1)):
        statuses = dict(zip(names, fibre, strict=True))
        result = get_joint_read_set(
            coordinates=frozenset(names),
            statuses=statuses,
            removal_order=names,
            read_primitives=_truth_reader(names, table, []),
        )
        certificates.append(
            tuple((k, statuses[k]) for k in sorted(result.live_coordinates))
        )
    assert certificates[0] != certificates[1]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(0, frozenset({"status"})), (1, frozenset({"status", "wealth"}))],
)
def test_branch_only_drops_wealth_where_kept_status_proves_it_unused(
    status: int, expected: frozenset[str]
) -> None:
    calls: list[dict[str, int]] = []
    expression: JSONValue = {"if": "status == 0", "then": 0, "else": "wealth"}
    result = get_joint_read_set(
        coordinates=frozenset({"status", "wealth"}),
        statuses={"status": status},
        removal_order=("status",),
        read_primitives=_expression_reader(expression, calls),
    )
    assert result.live_coordinates == expected
    assert {} in calls
    assert all(set(call) <= {"status"} for call in calls)
    assert partial(expression, {}).reads == frozenset({"status", "wealth"})


@pytest.mark.parametrize(
    "expression",
    [
        {
            "if": {"opaque": "same", "reads": ["status", "wealth", "choice", "beta"]},
            "then": 0,
            "else": 0,
        },
        {
            "if": "status == 0",
            "then": {"opaque": "same", "reads": ["wealth", "choice", "beta"]},
            "else": {"opaque": "same", "reads": ["wealth", "choice", "beta"]},
        },
    ],
)
def test_opaque_branch_tokens_do_not_prove_status_independence(
    expression: JSONValue,
) -> None:
    calls: list[dict[str, int]] = []
    assert not partial(expression, {"status": 0}).known
    result = get_joint_read_set(
        coordinates=frozenset({"status", "wealth"}),
        statuses={"status": 0},
        removal_order=("status",),
        read_primitives=_expression_reader(expression, calls),
    )
    assert result.live_coordinates == frozenset({"status", "wealth"})
    assert result.removed_statuses == ()
    assert all(set(call) <= {"status"} for call in calls)


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        (frozenset(), frozenset()),
        (frozenset({"wealth", "choice", "beta"}), frozenset({"wealth"})),
    ],
)
def test_nonstate_inputs_are_never_fixed_or_returned_as_state_coordinates(
    reads: frozenset[str], expected: frozenset[str]
) -> None:
    calls: list[dict[str, int]] = []

    def reader(*, fixed_statuses: Mapping[str, int]) -> frozenset[str]:
        calls.append(dict(fixed_statuses))
        return reads

    result = get_joint_read_set(
        coordinates=frozenset({"wealth", "unused"}),
        statuses={},
        removal_order=(),
        read_primitives=reader,
    )
    assert result.live_coordinates == expected
    assert calls and all(call == {} for call in calls)


def test_dead_statuses_are_removed_in_supplied_order_and_result_is_frozen() -> None:
    statuses = {"a": 0, "b": 1}
    order = ("b", "a")
    calls: list[dict[str, int]] = []
    result = get_joint_read_set(
        coordinates=frozenset({"a", "b", "wealth"}),
        statuses=statuses,
        removal_order=order,
        read_primitives=_expression_reader(0, calls),
    )
    assert result == JointReadSet(frozenset(), order, order)
    assert statuses == {"a": 0, "b": 1}
    assert {"a": 0} in calls and {} in calls
    with pytest.raises(FrozenInstanceError):
        setattr(result, "live_coordinates", frozenset({"a"}))


@pytest.mark.parametrize(
    ("statuses", "order"),
    [
        ({"unknown": 0}, ("unknown",)),
        ({"a": 0}, ()),
        ({"a": 0}, ("a", "a")),
        ({"a": 0}, ("unknown",)),
        ({"a": "not-an-integer"}, ("a",)),
        ({"a": 0.5}, ("a",)),
    ],
)
def test_malformed_status_or_removal_order_is_rejected(
    statuses: Mapping[str, int], order: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        get_joint_read_set(
            coordinates=frozenset({"a"}),
            statuses=statuses,
            removal_order=order,
            read_primitives=_expression_reader(0, []),
        )


def test_reader_cannot_add_an_uncovered_coordinate_after_joint_certification() -> None:
    responses = iter((frozenset(), frozenset({"wealth"})))

    def changing_reader(*, fixed_statuses: Mapping[str, int]) -> frozenset[str]:
        assert fixed_statuses == {}
        return next(responses)

    with pytest.raises(ValueError, match="changed its joint dependency certificate"):
        get_joint_read_set(
            coordinates=frozenset({"wealth"}),
            statuses={},
            removal_order=(),
            read_primitives=changing_reader,
        )


def _fixed_reads(*coordinates: str) -> JointReadSet:
    return JointReadSet(frozenset(coordinates), (), ())


def _literal_graph_closure(
    facts: tuple[FibreReadFacts, ...],
) -> tuple[FibreLiveness, ...]:
    nodes = tuple(
        (fibre.identifier, coordinate)
        for fibre in facts
        for coordinate in sorted(fibre.coordinates)
    )
    direct = {
        (fibre.identifier, coordinate)
        for fibre in facts
        for coordinate in fibre.direct.live_coordinates
    }
    candidates = []
    for mask in product((False, True), repeat=len(nodes)):
        live = {node for node, present in zip(nodes, mask, strict=True) if present}
        if not direct <= live:
            continue
        if all(
            (fibre.identifier, source_coordinate) in live
            for fibre in facts
            for link in fibre.links
            for update in link.updates
            if (link.target, update.coordinate) in live
            for source_coordinate in update.reads.live_coordinates
        ):
            candidates.append(live)
    assert candidates
    least = set.intersection(*candidates)
    assert least in candidates
    return tuple(
        FibreLiveness(
            fibre.identifier,
            frozenset(
                coordinate for owner, coordinate in least if owner == fibre.identifier
            ),
        )
        for fibre in facts
    )


def test_liveness_propagates_multiple_steps_and_preserves_input_order() -> None:
    facts = (
        FibreReadFacts("last", frozenset({"z"}), _fixed_reads("z"), ()),
        FibreReadFacts(
            "first",
            frozenset({"a", "dead"}),
            _fixed_reads(),
            (FibreLink("middle", (CoordinateRead("b", _fixed_reads("a")),)),),
        ),
        FibreReadFacts(
            "middle",
            frozenset({"b"}),
            _fixed_reads(),
            (FibreLink("last", (CoordinateRead("z", _fixed_reads("b")),)),),
        ),
    )
    expected = (
        FibreLiveness("last", frozenset({"z"})),
        FibreLiveness("first", frozenset({"a"})),
        FibreLiveness("middle", frozenset({"b"})),
    )
    assert _literal_graph_closure(facts) == expected
    result = calculate_fibre_liveness(facts)
    assert result == expected
    assert isinstance(result, tuple)
    with pytest.raises(FrozenInstanceError):
        setattr(result[0], "live_coordinates", frozenset())


def test_constant_update_needs_no_same_named_source_coordinate() -> None:
    facts = (
        FibreReadFacts(
            "source",
            frozenset(),
            _fixed_reads(),
            (FibreLink("target", (CoordinateRead("wealth", _fixed_reads()),)),),
        ),
        FibreReadFacts("target", frozenset({"wealth"}), _fixed_reads("wealth"), ()),
    )
    assert calculate_fibre_liveness(facts) == _literal_graph_closure(facts)
    assert calculate_fibre_liveness(facts)[0].live_coordinates == frozenset()


def test_only_live_target_updates_contribute_source_reads() -> None:
    facts = (
        FibreReadFacts(
            "source",
            frozenset({"shared", "irrelevant"}),
            _fixed_reads(),
            (
                FibreLink(
                    "target",
                    (
                        CoordinateRead("x", _fixed_reads("shared")),
                        CoordinateRead("y", _fixed_reads("shared")),
                        CoordinateRead("dead", _fixed_reads("irrelevant")),
                    ),
                ),
            ),
        ),
        FibreReadFacts(
            "target", frozenset({"x", "y", "dead"}), _fixed_reads("x", "y"), ()
        ),
    )
    assert calculate_fibre_liveness(facts) == (
        FibreLiveness("source", frozenset({"shared"})),
        FibreLiveness("target", frozenset({"x", "y"})),
    )


@pytest.mark.parametrize("has_output", [False, True])
def test_same_period_dependency_cycles_have_the_least_fixed_point(
    has_output: bool,
) -> None:
    facts = (
        FibreReadFacts(
            "first",
            frozenset({"x"}),
            _fixed_reads("x") if has_output else _fixed_reads(),
            (FibreLink("second", (CoordinateRead("y", _fixed_reads("x")),)),),
        ),
        FibreReadFacts(
            "second",
            frozenset({"y"}),
            _fixed_reads(),
            (
                FibreLink("first", (CoordinateRead("x", _fixed_reads("y")),)),
                FibreLink("second", (CoordinateRead("y", _fixed_reads("y")),)),
            ),
        ),
    )
    assert calculate_fibre_liveness(facts) == _literal_graph_closure(facts)
    assert all(
        bool(row.live_coordinates) == has_output
        for row in calculate_fibre_liveness(facts)
    )


@pytest.mark.parametrize("edge_mask", range(64))
def test_generated_tiny_graphs_match_literal_closed_set_enumeration(
    edge_mask: int,
) -> None:
    names = ("a", "b", "c")
    possible = tuple(
        (source, target) for source in names for target in names if source != target
    )
    edges = {edge for index, edge in enumerate(possible) if edge_mask & (1 << index)}
    for output_mask in range(8):
        facts = tuple(
            FibreReadFacts(
                name,
                frozenset({"x"}),
                _fixed_reads("x") if output_mask & (1 << index) else _fixed_reads(),
                tuple(
                    FibreLink(target, (CoordinateRead("x", _fixed_reads("x")),))
                    for source, target in sorted(edges)
                    if source == name
                ),
            )
            for index, name in enumerate(names)
        )
        assert calculate_fibre_liveness(facts) == _literal_graph_closure(facts)


def test_closure_uses_supplied_joint_certificates_without_reselecting_them() -> None:
    direct = JointReadSet(frozenset({"b", "c"}), ("a", "b", "c"), ("a",))
    update = JointReadSet(frozenset({"a", "c"}), ("b", "a", "c"), ("b",))
    facts = (
        FibreReadFacts(
            "source",
            frozenset({"a", "b", "c"}),
            direct,
            (FibreLink("target", (CoordinateRead("x", update),)),),
        ),
        FibreReadFacts("target", frozenset({"x"}), _fixed_reads("x"), ()),
    )
    before = deepcopy(facts)
    result = calculate_fibre_liveness(facts)
    assert result == _literal_graph_closure(facts)
    assert result[0].live_coordinates == frozenset({"a", "b", "c"})
    assert facts == before
    assert facts[0].direct is direct
    assert facts[0].links[0].updates[0].reads is update
    assert direct.removed_statuses == ("a",)
    assert update.removed_statuses == ("b",)


@pytest.mark.parametrize("removed_read", ["direct", "live_update"])
def test_literal_oracle_detects_a_missing_direct_or_live_update_read(
    removed_read: str,
) -> None:
    facts = (
        FibreReadFacts(
            "source",
            frozenset({"x"}),
            _fixed_reads(),
            (FibreLink("target", (CoordinateRead("y", _fixed_reads("x")),)),),
        ),
        FibreReadFacts("target", frozenset({"y"}), _fixed_reads("y"), ()),
    )
    expected = _literal_graph_closure(facts)
    assert calculate_fibre_liveness(facts) == expected
    if removed_read == "direct":
        mutated = (facts[0], replace(facts[1], direct=_fixed_reads()))
    else:
        mutated = (
            replace(
                facts[0],
                links=(FibreLink("target", (CoordinateRead("y", _fixed_reads()),)),),
            ),
            facts[1],
        )
    assert calculate_fibre_liveness(mutated) != expected


@pytest.mark.parametrize(
    "defect",
    [
        "empty_id",
        "duplicate_id",
        "missing_target",
        "direct_outside_source",
        "update_outside_source",
        "missing_update",
        "duplicate_update",
        "extra_update",
    ],
)
def test_invalid_fibre_or_update_inventory_is_rejected(defect: str) -> None:
    source = FibreReadFacts(
        "source",
        frozenset({"x"}),
        _fixed_reads(),
        (FibreLink("target", (CoordinateRead("y", _fixed_reads("x")),)),),
    )
    target = FibreReadFacts("target", frozenset({"y"}), _fixed_reads("y"), ())
    if defect == "empty_id":
        source = replace(source, identifier="")
    elif defect == "duplicate_id":
        source = replace(source, identifier="target")
    elif defect == "missing_target":
        source = replace(source, links=(replace(source.links[0], target="absent"),))
    elif defect == "direct_outside_source":
        source = replace(source, direct=_fixed_reads("not_declared"))
    elif defect == "update_outside_source":
        source = replace(
            source,
            links=(
                FibreLink(
                    "target", (CoordinateRead("y", _fixed_reads("not_declared")),)
                ),
            ),
        )
    elif defect == "missing_update":
        source = replace(source, links=(FibreLink("target", ()),))
    elif defect == "duplicate_update":
        source = replace(
            source, links=(FibreLink("target", source.links[0].updates * 2),)
        )
    elif defect == "extra_update":
        source = replace(
            source,
            links=(
                FibreLink(
                    "target",
                    source.links[0].updates
                    + (CoordinateRead("extra", _fixed_reads()),),
                ),
            ),
        )
    with pytest.raises(ValueError):
        calculate_fibre_liveness((source, target))
