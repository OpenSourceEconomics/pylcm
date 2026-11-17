"""Portable independent literal controls; no project imports or callbacks."""

from collections.abc import Callable, Mapping, Sequence
from _lcm_regime_finder.implicit.implicit_core import (
    Assignment,
    ReadCertificate as CoreReadCertificate,
)
from itertools import product, permutations
import json

import pytest

from _lcm_regime_finder.implicit.implicit_core import (
    MDD,
    ReadCertificate,
    Failure,
    joint_reads,
    pack,
    unpack,
)


def literal_joint(
    coordinates: frozenset[str],
    statuses: Mapping[str, int],
    order: Sequence[str],
    reader: Callable[[Mapping[str, int]], frozenset[str]],
) -> CoreReadCertificate:
    """Direct set algorithm, intentionally no diagram algebra or translator."""

    def reads(kept: frozenset[str]) -> frozenset[str]:
        fixed = {name: statuses[name] for name in order if name in kept}
        return frozenset(reader(fixed)) & coordinates

    kept = frozenset(statuses) | reads(frozenset())
    removed = []
    for name in order:
        proposed = kept - {name}
        if reads(proposed) <= proposed:
            kept = proposed
            removed.append(name)
    live = (kept & statuses.keys()) | reads(kept)
    if not live <= kept:
        raise ValueError("unstable certificate")
    return ReadCertificate(frozenset(live), tuple(order), tuple(removed))


def majority_reads(fixed: Mapping[str, int]) -> frozenset[str]:
    # Exact finite Boolean majority dependency for the fixed partial assignment.
    # Ordinary x is an always-live separate consumer.
    for value in (0, 1):
        if sum(code == value for code in fixed.values()) >= 2:
            return frozenset({"x"})
    return frozenset({"x", "a", "b", "c"}) - fixed.keys()


def test_demand_shares_unused_axis_without_deleting_addresses() -> None:
    d = MDD((("unused", tuple(range(100))), ("x", (4, 9))))
    root = d.demand(lambda env: env.code("x") == 4)
    assert d.attempts == 3
    assert len(list(d.cubes(root))) == 2
    for unused in (0, 31, 99):
        for x in (4, 9):
            assert d.evaluate(root, {"unused": unused, "x": x}) == (x == 4), (
                unused,
                x,
            )


def test_reverse_demand_order_reordered_exactly() -> None:
    d = MDD((("a", (0, 1)), ("b", (0, 1)), ("c", (0, 1))))

    def expression(env: Assignment) -> int:
        c = env.code("c")
        return (env.code("a") if c else env.code("b")) + c

    root = d.demand(expression)
    for a, b, c in product((0, 1), repeat=3):
        assert d.evaluate(root, dict(a=a, b=b, c=c)) == (a if c else b) + c, (a, b, c)
    assert all(
        node[0] == "leaf" or all(d.top(child) > node[1] for child in node[2])
        for node in d.nodes
    )


def test_simultaneous_majority_free_coordinates_all_orders() -> None:
    coordinates = frozenset({"a", "b", "c", "x"})
    for order in permutations(("a", "b", "c")):
        d = MDD(tuple((name, (0, 1)) for name in order))

        def reader(mask: frozenset[str]) -> int:
            return d.demand(
                lambda env: majority_reads(
                    {name: env.code(name) for name in order if name in mask}
                )
            )

        root = joint_reads(
            d, coordinates=coordinates, removal_order=order, reader=reader
        )
        for values in product((0, 1), repeat=3):
            statuses = dict(zip(order, values))
            assert d.evaluate(root, statuses) == literal_joint(
                coordinates, statuses, order, majority_reads
            ), (order, statuses)
            answer = d.evaluate(root, statuses)
            assert isinstance(answer, ReadCertificate), (order, statuses)
            assert len(answer.live_coordinates - {"x"}) == 2, (order, statuses)
        answer = d.evaluate(root, {name: 1 for name in order})
        assert isinstance(answer, ReadCertificate)
        assert answer.removed_statuses == (order[0],)


def test_generated_read_programs_with_joint_fixed_masks() -> None:
    coordinates = frozenset({"a", "b", "ordinary"})
    order = ("a", "b")
    for rule in range(12):

        def scalar(fixed: Mapping[str, int], rule: int = rule) -> frozenset[str]:
            if rule % 3 == 0:
                return coordinates  # Defined conservative unsupported result.
            if "a" in fixed and fixed["a"] == rule % 2:
                return frozenset({"ordinary"})
            if rule % 3 == 1 and "b" in fixed and fixed["b"] == 1:
                return frozenset()
            return coordinates - fixed.keys()

        d = MDD((("a", (0, 1)), ("b", (0, 1))))
        root = joint_reads(
            d,
            coordinates=coordinates,
            removal_order=order,
            reader=lambda mask: d.demand(
                lambda env: scalar(
                    {name: env.code(name) for name in order if name in mask}
                )
            ),
        )
        for a, b in product((0, 1), repeat=2):
            assert d.evaluate(root, {"a": a, "b": b}) == literal_joint(
                coordinates, {"a": a, "b": b}, order, scalar
            ), (rule, a, b)


def test_zero_axes_is_one_original_fibre() -> None:
    d = MDD(())
    root = joint_reads(
        d,
        coordinates=frozenset({"x"}),
        removal_order=(),
        reader=lambda mask: d.leaf(frozenset({"x"})),
    )
    assert d.evaluate(root, {}) == ReadCertificate(frozenset({"x"}), (), ())
    assert list(d.cubes(root)) == [((), ReadCertificate(frozenset({"x"}), (), ()))]


def test_duplicate_numeric_values_keep_distinct_nominal_codes() -> None:
    d = MDD((("choice", (10, 20)),))
    numeric = d.demand(lambda env: {10: 1.0, 20: 1.0}[env.code("choice")])
    code = d.demand(lambda env: env.code("choice"))
    assert d.top(numeric) == 1
    assert d.evaluate(code, {"choice": 10}) != d.evaluate(code, {"choice": 20})
    assert list(d.cubes(numeric))[0][0] == (frozenset({10, 20}),)


def test_native_tuple_json_list_are_distinct_leaves() -> None:
    d = MDD((("x", (0, 1)),))
    root = d.demand(lambda env: (1, 2) if env.code("x") else [1, 2])
    assert d.top(root) != 1
    assert unpack(pack((1, 2))) == (1, 2)
    assert unpack(pack([1, 2])) == [1, 2]
    assert d.leaf(False) != d.leaf(0)


def test_cubes_are_full_disjoint_partition() -> None:
    d = MDD((("a", (0, 1, 2)), ("b", (4, 5)), ("c", (8, 9))))
    root = d.demand(
        lambda env: env.code("b") + (env.code("a") if env.code("c") == 8 else 0)
    )
    observed = {}
    for box, value in d.cubes(root):
        for point in product(*box):
            assert point not in observed
            observed[point] = value
    assert len(observed) == 12
    for point in product(*(codes for _, codes in d.domains)):
        assert observed[point] == d.evaluate(root, dict(zip(("a", "b", "c"), point)))


def test_failure_is_not_silently_conservative_fact() -> None:
    d = MDD((("a", (0, 1)),))
    failed = Failure("ValueError", "invalid source")
    root = joint_reads(
        d,
        coordinates=frozenset({"a"}),
        removal_order=("a",),
        reader=lambda mask: d.leaf(failed),
    )
    assert d.evaluate(root, {"a": 0}) == failed
    assert d.evaluate(root, {"a": 1}) == failed


def test_invalid_original_domains_and_addresses_reject() -> None:
    for domains in (
        (("a", ()),),
        (("a", (0, 0)),),
        (("a", (False, 1)),),
        (("a", (0,)), ("a", (1,))),
    ):
        with pytest.raises(ValueError):
            MDD(domains)
    d = MDD((("a", (0, 1)),))
    root = d.leaf(True)
    for point in ({}, {"a": 2}, {"a": False}, {"a": 0, "extra": 1}):
        with pytest.raises(ValueError):
            d.evaluate(root, point)


def test_export_removes_unreachable_work_nodes() -> None:
    d = MDD((("a", (0, 1)),))
    d.leaf("not in final result")
    root = d.demand(lambda env: env.code("a"))
    wire = json.loads(json.dumps(d.export({"result": root})))
    assert len(wire["nodes"]) == 3
    assert wire["domains"] == [["a", [0, 1]]]


def test_historical_eval_three_fields_remain_byte_exact() -> None:
    from _lcm_regime_finder.implicit.implicit_core import Eval, UNKNOWN

    old = ["Eval", [["frozenset", []], ["Unknown", []], ["bool", True]]]
    assert pack(Eval(frozenset())) == old
    assert unpack(old) == Eval(frozenset(), UNKNOWN, True)


def test_eval_image_roundtrip_and_leaf_identity() -> None:
    from _lcm_regime_finder.implicit.implicit_core import Eval, UNKNOWN

    left = Eval(frozenset({"choice"}), UNKNOWN, True, (1, 2))
    right = Eval(frozenset({"choice"}), UNKNOWN, True, (3, 4))
    boolean = Eval(frozenset(), UNKNOWN, True, (False, True))
    for value in (left, right, boolean):
        assert unpack(pack(value)) == value
    assert pack(left) != pack(right)
    diagram = MDD(())
    assert diagram.leaf(left) != diagram.leaf(right)
    assert diagram.leaf(boolean) != diagram.leaf(
        Eval(frozenset(), UNKNOWN, True, (0, 1))
    )


def test_malformed_fourth_fields_are_rejected() -> None:
    from _lcm_regime_finder.records import JSONValue

    malformed: tuple[JSONValue, ...] = (
        ["NoneType", None],
        ["tuple", []],
        ["list", [["int", 1]]],
        ["tuple", [["bool", True], ["int", 1]]],
        ["tuple", [["int", 1], ["int", 1]]],
        ["tuple", [["int", 2**31]]],
        ["tuple", [["float", "0x1.0000000000000p+0"]]],
        ["tuple", [["dict", []]]],
        ["tuple", [["int", value] for value in range(65)]],
    )
    for image in malformed:
        wire: JSONValue = [
            "Eval",
            [["frozenset", []], ["Unknown", []], ["bool", True], image],
        ]
        with pytest.raises((TypeError, ValueError)):
            unpack(wire)


def test_fake_int_is_rejected_before_its_equality_or_hash() -> None:
    from _lcm_regime_finder.implicit.implicit_core import Eval, UNKNOWN

    class FakeInt(int):
        def __eq__(self, _other: object) -> bool:
            raise AssertionError("Support codec must not call fake-int equality")

        def __hash__(self) -> int:
            raise AssertionError("Support codec must not call fake-int hashing")

    with pytest.raises(ValueError):
        pack(Eval(frozenset(), UNKNOWN, True, (FakeInt(1),)))
