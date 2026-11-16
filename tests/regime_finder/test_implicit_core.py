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
    pack,
    unpack,
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


def test_native_tuple_json_list_are_distinct_leaves() -> None:
    d = MDD((("x", (0, 1)),))
    root = d.demand(lambda env: (1, 2) if env.code("x") else [1, 2])
    assert d.top(root) != 1
    assert unpack(pack((1, 2))) == (1, 2)
    assert unpack(pack([1, 2])) == [1, 2]
    assert d.leaf(False) != d.leaf(0)


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
