"""Portable independent literal controls; no project imports or callbacks."""

from collections.abc import Callable, Mapping, Sequence
from _lcm_regime_finder.fibre_source import _mapping
from _lcm_regime_finder.implicit.implicit_core import (
    Assignment,
    ReadCertificate as CoreReadCertificate,
)
from itertools import product, permutations
import json
import unittest

from _lcm_regime_finder.implicit.implicit_core import (
    MDD,
    ReadCertificate,
    Failure,
    joint_reads,
    pack,
    unpack,
)
from _lcm_regime_finder.implicit.verify_native import check_graph, same
from _lcm_regime_finder.implicit.binding_purity import certify_binding_cache
from tests.regime_finder.implicit_cache_witness import nested_records
from _lcm_regime_finder.fibre_source import _text


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


class DiagramTests(unittest.TestCase):
    def test_demand_shares_unused_axis_without_deleting_addresses(self) -> None:
        d = MDD((("unused", tuple(range(100))), ("x", (4, 9))))
        root = d.demand(lambda env: env.code("x") == 4)
        self.assertEqual(d.attempts, 3)
        self.assertEqual(len(list(d.cubes(root))), 2)
        for unused in (0, 31, 99):
            for x in (4, 9):
                with self.subTest(unused=unused, x=x):
                    self.assertEqual(
                        d.evaluate(root, {"unused": unused, "x": x}), x == 4
                    )

    def test_reverse_demand_order_reordered_exactly(self) -> None:
        d = MDD((("a", (0, 1)), ("b", (0, 1)), ("c", (0, 1))))

        def expression(env: Assignment) -> int:
            c = env.code("c")
            return (env.code("a") if c else env.code("b")) + c

        root = d.demand(expression)
        for a, b, c in product((0, 1), repeat=3):
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(
                    d.evaluate(root, dict(a=a, b=b, c=c)), (a if c else b) + c
                )
        self.assertTrue(
            all(
                node[0] == "leaf" or all(d.top(child) > node[1] for child in node[2])
                for node in d.nodes
            )
        )

    def test_simultaneous_majority_free_coordinates_all_orders(self) -> None:
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
                with self.subTest(order=order, statuses=statuses):
                    self.assertEqual(
                        d.evaluate(root, statuses),
                        literal_joint(coordinates, statuses, order, majority_reads),
                    )
                    answer = d.evaluate(root, statuses)
                    self.assertIsInstance(answer, ReadCertificate)
                    assert isinstance(answer, ReadCertificate)
                    self.assertEqual(len(answer.live_coordinates - {"x"}), 2)
            answer = d.evaluate(root, {name: 1 for name in order})
            assert isinstance(answer, ReadCertificate)
            self.assertEqual(answer.removed_statuses, (order[0],))

    def test_generated_read_programs_with_joint_fixed_masks(self) -> None:
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
                with self.subTest(rule=rule, a=a, b=b):
                    self.assertEqual(
                        d.evaluate(root, {"a": a, "b": b}),
                        literal_joint(coordinates, {"a": a, "b": b}, order, scalar),
                    )

    def test_zero_axes_is_one_original_fibre(self) -> None:
        d = MDD(())
        root = joint_reads(
            d,
            coordinates=frozenset({"x"}),
            removal_order=(),
            reader=lambda mask: d.leaf(frozenset({"x"})),
        )
        self.assertEqual(
            d.evaluate(root, {}), ReadCertificate(frozenset({"x"}), (), ())
        )
        self.assertEqual(
            list(d.cubes(root)), [((), ReadCertificate(frozenset({"x"}), (), ()))]
        )

    def test_duplicate_numeric_values_keep_distinct_nominal_codes(self) -> None:
        d = MDD((("choice", (10, 20)),))
        numeric = d.demand(lambda env: {10: 1.0, 20: 1.0}[env.code("choice")])
        code = d.demand(lambda env: env.code("choice"))
        self.assertEqual(d.top(numeric), 1)
        self.assertNotEqual(
            d.evaluate(code, {"choice": 10}), d.evaluate(code, {"choice": 20})
        )
        self.assertEqual(list(d.cubes(numeric))[0][0], (frozenset({10, 20}),))

    def test_native_tuple_json_list_are_distinct_leaves(self) -> None:
        d = MDD((("x", (0, 1)),))
        root = d.demand(lambda env: (1, 2) if env.code("x") else [1, 2])
        self.assertNotEqual(d.top(root), 1)
        self.assertEqual(unpack(pack((1, 2))), (1, 2))
        self.assertEqual(unpack(pack([1, 2])), [1, 2])
        self.assertNotEqual(d.leaf(False), d.leaf(0))

    def test_cubes_are_full_disjoint_partition(self) -> None:
        d = MDD((("a", (0, 1, 2)), ("b", (4, 5)), ("c", (8, 9))))
        root = d.demand(
            lambda env: env.code("b") + (env.code("a") if env.code("c") == 8 else 0)
        )
        observed = {}
        for box, value in d.cubes(root):
            for point in product(*box):
                self.assertNotIn(point, observed)
                observed[point] = value
        self.assertEqual(len(observed), 12)
        for point in product(*(codes for _, codes in d.domains)):
            self.assertEqual(
                observed[point], d.evaluate(root, dict(zip(("a", "b", "c"), point)))
            )

    def test_failure_is_not_silently_conservative_fact(self) -> None:
        d = MDD((("a", (0, 1)),))
        failed = Failure("ValueError", "invalid source")
        root = joint_reads(
            d,
            coordinates=frozenset({"a"}),
            removal_order=("a",),
            reader=lambda mask: d.leaf(failed),
        )
        self.assertEqual(d.evaluate(root, {"a": 0}), failed)
        self.assertEqual(d.evaluate(root, {"a": 1}), failed)

    def test_invalid_original_domains_and_addresses_reject(self) -> None:
        for domains in (
            (("a", ()),),
            (("a", (0, 0)),),
            (("a", (False, 1)),),
            (("a", (0,)), ("a", (1,))),
        ):
            with self.subTest(domains=domains):
                with self.assertRaises(ValueError):
                    MDD(domains)
        d = MDD((("a", (0, 1)),))
        root = d.leaf(True)
        for point in ({}, {"a": 2}, {"a": False}, {"a": 0, "extra": 1}):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    d.evaluate(root, point)

    def test_export_removes_unreachable_work_nodes(self) -> None:
        d = MDD((("a", (0, 1)),))
        d.leaf("not in final result")
        root = d.demand(lambda env: env.code("a"))
        wire = json.loads(json.dumps(d.export({"result": root})))
        self.assertEqual(len(wire["nodes"]), 3)
        self.assertEqual(wire["domains"], [["a", [0, 1]]])

    def test_independent_decoder_rejects_forged_domain_and_missing_root(self) -> None:
        d = MDD((("a", (0, 1)),))
        root = d.demand(lambda env: frozenset({env.code("a")}))
        wire = d.export({"exclude:choice": root})
        lookup = check_graph(wire, d.domains, {"exclude:choice"})
        self.assertTrue(same(lookup("exclude:choice", (0,)), frozenset({0})))
        changed = json.loads(json.dumps(wire))
        changed["domains"][0][1][0] = False
        with self.assertRaises(ValueError):
            check_graph(changed, d.domains, {"exclude:choice"})
        with self.assertRaises(ValueError):
            check_graph(wire, d.domains, {"exclude:choice", "direct"})

    def test_independent_comparison_does_not_alias_boolean_and_nominal_code(
        self,
    ) -> None:
        self.assertFalse(same(frozenset({False}), frozenset({0})))
        self.assertFalse(
            same(("support", frozenset({0}), 1), ("support", frozenset({0}), True))
        )
        self.assertFalse(same((1, 2), [1, 2]))

    def test_independent_decoder_rejects_cycles_and_noncoverage(self) -> None:
        d = MDD((("a", (0, 1)),))
        root = d.demand(lambda env: frozenset({env.code("a")}))
        wire = d.export({"result": root})
        branch = wire["roots"]["result"]
        bad = json.loads(json.dumps(wire))
        bad["nodes"][branch]["children"][0] = branch
        with self.assertRaises(ValueError):
            check_graph(bad, d.domains, {"result"})
        bad = json.loads(json.dumps(wire))
        bad["nodes"][branch]["children"].pop()
        with self.assertRaises(ValueError):
            check_graph(bad, d.domains, {"result"})

    def test_binding_height_proof_respects_exact_depth_boundary(self) -> None:
        records, long, short = nested_records()
        resolve = lambda ref: records.get(ref.get("capture_ref"))
        cold = certify_binding_cache(
            callbacks={"x": {"bindings": {"B": long}}}, records=records, resolve=resolve
        )
        shallow = certify_binding_cache(
            callbacks={"x": {"bindings": {"B": short}}},
            records=records,
            resolve=resolve,
        )
        self.assertFalse(cold["proved"])
        self.assertTrue(shallow["proved"])
        self.assertEqual(shallow["max_expanded_depth"], 64)

    def test_binding_height_proof_rejects_cycles_and_unknown_roots(self) -> None:
        records, long, _ = nested_records(2)
        records["B1"]["value"] = [long]
        resolve = lambda ref: records.get(ref.get("capture_ref"))
        self.assertFalse(
            certify_binding_cache(
                callbacks={"x": {"bindings": {"B": long}}},
                records=records,
                resolve=resolve,
            )["proved"]
        )
        self.assertFalse(
            certify_binding_cache(
                callbacks={
                    "x": {
                        "defaults": {
                            "B": {"kind": "callable-reference", "capture_ref": "absent"}
                        }
                    }
                },
                records={},
                resolve=lambda ref: None,
            )["proved"]
        )

    def test_opaque_binding_leaves_are_pure_failures_not_scalar_certificates(
        self,
    ) -> None:
        for kind in ("module", "callable", "primitive", "unresolved"):
            with self.subTest(kind=kind):
                result = certify_binding_cache(
                    callbacks={
                        "f": {
                            "bindings": {
                                "library": {
                                    "kind": kind,
                                    "arbitrary_opaque_fields": {
                                        "depth": "not interpreted"
                                    },
                                }
                            }
                        }
                    },
                    records={},
                    resolve=lambda ref: None,
                )
                self.assertTrue(result["proved"])
                self.assertEqual(result["max_expanded_depth"], 0)

    def test_resolved_callable_reference_is_invariant_binding_failure(self) -> None:
        target = {"kind": "callable", "capture_ref": "C", "source": "opaque"}
        reference = {"kind": "callable-reference", "capture_ref": "C"}
        result = certify_binding_cache(
            callbacks={"f": {"bindings": {"helper": reference}}},
            records={"C": target},
            resolve=lambda ref: target,
        )
        self.assertTrue(result["proved"])
        self.assertEqual(result["max_expanded_depth"], 0)
        unresolved = certify_binding_cache(
            callbacks={"f": {"bindings": {"helper": reference}}},
            records={},
            resolve=lambda ref: None,
        )
        self.assertFalse(unresolved["proved"])

    def test_binding_height_proof_counts_inline_wrappers(self) -> None:
        records, _, short = nested_records()
        result = certify_binding_cache(
            callbacks={"x": {"partial_keywords": {"B": [short]}}},
            records=records,
            resolve=lambda ref: records.get(_text(_mapping(ref)["capture_ref"])),
        )
        self.assertFalse(result["proved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
