"""Source parsing is shared; proof environments and changed source stay distinct."""

from copy import deepcopy
from typing import TypedDict


type PythonSource = str


class ConstantObject(TypedDict):
    type: str
    fields: dict[str, int]


class ConstantCallable(TypedDict):
    bound_self: ConstantObject
    capture_contract: str
    arguments: list[str]
    raw_arguments: list[str]
    variadic: bool
    bindings: dict[str, dict[str, str]]
    source: PythonSource


class ConstantCapture(ConstantObject):
    callable: ConstantCallable


from _lcm_regime_finder.semantic_check import SourceProof, _CallableRecords


def callback(source):
    return {
        "arguments": ["history", "age"],
        "original": {
            "kind": "callable",
            "arguments": ["history", "age"],
            "source": source,
        },
    }


def test_repeated_source_preserves_date_specific_reads() -> None:
    callbacks = {
        "utility": callback(
            "def utility(history, age):\n    return history if age < 30 else 0\n"
        )
    }
    records = _CallableRecords()
    for age in (20, 40, 20, 40):
        answer = SourceProof(
            callbacks, static={"age": age}, callable_records=records
        ).callback("utility")
        assert answer.proved
        assert (answer.reads) == (frozenset({"history"}) if age < 30 else frozenset())


def test_changed_source_with_same_name_cannot_reuse_old_tree() -> None:
    callbacks = {
        "utility": callback("def utility(history, age):\n    return history\n")
    }
    records = _CallableRecords()
    first = SourceProof(
        callbacks, static={"age": 40}, callable_records=records
    ).callback("utility")
    callbacks = deepcopy(callbacks)
    callbacks["utility"]["original"]["source"] = (
        "def utility(history, age):\n    return 0\n"
    )
    second = SourceProof(
        callbacks, static={"age": 40}, callable_records=records
    ).callback("utility")
    assert (first.reads) == (frozenset({"history"}))
    assert (second.reads) == (frozenset())


def test_unparseable_source_does_not_acquire_a_proof_from_cache() -> None:
    callbacks = {"utility": callback("def utility(history, age): ???")}
    records = _CallableRecords()
    for age in (20, 40):
        proof = SourceProof(callbacks, static={"age": age}, callable_records=records)
        assert not (proof.callback("utility").proved)
        assert proof.notes


def test_identity_reuses_syntax_but_checks_the_requested_state_each_time() -> None:
    from _lcm_regime_finder.semantic_check import _identity_callback

    cb = {"original": {"source": "def carry(history):\n    return history\n"}}
    records = _CallableRecords()
    assert _identity_callback(cb, "history", records)
    assert not (_identity_callback(cb, "assets", records))
    assert _identity_callback(cb, "history", records)


def test_constant_reuses_syntax_but_rechecks_its_bound_receiver() -> None:
    from _lcm_regime_finder.semantic_check import _closed_native_constant

    tag = "_lcm.regime_building.schedules._Constant"
    original: ConstantCapture = {
        "type": tag,
        "fields": {"value": 0},
        "callable": {
            "bound_self": {"type": tag, "fields": {"value": 0}},
            "capture_contract": "inspection-only-frozen-bindings-2",
            "arguments": [],
            "raw_arguments": ["self"],
            "variadic": False,
            "bindings": {"jnp": {"kind": "module", "name": "jax.numpy"}},
            "source": "def __call__(self):\n    return jnp.asarray(self.value)\n",
        },
    }
    records = _CallableRecords()
    assert _closed_native_constant(original, records)
    assert _closed_native_constant(original, records)
    changed = deepcopy(original)
    changed["callable"]["bound_self"]["fields"]["value"] = 1
    assert not (_closed_native_constant(changed, records))
