"""The actual CE primitive binds every registered execution behavior."""

import inspect
import types

import jax.numpy as jnp
import pytest
from jax._src import core as jax_core
from jax._src.interpreters import ad, batching, mlir
from jax.extend.core import Primitive

from _lcm.egm import ce_scaled_linear as scaled
from _lcm.solution import fingerprint as fingerprints
from lcm.typing import BoolND, FloatND, IntND

_SELECTED_PRIMITIVE = scaled._scaled_ce_p


def _actual_chain(direction: FloatND) -> FloatND:
    return scaled.join_ce_direction(
        logabs=jnp.zeros_like(direction),
        sign=jnp.ones_like(direction),
        base=jnp.zeros_like(direction),
        direction=direction,
    )


def _direct_bind(direction: FloatND) -> FloatND:
    return _SELECTED_PRIMITIVE.bind(direction, direction, direction, direction)


def test_actual_ce_function_fingerprint_is_stable_with_legitimate_rule_cycles() -> None:
    first = fingerprints._semantic_fingerprint(_actual_chain)
    assert fingerprints._semantic_fingerprint(_actual_chain) == first
    assert fingerprints._semantic_fingerprint(
        _direct_bind
    ) == fingerprints._semantic_fingerprint(_direct_bind)


@pytest.mark.parametrize(
    "member", ["impl", "abstract_eval", "multiple_results", "payload"]
)
def test_actual_ce_fingerprint_refuses_changed_primitive_state(
    *, monkeypatch: pytest.MonkeyPatch, member: str
) -> None:
    original = fingerprints._semantic_fingerprint(_actual_chain)
    replacement = True if member == "multiple_results" else scaled.scale_ce_direction
    with monkeypatch.context() as patch:
        patch.setitem(vars(scaled._scaled_ce_p), member, replacement)
        with pytest.raises(TypeError, match=r"CE primitive.*state"):
            fingerprints._semantic_fingerprint(_actual_chain)
    assert fingerprints._semantic_fingerprint(_actual_chain) == original


@pytest.mark.parametrize(
    "registry", ["jvp", "transpose", "batch", "lowering", "linearization", "platform"]
)
def test_actual_ce_fingerprint_refuses_changed_registered_behavior(
    *, monkeypatch: pytest.MonkeyPatch, registry: str
) -> None:
    primitive = scaled._scaled_ce_p
    fingerprints._semantic_fingerprint(_actual_chain)
    if registry == "jvp":
        replacement = next(
            rule for owner, rule in ad.primitive_jvps.items() if owner is not primitive
        )
        monkeypatch.setitem(ad.primitive_jvps, primitive, replacement)
    elif registry == "transpose":
        replacement = next(
            rule
            for owner, rule in ad.primitive_transposes.items()
            if owner is not primitive
        )
        monkeypatch.setitem(ad.primitive_transposes, primitive, replacement)
    elif registry == "batch":
        replacement = next(
            rule
            for owner, rule in batching.fancy_primitive_batchers.items()
            if owner is not primitive
        )
        monkeypatch.setitem(batching.fancy_primitive_batchers, primitive, replacement)
    elif registry == "lowering":
        replacement = next(
            rule for owner, rule in mlir._lowerings.items() if owner is not primitive
        )
        monkeypatch.setitem(mlir._lowerings, primitive, replacement)
    elif registry == "linearization":
        replacement = next(iter(ad.primitive_linearizations.values()))
        monkeypatch.setitem(ad.primitive_linearizations, primitive, replacement)
    else:
        monkeypatch.setitem(
            mlir._platform_specific_lowerings["cpu"],
            primitive,
            mlir._lowerings[primitive],
        )
    with pytest.raises(TypeError, match="CE primitive execution registration changed"):
        fingerprints._semantic_fingerprint(_actual_chain)


def test_same_named_primitive_cannot_borrow_original_capture_privilege(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints._semantic_fingerprint(_direct_bind)
    forged = Primitive(scaled._scaled_ce_p.name)
    monkeypatch.setitem(globals(), "_SELECTED_PRIMITIVE", forged)
    with pytest.raises(TypeError, match="same-named CE primitive"):
        fingerprints._semantic_fingerprint(_direct_bind)


def test_ce_fingerprint_refuses_changed_original_bind_method(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints._semantic_fingerprint(_actual_chain)
    monkeypatch.setattr(
        type(scaled._scaled_ce_p), "bind", type(scaled._scaled_ce_p).bind_with_trace
    )
    with pytest.raises(
        TypeError, match=r"original CE primitive bind|native class state"
    ):
        fingerprints._semantic_fingerprint(_actual_chain)


def _alternative_parts(*, logabs: FloatND) -> tuple[FloatND, IntND, BoolND]:
    return (
        logabs,
        jnp.zeros_like(logabs, dtype=jnp.int32),
        jnp.ones_like(logabs, dtype=jnp.bool_),
    )


def test_ce_fingerprint_walks_original_transitive_arithmetic_dependencies(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fingerprints._semantic_fingerprint(_actual_chain)
    monkeypatch.setattr(scaled, "_coefficient_parts", _alternative_parts)
    assert fingerprints._semantic_fingerprint(_actual_chain) != original


def test_ce_fingerprint_refuses_rebound_genuine_lowering_closure(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints._semantic_fingerprint(_actual_chain)
    lowering = mlir._lowerings[scaled._scaled_ce_p].rule
    assert isinstance(lowering, types.FunctionType)
    assert lowering.__closure__ is not None
    cell = next(
        cell
        for name, cell in zip(
            lowering.__code__.co_freevars, lowering.__closure__, strict=True
        )
        if name == "fun"
    )
    monkeypatch.setattr(cell, "cell_contents", _alternative_parts)
    with pytest.raises(TypeError, match="generated wrapper closure changed"):
        fingerprints._semantic_fingerprint(_actual_chain)


@pytest.mark.parametrize("binding", ["ShapedArray", "Zero", "UndefinedPrimal"])
def test_ce_fingerprint_refuses_same_named_metadata_class_at_original_path(
    *, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    fingerprints._semantic_fingerprint(_actual_chain)
    owner = scaled if binding == "ShapedArray" else scaled.ad
    original = getattr(owner, binding)
    forged = type(original.__name__, (), {"__module__": original.__module__})
    monkeypatch.setattr(owner, binding, forged)
    with pytest.raises(TypeError, match="native binding changed"):
        fingerprints._semantic_fingerprint(_actual_chain)


def test_ce_fingerprint_refuses_rebound_original_native_helper_export(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints._semantic_fingerprint(_actual_chain)
    monkeypatch.setattr(scaled.batching, "bdim_at_front", jax_core.typeof)
    with pytest.raises(TypeError, match="native binding changed"):
        fingerprints._semantic_fingerprint(_actual_chain)


@pytest.mark.parametrize(
    "metadata", ["__defaults__", "__kwdefaults__", "__annotations__"]
)
def test_ce_fingerprint_refuses_changed_rule_call_metadata(
    *, monkeypatch: pytest.MonkeyPatch, metadata: str
) -> None:
    original = fingerprints._semantic_fingerprint(_actual_chain)
    replacement = (None,) if metadata == "__defaults__" else {"wrong": None}
    with monkeypatch.context() as patch:
        if metadata == "__annotations__":
            # The annotations setter also clears Python's deferred annotation owner.
            patch.setitem(scaled._batch.__annotations__, "wrong", None)
        else:
            patch.setattr(scaled._batch, metadata, replacement)
        with pytest.raises(TypeError, match="callable code or metadata changed"):
            fingerprints._semantic_fingerprint(_actual_chain)
    assert fingerprints._semantic_fingerprint(_actual_chain) == original


def test_ce_fingerprint_refuses_rebound_generated_wrapper_global(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints._semantic_fingerprint(_actual_chain)
    wrapper = scaled._scaled_ce_p.abstract_eval
    assert isinstance(wrapper, types.FunctionType)
    monkeypatch.setitem(wrapper.__globals__, "no_effects", {"changed": True})
    with pytest.raises(TypeError, match="native binding changed"):
        fingerprints._semantic_fingerprint(_actual_chain)


def test_ce_fingerprint_keeps_original_fresh_builtin_binding() -> None:
    receiver = jax_core.typeof.__globals__["pytype_aval_mappings"]
    assert type(receiver) is dict
    first, second = receiver.get, receiver.get
    assert first is not second
    assert first.__self__ is receiver
    assert second.__self__ is receiver
    assert inspect.getattr_static(receiver, "get") is dict.get
    original = fingerprints._semantic_fingerprint(_actual_chain)
    assert fingerprints._semantic_fingerprint(_actual_chain) == original


def test_ce_fingerprint_refuses_equal_clone_of_original_native_receiver(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints._semantic_fingerprint(_actual_chain)
    original = jax_core.typeof.__globals__["pytype_aval_mappings"]
    assert type(original) is dict
    monkeypatch.setitem(
        jax_core.typeof.__globals__, "pytype_aval_mappings", dict(original)
    )
    with pytest.raises(TypeError, match="native binding changed"):
        fingerprints._semantic_fingerprint(_actual_chain)


def test_ce_fingerprint_refuses_rebound_original_native_descriptor(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints._semantic_fingerprint(_actual_chain)
    monkeypatch.setattr(
        jax_core.ShapedArray, "update", jax_core.ShapedArray.strip_weak_type
    )
    with pytest.raises(TypeError, match="native class state changed"):
        fingerprints._semantic_fingerprint(_actual_chain)
