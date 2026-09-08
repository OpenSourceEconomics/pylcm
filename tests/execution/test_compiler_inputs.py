"""Compiler-live paths describe actual dynamic leaves, including repeated aliases."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.compiler_inputs import compiler_input_paths
from lcm.exceptions import ExecutionPlanningError


def _nested_operation(
    *, payload: Mapping[str, Any], offset: jax.Array, first: bool
) -> jax.Array:
    pair = payload["pair"]
    return pair[0] + pair[1] + offset if first else payload["unused"]


def _arguments() -> dict[str, object]:
    aliased = jnp.arange(8, dtype=jnp.int32)
    return {
        "payload": {"pair": (aliased, aliased), "unused": jnp.ones(16)},
        "offset": jnp.asarray(3, dtype=jnp.int32),
    }


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("keep_unused", [False, True])
def test_real_compiler_paths_follow_static_selection_and_keep_unused(
    *, first: bool, keep_unused: bool
) -> None:
    """A static branch changes actual operands while two live alias paths stay two."""
    arguments = _arguments()
    compiled = (
        jax.jit(_nested_operation, static_argnames=("first",), keep_unused=keep_unused)
        .lower(**arguments, first=first)
        .compile()
    )
    paths = compiler_input_paths(compiled=compiled, arguments=arguments)
    pair_paths = {
        (
            jax.tree_util.DictKey("payload"),
            jax.tree_util.DictKey("pair"),
            jax.tree_util.SequenceKey(index),
        )
        for index in (0, 1)
    }
    first_paths = pair_paths | {(jax.tree_util.DictKey("offset"),)}
    other_path = (jax.tree_util.DictKey("payload"), jax.tree_util.DictKey("unused"))
    expected = (
        first_paths | {other_path}
        if keep_unused
        else (first_paths if first else {other_path})
    )
    assert isinstance(paths, frozenset)
    assert paths == expected
    assert all(isinstance(path[0], jax.tree_util.DictKey) for path in paths)


@pytest.mark.parametrize("mismatch", ["missing", "extra_static", "list_for_tuple"])
def test_mismatched_dynamic_tree_is_refused(*, mismatch: str) -> None:
    """A shifted locator cannot silently charge a different input tree."""
    arguments = _arguments()
    compiled = (
        jax.jit(_nested_operation, static_argnames=("first",))
        .lower(**arguments, first=True)
        .compile()
    )
    altered = dict(arguments)
    if mismatch == "missing":
        del altered["offset"]
    elif mismatch == "extra_static":
        altered["first"] = True
    else:
        payload = altered["payload"]
        assert isinstance(payload, dict)
        altered["payload"] = {**payload, "pair": list(payload["pair"])}
    with pytest.raises(ExecutionPlanningError):
        compiler_input_paths(compiled=compiled, arguments=altered)


def _shape_only(*, reference: jax.Array) -> jax.Array:
    return jnp.full_like(reference, 7)


def test_shape_only_signature_has_no_actual_input_path() -> None:
    """The original budget bug's dead reference is absent despite its public input."""
    arguments = {"reference": jnp.arange(32, dtype=jnp.int32)}
    compiled = jax.jit(_shape_only).lower(**arguments).compile()
    analysis = compiled.memory_analysis()
    assert analysis is not None
    assert analysis.argument_size_in_bytes == 0
    assert compiler_input_paths(compiled=compiled, arguments=arguments) == frozenset()


def test_unavailable_compiler_input_metadata_is_refused(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable public compiler report grants no argument exclusions."""
    arguments = {"reference": jnp.arange(4, dtype=jnp.int32)}
    compiled = jax.jit(_shape_only).lower(**arguments).compile()
    original = RuntimeError("compiler metadata unavailable")

    def unavailable(_compiled: jax.stages.Compiled) -> object:
        raise original

    monkeypatch.setattr(jax.stages.Compiled, "input_shardings", property(unavailable))
    with pytest.raises(ExecutionPlanningError, match="unavailable") as error:
        compiler_input_paths(compiled=compiled, arguments=arguments)
    assert error.value.__cause__ is original


def test_matching_tree_with_invalid_sharding_metadata_is_refused(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching tree alone cannot authorize an unrecognized leaf's exclusion."""
    arguments = {"reference": jnp.arange(4, dtype=jnp.int32)}
    compiled = jax.jit(_shape_only).lower(**arguments).compile()
    malformed = ((), {"reference": object()})
    monkeypatch.setattr(
        jax.stages.Compiled, "input_shardings", property(lambda _compiled: malformed)
    )
    with pytest.raises(ExecutionPlanningError, match="unsupported residency metadata"):
        compiler_input_paths(compiled=compiled, arguments=arguments)
