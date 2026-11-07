"""Diagnostic reductions hand back immutable mappings that survive JAX transforms."""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest

from _lcm.regime_building.diagnostics import _wrap_with_reduction
from lcm.typing import BoolND, FloatND, RegimeName


def _intermediates() -> tuple[
    FloatND, BoolND, FloatND, FloatND, MappingProxyType[RegimeName, FloatND]
]:
    values = jnp.array([[1.0, jnp.nan], [2.0, 3.0]])
    feasible = jnp.array([[True, True], [False, True]])
    return (
        values,
        feasible,
        values,
        values,
        MappingProxyType({"b": jnp.full((2, 2), 0.5)}),
    )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_diagnostic_reductions_are_immutable_at_both_levels(*, jit: bool) -> None:
    """The reductions and their per-target probabilities are immutable mappings."""
    reduce = _wrap_with_reduction(func=_intermediates, variable_names=("x", "a"))
    out = (jax.jit(reduce) if jit else reduce)()
    assert (type(out), type(out["regime_probs"])) == (
        MappingProxyType,
        MappingProxyType,
    )


def test_diagnostic_reductions_round_trip_through_jax_tree_utilities() -> None:
    """Flattening and unflattening the reductions restores the same mappings."""
    reduce = _wrap_with_reduction(func=_intermediates, variable_names=("x", "a"))
    out = reduce()
    leaves, treedef = jax.tree_util.tree_flatten(out)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert (type(rebuilt), dict(rebuilt["regime_probs"]).keys()) == (
        MappingProxyType,
        {"b"},
    )
