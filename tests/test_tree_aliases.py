from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.beartype_conf import INTERNAL_CONF
from _lcm.typing import ArgumentTree, FootprintTree

_ABSTRACT = jax.ShapeDtypeStruct((2,), jnp.float32)


@beartype(conf=INTERNAL_CONF)
def _take_arguments(*, tree: ArgumentTree) -> ArgumentTree:
    return tree


@beartype(conf=INTERNAL_CONF)
def _take_footprint(*, tree: FootprintTree) -> FootprintTree:
    return tree


@pytest.mark.parametrize(
    "tree",
    [
        {"wealth": jnp.ones(2)},
        {"wealth": _ABSTRACT},
        ({"wealth": _ABSTRACT},),
        [{"wealth": jnp.ones(2)}],
        {"grids": {"wealth": None}},
        {"key": jax.random.key(0)},
    ],
    ids=["concrete", "abstract", "abstract-in-tuple", "list", "none-leaf", "prng-key"],
)
def test_argument_tree_admits_concrete_and_abstract_name_keyed_trees(
    *, tree: ArgumentTree
) -> None:
    """Name-keyed trees of arrays, abstract leaves and keys pass the check."""
    assert _take_arguments(tree=tree) is tree


@pytest.mark.parametrize(
    "tree",
    ["wealth", {"wealth": "high"}, {0: jnp.ones(2)}, object()],
    ids=["string", "name-over-string", "period-keyed", "object"],
)
def test_argument_tree_rejects_strings_period_keys_and_objects(
    *,
    tree: str | Mapping[str, str] | Mapping[int, jax.Array] | object,  # noqa: PAN001 - includes a literal object to test rejection
) -> None:
    """A string leaf, a period-keyed level or an arbitrary object fails the check."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _take_arguments(tree=tree)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "tree",
    [
        {("working", "retired"): jnp.ones(2)},
        {0: {"wealth": jnp.ones(2)}},
        ({"wealth": _ABSTRACT}, [jnp.ones(2)]),
        ({"wealth": None},),
        {"key": jax.random.key(0)},
        {"drift": 1 + 2j},
    ],
    ids=[
        "edge-keyed",
        "period-keyed",
        "abstract-and-list",
        "none-leaf",
        "prng-key",
        "complex-leaf",
    ],
)
def test_footprint_tree_admits_any_hashable_keys_and_abstract_leaves(
    *, tree: FootprintTree
) -> None:
    """Trees keyed by names, periods or edges, with abstract leaves, pass the check."""
    assert _take_footprint(tree=tree) is tree


@pytest.mark.parametrize(
    "tree",
    ["wealth", {"wealth": "high"}, object()],
    ids=["string", "name-over-string", "object"],
)
def test_footprint_tree_rejects_strings_and_objects(
    *,
    tree: str | Mapping[str, str] | object,  # noqa: PAN001 - includes a literal object to test rejection
) -> None:
    """A string leaf or an arbitrary object fails the check."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _take_footprint(tree=tree)  # ty: ignore[invalid-argument-type]
