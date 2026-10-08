"""Binding a nested solver's outer node touches only the source's own branch.

The shared argument helper adds the outer post-decision node to the source's own
parameters and leaves every `edges` branch, and the `edges` object itself,
unchanged; the strict resolver still refuses a name owned both by the regime and
by its edges, whatever the values. These cases call the helper and the resolver
directly; `test_nested_edge_namespace` covers the NEGM and NNBEGM solve paths.
Deliberately colliding pools are negative controls, not supported models.
"""

from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import pytest

from _lcm.params.edges import regime_kernel_params
from _lcm.solution.negm import _with_outer_post_decision
from _lcm.typing import FlatParams
from lcm.exceptions import InvalidNameError

_SOURCES = ("alive", "retired")
_OUTER_NODE = "new_illiquid"
_EDGE_LAYOUTS = (
    "missing-root",
    "empty-root",
    "other-source-only",
    "empty-selected-source",
    "nonempty-selected-source",
)
_EDGE_KEYS = (
    "final_age_alive",
    "dead__rate",
    "dead__predicate__min_wealth",
    "dead__references__spouse__wealth__scale",
    "dead__routes__husband__fallback__solve__wealth__scale",
)


def _other_source(source):
    return "retired" if source == "alive" else "alive"


def _flat_params(*, source, has_own, edge_layout):
    """Use the alive/retired/dead layout of `test_nested_edge_namespace`."""
    own = {"utility__rho": jnp.asarray(2.0)} if has_own else {}
    root = {
        source: MappingProxyType(own),
        _other_source(source): MappingProxyType({"utility__rho": jnp.asarray(3.0)}),
        "dead": MappingProxyType({}),
    }
    if edge_layout != "missing-root":
        edge_root = {}
        if edge_layout != "empty-root":
            # Same spelling as a selected source slot, deliberately different
            # value: binding one source must not borrow the other source's leaf.
            edge_root[_other_source(source)] = MappingProxyType(
                {"final_age_alive": jnp.asarray(30.0)}
            )
        if edge_layout in ("empty-selected-source", "nonempty-selected-source"):
            selected = (
                {key: jnp.asarray(25.0 + index) for index, key in enumerate(_EDGE_KEYS)}
                if edge_layout == "nonempty-selected-source"
                else {}
            )
            edge_root[source] = MappingProxyType(selected)
        root["edges"] = MappingProxyType(edge_root)
    return cast("FlatParams", MappingProxyType(root))


def _bind(*, flat_params, source, value):
    return _with_outer_post_decision(
        flat_params=flat_params,
        regime_name=source,
        outer_post_decision=_OUTER_NODE,
        value=value,
    )


def _assert_binding_contract(*, original, bound, source, value):
    """Independent ownership checks followed by the actual production union."""
    _assert_only_the_own_branch_changed(
        original=original, bound=bound, source=source, value=value
    )
    _assert_the_kernel_pool_unites_own_and_edges(
        original=original, bound=bound, source=source, value=value
    )


def _assert_only_the_own_branch_changed(*, original, bound, source, value):
    """The own branch gains the outer node; every other root keeps its object."""
    assert bound is not original
    assert isinstance(bound, MappingProxyType)
    assert set(bound) == set(original)
    assert bound[source] is not original[source]
    assert isinstance(bound[source], MappingProxyType)
    assert set(bound[source]) == set(original[source]) | {_OUTER_NODE}
    assert bound[source][_OUTER_NODE] is value
    for key, leaf in original[source].items():
        if key != _OUTER_NODE:
            assert bound[source][key] is leaf
    for root_name, branch in original.items():
        if root_name != source:
            assert bound[root_name] is branch
    if "edges" in original:
        assert bound["edges"] is original["edges"]
        for edge_source, branch in original["edges"].items():
            assert bound["edges"][edge_source] is branch


def _assert_the_kernel_pool_unites_own_and_edges(*, original, bound, source, value):
    """The strict resolver unites the own branch and the source's edge branch."""
    selected_edges = original.get("edges", {}).get(source, {})
    pool = regime_kernel_params(bound, regime_name=source)
    assert set(pool) == set(original[source]) | {_OUTER_NODE} | set(selected_edges)
    assert pool[_OUTER_NODE] is value
    for key, leaf in original[source].items():
        if key != _OUTER_NODE:
            assert pool[key] is leaf
    for key, leaf in selected_edges.items():
        assert pool[key] is leaf
    if not selected_edges:
        assert pool is bound[source]


@pytest.mark.parametrize("source", _SOURCES)
@pytest.mark.parametrize("has_own", [False, True], ids=["empty-own", "nonempty-own"])
@pytest.mark.parametrize("edge_layout", _EDGE_LAYOUTS)
@pytest.mark.parametrize("outer_value", [-0.0, 5.0], ids=["negative-zero", "positive"])
def test_outer_binding_namespace_matrix(*, source, has_own, edge_layout, outer_value):
    """40 cases: own pools, missing/empty/nonempty edges, sources and nodes."""
    original = _flat_params(source=source, has_own=has_own, edge_layout=edge_layout)
    original_own_items = tuple(original[source].items())
    value = jnp.asarray(outer_value)
    bound = _bind(flat_params=original, source=source, value=value)
    _assert_binding_contract(original=original, bound=bound, source=source, value=value)
    assert tuple(original[source]) == tuple(key for key, _ in original_own_items)
    for key, leaf in original_own_items:
        assert original[source][key] is leaf
    assert _OUTER_NODE not in original[source]


@pytest.mark.parametrize("first_source", _SOURCES)
@pytest.mark.parametrize("has_own", [False, True], ids=["empty-own", "nonempty-own"])
def test_binding_multiple_sources_then_rebinding_keeps_prior_roots(
    *, first_source, has_own
):
    """Four cases: independent source binding and replacement of one outer node."""
    original = _flat_params(
        source=first_source,
        has_own=has_own,
        edge_layout="nonempty-selected-source",
    )
    previous = original
    history = []
    for source, scalar in (
        (first_source, -0.0),
        (_other_source(first_source), 5.0),
        (first_source, 9.0),
    ):
        value = jnp.asarray(scalar)
        bound = _bind(flat_params=previous, source=source, value=value)
        _assert_binding_contract(
            original=previous, bound=bound, source=source, value=value
        )
        assert bound["edges"] is original["edges"]
        history.append((bound, source, value))
        previous = bound
    # Later updates must leave the earlier immutable snapshots intact.
    for snapshot, source, value in history:
        assert snapshot[source][_OUTER_NODE] is value
        regime_kernel_params(snapshot, regime_name=source)
    assert _OUTER_NODE not in original[first_source]
    assert _OUTER_NODE not in original[_other_source(first_source)]


@pytest.mark.parametrize("source", _SOURCES)
@pytest.mark.parametrize("key", ["utility__rho", _OUTER_NODE])
@pytest.mark.parametrize(
    "same_value", [False, True], ids=["different-value", "same-object"]
)
def test_actual_own_edge_collision_remains_refused(*, source, key, same_value):
    """Eight negative controls: equal values do not make duplicate ownership legal."""
    value = jnp.asarray(5.0)
    edge_value = value if same_value else jnp.asarray(6.0)
    own = {key: value} if key != _OUTER_NODE else {}
    flat_params = cast(
        "FlatParams",
        MappingProxyType(
            {
                source: MappingProxyType(own),
                _other_source(source): MappingProxyType({}),
                "dead": MappingProxyType({}),
                "edges": MappingProxyType(
                    {source: MappingProxyType({key: edge_value})}
                ),
            }
        ),
    )
    bound = _bind(flat_params=flat_params, source=source, value=value)
    assert bound["edges"] is flat_params["edges"]
    with pytest.raises(
        InvalidNameError, match="names both a parameter of regime"
    ) as exc:
        regime_kernel_params(bound, regime_name=source)
    own_path = (
        f"params['{source}']['utility']['rho']"
        if key == "utility__rho"
        else f"params['{source}']['new_illiquid']"
    )
    edge_path = (
        f"params['edges']['{source}']['utility']['rho']"
        if key == "utility__rho"
        else f"params['edges']['{source}']['new_illiquid']"
    )
    assert own_path in str(exc.value)
    assert edge_path in str(exc.value)


@pytest.mark.parametrize("source", _SOURCES)
@pytest.mark.parametrize("copied_key", _EDGE_KEYS)
def test_copyback_mutation_is_detected_by_the_strict_resolver(*, source, copied_key):
    """Ten mutations: copying any edge shape into own is refused as a collision.

    A positive control reaches the real resolver first. Each local mutant then
    changes only the selected own mapping; the declared edges stay intact.
    This deliberately models the prohibited copyback without editing or
    monkeypatching project source or replacing the collision validator.
    """
    original = _flat_params(
        source=source, has_own=True, edge_layout="nonempty-selected-source"
    )
    value = jnp.asarray(5.0)
    bound = _bind(flat_params=original, source=source, value=value)
    _assert_binding_contract(original=original, bound=bound, source=source, value=value)
    copied_leaf = bound["edges"][source][copied_key]
    mutant = cast(
        "FlatParams",
        MappingProxyType(
            {
                **bound,
                source: MappingProxyType({**bound[source], copied_key: copied_leaf}),
            }
        ),
    )
    assert mutant["edges"] is original["edges"]
    assert copied_key not in bound[source]
    with pytest.raises(
        InvalidNameError, match="names both a parameter of regime"
    ) as exc:
        regime_kernel_params(mutant, regime_name=source)
    assert repr(copied_key) in str(exc.value)
