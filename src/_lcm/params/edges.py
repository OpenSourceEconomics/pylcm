"""The `edges` parameter namespace.

Parameters of the callables `Model(edges=...)` declares — a source's
regime-transition law, its gates, gate references and route fallbacks — live at
their declaration path under `params["edges"][source]`. The engine keeps them
there too: `flat_params["edges"][source]` is a flat mapping whose keys are the
rest of that path joined by `QNAME_DELIMITER`, e.g.

- `n_periods` for a law over all targets;
- `dead__rate` for a per-target cell, gated or not;
- `dead__predicate__min_wealth` for the gate on `dead`;
- `dead__references__spouse__wealth__scale` for a gate reference;
- `dead__routes__husband__fallback__solve__wealth__scale` for one phase of a
  route fallback.

A source's own kernels evaluate its law and its gated continuations, so they
bind from the union of the regime's own parameters and its edges' parameters
(`regime_kernel_params`), which refuses a key the two share.
"""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import cast

from dags.tree import tree_path_from_qname

from _lcm.typing import FlatEdgeParams, FlatRegimeParams, RegimeName
from lcm.exceptions import InvalidNameError

# Root of the edge namespace, in the user's params and in the engine's.
EDGES = "edges"

# Entries of a target's gate: its predicate, references and routes; and of one
# route's fallback below `ROUTES`.
PREDICATE = "predicate"
REFERENCES = "references"
ROUTES = "routes"
FALLBACK = "fallback"

# A gate, gate reference or route slot sits at least at `<target>__<entry>__<param>`.
_MIN_GATED_SLOT_DEPTH = 3

_EMPTY: FlatRegimeParams = MappingProxyType({})


# keyword-only-exempt: primary-argument=flat_params
def edge_params(
    # `object` leaves: the claw would otherwise check every leaf on every call.
    flat_params: Mapping[str, object],
    *,
    source: RegimeName,
) -> FlatRegimeParams:
    """Return the parameters of the callables `source`'s edges declare.

    Args:
        flat_params: The engine's params, with or without an edge level.
        source: The source regime.

    Returns:
        The flat mapping keyed by the declaration path below
        `params["edges"][source]`; empty for a source owning no slot.

    """
    edges = cast("FlatEdgeParams", flat_params.get(EDGES, _EMPTY))
    return edges.get(source, _EMPTY)


# keyword-only-exempt: primary-argument=flat_params
def regime_kernel_params(
    # `object` leaves: the claw would otherwise check every leaf on every call.
    flat_params: Mapping[str, object],
    *,
    regime_name: RegimeName,
) -> FlatRegimeParams:
    """Return everything a regime's own kernels bind by name.

    A source's kernels evaluate its regime-transition law and gate its gated
    continuations, so they read the regime's own parameters and its edges'
    parameters, by key.

    Args:
        flat_params: The engine's params.
        regime_name: The regime whose kernels are being called.

    Returns:
        The regime's own flat params, extended by its edges' flat params.

    Raises:
        InvalidNameError: If a key names both one of the regime's own parameters
            and one of its edges' parameters.

    """
    own = cast("FlatRegimeParams", flat_params.get(regime_name, _EMPTY))
    edges = edge_params(flat_params, source=regime_name)
    if not edges:
        return own
    _fail_if_own_and_edge_keys_collide(own=own, edges=edges, regime_name=regime_name)
    return MappingProxyType({**own, **edges})


def _fail_if_own_and_edge_keys_collide(
    *, own: FlatRegimeParams, edges: FlatRegimeParams, regime_name: RegimeName
) -> None:
    """Refuse a key a regime's own parameters and its edges' parameters share."""
    shared = sorted(set(own) & set(edges))
    if shared:
        key = shared[0]
        regime_path = user_path(path=(regime_name, *tree_path_from_qname(key)))
        edge_path = edge_user_path(source=regime_name, key=key)
        raise InvalidNameError(
            f"The parameter key {key!r} names both a parameter of regime "
            f"'{regime_name}' ({regime_path}) and one of its edges ({edge_path}). "
            "Rename the function, target or argument that spells it."
        )


def is_gated_cell_slot(key: str) -> bool:
    """Return whether an edge slot belongs to a gate, gate reference or route.

    Those slots are bound by the edge fold and the simulate router on the
    target's grid; every other slot belongs to the source's regime-transition
    law.

    Args:
        key: The slot's key in `flat_params["edges"][source]`.

    Returns:
        Whether the slot's second path segment is `predicate`, `references` or
        `routes`, below a target.

    """
    path = tree_path_from_qname(key)
    return len(path) >= _MIN_GATED_SLOT_DEPTH and path[1] in (
        PREDICATE,
        REFERENCES,
        ROUTES,
    )


def flat_namespaces(
    # `object` leaves: the claw would otherwise check every leaf on every call.
    flat_params: Mapping[str, object],
) -> Iterator[tuple[tuple[str, ...], FlatRegimeParams]]:
    """Yield every flat namespace of the engine's params with its path.

    Args:
        flat_params: The engine's params.

    Yields:
        Pairs of `(regime,)` and the regime's own flat params, and of
        `("edges", source)` and that source's edge slots.

    """
    for name, leaves in flat_params.items():
        if name == EDGES:
            for source, source_leaves in cast("FlatEdgeParams", leaves).items():
                yield (EDGES, source), source_leaves
        else:
            yield (name,), cast("FlatRegimeParams", leaves)


def edge_user_path(*, source: RegimeName, key: str) -> str:
    """Spell an edge slot the way a user writes it.

    Args:
        source: The source regime.
        key: The slot's key in `flat_params["edges"][source]`.

    Returns:
        The subscription chain, e.g. `params['edges']['work']['dead']['rate']`.

    """
    return user_path(path=(EDGES, source, *tree_path_from_qname(key)))


def user_path(*, path: tuple[str, ...]) -> str:
    """Spell a params path as the user's nested subscription chain.

    Args:
        path: The path from the params root to a leaf.

    Returns:
        The subscription chain, e.g. `params['edges']['work']['rate']`.

    """
    return "params" + "".join(f"[{segment!r}]" for segment in path)
