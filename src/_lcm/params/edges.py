"""The `edges` parameter namespace.

Parameters of the callables `Model(edges=...)` declares — a source's
regime-transition law, its gates, gate references and route fallbacks — live at
their declaration path under `params["edges"][source]`. The engine keeps them
there too: `flat_params["edges"][source]` is a flat mapping whose keys are the
rest of that path joined by `QNAME_DELIMITER`, e.g.

- `n_periods` for a law over all targets;
- `dead__probability__rate` for a gated cell's selection probability;
- `dead__gate__min_wealth` for its gate;
- `dead__gate_references__spouse__wealth__scale` for a gate reference;
- `dead__routes__husband__fallback__solve__wealth__scale` for one phase of a
  route fallback.

A source's own kernels evaluate its law and its gated continuations, so they
bind from the union of the regime's own parameters and its edges' parameters
(`regime_kernel_params`); the two key sets are disjoint because every regime key
starts with a function name and every edge key with a target or an argument.
"""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import cast

from dags.tree import tree_path_from_qname

from _lcm.typing import FlatEdgeParams, FlatRegimeParams, RegimeName

# Root of the edge namespace, in the user's params and in the engine's.
EDGES = "edges"

# Entries of a value-dependent cell: its selection probability, gate predicate,
# gate references and routes; and of one route's fallback below `ROUTES`.
PROBABILITY = "probability"
GATE = "gate"
GATE_REFERENCES = "gate_references"
ROUTES = "routes"
FALLBACK = "fallback"

# A gate, gate reference or route slot sits at least at `<target>__<entry>__<param>`.
_MIN_GATED_SLOT_DEPTH = 3

_EMPTY: FlatRegimeParams = MappingProxyType({})


# keyword-only-exempt: primary-argument=flat_params
def edge_params(
    flat_params: Mapping[str, object], *, source: RegimeName
) -> FlatRegimeParams:
    """Return the parameters of the callables `source`'s edges declare.

    Args:
        flat_params: The engine's params, with or without an edge level.
        source: The source regime.

    Returns:
        The flat mapping keyed by the declaration path below
        `params["edges"][source]`; empty for a source owning no slot.

    """
    edges = cast("FlatEdgeParams", flat_params.get(EDGES, MappingProxyType({})))
    return edges.get(source, _EMPTY)


# keyword-only-exempt: primary-argument=flat_params
def regime_kernel_params(
    flat_params: Mapping[str, object], *, regime_name: RegimeName
) -> FlatRegimeParams:
    """Return everything a regime's own kernels bind by name.

    A source's kernels evaluate its regime-transition law and gate its gated
    continuations, so they read the regime's own parameters and its edges'
    parameters. The two key sets are disjoint, so the union is unambiguous.

    Args:
        flat_params: The engine's params.
        regime_name: The regime whose kernels are being called.

    Returns:
        The regime's own flat params, extended by its edges' flat params.

    """
    own = cast("FlatRegimeParams", flat_params.get(regime_name, _EMPTY))
    edges = edge_params(flat_params, source=regime_name)
    if not edges:
        return own
    return MappingProxyType({**own, **edges})


def is_gated_cell_slot(key: str) -> bool:
    """Return whether an edge slot belongs to a gate, gate reference or route.

    Those slots are bound by the edge fold and the simulate router on the
    target's grid; every other slot belongs to the source's regime-transition
    law.

    Args:
        key: The slot's key in `flat_params["edges"][source]`.

    Returns:
        Whether the slot's second path segment is `gate`, `gate_references` or
        `routes`, below a target.

    """
    path = tree_path_from_qname(key)
    return len(path) >= _MIN_GATED_SLOT_DEPTH and path[1] in (
        GATE,
        GATE_REFERENCES,
        ROUTES,
    )


def flat_namespaces(
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
