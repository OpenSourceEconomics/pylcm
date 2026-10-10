"""Flatten and unflatten the qualified names a params pytree is keyed by.

`flatten_regime_namespace` joins a nested regime namespace into single
qualified keys and `unflatten_regime_namespace` inverts it, so one flat mapping
can address every regime's parameters.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import overload

from dags.tree import flatten_to_qnames, unflatten_from_qnames

from _lcm.typing import ParamsTemplate, QualifiedName, RegimeName
from lcm.typing import UserParams, UserParamsLeaf


class ParamsQnameDepth:
    """Depth (number of params-tree levels) of each flat-qname pattern.

    A flat param qname is a `__`-joined tree path. The params machinery
    classifies a qname by how many levels it carries; each attribute names one
    pattern. Two patterns share depth 3 — they are different patterns at the
    same depth, so they are kept as distinct names.
    """

    REGIME__FUNC__PARAM = 3
    REGIME__TARGETREGIME__FUNC__PARAM = 4
    TARGETREGIME__FUNC__PARAM = 3  # within-regime (regime prefix stripped)


@overload
def flatten_regime_namespace(
    d: ParamsTemplate,
) -> MappingProxyType[QualifiedName, str]: ...
@overload
def flatten_regime_namespace(
    d: UserParams,
) -> MappingProxyType[QualifiedName, UserParamsLeaf]: ...
@overload
def flatten_regime_namespace[Leaf](
    d: Mapping[RegimeName, Mapping[str, Leaf]],
) -> MappingProxyType[QualifiedName, Leaf]: ...
def flatten_regime_namespace[Leaf](
    d: ParamsTemplate | UserParams | Mapping[RegimeName, Mapping[str, Leaf]],
) -> (
    MappingProxyType[QualifiedName, str]
    | MappingProxyType[QualifiedName, UserParamsLeaf]
    | MappingProxyType[QualifiedName, Leaf]
):
    """Flatten a nested regime-keyed mapping to qualified names.

    A params template flattens to its type strings and a params tree to its
    leaves, at any depth. Any other namespace is two levels deep: regime names
    over names of grids or functions.

    Args:
        d: Mapping of regime names to nested values.

    Returns:
        Immutable mapping with keys like `"regime__variable"`.

    """
    return MappingProxyType(flatten_to_qnames(d))


def unflatten_regime_namespace[Leaf](
    d: dict[QualifiedName, Leaf],
) -> dict[RegimeName, dict[str, Leaf]]:
    """Unflatten two-part qualified names back to a regime-keyed dict.

    Args:
        d: Flat mapping with keys like `"regime__variable"`, each a regime name
            and one name below it.

    Returns:
        Dict keyed by regime name, each value keyed by the name below it.

    """
    return unflatten_from_qnames(d)  # ty: ignore[invalid-return-type]
