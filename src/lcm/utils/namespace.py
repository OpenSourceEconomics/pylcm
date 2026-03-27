from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from dags.tree import flatten_to_qnames, unflatten_from_qnames

from lcm.typing import RegimeName


def flatten_regime_namespace(d: Mapping[RegimeName, Any]) -> MappingProxyType[str, Any]:
    """Flatten a nested regime-keyed mapping to qualified names.

    Args:
        d: Mapping of regime names to nested values.

    Returns:
        Immutable mapping with keys like `"regime__variable"`.

    """
    return MappingProxyType(flatten_to_qnames(d))


def unflatten_regime_namespace(d: dict[str, Any]) -> dict[RegimeName, Any]:
    """Unflatten qualified names back to a nested regime-keyed dict.

    Args:
        d: Flat mapping with keys like `"regime__variable"`.

    Returns:
        Nested dict keyed by regime name.

    """
    return unflatten_from_qnames(d)  # ty: ignore[invalid-return-type]
