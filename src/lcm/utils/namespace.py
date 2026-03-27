from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from dags.tree import flatten_to_qnames, unflatten_from_qnames

from lcm.typing import RegimeName


def flatten_regime_namespace(d: Mapping[RegimeName, Any]) -> MappingProxyType[str, Any]:
    return MappingProxyType(flatten_to_qnames(d))


def unflatten_regime_namespace(d: dict[str, Any]) -> dict[RegimeName, Any]:
    return unflatten_from_qnames(d)  # ty: ignore[invalid-return-type]
