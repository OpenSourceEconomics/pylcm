from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Any, TypeVar, overload

import jax.numpy as jnp
from dags.tree import QNAME_DELIMITER, flatten_to_qnames, unflatten_from_qnames

from lcm.typing import Float1D, RegimeName

# Re-export for use in other modules. This is the separator used by dags to
# concatenate nested dictionary keys into qualified names (e.g., "work__next_wealth").
# User-defined regime names and function names must NOT contain this separator.
REGIME_SEPARATOR = QNAME_DELIMITER

T = TypeVar("T")


def find_duplicates(*containers: Iterable[T]) -> set[T]:
    combined = chain.from_iterable(containers)
    counts = Counter(combined)
    return {v for v, count in counts.items() if count > 1}


def first_non_none(*args: T | None) -> T:
    """Return the first non-None argument.

    Args:
        *args: Arguments to check.

    Returns:
        The first non-None argument.

    Raises:
        ValueError: If all arguments are None.

    """
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError("All arguments are None")


def flatten_regime_namespace(d: dict[RegimeName, Any]) -> dict[str, Any]:
    return flatten_to_qnames(d)


def unflatten_regime_namespace(d: dict[str, Any]) -> dict[RegimeName, Any]:
    return unflatten_from_qnames(d)  # ty: ignore[invalid-return-type]


@overload
def normalize_regime_transition_probs(
    probs: Mapping[str, float],
    active_regimes: list[str],
) -> MappingProxyType[str, float]: ...


@overload
def normalize_regime_transition_probs(
    probs: Mapping[str, Float1D],
    active_regimes: list[str],
) -> MappingProxyType[str, Float1D]: ...


def normalize_regime_transition_probs(
    probs: Mapping[str, float] | Mapping[str, Float1D],
    active_regimes: list[str],
) -> MappingProxyType[str, float] | MappingProxyType[str, Float1D]:
    """Normalize regime transition probabilities over active regimes only."""
    active_probs = jnp.array([probs[r] for r in active_regimes])
    total = jnp.sum(active_probs, axis=0)
    return MappingProxyType({r: probs[r] / total for r in active_regimes})
