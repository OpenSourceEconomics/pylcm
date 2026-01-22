from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Any, TypeVar

import jax.numpy as jnp
from dags.tree import QNAME_DELIMITER, flatten_to_qnames, unflatten_from_qnames
from jax import Array

from lcm.typing import RegimeName

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


def flatten_regime_namespace(d: Mapping[RegimeName, Any]) -> dict[str, Any]:
    return flatten_to_qnames(d)


def unflatten_regime_namespace(d: dict[str, Any]) -> dict[RegimeName, Any]:
    return unflatten_from_qnames(d)  # ty: ignore[invalid-return-type]


def normalize_regime_transition_probs(
    probs: MappingProxyType[str, Array],
    active_regimes_next_period: tuple[str, ...],
) -> MappingProxyType[str, Array]:
    """Normalize regime transition probabilities over active regimes only.

    Args:
        probs: Mapping of regime names to probability arrays.
        active_regimes_next_period: Tuple of regime names that are active in the
            next period.

    Returns:
        Normalized probabilities mapping with same structure as input. Inactive regimes
        have probability 0, active regimes sum to 1.

    """
    # Get probabilities for active regimes only
    active_probs = {
        name: prob for name, prob in probs.items() if name in active_regimes_next_period
    }

    if not active_probs:
        return MappingProxyType(dict(probs))

    # Stack active probabilities and compute total
    stacked = jnp.stack(list(active_probs.values()), axis=0)
    total = jnp.sum(stacked, axis=0, keepdims=True).squeeze(0)

    # Normalize active regimes
    result = {}
    for name, prob in probs.items():
        if name in active_regimes_next_period:
            result[name] = prob / total
        else:
            result[name] = jnp.zeros_like(prob)

    return MappingProxyType(result)
