from collections import Counter
from collections.abc import Iterable
from itertools import chain
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


def flatten_regime_namespace(d: dict[RegimeName, Any]) -> dict[str, Any]:
    return flatten_to_qnames(d)


def unflatten_regime_namespace(d: dict[str, Any]) -> dict[RegimeName, Any]:
    return unflatten_from_qnames(d)  # ty: ignore[invalid-return-type]


def normalize_regime_transition_probs(
    probs: Array,
    active_regime_ids: Array,
) -> Array:
    """Normalize regime transition probabilities over active regimes only.

    Args:
        probs: Array of transition probabilities indexed by regime ID.
            Shape [n_regimes] (solve) or [n_regimes, n_subjects] (simulate).
        active_regime_ids: 1D array of regime IDs that are active in the next period.

    Returns:
        Normalized probabilities array with same shape as input. Inactive regimes
        have probability 0, active regimes sum to 1.

    """
    # Create mask for active regimes
    active_mask = jnp.isin(jnp.arange(probs.shape[0]), active_regime_ids)

    # Expand mask dimensions to match probs shape
    if probs.ndim > 1:
        active_mask = active_mask[:, None]

    # Zero out inactive regimes and normalize
    masked_probs = jnp.where(active_mask, probs, 0.0)
    total = jnp.sum(masked_probs, axis=0, keepdims=True)
    return masked_probs / total


def normalize_regime_transition_probs_dict(
    probs: dict[str, Array],
    active_regimes: list[RegimeName],
) -> dict[str, Array]:
    """Normalize regime transition probabilities over active regimes (dict version).

    This is the dict-based version for simulation, where regime transition
    probabilities are returned as a dict mapping regime names to probability arrays.

    Args:
        probs: Dict mapping regime names to probability arrays.
        active_regimes: List of regime names that are active in the next period.

    Returns:
        Normalized probabilities dict with same structure as input. Inactive regimes
        have probability 0, active regimes sum to 1.

    """
    active_set = set(active_regimes)

    # Get probabilities for active regimes only
    active_probs = {name: prob for name, prob in probs.items() if name in active_set}

    if not active_probs:
        return probs

    # Stack active probabilities and compute total
    stacked = jnp.stack(list(active_probs.values()), axis=0)
    total = jnp.sum(stacked, axis=0, keepdims=True)

    # Normalize active regimes
    result = {}
    for name, prob in probs.items():
        if name in active_set:
            result[name] = prob / total.squeeze(0)
        else:
            result[name] = jnp.zeros_like(prob)

    return result
