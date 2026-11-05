"""Associated original NB candidates in their pre-inversion resource coordinates."""

from collections.abc import Iterable
from typing import NamedTuple, cast

import jax

from _lcm.typing import PytreeChild
from lcm.typing import Float1D, ScalarBool, ScalarFloat


class NBEGMResourcesBank(NamedTuple):
    """Retain original Euler and endpoint candidates under one resource domain."""

    endog_grid: Float1D
    """Original pre-inversion resource coordinates, with NaN-dead candidates."""
    value: Float1D
    """Economic value associated with each original candidate."""
    policy: Float1D
    """Consumption associated with the same stored candidate."""
    marginal: Float1D
    """Economic marginal with respect to resources, separate from value."""
    savings: Float1D
    """Original saving node associated with each candidate and endpoint corner."""
    segment_id: Float1D
    """Original fold/endpoint identities consumed by the native tie comparator."""
    resources_support: Float1D
    """Sorted common corner support over every live original state/joint row."""
    borrowing_limit: ScalarFloat
    """Original literal first saving node."""
    upper_savings: ScalarFloat
    """Original final saving node bounding retained endpoint candidates."""
    construction_invalid: ScalarBool
    """Mandatory original-input, source-domain and future-integrity failure."""
    unsupported_economic_minus_inf: ScalarBool
    """A feasible original candidate needs an unrepresented economic value kind."""


def _flatten_nbegm_resources(
    bank: NBEGMResourcesBank,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(bank), None


def _flatten_nbegm_resources_with_keys(
    bank: NBEGMResourcesBank,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(NBEGMResourcesBank._fields, bank, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_nbegm_resources(
    metadata: None, children: Iterable[PytreeChild]
) -> NBEGMResourcesBank:
    """Restore the original record for JAX's abstract, tangent and role leaves."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(NBEGMResourcesBank._fields):
        raise TypeError("NB resources reconstruction requires its exact fields.")
    return cast("NBEGMResourcesBank", tuple.__new__(NBEGMResourcesBank, leaves))


jax.tree_util.register_pytree_with_keys(
    NBEGMResourcesBank,
    _flatten_nbegm_resources_with_keys,
    _unflatten_nbegm_resources,
    _flatten_nbegm_resources,
)
