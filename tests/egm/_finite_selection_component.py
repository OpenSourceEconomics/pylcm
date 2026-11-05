"""Ordinary-winning component view of a complete finite selection."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.finite_negm_query import (
    _FiniteOuterProfile,
    _select_finite_outer_profiles,
)
from lcm.typing import BoolND, FloatND


class _FiniteOuterRead(NamedTuple):
    """Numeric side payloads for component controls."""

    value: FloatND
    right_marginal: FloatND
    left_marginal: FloatND
    right_dead: BoolND
    left_dead: BoolND
    coverage_missing: BoolND


def component_view(*, profile: _FiniteOuterProfile) -> _FiniteOuterRead:
    """Read a single O whose ordinary point beats its absent or lower corner."""
    assert bool(jnp.all(profile.right.supported))
    assert bool(
        jnp.all(
            ~profile.corner.supported | (profile.right.value > profile.corner.value)
        )
    )
    selection = _select_finite_outer_profiles(
        profiles=jax.tree.map(lambda leaf: leaf[..., None], profile)
    )
    right, left = selection.right, selection.left
    return _FiniteOuterRead(
        value=selection.value[..., 0],
        right_marginal=jnp.where(
            right.use_corner, right.corner.marginal, right.ordinary.marginal
        )[..., 0],
        left_marginal=jnp.where(
            left.use_corner, left.corner.marginal, left.ordinary.marginal
        )[..., 0],
        right_dead=jnp.where(
            right.use_corner, ~right.corner.supported, right.ordinary.dead
        )[..., 0],
        left_dead=jnp.where(
            left.use_corner, ~left.corner.supported, left.ordinary.dead
        )[..., 0],
        coverage_missing=profile.coverage_missing,
    )
