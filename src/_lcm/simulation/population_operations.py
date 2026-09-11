"""Pure population setup bodies shared by profiling and concrete execution."""

import jax
import jax.numpy as jnp

from _lcm.regime_building.collective import NO_ROLE


def default_roles(*, regime_ids: jax.Array) -> jax.Array:
    """Allocate the existing no-role sentinel at the population's shape."""
    return jnp.full_like(regime_ids, NO_ROLE, dtype=jnp.int32)


def regime_is_occupied(*, regime_ids: jax.Array, regime_id: jax.Array) -> jax.Array:
    """Return the existing occupied-regime diagnostic summary."""
    return jnp.any(regime_ids == regime_id)


def canonical_roles(
    *, declared: jax.Array, known_role_ids: tuple[int | jax.Array, ...]
) -> tuple[jax.Array, jax.Array]:
    """Cast role codes and validate the model-wide vocabulary in one body."""
    roles = jnp.asarray(declared, dtype=jnp.int32)
    known = jnp.asarray([NO_ROLE, *known_role_ids], dtype=jnp.int32)
    return roles, jnp.all(jnp.isin(roles, known))


def role_mismatch(
    *,
    roles: jax.Array,
    regime_ids: jax.Array,
    regime_id: jax.Array,
    role_ids: tuple[int | jax.Array, ...],
) -> jax.Array:
    """Check the original regime-specific role predicate without extra staging."""
    starts_here = regime_ids == regime_id
    declared_here = jnp.asarray(role_ids, dtype=jnp.int32)
    return jnp.any(starts_here & ~jnp.isin(roles, declared_here))


def starting_periods(
    *, initial_ages: jax.Array, age_values: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Preserve searchsorted and the existing isclose age-validity predicate."""
    age_values = jnp.asarray(age_values)
    periods = jnp.searchsorted(age_values, initial_ages)
    safe_idx = jnp.clip(periods, 0, len(age_values) - 1)
    in_bounds = periods < len(age_values)
    valid = in_bounds & jnp.isclose(age_values[safe_idx], initial_ages)
    return periods, valid, jnp.all(valid)
