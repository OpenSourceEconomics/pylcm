"""Array operations for exact incoming-edge provenance and once-only routing."""

from collections.abc import Mapping

import jax.numpy as jnp

from lcm.typing import BoolND, FloatND, IntND


def pending_arrival_rows(
    *, regime_ids: IntND, entered_through: IntND, source_id: IntND, target_id: IntND
) -> BoolND:
    """Select only rows carrying this scheduled incoming edge's provenance."""
    return (regime_ids == target_id) & (entered_through == source_id)


def record_arrival_entries(
    *,
    drawn_ids: IntND,
    routed_ids: IntND,
    holders: BoolND,
    entries: IntND,
    source_id: int,
    target_ids: tuple[int, ...],
) -> IntND:
    """Remember actual scheduled arrivals, excluding other edges' fallbacks."""
    selected = jnp.zeros_like(holders)
    for target in target_ids:
        selected |= (drawn_ids == target) & (routed_ids == target)
    return jnp.where(holders & selected, source_id, entries)


def arrival_route_codes(
    *,
    original: IntND,
    incoming: IntND,
    pending: BoolND,
    source_roles: IntND,
    codes: tuple[tuple[int | None, int, int], ...],
) -> IntND:
    """Encode each original role/Boolean branch without merging value tuples."""
    result = original
    for role, branch, code in codes:
        selected = pending & (incoming == branch)
        if role is not None:
            selected &= source_roles == role
        result = jnp.where(selected, code, result)
    return result


def ordinary_exit_rows(*, holders: BoolND, pending: tuple[BoolND, ...]) -> BoolND:
    """Only pending incoming rows bypass the actor's ordinary Full policy."""
    for rows in pending:
        holders &= ~rows
    return holders


def arrival_frame_bases(
    *,
    bases: Mapping[str, FloatND],
    produced: Mapping[str, BoolND],
    incoming_bases: Mapping[str, Mapping[str, FloatND]],
    incoming_available: Mapping[str, Mapping[str, BoolND]],
    pending: Mapping[str, BoolND],
    physical_names: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[dict[str, FloatND], dict[str, BoolND]]:
    """Reveal only the selected producing frame's bases to the final actor."""
    merged, available = dict(bases), dict(produced)
    physical = dict(physical_names)
    for edge, columns in incoming_bases.items():
        selected = pending[edge]
        for name, value in columns.items():
            merged[name] = jnp.where(
                selected, value, merged.get(name, jnp.zeros_like(value))
            )
        for name in dict.fromkeys((*available, *columns)):
            held = (
                incoming_available[edge].get(name, jnp.ones_like(selected))
                if name in physical[edge]
                else jnp.zeros_like(selected)
            )
            available[name] = jnp.where(
                selected, held, available.get(name, jnp.zeros_like(selected))
            )
    return merged, available
