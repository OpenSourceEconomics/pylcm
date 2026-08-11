"""The realized regime draw over per-subject probability distributions."""

from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.simulation.transitions import draw_key_from_dict

_REGIME_IDS = MappingProxyType({"working": jnp.int32(0), "dead": jnp.int32(1)})


def test_regime_draw_broadcasts_unbatched_distribution() -> None:
    """A transition reading no per-subject state or action (e.g. only `age`)
    yields one shared distribution; the draw broadcasts it across every
    subject's key instead of failing the per-subject vmap."""
    keys = jax.random.split(jax.random.key(0), 4)
    probs = MappingProxyType({"working": jnp.asarray(1.0), "dead": jnp.asarray(0.0)})
    ids = draw_key_from_dict(d=probs, regime_names_to_ids=_REGIME_IDS, keys=keys)
    assert ids.shape == (4,)
    assert bool((ids == 0).all())


def test_regime_draw_uses_per_subject_distributions() -> None:
    """Per-subject probability vectors give each subject its own draw."""
    keys = jax.random.split(jax.random.key(0), 2)
    probs = MappingProxyType(
        {"working": jnp.asarray([1.0, 0.0]), "dead": jnp.asarray([0.0, 1.0])}
    )
    ids = draw_key_from_dict(d=probs, regime_names_to_ids=_REGIME_IDS, keys=keys)
    assert ids.tolist() == [0, 1]


def test_regime_draw_is_invariant_to_dict_insertion_order() -> None:
    """The realized draw depends on regime id, not `d`'s insertion order.

    A caller may hand `d` in any order (e.g. a reachability graph's internal
    alphabetical candidate-listing convention, versus a model's declared
    regime-id order) — the draw for a fixed key must be identical either way,
    since `regime_names_to_ids` is the only thing that should determine which
    probability row pairs with which regime id.
    """
    regime_names_to_ids = MappingProxyType(
        {"working": jnp.int32(0), "retired": jnp.int32(1), "dead": jnp.int32(2)}
    )
    keys = jax.random.split(jax.random.key(0), 8)
    probs_by_name = {
        "working": jnp.full(8, 0.2),
        "retired": jnp.full(8, 0.3),
        "dead": jnp.full(8, 0.5),
    }
    declared_order = MappingProxyType(
        {name: probs_by_name[name] for name in ("working", "retired", "dead")}
    )
    alphabetical_order = MappingProxyType(
        {name: probs_by_name[name] for name in sorted(probs_by_name)}
    )
    got_declared = draw_key_from_dict(
        d=declared_order, regime_names_to_ids=regime_names_to_ids, keys=keys
    )
    got_alphabetical = draw_key_from_dict(
        d=alphabetical_order, regime_names_to_ids=regime_names_to_ids, keys=keys
    )
    assert got_declared.tolist() == got_alphabetical.tolist()
