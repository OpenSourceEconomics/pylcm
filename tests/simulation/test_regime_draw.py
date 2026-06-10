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
