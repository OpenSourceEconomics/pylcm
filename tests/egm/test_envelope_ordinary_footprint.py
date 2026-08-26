"""The ordinary envelope reads its candidates without materialising them.

Both arithmetics answer the same question over the same candidate set, and the
ordinary one is the cheaper read per candidate. That saving is only real if the
reduction streams: a comparison that first builds one
`(n_query, n_candidate)` array per channel spends memory the certified path —
whose reduction happens inside its own kernel — never asks for, and the excess
grows with a candidate count the solver does not control.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.upper_envelope.query import envelope_at_query

_N_QUERY = 256


def _candidates(*, n_candidates: int) -> dict[str, jnp.ndarray]:
    """A single rising branch of `n_candidates` links, plus the query grid."""
    endog = jnp.asarray(np.linspace(0.5, 40.0, n_candidates), dtype=jnp.float32)
    value = jnp.sqrt(endog)
    return {
        "endog_grid": endog,
        "policy": 0.5 * endog,
        "value": value,
        "marginal": 0.5 / jnp.sqrt(endog),
        "segment_id": jnp.zeros_like(endog),
        "x_query": jnp.asarray(np.linspace(0.5, 40.0, _N_QUERY), dtype=jnp.float32),
    }


def _temp_bytes(*, n_candidates: int) -> int:
    """Scratch memory the compiled ordinary envelope reserves for one call."""
    args = _candidates(n_candidates=n_candidates)
    compiled = (
        jax.jit(envelope_at_query, static_argnames=("arithmetic",))
        .lower(arithmetic="ordinary", **args)
        .compile()
    )
    analysis = compiled.memory_analysis()
    assert analysis is not None, "the backend published no memory analysis"
    return analysis.temp_size_in_bytes


def test_the_ordinary_envelope_holds_its_scratch_memory_as_candidates_grow():
    """Eight times the candidates must not cost eight times the scratch memory.

    A streamed reduction reserves scratch for one block of candidates and the
    running winner, so its footprint is governed by the query count. A reduction
    that materialises every candidate against every query reserves the product,
    and doubling either axis doubles the reservation.
    """
    small = _temp_bytes(n_candidates=512)
    large = _temp_bytes(n_candidates=4096)

    assert large < 2 * small, f"{small=} {large=} ratio={large / small:.1f}"
