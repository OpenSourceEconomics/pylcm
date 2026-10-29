"""A refined MSS row can omit an interior owner.

Simulation reads the published row policy-first under every envelope backend,
so this is a known limit of the read rather than a gate: where MSS emits no
record for a branch that wins strictly inside one candidate interval, the read
returns a neighbouring branch's policy (feasible, possibly suboptimal). The
witness below pins the omission so a backend change that closes it is noticed.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.mss import refine_envelope


def test_mss_omits_an_interior_owner_that_the_read_would_need():
    """MSS can publish a row whose interior owner was never enumerated.

    Three branches meet so that the middle one, `consumption == 3`, strictly owns
    `R = 0.8` without owning either bracketing candidate abscissa. The refinement
    emits no record for it, so a read at `R = 0.8` returns some other branch's
    policy.
    """
    refined_grid, refined_policy, _refined_value, _n_kept = refine_envelope(
        endog_grid=jnp.asarray([0.0, 1.0, 0.0, 1.0, 0.6, 1.2]),
        policy=jnp.asarray([10.0, 10.0, 3.0, 3.0, 1.0, 1.0]),
        value=jnp.asarray([1.0, 1.1, 0.84, 0.84 + 1 / 3, 0.8, 1.4]),
        n_refined=16,
    )
    got = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(0.8), xp=refined_grid, fp=refined_policy
        )
    )
    assert not np.isclose(got, 3.0), (
        "MSS now enumerates this owner; the read's documented limit is gone"
    )
