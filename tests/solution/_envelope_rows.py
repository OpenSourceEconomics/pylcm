"""Reading a refined envelope row the way production reads it.

Every upper-envelope backend publishes NaN-padded rows and is judged on the
function those rows *represent* under interpolation, not on which raw candidates
survive. The two helpers here fix that reading convention in one place, so a
change to the padding convention or the reference interpolation cannot leave one
backend's parity test comparing against a stale one.
"""

import jax.numpy as jnp
import numpy as np


def drop_nan(arr: jnp.ndarray) -> np.ndarray:
    """Return the kept prefix of a NaN-padded refined row."""
    out = np.asarray(arr)
    return out[~np.isnan(out)]


def envelope_interp(
    grid: jnp.ndarray, value: jnp.ndarray, x_query: float | np.ndarray
) -> np.ndarray:
    """Interpolate a refined row's values at `x_query`, ignoring the NaN tail."""
    keep = ~np.isnan(np.asarray(grid))
    return np.interp(x_query, np.asarray(grid)[keep], np.asarray(value)[keep])
