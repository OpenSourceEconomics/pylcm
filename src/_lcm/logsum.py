"""Smoothed maximum (EV1 logsum) and choice probabilities over discrete axes.

Shared by the brute-force taste-shock aggregation and the DC-EGM choice
aggregation. The smoothed maximum of choice-specific values $v_d$ under IID
EV1 taste shocks with scale $\\lambda > 0$ is the logsum

```{math}
W = \\lambda \\log \\sum_d e^{v_d / \\lambda},
```

with conditional choice probabilities $P_d = \\text{softmax}(v_d / \\lambda)$.
`-inf` (infeasible) entries contribute zero weight and zero probability; a
slice that is `-inf` everywhere yields `W = -inf` and zero probabilities.

The logsum equals the expected maximum $\\mathbb{E}[\\max_d (v_d + \\lambda
\\varepsilon_d)]$ only for **mean-zero** EV1 shocks. A standard Gumbel(0, 1)
draw has mean $\\gamma$ (the Euler-Mascheroni constant), so the simulation
centers its draws by `EULER_GAMMA` to keep the simulated expected maximum
consistent with this solved value.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from lcm.typing import FloatND, ScalarFloat

# Euler-Mascheroni constant: the mean of a standard Gumbel(0, 1) draw.
EULER_GAMMA = 0.5772156649015329


def logsum_and_softmax(
    *,
    values: FloatND,
    scale: float | ScalarFloat,
    axes: tuple[int, ...],
) -> tuple[FloatND, FloatND]:
    """Compute the EV1 smoothed maximum and choice probabilities.

    Args:
        values: Choice-specific values; infeasible entries are `-inf`.
        scale: Taste-shock scale; a 0-d array or float, strictly positive.
        axes: Axes to aggregate over (the discrete-choice axes).

    Returns:
        Tuple of the smoothed maximum (shape of `values` with `axes` reduced)
        and the choice probabilities (shape of `values`).

    """
    # Subtract the per-slice max before dividing by the scale: `values / scale`
    # would overflow to `inf` for a tiny scale (e.g. float32 values near the
    # max), but `(values - max) / scale <= 0` cannot. A fully infeasible slice
    # has max `-inf`; shift it by `0` so the slice stays `-inf` instead of NaN.
    v_max = jnp.max(values, axis=axes, keepdims=True)
    finite_max = jnp.where(jnp.isneginf(v_max), 0.0, v_max)
    shifted = (values - finite_max) / scale
    smoothed = jnp.squeeze(finite_max, axis=axes) + scale * logsumexp(
        shifted, axis=axes
    )
    # `logsumexp` returns `-inf` for a fully infeasible slice without NaN, but
    # `jax.nn.softmax` would divide `-inf` by `-inf` there, so guard the probs.
    all_masked = jnp.all(jnp.isneginf(values), axis=axes, keepdims=True)
    probs = jnp.where(all_masked, 0.0, jax.nn.softmax(shifted, axis=axes))
    return smoothed, probs
