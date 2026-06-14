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
    scaled = values / scale
    smoothed = scale * logsumexp(scaled, axis=axes)
    # A fully infeasible slice has no valid choice: zero probability everywhere.
    # `logsumexp` returns `-inf` for it without NaN, but `jax.nn.softmax` would
    # divide `-inf` by `-inf` and return NaN, so guard the probabilities.
    all_masked = jnp.all(jnp.isneginf(values), axis=axes, keepdims=True)
    probs = jnp.where(all_masked, 0.0, jax.nn.softmax(scaled, axis=axes))
    return smoothed, probs
