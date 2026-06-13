"""Smoothed maximum (EV1 logsum) and choice probabilities over discrete axes.

Shared by the brute-force taste-shock aggregation and the DC-EGM choice
aggregation. The smoothed maximum of choice-specific values $v_d$ under IID
EV1 taste shocks with scale $\\lambda$ is the logsum

```{math}
W = v_{\\max} + \\lambda \\log \\sum_d e^{(v_d - v_{\\max}) / \\lambda},
```

with conditional choice probabilities $P_d = \\text{softmax}((v_d - v_{\\max}) /
\\lambda)$. Max-rescaling makes `-inf` (infeasible) entries safe: they
contribute zero weight and zero probability.

The logsum equals the expected maximum $\\mathbb{E}[\\max_d (v_d + \\lambda
\\varepsilon_d)]$ only for **mean-zero** EV1 shocks. A standard Gumbel(0, 1)
draw has mean $\\gamma$ (the Euler-Mascheroni constant), so the simulation
centers its draws by `EULER_GAMMA` to keep the simulated expected maximum
consistent with this solved value.
"""

import jax.numpy as jnp

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

    Behavior by case:

    - `scale > 0` ⇒ logsum and softmax over `axes`.
    - `scale = 0` ⇒ the hard maximum and a one-hot probability array (ties
      broken toward the first flat index).
    - A slice that is `-inf` everywhere ⇒ smoothed value `-inf`; probabilities
      are zero for `scale > 0` (one-hot at index 0 in the degenerate
      `scale = 0` case).

    Args:
        values: Choice-specific values; infeasible entries are `-inf`.
        scale: Taste-shock scale; a 0-d array or float, may be zero.
        axes: Axes to aggregate over (the discrete-choice axes).

    Returns:
        Tuple of the smoothed maximum (shape of `values` with `axes` reduced)
        and the choice probabilities (shape of `values`).

    """
    v_max = jnp.max(values, axis=axes, keepdims=True)
    # Keep the shift finite where the whole slice is -inf so that
    # `values - finite_max` stays -inf instead of becoming NaN.
    finite_max = jnp.where(jnp.isneginf(v_max), 0.0, v_max)
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    exp_shifted = jnp.exp((values - finite_max) / safe_scale)
    sum_exp = jnp.sum(exp_shifted, axis=axes, keepdims=True)

    smoothed = finite_max + scale * jnp.log(sum_exp)
    all_masked = jnp.isneginf(v_max)
    smoothed = jnp.where(all_masked, -jnp.inf, smoothed)
    result_values = jnp.where(scale > 0, smoothed, v_max)

    softmax = jnp.where(sum_exp > 0, exp_shifted / sum_exp, 0.0)
    one_hot = _one_hot_argmax(values=values, axes=axes)
    result_probs = jnp.where(scale > 0, softmax, one_hot)

    return jnp.squeeze(result_values, axis=axes), result_probs


def _one_hot_argmax(*, values: FloatND, axes: tuple[int, ...]) -> FloatND:
    """One-hot indicator of the (first) argmax over the given axes."""
    moved = jnp.moveaxis(values, axes, tuple(range(len(axes))))
    lead_shape = moved.shape[: len(axes)]
    flat = moved.reshape((-1, *moved.shape[len(axes) :]))
    indicator = (
        jnp.arange(flat.shape[0]).reshape((-1,) + (1,) * (flat.ndim - 1))
        == jnp.argmax(flat, axis=0)
    ).astype(values.dtype)
    return jnp.moveaxis(
        indicator.reshape(lead_shape + moved.shape[len(axes) :]),
        tuple(range(len(axes))),
        axes,
    )
