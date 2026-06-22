"""Triangular-mesh barycentric geometry for the 2-D G2EGM upper envelope.

The G2EGM upper envelope maps each constraint segment's endogenous candidate cloud
(a regular source grid imaged into the current-state $(m, n)$ plane) onto a common
state grid by interpolating policies and recomputing the Bellman objective. The
geometry is **triangular**: each regular source cell is split into two triangles, and
a target state is located in a triangle by its barycentric weights. Triangles, unlike
mapped quadrilaterals, are convex simplices with a unique affine inverse — a mapped
quad can fold (a sign-changing bilinear Jacobian) and have no unique inverse, so the
reference method triangulates.

A target is an **admissible** candidate of a triangle when every barycentric weight
exceeds a (negative) extrapolation threshold: a mildly extrapolated triangle stays a
candidate (the within-segment envelope decides), rather than being dropped unless no
triangle strictly covers the target.
"""

import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, Float2D, FloatND, Int2D


def triangulate_regular_grid(*, n_rows: int, n_cols: int) -> Int2D:
    """Split each cell of an `n_rows` by `n_cols` regular grid into two triangles.

    Node $(i, j)$ has the row-major flat index `i * n_cols + j`. Each cell
    $(i, j)$ contributes the two triangles `(n00, n10, n01)` and `(n10, n11, n01)`
    — the diagonal split that keeps both halves non-degenerate even when the cell's
    image folds.

    Args:
        n_rows: Number of source-grid rows.
        n_cols: Number of source-grid columns.

    Returns:
        Integer array of shape `(2 * (n_rows - 1) * (n_cols - 1), 3)`; each row is the
        three flat node indices of one triangle.

    """
    rows = jnp.arange(n_rows - 1)[:, None]
    cols = jnp.arange(n_cols - 1)[None, :]
    n00 = (rows * n_cols + cols).reshape(-1)
    n01 = n00 + 1
    n10 = n00 + n_cols
    n11 = n10 + 1
    lower = jnp.stack([n00, n10, n01], axis=1)
    upper = jnp.stack([n10, n11, n01], axis=1)
    return jnp.concatenate([lower, upper], axis=0).astype(jnp.int32)


def barycentric_weights(*, triangle: Float2D, query: Float1D) -> Float1D:
    """Compute the barycentric weights of `query` with respect to `triangle`.

    The weights $(w_0, w_1, w_2)$ satisfy $\\sum_k w_k = 1$ and
    $\\sum_k w_k\\,p_k = q$ for triangle vertices $p_k$ and query $q$. A query inside
    the triangle has all weights in $[0, 1]$; a negative weight marks extrapolation
    beyond the opposite edge. The weights are exact for an affine (triangular) map.

    Args:
        triangle: Vertex coordinates, shape `(3, 2)`.
        query: Query point, shape `(2,)`.

    Returns:
        Barycentric weights, shape `(3,)`.

    """
    p0, p1, p2 = triangle[0], triangle[1], triangle[2]
    # Solve [p1 - p0, p2 - p0] @ (w1, w2) = q - p0 by Cramer's rule; w0 = 1 - w1 - w2.
    e1 = p1 - p0
    e2 = p2 - p0
    rhs = query - p0
    det = e1[0] * e2[1] - e1[1] * e2[0]
    w1 = (rhs[0] * e2[1] - rhs[1] * e2[0]) / det
    w2 = (e1[0] * rhs[1] - e1[1] * rhs[0]) / det
    w0 = 1.0 - w1 - w2
    return jnp.stack([w0, w1, w2])


def is_admissible(*, weights: Float1D, threshold: float) -> BoolND:
    """Whether a triangle is an admissible candidate for its query.

    A triangle is admissible when every barycentric weight is at least `-threshold`
    (`threshold >= 0`) and finite: strictly covering triangles (weights non-negative)
    qualify, and so do mildly extrapolated ones within the band. Equality at the
    boundary `-threshold` is admissible, matching the reference rejection rule (a weight
    is rejected only when strictly below `-threshold`). A degenerate (collinear-image)
    triangle yields non-finite weights and is inadmissible. The within-segment envelope
    then maximizes over all admissible triangles rather than only the covering one.

    Args:
        weights: Barycentric weights, shape `(3,)`.
        threshold: Non-negative extrapolation tolerance on each weight.

    Returns:
        Boolean scalar: `True` if every weight is finite and at least `-threshold`.

    """
    return jnp.all((weights >= -threshold) & jnp.isfinite(weights))


def interpolate_on_triangle(*, node_values: Float2D, weights: Float1D) -> FloatND:
    """Barycentrically interpolate node values at the query.

    Args:
        node_values: Per-vertex values, shape `(3, n_fields)` (e.g. the policy
            `(c, d)` at the triangle's three vertices).
        weights: Barycentric weights, shape `(3,)`.

    Returns:
        Interpolated values, shape `(n_fields,)`.

    """
    return weights @ node_values
