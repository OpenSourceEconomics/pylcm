"""Point-in-quad location on a curvilinear image mesh, with bilinear reads.

The two-continuous-state EGM step maps a regular post-decision grid in $(a, b)$
onto an irregular quadrilateral mesh in endogenous current-state space $(m, n)$:
each source cell of the product grid becomes one quad in the image, with fixed
product topology (source cell $(i, j)$ has image corners at nodes $(i, j)$,
$(i{+}1, j)$, $(i, j{+}1)$, $(i{+}1, j{+}1)$). To read a continuation on a
regular target $(m, n)$ grid, every target point must be located in the quad
that contains it together with its bilinear coordinates inside that quad.

Two shortcuts are wrong and this module avoids them:

- **Independent marginal search.** Two separate 1-D `searchsorted`s on the
  marginal image coordinates do not locate the cell in a *coupled* monotone
  image — a shear `F(a,b)=(a+0.4b, b+0.4a)` is strictly increasing in each
  coordinate yet the marginal search lands in the wrong source cell. The locator
  instead inverts the full bilinear cell map and tests $(\\xi, \\eta) \\in [0,1]^2$.
- **Constant-sign-Jacobian acceptance.** A positive Jacobian everywhere gives
  only *local* invertibility. A polar map `F(a,b)=(e^a\\cos b, e^a\\sin b)` has a
  positive Jacobian throughout yet folds over itself. `validate_quad_mesh`
  detects folded / self-intersecting / inconsistently oriented cells and raises.

The locator computes, for each query and *every* source cell, the analytic
inverse-bilinear coordinates, marks the cell that brackets the query, and
selects it — a static-shape, `vmap`-friendly scan with no data-dependent control
flow. `read_bilinear` then reads a value and a gradient at the located
coordinates. The bilinear value and the gradient are two distinct approximants:
the value is the bilinear interpolant of the node values, the gradient is that
interpolant's spatial derivative through the cell's geometric Jacobian.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from lcm.exceptions import PyLCMError
from lcm.typing import (
    BoolND,
    Float1D,
    Float2D,
    FloatND,
    Int1D,
    ScalarFloat,
    ScalarInt,
)

# Slack on the [0, 1] membership test, to keep shared edges/corners owned.
_INSIDE_TOL = 1e-7
# Below this absolute signed area a cell counts as degenerate (non-invertible).
_DEGENERATE_AREA_TOL = 1e-12
# Slack on the boundary-turn sign used to detect a self-intersecting quad.
_TURN_SIGN_TOL = 1e-12
# Below this the bilinear cross-term degenerates and the cell map is affine.
_AFFINE_CELL_TOL = 1e-12


@dataclass(frozen=True)
class LocatedQueries:
    """Per-query location of a batch of points in a quadrilateral image mesh.

    Carries the located cell's geometric Jacobian at the query coordinates so a
    bilinear read can map $(\\xi, \\eta)$-partials to $(m, n)$-partials without
    re-touching the image mesh.
    """

    cell_i: Int1D
    """Source-cell row index $i$ (the $a$-axis cell) containing each query."""
    cell_j: Int1D
    """Source-cell column index $j$ (the $b$-axis cell) containing each query."""
    xi: Float1D
    """Bilinear coordinate $\\xi \\in [0, 1]$ along the cell's $a$-edge."""
    eta: Float1D
    """Bilinear coordinate $\\eta \\in [0, 1]$ along the cell's $b$-edge."""
    geom_dm_dxi: Float1D
    """$\\partial m / \\partial \\xi$ of the cell's geometric map at the query."""
    geom_dm_deta: Float1D
    """$\\partial m / \\partial \\eta$ of the cell's geometric map at the query."""
    geom_dn_dxi: Float1D
    """$\\partial n / \\partial \\xi$ of the cell's geometric map at the query."""
    geom_dn_deta: Float1D
    """$\\partial n / \\partial \\eta$ of the cell's geometric map at the query."""


@dataclass(frozen=True)
class BilinearRead:
    """Bilinear value and spatial gradient at located query coordinates."""

    value: Float1D
    """Bilinear interpolant of the node values at each query."""
    grad_m: Float1D
    """Partial derivative of the value w.r.t. $m$ (the first image coordinate)."""
    grad_n: Float1D
    """Partial derivative of the value w.r.t. $n$ (the second image coordinate)."""


def locate_in_quad_mesh(
    *,
    m_image: Float2D,
    n_image: Float2D,
    queries: Float2D,
) -> LocatedQueries:
    """Locate each query in the source quad that contains it.

    For every query and every source cell the analytic inverse-bilinear map
    yields candidate coordinates $(\\xi, \\eta)$; a query lies in a cell when those
    fall in $[0, 1]^2$. The first such cell in row-major order is selected (shared
    edges and corners are owned by the lowest-index neighbour). A query outside
    the whole image falls back to the nearest boundary cell with clamped
    coordinates, so the read stays defined.

    Args:
        m_image: First image coordinate of every source node, shape `(n_a, n_b)`.
        n_image: Second image coordinate of every source node, shape `(n_a, n_b)`.
        queries: Query points, shape `(n_queries, 2)` as `(m, n)` pairs.

    Returns:
        Located cell indices and bilinear coordinates, one entry per query.

    """
    corners = _cell_corners(m_image=m_image, n_image=n_image)
    flat_corners = tuple(corner.reshape(-1, 2) for corner in corners)
    n_j = m_image.shape[1] - 1

    def locate_one(
        query: Float1D,
    ) -> tuple[
        ScalarInt,
        ScalarInt,
        ScalarFloat,
        ScalarFloat,
        ScalarFloat,
        ScalarFloat,
        ScalarFloat,
        ScalarFloat,
    ]:
        xi, eta = _inverse_bilinear_all_cells(corners=corners, query=query)
        inside = _is_inside(xi, eta)
        # Distance-to-cell-centre penalty breaks ties / drives the no-hit
        # fallback to the nearest boundary cell.
        center_penalty = (xi - 0.5) ** 2 + (eta - 0.5) ** 2
        score = jnp.where(inside, -1.0, center_penalty)
        flat = jnp.argmin(score).astype(jnp.int32)
        i = flat // n_j
        j = flat % n_j
        xi_sel = jnp.clip(xi[flat], 0.0, 1.0)
        eta_sel = jnp.clip(eta[flat], 0.0, 1.0)
        jac = _geometric_jacobian(
            flat_corners=flat_corners, flat=flat, xi=xi_sel, eta=eta_sel
        )
        return i.astype(jnp.int32), j.astype(jnp.int32), xi_sel, eta_sel, *jac

    cell_i, cell_j, xi, eta, dm_dxi, dm_deta, dn_dxi, dn_deta = jax.vmap(locate_one)(
        queries
    )
    return LocatedQueries(
        cell_i=cell_i,
        cell_j=cell_j,
        xi=xi,
        eta=eta,
        geom_dm_dxi=dm_dxi,
        geom_dm_deta=dm_deta,
        geom_dn_dxi=dn_dxi,
        geom_dn_deta=dn_deta,
    )


def read_bilinear(
    *,
    node_values: Float2D,
    located: LocatedQueries,
) -> BilinearRead:
    """Read a bilinear value and spatial gradient at located coordinates.

    The value is the bilinear interpolant of the four cell-corner node values at
    $(\\xi, \\eta)$. The gradient is that interpolant's derivative w.r.t. $(m, n)$,
    obtained by pushing the $(\\xi, \\eta)$ partials through the inverse of the
    cell's geometric Jacobian $\\partial(m, n)/\\partial(\\xi, \\eta)$. Value and
    gradient are therefore separate approximants of the underlying field.

    Args:
        node_values: Scalar field sampled on every source node, shape
            `(n_a, n_b)` — aligned with the image-coordinate arrays.
        located: Located cells and coordinates from `locate_in_quad_mesh`.

    Returns:
        Bilinear value and gradient components per query.

    """
    i = located.cell_i
    j = located.cell_j
    v00 = node_values[i, j]
    v10 = node_values[i + 1, j]
    v01 = node_values[i, j + 1]
    v11 = node_values[i + 1, j + 1]
    xi = located.xi
    eta = located.eta

    value = (
        (1 - xi) * (1 - eta) * v00
        + xi * (1 - eta) * v10
        + (1 - xi) * eta * v01
        + xi * eta * v11
    )
    dv_dxi = (1 - eta) * (v10 - v00) + eta * (v11 - v01)
    dv_deta = (1 - xi) * (v01 - v00) + xi * (v11 - v10)
    grad_m, grad_n = _push_gradient_through_geometry(
        located=located, dv_dxi=dv_dxi, dv_deta=dv_deta
    )
    return BilinearRead(value=value, grad_m=grad_m, grad_n=grad_n)


def validate_quad_mesh(*, m_image: Float2D, n_image: Float2D) -> None:
    """Reject a folded, self-intersecting, or inconsistently oriented image mesh.

    Every source cell maps to a quad. A valid mesh — locally invertible *and*
    globally one-to-one over the domain — must pass three checks:

    - **Orientation.** Each quad's signed area (shoelace over its corners in
      boundary order) must be nonzero and share one sign across all cells. A
      zero area is a degenerate (non-invertible) cell; a flipped sign is a cell
      folded relative to its neighbours.
    - **Convexity.** No quad may self-intersect (a bow-tie), detected by a turn
      sign-flip while walking the cell boundary.
    - **No global overlap.** No cell's centroid may fall inside any *other*
      cell. A consistent local orientation does not rule out the image folding
      back over itself far away in index space — the polar map
      `F(a,b)=(e^a\\cos b, e^a\\sin b)` keeps every cell positively oriented yet
      wraps so distant cells cover the same region. The overlap check catches
      that global fold a per-cell test cannot.

    A positive geometric Jacobian everywhere does not imply any of these.

    Args:
        m_image: First image coordinate of every source node, shape `(n_a, n_b)`.
        n_image: Second image coordinate of every source node, shape `(n_a, n_b)`.

    Raises:
        PyLCMError: If any cell is degenerate, self-intersecting, oriented
            against the others, or overlaps another cell (a fold or overlap).

    """
    corners = _cell_corners(m_image=m_image, n_image=n_image)
    p00, p10, p11, p01 = corners[0], corners[1], corners[3], corners[2]

    signed_area = 0.5 * (_cross(p10 - p00, p01 - p00) + _cross(p11 - p10, p01 - p11))
    areas = jnp.asarray(signed_area).ravel()

    if bool(jnp.any(jnp.abs(areas) < _DEGENERATE_AREA_TOL)):
        msg = (
            "Degenerate image cell: a source cell maps to a zero-area quad, so "
            "the mesh is not locally invertible. The mesh must have strictly "
            "oriented cells (no fold / overlap)."
        )
        raise PyLCMError(msg)

    positive = bool(jnp.all(areas > 0))
    negative = bool(jnp.all(areas < 0))
    if not (positive or negative):
        msg = (
            "Inconsistently oriented image cells: the signed areas of the source "
            "cells do not all share one sign, so the image folds over itself "
            "(overlapping cells). The mesh is not globally one-to-one."
        )
        raise PyLCMError(msg)

    if _has_self_intersecting_cell(corners=corners):
        msg = (
            "Self-intersecting image cell: a source cell maps to a bow-tie quad "
            "whose edges cross, so the cell folds on itself. The mesh must have "
            "convex, consistently oriented cells (no fold / overlap)."
        )
        raise PyLCMError(msg)

    if _has_overlapping_cells(corners=corners):
        msg = (
            "Overlapping image cells: a source cell's centroid falls inside "
            "another cell, so the image folds back over itself and is not "
            "globally one-to-one (a fold / overlap)."
        )
        raise PyLCMError(msg)


def _cell_corners(*, m_image: Float2D, n_image: Float2D) -> FloatND:
    """Stack the four corner image-points of every cell.

    Returns an array of shape `(4, n_a-1, n_b-1, 2)` whose leading axis runs over
    the corners $(i, j)$, $(i{+}1, j)$, $(i, j{+}1)$, $(i{+}1, j{+}1)$ as
    `(m, n)` pairs.
    """
    points = jnp.stack([m_image, n_image], axis=-1)
    p00 = points[:-1, :-1]
    p10 = points[1:, :-1]
    p01 = points[:-1, 1:]
    p11 = points[1:, 1:]
    return jnp.stack([p00, p10, p01, p11], axis=0)


def _inverse_bilinear_all_cells(
    *, corners: FloatND, query: Float1D
) -> tuple[FloatND, FloatND]:
    """Solve the inverse-bilinear cell map for one query against every cell.

    Returns flattened $(\\xi, \\eta)$ over all cells (row-major), each of shape
    `(n_cells,)`. For each cell the bilinear map
    $Q - A = B\\,\\xi + C\\,\\eta + D\\,\\xi\\eta$ is inverted by solving a quadratic
    in $\\eta$ (linear when the bilinear cross-term degenerates to affine) and
    back-substituting $\\xi$; the root yielding coordinates closest to the unit
    square is kept.
    """
    p00 = corners[0].reshape(-1, 2)
    p10 = corners[1].reshape(-1, 2)
    p01 = corners[2].reshape(-1, 2)
    p11 = corners[3].reshape(-1, 2)

    coeff_a = p00
    coeff_b = p10 - p00
    coeff_c = p01 - p00
    coeff_d = p00 - p10 - p01 + p11
    q_minus_a = query[None, :] - coeff_a

    quad_a = _cross(-coeff_c, coeff_d)
    quad_b = _cross(q_minus_a, coeff_d) + _cross(-coeff_c, coeff_b)
    quad_c = _cross(q_minus_a, coeff_b)

    eta_linear = -quad_c / _nonzero(quad_b)
    disc = jnp.maximum(quad_b**2 - 4 * quad_a * quad_c, 0.0)
    sqrt_disc = jnp.sqrt(disc)
    eta_root1 = (-quad_b + sqrt_disc) / _nonzero(2 * quad_a)
    eta_root2 = (-quad_b - sqrt_disc) / _nonzero(2 * quad_a)

    is_affine = jnp.abs(quad_a) < _AFFINE_CELL_TOL
    xi1, eta1 = _xi_from_eta(coeff_b, coeff_c, coeff_d, q_minus_a, eta_root1)
    xi2, eta2 = _xi_from_eta(coeff_b, coeff_c, coeff_d, q_minus_a, eta_root2)
    xi_lin, eta_lin = _xi_from_eta(coeff_b, coeff_c, coeff_d, q_minus_a, eta_linear)

    # Prefer the root that lands inside the unit square; fall back to the one
    # closest to the cell centre.
    pick_root1 = _prefer(xi1, eta1, xi2, eta2)
    xi_quad = jnp.where(pick_root1, xi1, xi2)
    eta_quad = jnp.where(pick_root1, eta1, eta2)

    xi = jnp.where(is_affine, xi_lin, xi_quad)
    eta = jnp.where(is_affine, eta_lin, eta_quad)
    return xi, eta


def _geometric_jacobian(
    *,
    flat_corners: tuple[FloatND, FloatND, FloatND, FloatND],
    flat: ScalarInt,
    xi: ScalarFloat,
    eta: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
    """Bilinear geometric Jacobian of the selected cell at $(\\xi, \\eta)$.

    Returns $(\\partial m/\\partial\\xi, \\partial m/\\partial\\eta,
    \\partial n/\\partial\\xi, \\partial n/\\partial\\eta)$ for the cell indexed by
    `flat` in the flattened (row-major) cell ordering.
    """
    p00 = flat_corners[0][flat]
    p10 = flat_corners[1][flat]
    p01 = flat_corners[2][flat]
    p11 = flat_corners[3][flat]
    d_dxi = (1 - eta) * (p10 - p00) + eta * (p11 - p01)
    d_deta = (1 - xi) * (p01 - p00) + xi * (p11 - p10)
    return d_dxi[0], d_deta[0], d_dxi[1], d_deta[1]


def _xi_from_eta(
    coeff_b: FloatND,
    coeff_c: FloatND,
    coeff_d: FloatND,
    q_minus_a: FloatND,
    eta: FloatND,
) -> tuple[FloatND, FloatND]:
    """Back-substitute $\\xi$ from a chosen $\\eta$ via the better-conditioned row.

    From $\\xi\\,(B + D\\eta) = (Q - A) - C\\eta$, divide by whichever component of
    $B + D\\eta$ has larger magnitude to avoid cancellation.
    """
    denom = coeff_b + coeff_d * eta[:, None]
    rhs = q_minus_a - coeff_c * eta[:, None]
    use_first = jnp.abs(denom[:, 0]) >= jnp.abs(denom[:, 1])
    xi = jnp.where(
        use_first,
        rhs[:, 0] / _nonzero(denom[:, 0]),
        rhs[:, 1] / _nonzero(denom[:, 1]),
    )
    return xi, eta


def _prefer(xi1: FloatND, eta1: FloatND, xi2: FloatND, eta2: FloatND) -> BoolND:
    """Boolean mask selecting root 1 over root 2 per cell."""
    inside1 = _is_inside(xi1, eta1)
    inside2 = _is_inside(xi2, eta2)
    dist1 = (xi1 - 0.5) ** 2 + (eta1 - 0.5) ** 2
    dist2 = (xi2 - 0.5) ** 2 + (eta2 - 0.5) ** 2
    return inside1 | (~inside2 & (dist1 <= dist2))


def _is_inside(xi: FloatND, eta: FloatND) -> BoolND:
    """Boolean mask: coordinates within the unit square up to a small slack."""
    return (
        (xi >= -_INSIDE_TOL)
        & (xi <= 1 + _INSIDE_TOL)
        & (eta >= -_INSIDE_TOL)
        & (eta <= 1 + _INSIDE_TOL)
    )


def _push_gradient_through_geometry(
    *, located: LocatedQueries, dv_dxi: Float1D, dv_deta: Float1D
) -> tuple[Float1D, Float1D]:
    """Map $(\\xi, \\eta)$-partials to $(m, n)$-partials via the inverse Jacobian.

    The value's spatial gradient is
    $\\nabla_{m,n} V = J^{-1\\top} \\, (\\partial V/\\partial\\xi,
    \\partial V/\\partial\\eta)$, with $J = \\partial(m, n)/\\partial(\\xi, \\eta)$
    the cell's geometric Jacobian evaluated at the located coordinates. The
    gradient is read from the same located cell as the value but is its own
    bilinear approximant.
    """
    j_m_xi = located.geom_dm_dxi
    j_m_eta = located.geom_dm_deta
    j_n_xi = located.geom_dn_dxi
    j_n_eta = located.geom_dn_deta
    det = j_m_xi * j_n_eta - j_m_eta * j_n_xi
    # Floor the magnitude (keeping the sign) so a degenerate cell never divides
    # by zero; a valid mesh has det well away from zero here.
    safe_sign = jnp.where(det < 0, -1.0, 1.0)
    det = safe_sign * jnp.maximum(jnp.abs(det), 1e-12)
    # Inverse-transpose times the (xi, eta) gradient.
    grad_m = (j_n_eta * dv_dxi - j_n_xi * dv_deta) / det
    grad_n = (-j_m_eta * dv_dxi + j_m_xi * dv_deta) / det
    return grad_m, grad_n


def _has_self_intersecting_cell(*, corners: FloatND) -> bool:
    """Detect a bow-tie quad via a sign flip across a splitting diagonal."""
    p00 = corners[0]
    p10 = corners[1]
    p01 = corners[2]
    p11 = corners[3]
    # Walk the boundary p00 -> p10 -> p11 -> p01; a convex (non-self-crossing)
    # quad keeps the same turn sign at every vertex.
    turn0 = _cross(p10 - p00, p11 - p10)
    turn1 = _cross(p11 - p10, p01 - p11)
    turn2 = _cross(p01 - p11, p00 - p01)
    turn3 = _cross(p00 - p01, p10 - p00)
    turns = jnp.stack(
        [turn0.ravel(), turn1.ravel(), turn2.ravel(), turn3.ravel()], axis=0
    )
    all_nonneg = jnp.all(turns >= -_TURN_SIGN_TOL, axis=0)
    all_nonpos = jnp.all(turns <= _TURN_SIGN_TOL, axis=0)
    convex = all_nonneg | all_nonpos
    return bool(jnp.any(~convex))


def _has_overlapping_cells(*, corners: FloatND) -> bool:
    """Detect a global fold: any cell centroid landing inside a different cell.

    For every cell centroid the inverse-bilinear map is solved against all cells;
    a valid mesh contains each centroid in exactly one cell (its own). A centroid
    contained by any other cell means two source cells cover the same image
    region — a fold the per-cell orientation check misses.
    """
    p00 = corners[0].reshape(-1, 2)
    p10 = corners[1].reshape(-1, 2)
    p01 = corners[2].reshape(-1, 2)
    p11 = corners[3].reshape(-1, 2)
    centroids = (p00 + p10 + p01 + p11) / 4.0

    def count_containers(centroid: Float1D) -> ScalarInt:
        xi, eta = _inverse_bilinear_all_cells(corners=corners, query=centroid)
        return jnp.sum(_is_inside(xi, eta).astype(jnp.int32)).astype(jnp.int32)

    container_counts = jax.vmap(count_containers)(centroids)
    # Each centroid is inside its own cell; >1 means an overlap with another.
    return bool(jnp.any(container_counts > 1))


def _cross(u: FloatND, v: FloatND) -> FloatND:
    """2-D scalar cross product $u_x v_y - u_y v_x$ over the last axis."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _nonzero(x: FloatND) -> FloatND:
    """Replace exact zeros with a tiny signed value to keep divisions finite."""
    return jnp.where(x == 0, 1e-30, x)
