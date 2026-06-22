"""Triangular-mesh barycentric geometry for the 2-D G2EGM envelope is correct.

The geometry is the foundation of the upper-envelope selection: each source cell
splits into two triangles, a target is located by its barycentric weights, and a
triangle is an admissible candidate when every weight exceeds a negative threshold.
The tests pin the basic invariants and the two counterexamples an adversarial audit
used to reject a quadrilateral, cover-first design: a folded cell whose triangles stay
non-degenerate, and an extrapolated target that is admissible under the reference
threshold.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.mesh_geometry import (
    barycentric_weights,
    interpolate_on_triangle,
    is_admissible,
    triangulate_regular_grid,
)


def test_triangulate_one_cell_splits_into_two_triangles():
    """A 2x2 grid's single cell becomes two triangles on the shared diagonal."""
    simplices = np.asarray(triangulate_regular_grid(n_rows=2, n_cols=2))
    np.testing.assert_array_equal(simplices, np.array([[0, 2, 1], [2, 3, 1]]))


def test_triangulate_cell_count():
    """An `n_rows` by `n_cols` grid yields `2*(n_rows-1)*(n_cols-1)` triangles."""
    simplices = triangulate_regular_grid(n_rows=4, n_cols=5)
    assert simplices.shape == (2 * 3 * 4, 3)


def test_barycentric_weights_at_vertex_are_one_hot():
    """The barycentric weights at a triangle vertex select that vertex."""
    triangle = jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    weights = barycentric_weights(triangle=triangle, query=triangle[1])
    np.testing.assert_allclose(np.asarray(weights), [0.0, 1.0, 0.0], atol=1e-12)


def test_barycentric_weights_at_centroid_are_uniform():
    """The barycentric weights at the centroid are all one third."""
    triangle = jnp.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    centroid = jnp.mean(triangle, axis=0)
    weights = barycentric_weights(triangle=triangle, query=centroid)
    np.testing.assert_allclose(np.asarray(weights), [1 / 3, 1 / 3, 1 / 3], atol=1e-12)


def test_folded_quad_triangles_stay_non_degenerate():
    """The audit's folding quad splits into two non-degenerate triangles.

    Mapping a unit cell to corners with a sign-changing bilinear Jacobian makes the
    quad's inverse non-unique, but the diagonal split `(p00,p10,p01)`, `(p10,p11,p01)`
    yields two triangles with non-zero signed area, each separately evaluable.
    """
    p00, p10, p11, p01 = (
        jnp.array([0.0, 0.0]),
        jnp.array([1.0, 0.0]),
        jnp.array([0.4, 0.4]),
        jnp.array([0.0, 1.0]),
    )

    def signed_area(a, b, c):
        return 0.5 * float(
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        )

    assert abs(signed_area(p00, p10, p01)) > 1e-6
    assert abs(signed_area(p10, p11, p01)) > 1e-6


def test_extrapolated_target_is_admissible_under_reference_threshold():
    """A target just outside a triangle is an admissible candidate (weights > -0.25).

    For the triangle `(0,0),(2,0),(0,2)` the target `(1.2, 1.2)` has weights
    `(-0.2, 0.6, 0.6)`; it is admissible at the reference threshold 0.25 but not at a
    stricter 0.1 — so a cover-first rule that drops it can miss the winning branch.
    """
    triangle = jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    weights = barycentric_weights(triangle=triangle, query=jnp.array([1.2, 1.2]))
    np.testing.assert_allclose(np.asarray(weights), [-0.2, 0.6, 0.6], atol=1e-12)
    assert bool(is_admissible(weights=weights, threshold=0.25))
    assert not bool(is_admissible(weights=weights, threshold=0.1))


def test_weight_exactly_at_negative_threshold_is_admissible():
    """A weight exactly at `-threshold` is admissible (reference rejects only below it).

    For the triangle `(0,0),(1,0),(0,1)` the target `(0.625, 0.625)` has weights
    `(-0.25, 0.625, 0.625)`; at the reference threshold 0.25 the boundary weight `-0.25`
    must keep the triangle admissible, so a competing simplex cannot win by a hair.
    """
    triangle = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    weights = barycentric_weights(triangle=triangle, query=jnp.array([0.625, 0.625]))
    np.testing.assert_allclose(np.asarray(weights), [-0.25, 0.625, 0.625], atol=1e-12)
    assert bool(is_admissible(weights=weights, threshold=0.25))


def test_degenerate_triangle_is_inadmissible():
    """A collinear-image triangle yields non-finite weights and is inadmissible.

    The mapped triangle `(0,0),(1,0),(2,0)` is degenerate (zero area), so its weights
    are non-finite; admissibility must reject it rather than let a NaN weight slip
    through as a candidate.
    """
    triangle = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    weights = barycentric_weights(triangle=triangle, query=jnp.array([1.0, 0.0]))
    assert not np.all(np.isfinite(np.asarray(weights)))
    assert not bool(is_admissible(weights=weights, threshold=0.25))


def test_interpolate_reproduces_affine_field():
    """Barycentric interpolation reproduces an affine field exactly at any query."""
    triangle = jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    # Affine field f(x, y) = (1 + 2x + 3y, 5 - x): exact under barycentric weights.
    node_values = jnp.array([[1.0, 5.0], [5.0, 3.0], [7.0, 5.0]])
    query = jnp.array([0.5, 0.5])
    weights = barycentric_weights(triangle=triangle, query=query)
    interpolated = interpolate_on_triangle(node_values=node_values, weights=weights)
    np.testing.assert_allclose(np.asarray(interpolated), [3.5, 4.5], atol=1e-12)
