"""Point-in-quad location on a curvilinear image mesh.

The inverse-Euler step of the two-continuous-state EGM foundation maps a
regular post-decision grid in $(a, b)$ onto an irregular quadrilateral mesh in
endogenous current-state space $(m, n)$ — one quad per source cell, with fixed
product topology. To read a continuation on a regular target $(m, n)$ grid, the
locator must, for each target point, find the source quad that contains it and
the bilinear coordinates $(\\xi, \\eta) \\in [0, 1]^2$ inside that quad, then read
a bilinear value and gradient there.

These tests pin the behavior that two naive shortcuts get wrong:

- Two independent 1-D searches on the marginal image coordinates do **not**
  locate the cell in a coupled monotone image (the shear case).
- A nonzero constant-sign Jacobian gives only *local* invertibility; a folded
  image (positive Jacobian everywhere, yet not globally one-to-one) must be
  rejected by the geometry gate, not silently mislocated.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator

from _lcm.egm.cell_locator import (
    locate_in_quad_mesh,
    read_bilinear,
    validate_quad_mesh,
)
from lcm.exceptions import PyLCMError


def _affine_image(a, b, matrix, shift=(0.0, 0.0)):
    """Image coordinates of an affine map on a product grid of `a` x `b`."""
    a_grid, b_grid = jnp.meshgrid(jnp.asarray(a), jnp.asarray(b), indexing="ij")
    m = matrix[0][0] * a_grid + matrix[0][1] * b_grid + shift[0]
    n = matrix[1][0] * a_grid + matrix[1][1] * b_grid + shift[1]
    return m, n


def _values_on_nodes(m_image, n_image, func):
    """Sample a scalar field on the image nodes (host-side reference truth)."""
    return func(np.asarray(m_image), np.asarray(n_image))


def _host_brute_locate(m_image, n_image, query):
    """Brute point-in-quad scan returning the containing cell, or None.

    Uses the same inverse-bilinear test the locator uses, but in plain NumPy,
    so it is an independent oracle for which cell the query lands in.
    """
    m_image = np.asarray(m_image)
    n_image = np.asarray(n_image)
    n_i, n_j = m_image.shape[0] - 1, m_image.shape[1] - 1
    for i in range(n_i):
        for j in range(n_j):
            xi, eta = _host_inverse_bilinear(m_image, n_image, i, j, query)
            if xi is None:
                continue
            if -1e-9 <= xi <= 1 + 1e-9 and -1e-9 <= eta <= 1 + 1e-9:
                return (i, j, xi, eta)
    return None


def _host_inverse_bilinear(m_image, n_image, i, j, query):
    """Solve the cell's inverse-bilinear map for `query` via Newton (host)."""
    corners_m = np.array(
        [m_image[i, j], m_image[i + 1, j], m_image[i, j + 1], m_image[i + 1, j + 1]]
    )
    corners_n = np.array(
        [n_image[i, j], n_image[i + 1, j], n_image[i, j + 1], n_image[i + 1, j + 1]]
    )
    xi, eta = 0.5, 0.5
    for _ in range(50):
        weights = np.array(
            [(1 - xi) * (1 - eta), xi * (1 - eta), (1 - xi) * eta, xi * eta]
        )
        residual = np.array(
            [corners_m @ weights - query[0], corners_n @ weights - query[1]]
        )
        d_xi = np.array([-(1 - eta), (1 - eta), -eta, eta])
        d_eta = np.array([-(1 - xi), -xi, (1 - xi), xi])
        jac = np.array(
            [
                [corners_m @ d_xi, corners_m @ d_eta],
                [corners_n @ d_xi, corners_n @ d_eta],
            ]
        )
        if abs(np.linalg.det(jac)) < 1e-14:
            return None, None
        step = np.linalg.solve(jac, residual)
        xi, eta = xi - step[0], eta - step[1]
        if np.max(np.abs(step)) < 1e-12:
            break
    return xi, eta


# Identity and affine meshes recover the inverse exactly.


@pytest.mark.parametrize(
    ("matrix", "query", "expected_cell", "expected_xi_eta"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], (0.25, 0.75), (0, 1), (0.5, 0.5)),
        ([[2.0, 0.0], [0.0, 3.0]], (0.5, 2.25), (0, 1), (0.5, 0.5)),
    ],
)
def test_locate_affine_recovers_exact_cell_and_coords(
    matrix, query, expected_cell, expected_xi_eta
):
    """On an axis-aligned affine mesh the locator returns the exact cell and coords."""
    nodes = jnp.array([0.0, 0.5, 1.0])
    m_image, n_image = _affine_image(nodes, nodes, matrix)
    located = locate_in_quad_mesh(
        m_image=m_image,
        n_image=n_image,
        queries=jnp.array([query]),
    )
    np.testing.assert_array_equal(
        np.array([located.cell_i[0], located.cell_j[0]]), np.array(expected_cell)
    )
    np.testing.assert_allclose(
        np.array([located.xi[0], located.eta[0]]),
        np.array(expected_xi_eta),
        atol=1e-9,
    )


def test_locate_shear_picks_coupled_cell_not_marginal_cell():
    """The shear $F(a,b)=(a+0.4b, b+0.4a)$ defeats independent marginal search.

    For query $(m,n)=(0.72, 0.96)$ the true inverse is $a=0.4, b=0.8$, which lies
    in source cell $(0, 1)$ with bilinear coordinates $(\\xi, \\eta)=(0.8, 0.6)$.
    Two independent 1-D searches on the marginal image coordinates land in a
    different $a$-cell; the locator must return the coupled cell.
    """
    nodes = jnp.array([0.0, 0.5, 1.0])
    m_image, n_image = _affine_image(nodes, nodes, [[1.0, 0.4], [0.4, 1.0]])
    located = locate_in_quad_mesh(
        m_image=m_image,
        n_image=n_image,
        queries=jnp.array([[0.72, 0.96]]),
    )
    np.testing.assert_array_equal(
        np.array([located.cell_i[0], located.cell_j[0]]), np.array([0, 1])
    )
    np.testing.assert_allclose(
        np.array([located.xi[0], located.eta[0]]),
        np.array([0.8, 0.6]),
        atol=1e-9,
    )


def test_locate_shear_marginal_search_would_have_failed():
    """Independent marginal `searchsorted` lands in the wrong $a$-cell here.

    Documents *why* the coupled test is necessary: the marginal-image search the
    locator must avoid picks a different cell than the true containing one.
    """
    nodes = np.array([0.0, 0.5, 1.0])
    a_grid, b_grid = np.meshgrid(nodes, nodes, indexing="ij")
    m_image = a_grid + 0.4 * b_grid
    n_image = b_grid + 0.4 * a_grid
    # Marginal image coordinates along each axis at the opposite-axis origin.
    m_marginal = m_image[:, 0]
    n_marginal = n_image[0, :]
    marginal_i = int(np.searchsorted(m_marginal, 0.72) - 1)
    marginal_j = int(np.searchsorted(n_marginal, 0.96) - 1)
    # The true containing cell is (0, 1); marginal search disagrees on the i-axis.
    assert (marginal_i, marginal_j) != (0, 1)


def _curvilinear_image(a, b):
    """A smooth, monotone, fold-free curvilinear deformation of a product grid.

    Strictly increasing in each coordinate with a coupled cross-term, so it is a
    genuine non-affine quad mesh while staying globally one-to-one on the domain.
    """
    a_grid, b_grid = np.meshgrid(np.asarray(a), np.asarray(b), indexing="ij")
    m = a_grid + 0.15 * b_grid + 0.1 * a_grid * b_grid
    n = b_grid + 0.12 * a_grid + 0.08 * a_grid**2
    return jnp.asarray(m), jnp.asarray(n)


def _curvilinear_query(q_a, q_b):
    """Image of host-side source points under the curvilinear deformation."""
    qm = q_a + 0.15 * q_b + 0.1 * q_a * q_b
    qn = q_b + 0.12 * q_a + 0.08 * q_a**2
    return qm, qn


@pytest.mark.parametrize("n_nodes", [9, 17])
def test_locate_curvilinear_value_matches_host_interpolator(n_nodes):
    """Located bilinear value matches a host Delaunay interpolator on dense probes.

    On a smooth fold-free curvilinear mesh, the located cell and its
    inverse-bilinear value read agree with `scipy`'s `LinearNDInterpolator`
    (built on the same nodes) within tolerance over many random interior probes.
    """
    nodes = np.linspace(0.0, 1.0, n_nodes)
    m_image, n_image = _curvilinear_image(nodes, nodes)

    def field(m, n):
        return 1.0 + 0.7 * m + 0.5 * n

    node_values = jnp.asarray(_values_on_nodes(m_image, n_image, field))

    points = np.column_stack([np.asarray(m_image).ravel(), np.asarray(n_image).ravel()])
    host = LinearNDInterpolator(points, np.asarray(node_values).ravel())

    rng = np.random.default_rng(20240601)
    # Sample queries well inside the image to avoid Delaunay convex-hull edge cases.
    q_a = rng.uniform(0.1, 0.9, size=400)
    q_b = rng.uniform(0.1, 0.9, size=400)
    qm, qn = _curvilinear_query(q_a, q_b)
    queries = jnp.asarray(np.column_stack([qm, qn]))

    located = locate_in_quad_mesh(m_image=m_image, n_image=n_image, queries=queries)
    read = read_bilinear(node_values=node_values, located=located)

    host_values = host(np.column_stack([qm, qn]))
    finite = np.isfinite(host_values)
    assert finite.mean() > 0.9
    np.testing.assert_allclose(
        np.asarray(read.value)[finite], host_values[finite], atol=2e-3
    )


def test_locate_curvilinear_value_converges_under_refinement():
    """The bilinear value read converges to the true field as the mesh refines."""

    def field(m, n):
        return np.sin(1.3 * m) + np.cos(0.9 * n) + 0.4 * m * n

    rng = np.random.default_rng(7)
    q_a = rng.uniform(0.15, 0.85, size=200)
    q_b = rng.uniform(0.15, 0.85, size=200)
    qm, qn = _curvilinear_query(q_a, q_b)
    queries = jnp.asarray(np.column_stack([qm, qn]))
    truth = field(qm, qn)

    errors = []
    for n_nodes in (9, 33):
        nodes = np.linspace(0.0, 1.0, n_nodes)
        m_image, n_image = _curvilinear_image(nodes, nodes)
        node_values = jnp.asarray(_values_on_nodes(m_image, n_image, field))
        located = locate_in_quad_mesh(m_image=m_image, n_image=n_image, queries=queries)
        read = read_bilinear(node_values=node_values, located=located)
        errors.append(float(np.max(np.abs(np.asarray(read.value) - truth))))

    # Roughly second-order: a 4x finer mesh cuts the worst-case error by >5x.
    assert errors[1] < errors[0] / 5.0


def test_locate_curvilinear_gradient_converges_under_refinement():
    """The bilinear gradient read converges to the true gradient under refinement.

    The value and the gradient are two *separate* bilinear approximants: the
    gradient is read from the same cell map but is its own quantity, so its
    convergence is asserted independently of the value's.
    """

    def field(m, n):
        return 0.6 * m**2 + 0.5 * n**2 + 0.3 * m * n

    def grad_field(m, n):
        return np.column_stack([1.2 * m + 0.3 * n, 1.0 * n + 0.3 * m])

    rng = np.random.default_rng(11)
    q_a = rng.uniform(0.2, 0.8, size=150)
    q_b = rng.uniform(0.2, 0.8, size=150)
    qm, qn = _curvilinear_query(q_a, q_b)
    queries = jnp.asarray(np.column_stack([qm, qn]))
    truth = grad_field(qm, qn)

    errors = []
    for n_nodes in (9, 33):
        nodes = np.linspace(0.0, 1.0, n_nodes)
        m_image, n_image = _curvilinear_image(nodes, nodes)
        node_values = jnp.asarray(_values_on_nodes(m_image, n_image, field))
        located = locate_in_quad_mesh(m_image=m_image, n_image=n_image, queries=queries)
        read = read_bilinear(node_values=node_values, located=located)
        grad = np.column_stack([np.asarray(read.grad_m), np.asarray(read.grad_n)])
        errors.append(float(np.max(np.abs(grad - truth))))

    assert errors[1] < errors[0] / 3.0


def test_locate_matches_host_brute_scan_on_curvilinear_cell_choice():
    """The located cell index agrees with an independent host brute-scan."""
    nodes = np.linspace(0.0, 1.0, 11)
    m_image, n_image = _curvilinear_image(nodes, nodes)

    rng = np.random.default_rng(99)
    q_a = rng.uniform(0.1, 0.9, size=60)
    q_b = rng.uniform(0.1, 0.9, size=60)
    qm, qn = _curvilinear_query(q_a, q_b)
    queries = jnp.asarray(np.column_stack([qm, qn]))

    located = locate_in_quad_mesh(m_image=m_image, n_image=n_image, queries=queries)

    for k in range(queries.shape[0]):
        host = _host_brute_locate(m_image, n_image, (qm[k], qn[k]))
        assert host is not None
        assert (int(located.cell_i[k]), int(located.cell_j[k])) == (host[0], host[1])


# The geometry gate rejects a folded image (F6).


def test_geometry_gate_accepts_valid_curvilinear_mesh():
    """A smooth fold-free monotone mesh passes the geometry gate without error."""
    nodes = np.linspace(0.0, 1.0, 7)
    m_image, n_image = _curvilinear_image(nodes, nodes)
    # Must not raise.
    validate_quad_mesh(m_image=m_image, n_image=n_image)


def test_geometry_gate_rejects_folded_polar_mesh():
    """The polar map $F(a,b)=(e^a\\cos b, e^a\\sin b)$ wraps and must be rejected.

    Its Jacobian is positive everywhere, so a constant-sign-Jacobian check would
    pass it, yet it folds over itself across $b \\in [0, 4\\pi]$ — the image is not
    globally one-to-one. The geometry gate must detect the fold (sign-flipped /
    overlapping cells) and raise rather than let the locator mislocate.
    """
    a = np.linspace(0.0, 1.0, 6)
    b = np.linspace(0.0, 4 * np.pi, 24)
    a_grid, b_grid = np.meshgrid(a, b, indexing="ij")
    m_image = jnp.asarray(np.exp(a_grid) * np.cos(b_grid))
    n_image = jnp.asarray(np.exp(a_grid) * np.sin(b_grid))
    with pytest.raises(PyLCMError, match=r"fold|orient|overlap"):
        validate_quad_mesh(m_image=m_image, n_image=n_image)


def test_geometry_gate_rejects_self_intersecting_cell():
    """A single bow-tie (self-intersecting) quad fails the gate.

    Twisting one interior corner of an otherwise-regular grid produces a
    non-convex, self-crossing quad; the signed-area / orientation check must
    catch it.
    """
    nodes = np.linspace(0.0, 1.0, 3)
    a_grid, b_grid = np.meshgrid(nodes, nodes, indexing="ij")
    m_image = np.array(a_grid, dtype=float)
    n_image = np.array(b_grid, dtype=float)
    # Twist one interior corner so the adjacent cells become bow-ties.
    m_image[1, 1], n_image[1, 1] = -0.5, -0.5
    with pytest.raises(PyLCMError, match=r"fold|orient|overlap"):
        validate_quad_mesh(m_image=jnp.asarray(m_image), n_image=jnp.asarray(n_image))
