"""Validated interpolation of candidate surfaces across the outer-node axis.

The continuous-outer solver evaluates exact inner solves only at a shared set
of outer nodes; everything between the nodes is this module's job. It builds
a *local cubic Hermite* interpolant along the leading candidate axis of a
stacked surface (value, inner policy, or liquid marginal), vectorized over
all trailing state axes, and it is explicit about where interpolation is not
trustworthy:

- an interval with a nonfinite endpoint value is **invalid** — the read
  reports `-inf` (`fmax` semantics: an infeasible candidate must lose every
  comparison) instead of fabricating a bridge across an infeasible gap;
- the caller can declare additional invalid intervals (solver-declared
  discontinuities, intervals whose midpoint validation failed);
- out-of-domain queries are invalid, never extrapolated;
- a nonfinite neighbor cannot poison an adjacent *valid* interval: node
  slopes fall back to the one-sided secant of the finite side.

Node slopes differentiate the local cubic Lagrange fit through four
consecutive nodes (three-point parabolic fallback near nonfinite values), so
the interpolant reproduces cubics exactly, is C¹ across shared nodes, and
keeps `O(h^4)` accuracy on *graded* meshes — which adaptive refinement
depends on (see `_node_slopes`). Interval location uses `jnp.searchsorted`;
nodes must be strictly increasing (validated eagerly — the mesh is static
under JIT).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.errors
import jax.numpy as jnp

from lcm.typing import BoolND, FloatND


@runtime_checkable
class OuterInterpolant(Protocol):
    """Evaluates a stacked candidate surface between shared outer nodes."""

    def evaluate(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
    ) -> FloatND:
        """Interpolated surface at `query`; `-inf` where invalid."""
        ...

    def evaluate_with_derivative(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
    ) -> tuple[FloatND, FloatND]:
        """Interpolated surface and its outer derivative at `query`."""
        ...


@dataclass(frozen=True, kw_only=True)
class LocalCubicOuterInterpolant:
    """Local cubic Hermite interpolation with explicit validity diagnostics.

    `nodes` has shape `(C,)` (strictly increasing); `values` has shape
    `(C, *S)` with arbitrary trailing state axes; `query` must broadcast
    against `S`. Reads outside `[nodes[0], nodes[-1]]`, inside an interval
    with a nonfinite endpoint, or inside a caller-declared invalid interval
    report `-inf` (and derivative `0`) rather than a fabricated bridge.
    """

    def evaluate(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
        interval_valid: BoolND | None = None,
    ) -> FloatND:
        """Interpolated surface at `query`; `-inf` where invalid.

        Args:
            nodes: Shared outer nodes, shape `(C,)`, strictly increasing.
            values: Stacked candidate surface, shape `(C, *S)`.
            query: Outer abscissae to read, broadcastable against `S`.
            interval_valid: Optional extra per-interval validity, shape
                broadcastable against `(C - 1, *S)`; `False` marks an
                interval the interpolant must not bridge.

        Returns:
            The interpolated read, shape `broadcast(query, S)`.

        """
        value, _, valid = self._evaluate(
            nodes=nodes, values=values, query=query, interval_valid=interval_valid
        )
        return jnp.where(valid, value, -jnp.inf)

    def evaluate_with_derivative(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
        interval_valid: BoolND | None = None,
    ) -> tuple[FloatND, FloatND]:
        """Interpolated surface and its outer derivative at `query`.

        Invalid reads report `(-inf, 0.0)`.

        Args:
            nodes: Shared outer nodes, shape `(C,)`, strictly increasing.
            values: Stacked candidate surface, shape `(C, *S)`.
            query: Outer abscissae to read, broadcastable against `S`.
            interval_valid: Optional extra per-interval validity, shape
                broadcastable against `(C - 1, *S)`.

        Returns:
            The pair `(value, d value / d outer)`.

        """
        value, derivative, valid = self._evaluate(
            nodes=nodes, values=values, query=query, interval_valid=interval_valid
        )
        return (
            jnp.where(valid, value, -jnp.inf),
            jnp.where(valid, derivative, 0.0),
        )

    def evaluate_validity(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
        interval_valid: BoolND | None = None,
    ) -> BoolND:
        """Where a read at `query` would be trustworthy (see `evaluate`)."""
        _, _, valid = self._evaluate(
            nodes=nodes, values=values, query=query, interval_valid=interval_valid
        )
        return valid

    def _evaluate(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
        interval_valid: BoolND | None,
    ) -> tuple[FloatND, FloatND, BoolND]:
        nodes = jnp.asarray(nodes)
        values = jnp.asarray(values)
        _validate_nodes(nodes=nodes, values=values)
        n_nodes = nodes.shape[0]
        state_shape = values.shape[1:]
        query = jnp.asarray(query)
        out_shape = jnp.broadcast_shapes(query.shape, state_shape)
        query_b = jnp.broadcast_to(query, out_shape)
        values_b = jnp.broadcast_to(
            _pad_state_axes(values, out_ndim=len(out_shape)),
            (n_nodes, *out_shape),
        )

        # Containing interval per read; reads at the last node land in the
        # final interval (t = 1), out-of-domain reads are flagged below.
        interval = jnp.clip(
            jnp.searchsorted(nodes, query_b, side="right") - 1, 0, n_nodes - 2
        )
        in_domain = (query_b >= nodes[0]) & (query_b <= nodes[-1])

        slopes = _node_slopes(nodes=nodes, values=values_b)
        y0 = jnp.take_along_axis(values_b, interval[None], axis=0)[0]
        y1 = jnp.take_along_axis(values_b, interval[None] + 1, axis=0)[0]
        m0 = jnp.take_along_axis(slopes, interval[None], axis=0)[0]
        m1 = jnp.take_along_axis(slopes, interval[None] + 1, axis=0)[0]
        x0 = nodes[interval]
        h = nodes[interval + 1] - x0
        t = (query_b - x0) / h

        # Cubic Hermite basis in the local coordinate t = (f - f_j) / h.
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        value = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
        derivative = (
            (6.0 * t2 - 6.0 * t) * (y0 - y1) / h
            + (3.0 * t2 - 4.0 * t + 1.0) * m0
            + (3.0 * t2 - 2.0 * t) * m1
        )

        endpoint_finite = jnp.isfinite(y0) & jnp.isfinite(y1)
        valid = in_domain & endpoint_finite
        if interval_valid is not None:
            declared = jnp.broadcast_to(
                _pad_state_axes(jnp.asarray(interval_valid), out_ndim=len(out_shape)),
                (n_nodes - 1, *out_shape),
            )
            valid = valid & jnp.take_along_axis(declared, interval[None], axis=0)[0]
        return value, derivative, valid


def _pad_state_axes(stacked: FloatND | BoolND, *, out_ndim: int) -> FloatND | BoolND:
    """Insert singleton state axes after the leading candidate axis.

    Broadcasting aligns from the right, so a stacked `(C, *S)` array must
    have `S` padded on its left to the output rank before the whole array
    can broadcast against `(C, *out_shape)`.
    """
    state_ndim = stacked.ndim - 1
    pad = (1,) * (out_ndim - state_ndim)
    return stacked.reshape(stacked.shape[0], *pad, *stacked.shape[1:])


def _node_slopes(*, nodes: FloatND, values: FloatND) -> FloatND:
    """Slope estimates at every node: 4-point where finite, 3-point fallback.

    The primary estimate differentiates the cubic Lagrange fit through four
    consecutive nodes containing the target (cubic-exact, `O(h^3)` slope
    error). That order matters on *graded* meshes: with the cheaper
    parabolic slopes the interval midpoint error is `O(h^4)` on uniform
    spacing (symmetric slope errors cancel) but degrades to `O(h^3)` at a
    grading boundary, which makes adaptive refinement creep one interval
    per round; the 4-point estimate keeps `O(h^4)` everywhere.

    Where the 4-point stencil touches a nonfinite value the slope falls
    back to the 3-point parabolic estimate, then to the finite adjacent
    secant, then to `0` — so a nonfinite neighbor cannot poison a valid
    interval; that interval's own endpoints are checked separately.
    """
    four_point = (
        _four_point_slopes(nodes=nodes, values=values)
        if nodes.shape[0] >= 4  # noqa: PLR2004
        else None
    )
    three_point = _three_point_slopes(nodes=nodes, values=values)
    if four_point is None:
        return three_point
    return jnp.where(jnp.isfinite(four_point), four_point, three_point)


def _four_point_slopes(*, nodes: FloatND, values: FloatND) -> FloatND:
    """Derivative at each node of the cubic Lagrange fit through the four
    consecutive nodes surrounding it (near-centered; clipped at the ends).

    A nonfinite value anywhere in a stencil makes that node's estimate
    nonfinite — the caller falls back to a shorter stencil.
    """
    n_nodes = nodes.shape[0]
    base = jnp.clip(jnp.arange(n_nodes) - 1, 0, n_nodes - 4)
    stencil = base[:, None] + jnp.arange(4)[None, :]  # (C, 4)
    target = jnp.arange(n_nodes) - base  # position of the node in its stencil
    xs = nodes[stencil]  # (C, 4)
    xt = nodes[:, None]  # (C, 1)
    is_target = jnp.arange(4)[None, :] == target[:, None]  # (C, 4)

    # Lagrange-basis derivatives at the target abscissa. For i != t:
    # l_i'(x_t) = prod_{j != i,t}(x_t - x_j) / prod_{j != i}(x_i - x_j);
    # for the target itself: l_t'(x_t) = sum_{j != t} 1 / (x_t - x_j).
    d = jnp.where(is_target, 1.0, xt - xs)  # (C, 4); target slot neutralized
    product = jnp.prod(d, axis=1, keepdims=True)  # prod over j != t
    numerator = product / d
    pair_diffs = xs[:, :, None] - xs[:, None, :]  # (C, 4, 4)
    pair_diffs = jnp.where(jnp.eye(4, dtype=bool)[None], 1.0, pair_diffs)
    denominator = jnp.prod(pair_diffs, axis=2)  # (C, 4): prod_{j != i}
    weights = numerator / denominator
    target_weight = jnp.sum(jnp.where(is_target, 0.0, 1.0 / d), axis=1)
    weights = jnp.where(is_target, target_weight[:, None], weights)

    ys = values[stencil]  # (C, 4, *S)
    weights_b = weights.reshape(*weights.shape, *([1] * (values.ndim - 1)))
    return jnp.sum(weights_b * ys, axis=1)


def _three_point_slopes(*, nodes: FloatND, values: FloatND) -> FloatND:
    """Non-uniform three-point (parabolic) slope estimates at every node.

    Quadratic-exact; one-sided at the ends. Nonfinite stencils fall back to
    the finite adjacent secant, then to `0`.
    """
    h = (nodes[1:] - nodes[:-1]).reshape((-1,) + (1,) * (values.ndim - 1))
    secants = (values[1:] - values[:-1]) / h
    left = secants[:-1]
    right = secants[1:]
    h_left = h[:-1]
    h_right = h[1:]
    parabolic = (h_right * left + h_left * right) / (h_left + h_right)
    interior = jnp.where(
        jnp.isfinite(left) & jnp.isfinite(right),
        parabolic,
        jnp.where(
            jnp.isfinite(left),
            left,
            jnp.where(jnp.isfinite(right), right, 0.0),
        ),
    )
    if secants.shape[0] == 1:
        first = jnp.where(jnp.isfinite(secants[:1]), secants[:1], 0.0)
        last = first
    else:
        first = _one_sided_parabolic_slope(
            near=secants[:1], far=secants[1:2], h_near=h[:1], h_far=h[1:2]
        )
        last = _one_sided_parabolic_slope(
            near=secants[-1:], far=secants[-2:-1], h_near=h[-1:], h_far=h[-2:-1]
        )
    return jnp.concatenate([first, interior, last], axis=0)


def _one_sided_parabolic_slope(
    *, near: FloatND, far: FloatND, h_near: FloatND, h_far: FloatND
) -> FloatND:
    """Quadratic-fit endpoint slope from the two nearest secants.

    `near` is the secant of the interval touching the endpoint, `far` the
    next one in; nonfinite stencils fall back to the finite secant (else 0).
    """
    parabolic = ((2.0 * h_near + h_far) * near - h_near * far) / (h_near + h_far)
    return jnp.where(
        jnp.isfinite(near) & jnp.isfinite(far),
        parabolic,
        jnp.where(jnp.isfinite(near), near, 0.0),
    )


def _validate_nodes(*, nodes: FloatND, values: FloatND) -> None:
    if nodes.ndim != 1:
        msg = f"nodes must be one-dimensional, got shape {nodes.shape}."
        raise ValueError(msg)
    if nodes.shape[0] < 2:  # noqa: PLR2004
        msg = f"need at least two nodes, got {nodes.shape[0]}."
        raise ValueError(msg)
    if values.shape[0] != nodes.shape[0]:
        msg = (
            "leading axis of values must match the number of nodes, got "
            f"values {values.shape} for {nodes.shape[0]} nodes."
        )
        raise ValueError(msg)
    # Under an active trace (e.g. this read is the golden-section objective
    # inside `fori_loop`) even constant-derived comparisons are abstract and
    # cannot be checked; the eager top-level calls on the same mesh (mesh
    # driver, bank build, the pre-loop probes) have already validated it.
    try:
        increasing = bool(jnp.all(nodes[1:] > nodes[:-1]))
    except jax.errors.ConcretizationTypeError:
        return
    if not increasing:
        msg = "nodes must be strictly increasing."
        raise ValueError(msg)
