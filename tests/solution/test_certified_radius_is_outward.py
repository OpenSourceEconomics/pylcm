"""The certified radius never understates the interior evaluation's own error.

Selection rests on the claim that the exact interpolant lies inside
`[value_hi + value_lo - radius, value_hi + value_lo + radius]`. A radius that is
outward is the whole content of that claim: one that understates by even a
fraction of an ULP turns a certificate into an estimate, and the failure is
silent — the envelope keeps naming a winner, and the winner is right almost
always, so nothing downstream looks wrong.

Every other artifact in this chain tests published VALUES. This one tests the
CERTIFICATE, which is the gap the round-15 completion record names as the thing
it most wants attacked: the radius is the same quantity as before, but it is now
computed from `_framed_affine`'s own framed addends rather than beside them, and
a subtly-too-small radius would not surface anywhere else.

The oracle is exact rational arithmetic over the stored floats, so it cannot
pass vacuously against a tolerance.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import _candidate_terms

_SEED = 20260801


def _exact_affine(*, x0: float, x1: float, v0: float, v1: float, x: float) -> Fraction:
    """The link's affine value at `x`, exactly, over the stored floats."""
    if x == x0:
        return Fraction(v0)
    if x == x1:
        return Fraction(v1)
    lo, hi, query = Fraction(x0), Fraction(x1), Fraction(x)
    return (Fraction(v0) * (hi - query) + Fraction(v1) * (query - lo)) / (hi - lo)


def _geometries(rng, dtype, n_links):
    """Links spanning many widths and value scales, as `(n_links, 8)` rows."""
    rows = []
    for _ in range(n_links):
        # Width and value scale drawn over many binades independently, so narrow
        # links carrying huge values and wide links carrying tiny ones both occur.
        width_exp = int(rng.integers(-40, 40))
        value_exp = int(rng.integers(-40, 40))
        x0 = dtype(np.ldexp(rng.uniform(-2.0, 2.0), int(rng.integers(-30, 30))))
        width = dtype(abs(np.ldexp(rng.uniform(0.5, 2.0), width_exp)))
        x1 = dtype(x0 + width)
        if not np.isfinite(x1) or x1 == x0:
            continue
        v0 = dtype(np.ldexp(rng.uniform(-2.0, 2.0), value_exp))
        v1 = dtype(np.ldexp(rng.uniform(-2.0, 2.0), value_exp))
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        p0 = dtype(rng.uniform(-1.0, 1.0))
        p1 = dtype(rng.uniform(-1.0, 1.0))
        m0 = dtype(rng.uniform(-1.0, 1.0))
        m1 = dtype(rng.uniform(-1.0, 1.0))
        rows.append([x0, x1, v0, v1, p0, p1, m0, m1])
    assert rows, "generator produced no usable geometry"
    return np.asarray(rows, dtype=dtype)


def _queries(rows, rng, dtype):
    """Endpoints, their representable neighbours, and interior points."""
    qs = []
    for x0, x1 in zip(rows[:, 0], rows[:, 1], strict=True):
        qs.extend([x0, x1])
        qs.append(np.nextafter(x0, dtype(np.inf), dtype=dtype))
        qs.append(np.nextafter(x1, dtype(-np.inf), dtype=dtype))
        qs.extend(
            dtype(x0 + dtype(frac) * (x1 - x0)) for frac in (0.5, rng.uniform(0.0, 1.0))
        )
    finite = [q for q in qs if np.isfinite(q)]
    return np.asarray(finite, dtype=dtype)


def _check_rows(rows, dtype):
    """Return (checked, violations) for one geometry set."""
    rng = np.random.default_rng(_SEED)
    flat = _queries(rows, rng, dtype)
    terms = _candidate_terms(
        block=jnp.asarray(rows),
        live=jnp.ones(rows.shape[0], dtype=bool),
        flat=jnp.asarray(flat),
    )
    value_hi = np.asarray(terms.value_hi)
    value_lo = np.asarray(terms.value_lo)
    radius = np.asarray(terms.radius)
    brackets = np.asarray(terms.brackets)

    checked = 0
    violations = []
    for i, q in enumerate(flat):
        for j, row in enumerate(rows):
            if not brackets[i, j]:
                continue
            x0, x1, v0, v1 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            exact = _exact_affine(x0=x0, x1=x1, v0=v0, v1=v1, x=float(q))
            published = Fraction(float(value_hi[i, j])) + Fraction(
                float(value_lo[i, j])
            )
            bound = Fraction(float(radius[i, j]))
            checked += 1
            if abs(exact - published) > bound:
                violations.append(
                    {
                        "x0": x0,
                        "x1": x1,
                        "v0": v0,
                        "v1": v1,
                        "q": float(q),
                        "error_ulp_of_bound": (
                            float(abs(exact - published) / bound) if bound else None
                        ),
                        "abs_error": float(abs(exact - published)),
                        "radius": float(bound),
                    }
                )
    return checked, violations


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_the_certified_radius_contains_the_exact_interpolant(dtype):
    """`|exact - (value_hi + value_lo)| <= radius` on every bracketing lane."""
    rng = np.random.default_rng(_SEED)
    rows = _geometries(rng, np.dtype(jnp.dtype(dtype)).type, 120)
    checked, violations = _check_rows(rows, dtype)
    assert checked > 0, "no bracketing lane was exercised — the check is vacuous"
    assert not violations, (
        f"{len(violations)}/{checked} lanes have an exact value OUTSIDE the "
        f"certified radius; worst: {max(violations, key=lambda v: v['abs_error'])}"
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_the_radius_survives_a_large_common_value_level(dtype):
    """The same geometry lifted onto a common level, where reads least separate.

    Adding a large constant to both endpoint values leaves the exact interpolant
    shifted by exactly that constant but destroys the relative separation of the
    stored reads, which is where a residual-based bound is most easily
    understated.
    """
    scalar = np.dtype(jnp.dtype(dtype)).type
    rng = np.random.default_rng(_SEED + 1)
    rows = _geometries(rng, scalar, 80)
    level = scalar(np.ldexp(1.0, 30))
    lifted = rows.copy()
    lifted[:, 2] = (rows[:, 2] + level).astype(rows.dtype)
    lifted[:, 3] = (rows[:, 3] + level).astype(rows.dtype)
    keep = np.isfinite(lifted[:, 2]) & np.isfinite(lifted[:, 3])
    lifted = lifted[keep]
    assert lifted.shape[0] > 0, "lifting left no finite geometry"

    checked, violations = _check_rows(lifted, dtype)
    assert checked > 0, "no bracketing lane was exercised — the check is vacuous"
    assert not violations, (
        f"{len(violations)}/{checked} lifted lanes fall outside the certified "
        f"radius; worst: {max(violations, key=lambda v: v['abs_error'])}"
    )
