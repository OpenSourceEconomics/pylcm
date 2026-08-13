"""A link far narrower than its rival keeps its strict order against it.

The certificate may read a comparison, or refuse it, but it may never report an
exact tie between two lines whose exact values at the query differ. This module
states that as a *property* of the inputs rather than as a construction, and
covers the two faces of the same class:

- **a separation that arrives subnormal.** Every abscissa is normal by its
  exponent bits, while the exact difference between two endpoints of one link is
  a positive subnormal, so a backend that flushes subnormals loses the whole
  distance before anything else happens to it.
- **a separation that arrives perfectly ordinary.** Nothing is near the bottom of
  the range — the narrow link's endpoints and their difference are all ordinary
  normal numbers — but the rival is wide enough that a scale chosen for the pair
  drives that difference under the smallest normal. This is the wider face: what
  the class needs is the *ratio* between the widest operand and the narrowest
  difference, not a subnormal input.

Witnesses are reached by stepping raw bit patterns, and the ground truth is
computed in exact rational arithmetic, so neither the witness nor the oracle
shares machinery with the predicate under test.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from tests.conftest import X64_ENABLED

type Geometry = tuple[np.floating, np.floating]


def _working_dtypes() -> tuple[type[np.floating], jnp.dtype]:
    return (np.float64, jnp.float64) if X64_ENABLED else (np.float32, jnp.float32)


def _uint_view(dtype: type[np.floating]) -> type[np.unsignedinteger]:
    return np.uint64 if dtype is np.float64 else np.uint32


def _step_bits(
    value: np.floating, *, steps: int, dtype: type[np.floating]
) -> np.floating:
    """Move `value` by `steps` representable places using its bit pattern.

    A negative step is what walks the query below the link, and `uint(steps)`
    cannot represent one. The arithmetic therefore runs on a Python integer,
    which is unbounded and so cannot wrap at either end of the unsigned range.
    """
    uint = _uint_view(dtype)
    bits = int(np.asarray(dtype(value)).view(uint)) + steps
    return dtype(np.asarray(uint(bits)).view(dtype)[()])


def _is_normal(value: np.floating, *, dtype: type[np.floating]) -> bool:
    return bool(abs(float(value)) >= float(np.finfo(dtype).tiny))


def _separation_is_subnormal(
    *, low: np.floating, high: np.floating, dtype: type[np.floating]
) -> bool:
    """The exact gap is positive and below the smallest normal of the format."""
    gap = Fraction(float(high)) - Fraction(float(low))
    return 0 < gap < Fraction(float(np.finfo(dtype).tiny))


def _exact_value_at(
    *,
    x0: np.floating,
    x1: np.floating,
    v0: np.floating,
    v1: np.floating,
    query: np.floating,
) -> Fraction:
    """The affine line through the two stored nodes, evaluated exactly."""
    fx0, fx1, fv0, fv1, fq = (Fraction(float(term)) for term in (x0, x1, v0, v1, query))
    return (fv0 * (fx1 - fq) + fv1 * (fq - fx0)) / (fx1 - fx0)


def _narrow_links(dtype: type[np.floating]) -> list[Geometry]:
    """Links `(x0, x1)` of normal endpoints separated by a subnormal gap.

    The base of each link is a power of two stepped up from `tiny`, so the sweep
    walks the bottom binades where a representable step is still smaller than the
    smallest normal. `x1` is reached by adding to the raw significand bits.
    """
    tiny = dtype(np.finfo(dtype).tiny)
    significand_bits = 52 if dtype is np.float64 else 23
    links: list[Geometry] = []
    for binade in range(0, significand_bits, 4):
        base = dtype(np.ldexp(float(tiny), binade))
        for separation in (2, 3, 5, 8, 17, 64):
            upper = _step_bits(base, steps=separation, dtype=dtype)
            if not (_is_normal(base, dtype=dtype) and _is_normal(upper, dtype=dtype)):
                continue
            if not _separation_is_subnormal(low=base, high=upper, dtype=dtype):
                continue
            links.append((base, upper))
    return links


def _wide_half_widths(dtype: type[np.floating]) -> list[np.floating]:
    """Magnitudes for the rival link, spanning the ones that pin a shared exponent."""
    exponents = (0, -20, -100) if dtype is np.float64 else (0, -10, -40)
    return [dtype(np.ldexp(1.0, exponent)) for exponent in exponents]


def _query_for(
    *, x0: np.floating, x1: np.floating, placement: str, dtype: type[np.floating]
) -> np.floating | None:
    """A query inside the link, or outside it on either side."""
    separation = int(
        np.asarray(dtype(x1)).view(_uint_view(dtype))
        - np.asarray(dtype(x0)).view(_uint_view(dtype))
    )
    match placement:
        case "inside":
            query = _step_bits(x0, steps=max(1, separation // 2), dtype=dtype)
        case "below":
            query = _step_bits(x0, steps=-max(1, separation), dtype=dtype)
        case "above":
            query = _step_bits(x1, steps=max(1, separation), dtype=dtype)
        case _:  # pragma: no cover - guarded by parametrization
            raise AssertionError(placement)
    return query if _is_normal(query, dtype=dtype) else None


def _certified(
    *,
    wide_half: np.floating,
    wide_value: np.floating,
    narrow: Geometry,
    narrow_value: np.floating,
    query: np.floating,
    jax_dtype: jnp.dtype,
) -> int:
    x0, x1 = narrow
    return int(
        certified_margin_sign(
            a_x0=jnp.asarray(-wide_half, dtype=jax_dtype),
            a_x1=jnp.asarray(wide_half, dtype=jax_dtype),
            a_v0=jnp.asarray(wide_value, dtype=jax_dtype),
            a_v1=jnp.asarray(wide_value, dtype=jax_dtype),
            b_x0=jnp.asarray(x0, dtype=jax_dtype),
            b_x1=jnp.asarray(x1, dtype=jax_dtype),
            b_v0=jnp.asarray(narrow_value, dtype=jax_dtype),
            b_v1=jnp.asarray(narrow_value, dtype=jax_dtype),
            x_query=jnp.asarray(query, dtype=jax_dtype),
        )
    )


def _ordinary_links(dtype: type[np.floating]) -> list[Geometry]:
    """Links whose endpoints *and* separation are ordinary normal numbers.

    Each base is a power of two well clear of the bottom of the range, and `x1`
    is eight representable steps above it, so the separation is normal by many
    binades. Nothing here is near a subnormal on input.
    """
    significand_bits = 52 if dtype is np.float64 else 23
    smallest_normal_exponent = int(np.finfo(dtype).minexp)
    exponents = (
        (0, -100, -300, -600, -900) if dtype is np.float64 else (0, -20, -50, -80)
    )
    links: list[Geometry] = []
    for exponent in exponents:
        base = dtype(np.ldexp(1.0, exponent))
        upper = _step_bits(base, steps=8, dtype=dtype)
        separation = Fraction(float(upper)) - Fraction(float(base))
        if (
            separation <= 0
            or exponent - significand_bits + 3 <= smallest_normal_exponent
        ):
            continue
        links.append((base, upper))
    return links


def _widest_rival_half_width(dtype: type[np.floating]) -> np.floating:
    """A rival link spanning close to the whole representable range."""
    return dtype(np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 5))


@pytest.mark.parametrize("placement", ["inside", "below", "above"])
@pytest.mark.parametrize("narrow_is_higher", [False, True])
def test_a_wide_rival_never_certifies_a_tie_against_an_ordinary_narrow_link(
    placement: str,
    narrow_is_higher: bool,  # noqa: FBT001
) -> None:
    """No input is subnormal, and a strict exact ordering still survives.

    The narrow link's separation is an ordinary normal number, so this holds the
    class open at its wider face: a scale chosen for the widest operand can drive
    a healthy difference under the smallest normal, and the verdict must still be
    the exact one or a refusal.
    """
    dtype, jax_dtype = _working_dtypes()
    lower = dtype(0.75)
    higher = _step_bits(lower, steps=64, dtype=dtype)
    wide_value, narrow_value = (lower, higher) if narrow_is_higher else (higher, lower)
    wide_half = _widest_rival_half_width(dtype)
    tiny = float(np.finfo(dtype).tiny)

    violations: list[str] = []
    examined = 0
    for x0, x1 in _ordinary_links(dtype):
        query = _query_for(x0=x0, x1=x1, placement=placement, dtype=dtype)
        if query is None:
            continue
        separation = Fraction(float(x1)) - Fraction(float(x0))
        assert separation >= Fraction(tiny), (
            "witness is vacuous if the separation arrives subnormal"
        )
        exact_wide = _exact_value_at(
            x0=-wide_half, x1=wide_half, v0=wide_value, v1=wide_value, query=query
        )
        exact_narrow = _exact_value_at(
            x0=x0, x1=x1, v0=narrow_value, v1=narrow_value, query=query
        )
        if exact_wide == exact_narrow:
            continue
        expected = 1 if exact_wide > exact_narrow else -1
        examined += 1
        sign = _certified(
            wide_half=wide_half,
            wide_value=wide_value,
            narrow=(x0, x1),
            narrow_value=narrow_value,
            query=query,
            jax_dtype=jax_dtype,
        )
        if sign not in (expected, UNRESOLVED_SIGN):
            kind = "false exact tie" if sign == 0 else "wrong strict sign"
            violations.append(
                f"{kind}: narrow=[{float(x0):.17g}, {float(x1):.17g}] "
                f"separation={float(separation):.6g} "
                f"wide_half={float(wide_half):.17g} query={float(query):.17g} "
                f"expected={expected:+d} got={sign:+d}"
            )

    assert examined > 0, "the sweep produced no witness satisfying the property"
    assert not violations, (
        f"{len(violations)} of {examined} comparisons decided against the exact "
        f"ordering:\n" + "\n".join(violations[:12])
    )


@pytest.mark.parametrize("placement", ["inside", "below", "above"])
@pytest.mark.parametrize("narrow_is_higher", [False, True])
def test_subnormal_separation_never_certifies_a_tie_between_ordered_lines(
    placement: str,
    narrow_is_higher: bool,  # noqa: FBT001
) -> None:
    """A strict exact ordering is certified correctly or refused, never tied."""
    dtype, jax_dtype = _working_dtypes()
    lower = dtype(0.75)
    higher = _step_bits(lower, steps=64, dtype=dtype)
    wide_value, narrow_value = (lower, higher) if narrow_is_higher else (higher, lower)

    violations: list[str] = []
    examined = 0
    for narrow in _narrow_links(dtype):
        x0, x1 = narrow
        query = _query_for(x0=x0, x1=x1, placement=placement, dtype=dtype)
        if query is None:
            continue
        for wide_half in _wide_half_widths(dtype):
            exact_wide = _exact_value_at(
                x0=-wide_half, x1=wide_half, v0=wide_value, v1=wide_value, query=query
            )
            exact_narrow = _exact_value_at(
                x0=x0, x1=x1, v0=narrow_value, v1=narrow_value, query=query
            )
            if exact_wide == exact_narrow:
                continue
            expected = 1 if exact_wide > exact_narrow else -1
            examined += 1
            sign = _certified(
                wide_half=wide_half,
                wide_value=wide_value,
                narrow=narrow,
                narrow_value=narrow_value,
                query=query,
                jax_dtype=jax_dtype,
            )
            if sign not in (expected, UNRESOLVED_SIGN):
                kind = "false exact tie" if sign == 0 else "wrong strict sign"
                violations.append(
                    f"{kind}: narrow=[{float(x0):.17g}, {float(x1):.17g}] "
                    f"wide_half={float(wide_half):.17g} query={float(query):.17g} "
                    f"expected={expected:+d} got={sign:+d}"
                )

    assert examined > 0, "the sweep produced no witness satisfying the property"
    assert not violations, (
        f"{len(violations)} of {examined} comparisons decided against the exact "
        f"ordering:\n" + "\n".join(violations[:12])
    )
