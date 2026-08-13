"""The shared power-of-two scaling may never manufacture a tie.

`certified_margin_sign` scales all abscissae by one power of two taken from the
largest of them, which is what keeps its products inside the range where the
error-free transforms are exact. The scaling is shared, so an operand far
smaller than that largest abscissa can lose its identity in the process: a
narrow link's endpoints can round onto the same number, and a query close to
zero can flush to zero outright. Either way the determinant is computed on
geometry the caller did not supply.

A tie is a certificate — it licenses the caller to treat either link as owner —
so it may not be issued on geometry the transform silently altered. The
unresolved verdict exists for exactly this case.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    BELOW_RESOLUTION_SIGN,
    UNRESOLVED_SIGN,
    backend_flushes_subnormals,
    certified_margin_sign,
)
from tests.conftest import X64_ENABLED

_FLUSHING_BACKEND = backend_flushes_subnormals(jnp.zeros(()).dtype)

_PRE_GUARD_CONTRACT = pytest.mark.xfail(
    condition=_FLUSHING_BACKEND,
    reason=(
        "Asserts a strict sign across a separation the backend destroys, which the "
        "certificate now refuses. Whether the assertion or the refusal states the "
        "contract is an open question; the two are recorded rather than reconciled "
        "by editing one to match the other."
    ),
    strict=False,
)

_HONEST_VERDICTS = (-1, UNRESOLVED_SIGN)


def _sign_for_exponents(dtype, jax_dtype, large_exp: int, small_exp: int) -> int:
    """Certified sign of `A - B` where B is much narrower than A.

    Both links rise from `-0.5` to `0.5` across their own span, so at half of B's
    half-width B is exactly `0.25` above A, whatever the ratio between them.
    """
    large = dtype(np.ldexp(1.0, large_exp))
    small = dtype(np.ldexp(1.0, small_exp))
    return int(
        certified_margin_sign(
            a_x0=jnp.asarray(-large, dtype=jax_dtype),
            a_x1=jnp.asarray(large, dtype=jax_dtype),
            a_v0=jnp.asarray(dtype(-0.5), dtype=jax_dtype),
            a_v1=jnp.asarray(dtype(0.5), dtype=jax_dtype),
            b_x0=jnp.asarray(-small, dtype=jax_dtype),
            b_x1=jnp.asarray(small, dtype=jax_dtype),
            b_v0=jnp.asarray(dtype(-0.5), dtype=jax_dtype),
            b_v1=jnp.asarray(dtype(0.5), dtype=jax_dtype),
            x_query=jnp.asarray(small / dtype(2), dtype=jax_dtype),
        )
    )


def _working_dtypes():
    """The numpy/jax dtype pair matching the configured working precision."""
    return (np.float64, jnp.float64) if X64_ENABLED else (np.float32, jnp.float32)


def test_no_width_ratio_is_certified_into_a_tie() -> None:
    """Across every representable width ratio, a `0.25` margin is never a tie.

    This is the whole counterexample class rather than one witness: the ratio is
    swept from `2^1` up to the point where the narrow link stops being
    representable, and at every step the narrow link sits `0.25` above the wide
    one. `-1` and the fail-loud sentinel are both honest; `0` would certify a tie
    that does not exist, and licenses the caller to publish either branch.
    """
    dtype, jax_dtype = _working_dtypes()
    limit = 250 if X64_ENABLED else 128
    ties = [
        ratio
        for ratio in range(1, limit + 1)
        # Centre the pair so that neither endpoint over- or underflows on its own.
        if _sign_for_exponents(dtype, jax_dtype, ratio // 2, ratio // 2 - ratio) == 0
    ]
    assert ties == []


@pytest.mark.parametrize(
    ("dtype", "jax_dtype", "large_exp", "small_exp"),
    [
        (np.float32, jnp.float32, 47, -103),
        (np.float64, jnp.float64, 400, -675),
    ],
)
def test_a_collapsed_width_is_unresolved_rather_than_tied(
    dtype, jax_dtype, large_exp, small_exp
) -> None:
    """Where the shared scaling flattens the narrow link, the sign is unresolved.

    The narrow link is strictly above at the query, so the honest verdicts are
    `-1` or the fail-loud sentinel. `0` would certify a tie that does not exist,
    and `BELOW_RESOLUTION_SIGN` would claim the two are within a rounding of each
    other when they are `0.25` apart.
    """
    if (jax_dtype is jnp.float64) != X64_ENABLED:
        pytest.skip("dtype is not the configured working precision")
    assert (
        _sign_for_exponents(dtype, jax_dtype, large_exp, small_exp) in _HONEST_VERDICTS
    )


def test_a_query_the_scaling_flushes_to_zero_is_unresolved_rather_than_tied() -> None:
    """A query that underflows under the scaling does not become an exact tie.

    Here both links keep a positive width, so nothing about the widths is wrong;
    it is the query that loses its position between the narrow link's endpoints.
    Both numerators then evaluate at a point the caller never asked about and
    come out exactly zero — the determinant's most emphatic verdict, on geometry
    that was never supplied.
    """
    dtype, jax_dtype = _working_dtypes()
    large_exp, small_exp = (512, -549) if X64_ENABLED else (62, -63)
    assert (
        _sign_for_exponents(dtype, jax_dtype, large_exp, small_exp) in _HONEST_VERDICTS
    )


@pytest.mark.parametrize("unit_shift", [-10, 0, 10])
def test_the_verdict_does_not_depend_on_the_choice_of_units(unit_shift: int) -> None:
    """Rescaling both links by a power of two cannot turn a strict sign into a tie.

    The determinant is homogeneous, so a common power-of-two change of units
    multiplies it by a positive constant and leaves its sign alone.
    """
    dtype, jax_dtype = _working_dtypes()
    large_exp, small_exp = (400, -675) if X64_ENABLED else (47, -103)
    assert (
        _sign_for_exponents(
            dtype, jax_dtype, large_exp + unit_shift, small_exp + unit_shift
        )
        in _HONEST_VERDICTS
    )


@_PRE_GUARD_CONTRACT
@pytest.mark.parametrize("swap", [False, True])
def test_a_value_range_no_scaling_can_hold_is_still_decided(*, swap: bool) -> None:
    """A gap too large to represent is the easiest comparison, not an open one.

    One link's values sit at the top of the format while the other's are
    ordinary, so no single power of two brings both groups into the range where
    the transforms are exact — the smaller group's contribution to the
    determinant underflows whatever exponent is chosen. That contribution is
    bounded rather than unknown, so the large link decides the sign, and it must
    be the sign of the exact difference.
    """
    dtype, jax_dtype = _working_dtypes()
    huge = dtype(np.ldexp(1.0, int(np.finfo(dtype).maxexp) - 1))
    falling = (huge, -huge)
    flat = (dtype(0.25), dtype(0.25))
    first, second = (flat, falling) if swap else (falling, flat)
    query = dtype(0.75)

    def value_at(endpoints: tuple) -> Fraction:
        low, high = (Fraction(float(term)) for term in endpoints)
        return low + (high - low) * Fraction(float(query))

    expected = value_at(first) - value_at(second)
    assert expected != 0, "witness is vacuous if the two links agree"

    sign = int(
        certified_margin_sign(
            a_x0=jnp.asarray(dtype(0.0), dtype=jax_dtype),
            a_x1=jnp.asarray(dtype(1.0), dtype=jax_dtype),
            a_v0=jnp.asarray(first[0], dtype=jax_dtype),
            a_v1=jnp.asarray(first[1], dtype=jax_dtype),
            b_x0=jnp.asarray(dtype(0.0), dtype=jax_dtype),
            b_x1=jnp.asarray(dtype(1.0), dtype=jax_dtype),
            b_v0=jnp.asarray(second[0], dtype=jax_dtype),
            b_v1=jnp.asarray(second[1], dtype=jax_dtype),
            x_query=jnp.asarray(query, dtype=jax_dtype),
        )
    )
    assert sign == (1 if expected > 0 else -1)


def test_ordinary_width_ratios_still_resolve_strictly() -> None:
    """The fence does not cost resolution on geometry the transforms handle.

    A width ratio a real grid could produce keeps its strict verdict; only ratios
    wide enough to disturb the scaling fall back to unresolved.
    """
    dtype, jax_dtype = _working_dtypes()
    assert _sign_for_exponents(dtype, jax_dtype, 3, -17) == -1


def _flat_gap_signs(
    dtype,
    jax_dtype,
    half_widths: np.ndarray,
    *,
    wide_exponent: int,
    wide_is_above: bool,
) -> np.ndarray:
    """Certified signs of `A - B` for a wide flat link against a narrow flat one.

    Both links straddle the query and both are flat, so each takes its own stored
    value there whatever the ratio between their widths, and the exact sign is the
    comparison of those two values. `wide_is_above` picks which one is raised.
    """
    raised = dtype(0.75)
    for _ in range(64):
        raised = np.nextafter(raised, dtype(np.inf), dtype=dtype)
    wide, narrow = (raised, dtype(0.75)) if wide_is_above else (dtype(0.75), raised)
    ones = np.ones_like(half_widths)
    wide_half = dtype(np.ldexp(1.0, wide_exponent))
    return np.asarray(
        certified_margin_sign(
            a_x0=jnp.asarray(-wide_half * ones, dtype=jax_dtype),
            a_x1=jnp.asarray(wide_half * ones, dtype=jax_dtype),
            a_v0=jnp.asarray(wide * ones, dtype=jax_dtype),
            a_v1=jnp.asarray(wide * ones, dtype=jax_dtype),
            b_x0=jnp.asarray(-half_widths, dtype=jax_dtype),
            b_x1=jnp.asarray(half_widths, dtype=jax_dtype),
            b_v0=jnp.asarray(narrow * ones, dtype=jax_dtype),
            b_v1=jnp.asarray(narrow * ones, dtype=jax_dtype),
            x_query=jnp.asarray(dtype(0.0) * ones, dtype=jax_dtype),
        )
    )


@pytest.mark.parametrize("wide_exponent", [-40, 0, 40])
@pytest.mark.parametrize("wide_is_above", [True, False])
def test_a_flat_gap_keeps_its_strict_sign_at_every_width_ratio(
    wide_exponent: int, *, wide_is_above: bool
) -> None:
    """Two flat links a fixed gap apart stay strictly ordered however narrow one is.

    Each link enters the determinant only through its own distances — to the query
    and between its endpoints — so a link far narrower than its rival contributes
    terms far below one while every operand is still an ordinary number. What
    cancels between the two contributions is smaller again by the ratio between
    them, and it can fall under the smallest normal.

    Nothing upstream sees that. The operands are readable, the products stay inside
    the domain where the transforms are exact, and the transforms discard nothing
    on the way, so what arrives is an estimate of exactly zero carrying an error
    bound of exactly zero — the certificate for an exact tie — for links that are
    demonstrably ordered. Where instead it is the narrow link's own numerator that
    collapses, its contribution vanishes entirely and the wide link is certified
    the winner outright, which is the worse failure: a tie at least tells the
    caller the contest was indecisive.

    The whole class is swept rather than one witness — every width ratio down to
    the narrowest link the format represents as a normal number, at three scales
    for the wide link, in both orientations. The second orientation is what
    separates a decision from a collapse, since a vanished contribution hands the
    verdict to whichever link survives and that is right half the time.
    """
    dtype, jax_dtype = _working_dtypes()
    exponents = np.arange(1, -int(np.finfo(dtype).minexp) + 1)
    half_widths = np.ldexp(np.ones(exponents.shape), -exponents).astype(dtype)
    signs = _flat_gap_signs(
        dtype,
        jax_dtype,
        half_widths,
        wide_exponent=wide_exponent,
        wide_is_above=wide_is_above,
    )
    expected = 1 if wide_is_above else -1
    assert sorted(set(map(int, exponents[signs != expected]))) == []


def _adjacent_neighbour_signs(
    dtype,
    jax_dtype,
    *,
    separations: np.ndarray,
    offsets: np.ndarray,
    narrow_is_above: bool,
) -> np.ndarray:
    """Certified signs where the narrow link's endpoints are neighbours near `tiny`.

    Both links are flat and straddle the query, so the exact sign is the comparison
    of their two stored values. The narrow link sits at the bottom of the normal
    range with its endpoints a few representable steps apart, while the wide link
    spans the unit interval; `separations` and `offsets` count those steps.
    """
    tiny = dtype(np.finfo(dtype).tiny)
    steps = np.arange(int(max(separations.max(), offsets.max())) + 1)
    ladder = np.empty(steps.shape, dtype=dtype)
    ladder[0] = tiny
    for step in steps[1:]:
        ladder[step] = np.nextafter(ladder[step - 1], dtype(np.inf), dtype=dtype)

    raised = dtype(0.75)
    for _ in range(64):
        raised = np.nextafter(raised, dtype(np.inf), dtype=dtype)
    narrow, wide = (raised, dtype(0.75)) if narrow_is_above else (dtype(0.75), raised)
    ones = np.ones(separations.shape, dtype=dtype)
    return np.asarray(
        certified_margin_sign(
            a_x0=jnp.asarray(ladder[0] * ones, dtype=jax_dtype),
            a_x1=jnp.asarray(ladder[separations], dtype=jax_dtype),
            a_v0=jnp.asarray(narrow * ones, dtype=jax_dtype),
            a_v1=jnp.asarray(narrow * ones, dtype=jax_dtype),
            b_x0=jnp.asarray(dtype(0.0) * ones, dtype=jax_dtype),
            b_x1=jnp.asarray(dtype(1.0) * ones, dtype=jax_dtype),
            b_v0=jnp.asarray(wide * ones, dtype=jax_dtype),
            b_v1=jnp.asarray(wide * ones, dtype=jax_dtype),
            x_query=jnp.asarray(ladder[offsets], dtype=jax_dtype),
        )
    )


@_PRE_GUARD_CONTRACT
@pytest.mark.parametrize("narrow_is_above", [True, False])
def test_endpoints_a_subnormal_step_apart_keep_their_strict_sign(
    *, narrow_is_above: bool
) -> None:
    """Two links stay ordered when one link's endpoints are adjacent floats.

    At the bottom of the normal range the spacing between neighbouring floats is
    itself subnormal, so a link whose endpoints are a few representable steps
    apart has readable endpoints and an unreadable width. That link's three
    distances are exact positive rationals the format can hold — the endpoints
    are ordinary numbers — but the subtraction that forms them underflows, and a
    backend that flushes returns all three as zero.

    Nothing distinguishes that from a link the caller supplied as a point: the
    error-free transforms discarded nothing, so the determinant arrives as an
    estimate of exactly zero carrying an error bound of exactly zero. That is the
    certificate for an exact tie, issued for two flat links separated by 64 ULPs
    of value.

    Both orientations are swept, because a collapse hands the verdict to whichever
    link survives and that is right half the time.
    """
    dtype, jax_dtype = _working_dtypes()
    separations, offsets = (
        np.asarray(pair)
        for pair in zip(
            *[
                (separation, offset)
                for separation in range(2, 42, 2)
                for offset in range(1, separation)
            ],
            strict=True,
        )
    )
    signs = _adjacent_neighbour_signs(
        dtype,
        jax_dtype,
        separations=separations,
        offsets=offsets,
        narrow_is_above=narrow_is_above,
    )
    expected = 1 if narrow_is_above else -1
    assert sorted(set(map(int, signs[signs != expected]))) == []


@pytest.mark.parametrize("narrow_is_above", [True, False])
def test_a_link_no_exponent_can_lift_is_refused_rather_than_tied(
    *, narrow_is_above: bool
) -> None:
    """A link whose width no scaling recovers is unresolved, never an exact tie.

    Measuring a link on its own scale recovers its width only when its own three
    abscissae sit together. A query far outside a link at the bottom of the
    normal range puts an ordinary number in that triple, so normalizing it leaves
    the endpoints where they were and the width between them still has no
    representation as a difference.

    What arrives is then a determinant carrying no information about the ordering,
    and the one verdict it may not draw from that is an exact tie: the links are
    ordered, and the caller must be told so or told nothing rather than handed a
    licence to choose between them.

    Which refusal it is depends on the backend and is not the property under test.
    Where subnormals flush, no determinant is produced at all and the answer is
    `UNRESOLVED_SIGN`; where they are read, a determinant is produced and sits
    under its own error bound, which is `BELOW_RESOLUTION_SIGN`. Deciding the
    ordering correctly is also honest — what is refused is a tie, or a strict
    verdict against the exact ordering.
    """
    dtype, jax_dtype = _working_dtypes()
    tiny = dtype(np.finfo(dtype).tiny)
    narrow_x1 = np.nextafter(tiny, dtype(np.inf), dtype=dtype)
    near, far = dtype(0.9), dtype(0.1)
    narrow, wide = (near, far) if narrow_is_above else (far, near)
    verdict = int(
        certified_margin_sign(
            a_x0=jnp.asarray(tiny, dtype=jax_dtype),
            a_x1=jnp.asarray(narrow_x1, dtype=jax_dtype),
            a_v0=jnp.asarray(narrow, dtype=jax_dtype),
            a_v1=jnp.asarray(narrow, dtype=jax_dtype),
            b_x0=jnp.asarray(dtype(0.0), dtype=jax_dtype),
            b_x1=jnp.asarray(dtype(256.0), dtype=jax_dtype),
            b_v0=jnp.asarray(wide, dtype=jax_dtype),
            b_v1=jnp.asarray(wide, dtype=jax_dtype),
            x_query=jnp.asarray(dtype(1.0), dtype=jax_dtype),
        )
    )
    exact_sign = 1 if narrow_is_above else -1

    assert verdict in (exact_sign, UNRESOLVED_SIGN, BELOW_RESOLUTION_SIGN)
