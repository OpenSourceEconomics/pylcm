"""How far an NBEGM solve's published affine reads stay from the format's floor.

An affine read forms `v0*(x1 - x) + v1*(x - x0)` and divides it by the link width to
publish one of the value/policy/marginal channels. Where that quotient's exact result
lands among the subnormals, the two arithmetics part company. The ordinary reader
divides in the target format, and XLA:CPU cannot hold the result: every operand is
normal, the true answer is genuinely below the smallest normal, and what is published
is a zero the caller cannot distinguish from an earned one. The certified reader forms
the same quotient as an exact signed rational and rounds once, so it publishes the
subnormal band as values -- there a subnormal is an answer, not a refusal.

So this bound is not a claim that the band is unpublishable. It is a claim about
*distance*: how close an ordinary calibration brings the model's published reads to
the floor, and therefore how much margin stands between this model and the arithmetic
that cannot publish there. The margin is what makes `arithmetic` a free choice rather
than a correctness-relevant one on this model, which is why it is worth pinning even
while the solve runs certified. The measured headroom on the certified reader is 35.2
orders of magnitude, over 9 calls whose closest published quotient sits at 2^-9 against
a floor of 2^-126; the assertion demands only ten, far enough below the observation to
stay quiet under grid and solver churn and still fail the day the distance genuinely
collapses.

**Which reader this instruments is not cosmetic, and it has already moved once.**
`envelope_at_query` returns `_envelope_exact(...).published` outright under
`arithmetic="certified"`; only the ordinary path descends through `_along_link` to
`affine_numerator`. This test first wrapped `affine_numerator` and went silently
unreached the moment the certified reader took ownership of the read -- caught only
because the reachability guard below refused to state a bound over zero calls. It now
wraps `exact_affine_read`, which is where the published quotient is formed on the
route this model actually takes.

Three things this test is careful about, each of which cost a real measurement first:

- **The gate must be shown to be reached.** A minimum over zero calls is not a bound,
  it is an untouched instrument, and it reads exactly like a comfortable result. The
  DC-EGM models never take this path at all, so pointing the instrument at one returns
  a clean and entirely meaningless zero. The fixture fires the wrapper *before* the
  solve and asserts it was seen, then clears the counters. This guard is the primary
  assertion, not a precondition for one: a route change must fail loudly here rather
  than pass vacuously below.

- **The measurement must not perform the operation it is watching for.** Forming
  `numerator / width` to see whether it underflows evaluates the very division that
  underflows -- the result is flushed, and a flushed quotient is indistinguishable
  from a genuinely zero one. `frexp` reports each operand's exponent without dividing,
  so the quotient's exponent is a *difference of exponents* and cannot underflow.
  The float64 host cross-check that established this -- 31.9 orders reported here
  against 32.4 measured there, conservative by under two exponents, which is the
  direction a lower bound must err in -- was run against `affine_numerator`, the
  reader this test no longer watches. It is quoted as evidence about `frexp`, which
  is unchanged, and not as a figure for the certified route; that route's number is
  the 35.2 above and has not been cross-checked against a host recomputation.

  `frexp` has its own blind spot, and it happens to fall on the safe side here. Given
  a *subnormal* operand it does not report that operand's exponent: every subnormal
  returns the format's minimum (-149 at fp32, -1074 at fp64), disagreeing with the
  host by up to twenty exponents, silently and with a plausible-looking integer. That
  understates the numerator, which understates the quotient, which understates the
  headroom -- so it can only make the assertion below fire earlier, never later. The
  same primitive used to *upper*-bound a magnitude would be unsafe in exactly the way
  this use is safe, so anywhere else it appears the direction has to be re-argued
  rather than inherited from here.

- **The precision is pinned, and pinned to the tighter case.** The suite defaults to
  `--precision=64`, where the smallest normal is ~2.2e-308 and this band is far
  further away; fp32 is both the harder test and what an unconfigured user gets.
  Pinning also keeps the runtime honest -- the same solve costs seconds at fp32 and
  about an hour at fp64, and an hour is past the suite's own ceiling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope import query
from lcm.typing import FloatND, IntND
from tests.test_models import nbegm_two_cliff_toy as toy

# The observed headroom is ~32 orders. Demanding ten leaves the assertion insensitive
# to ordinary retuning of the grids while still catching a collapse toward the band.
MIN_ORDERS_ABOVE_TINY = 10
_LOG10_2 = float(np.log10(2.0))


class _Reads:
    """Closest approach to the subnormal band, in exponents, over one solve."""

    def __init__(self) -> None:
        self.calls = 0
        self.min_exponent = np.inf

    def clear(self) -> None:
        self.calls = 0
        self.min_exponent = np.inf


def _quotient_exponent(*, numerator: FloatND, width: FloatND) -> FloatND:
    """Lower-bound `log2|numerator / width|` without ever forming the quotient.

    `frexp` splits each operand into a mantissa in [0.5, 1) and an exponent, so the
    quotient's exponent is the difference of the two exponents, give or take the one
    bit the mantissa ratio can carry. Subtracting that bit keeps the result a lower
    bound, which is the safe direction: it can only understate the headroom.
    """
    _, exponent_numerator = jnp.frexp(numerator)
    _, exponent_width = jnp.frexp(width)
    return (exponent_numerator - exponent_width - 1).astype(jnp.int32)


@pytest.fixture(name="instrumented_reads")
def _fixture_instrumented_reads(
    *,
    monkeypatch: pytest.MonkeyPatch,
    x64_disabled: None,  # noqa: ARG001
) -> _Reads:
    """Wrap `exact_affine_read`, prove it fires, then hand back cleared counters."""
    reads = _Reads()
    real = query.exact_affine_read

    def observe(exponent: FloatND) -> None:
        reads.calls += 1
        reads.min_exponent = min(reads.min_exponent, int(np.asarray(exponent)))

    def wrapped(
        *, x0: FloatND, x1: FloatND, v0: FloatND, v1: FloatND, x_query: FloatND
    ) -> tuple[FloatND, IntND]:
        out = real(x0=x0, x1=x1, v0=v0, v1=v1, x_query=x_query)
        numerator = v0 * (x1 - x_query) + v1 * (x_query - x0)
        width = x1 - x0
        usable = jnp.isfinite(numerator) & jnp.isfinite(width) & (width != 0.0)
        usable &= numerator != 0.0
        exponent = jnp.where(
            usable, _quotient_exponent(numerator=numerator, width=width), jnp.int32(0)
        )
        # One scalar per call, not one array. An earlier version shipped every cell
        # to the host and took 59 minutes; the bound it produced was the same.
        jax.debug.callback(observe, jnp.min(exponent))
        return out

    monkeypatch.setattr(query, "exact_affine_read", wrapped)

    # POSITIVE CONTROL. Without this, a solve that never reaches the gate would
    # satisfy every assertion below while measuring nothing whatsoever.
    dtype = jnp.zeros(()).dtype
    probe = wrapped(
        x0=jnp.asarray(0.0, dtype),
        x1=jnp.asarray(2.0, dtype),
        v0=jnp.asarray(1.0, dtype),
        v1=jnp.asarray(3.0, dtype),
        x_query=jnp.asarray(1.0, dtype),
    )
    jax.block_until_ready(probe)
    assert reads.calls >= 1, "the instrument does not observe `exact_affine_read`"
    reads.clear()
    return reads


def test_no_published_affine_read_approaches_the_subnormal_band(
    instrumented_reads: _Reads,
) -> None:
    """An NBEGM solve publishes nothing within ten orders of the smallest normal."""
    finfo = jnp.finfo(jnp.zeros(()).dtype)
    assert finfo.bits == 32, "this bound is stated at fp32; see the module docstring"

    model = toy.build_model(variant="nbegm")
    solution = model.solve(params=toy.build_params(), log_level="warning")
    jax.block_until_ready(solution)

    reads = instrumented_reads
    assert reads.calls > 0, (
        "the solve never reached `exact_affine_read`, so the bound below would be a "
        "statement about an instrument rather than about the model; if the published "
        "read has moved again, re-point the wrapper rather than relaxing this"
    )

    orders = (reads.min_exponent - finfo.minexp) * _LOG10_2
    assert orders > MIN_ORDERS_ABOVE_TINY, (
        f"closest published affine read is 2^{reads.min_exponent}, only "
        f"{orders:.1f} orders above the smallest normal 2^{finfo.minexp} "
        f"({finfo.tiny:.3e}); this model has always kept a "
        f"{MIN_ORDERS_ABOVE_TINY}-order margin, so the band is now reachable from "
        "an ordinary calibration and the publication contract matters here"
    )
