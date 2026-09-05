"""Agreement of the streamed NB-EGM interval read-and-fold with the one-shot merge.

The algebra and standalone reduction tests prove the winner monoid independently.
These maintainer-owned tests use the project runtime and the built exact-affine
payload to check that the production step publishes the same envelope for one-shot
and streamed interval layouts at every representative partition width.

`interval_batch_size` partitions a computation it does not change: the same
candidates are compared under the same total order, and the standing winner
re-enters each block under its global stored-link index. What a partition does
change is the vmap width each block is compiled for, so the two routes can name the
same real number with adjacent bit patterns. Ownership is therefore asserted
exactly — the global stored-link identity that owns each node, and with it the
published feasible set, both of which a mis-ordered fold moves by a finite amount —
and the published levels in units of the working format's spacing. The normal
precision fixture runs this module at float64 and with ``--precision=32``.
"""

from collections.abc import Callable
from functools import cache
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_step import nbegm_per_interval_continuation_step_savings
from _lcm.egm.preferences import Preferences
from _lcm.egm.upper_envelope._exact_affine.ffi import (
    kernel_built_for_current_backend,
)
from _lcm.egm.upper_envelope.query import NO_OWNER, ComparisonArithmetic
from lcm.typing import Float1D, FloatND, IntND, ScalarFloat
from tests.conftest import assert_agrees_to_ulp
from tests.solution._crra_preferences import crra_preferences

_N_INTERVALS = 7
_N_SAVINGS = 36
_N_LIQUID = 31

_CHANNELS = ("value", "marginal", "policy")

# The spacing budget a change of compiled vmap width may spend. It is the constant
# this repository already uses for a partition knob, and it is two orders of
# magnitude below the smallest finite gap a mis-ordered fold would open.
_PARTITION_ULP = 16


class _Geometry(TypedDict):
    """The interval budget the step solves, apart from its continuation rows."""

    liquid_grid: Float1D
    savings_grid: Float1D
    discount_factor: ScalarFloat
    preferences: Preferences
    coh_slopes: Float1D
    coh_intercepts: Float1D
    breakpoints: Float1D


def _geometry() -> _Geometry:
    return {
        "liquid_grid": jnp.linspace(0.1, 32.0, _N_LIQUID),
        "savings_grid": jnp.linspace(0.0, 29.0, _N_SAVINGS),
        "discount_factor": jnp.asarray(0.96),
        "preferences": crra_preferences(crra=2.0),
        "coh_slopes": jnp.linspace(0.95, 1.35, _N_INTERVALS),
        "coh_intercepts": jnp.linspace(0.4, 2.1, _N_INTERVALS),
        "breakpoints": jnp.asarray([1.8, 5.2, 9.1, 14.7, 21.3, 27.4]),
    }


def _continuation() -> tuple[FloatND, FloatND]:
    """The continuation rows, asserted finite in the format that will be used."""
    shift = jnp.linspace(0.0, 0.9, _N_INTERVALS)[:, None]
    cont_value = -1.0 / jnp.linspace(0.45, 5.5, _N_SAVINGS)[None, :] + shift
    cont_marginal = jnp.linspace(2.4, 0.04, _N_SAVINGS)[None, :] + 0.07 * shift
    assert bool(jnp.isfinite(cont_value).all())
    assert bool(jnp.isfinite(cont_marginal).all())
    return cont_value, cont_marginal


def _one_shot(
    *, cont_value: FloatND, cont_marginal: FloatND, arithmetic: ComparisonArithmetic
) -> tuple[FloatND, ...]:
    return nbegm_per_interval_continuation_step_savings(
        **_geometry(),
        cont_value=cont_value,
        cont_marginal=cont_marginal,
        arithmetic=arithmetic,
        interval_batch_size=0,
        return_owner=True,
    )


def _streamed(
    *,
    cont_value: FloatND,
    cont_marginal: FloatND,
    arithmetic: ComparisonArithmetic,
    interval_batch_size: int,
) -> tuple[FloatND, ...]:

    def read(interval_indices: IntND) -> tuple[FloatND, FloatND]:
        return cont_value[interval_indices], cont_marginal[interval_indices]

    return nbegm_per_interval_continuation_step_savings(
        **_geometry(),
        cont_value=None,
        cont_marginal=None,
        arithmetic=arithmetic,
        interval_block_reader=read,
        interval_batch_size=interval_batch_size,
        return_owner=True,
    )


@cache
def _published(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> tuple[np.ndarray, ...]:
    """Publish the three channels and the owner once per (arithmetic, width)."""
    cont_value, cont_marginal = _continuation()
    if interval_batch_size == 0:
        solve: Callable[..., tuple[FloatND, ...]] = jax.jit(
            lambda value, marginal: _one_shot(
                cont_value=value, cont_marginal=marginal, arithmetic=arithmetic
            )
        )
    else:
        solve = jax.jit(
            lambda value, marginal: _streamed(
                cont_value=value,
                cont_marginal=marginal,
                arithmetic=arithmetic,
                interval_batch_size=interval_batch_size,
            )
        )
    return tuple(np.asarray(channel) for channel in solve(cont_value, cont_marginal))


def _skip_without_payload(arithmetic: ComparisonArithmetic) -> None:
    if arithmetic == "certified" and not kernel_built_for_current_backend():
        pytest.skip("the certified exact-affine payload is not built for this backend")


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [1, 2, 4, 7])
def test_streamed_step_publishes_the_one_shot_feasible_set(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """Singleton, divisor, non-divisor, and full-width partitions own the same nodes."""
    _skip_without_payload(arithmetic)
    reference = _published(arithmetic=arithmetic, interval_batch_size=0)
    candidate = _published(
        arithmetic=arithmetic, interval_batch_size=interval_batch_size
    )

    np.testing.assert_array_equal(
        [np.isfinite(channel) for channel in candidate[:3]],
        [np.isfinite(channel) for channel in reference[:3]],
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [1, 2, 4, 7])
def test_streamed_step_publishes_the_one_shot_owner_at_every_node(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """The same global stored-link identity owns each node at every partition."""
    _skip_without_payload(arithmetic)
    np.testing.assert_array_equal(
        _published(arithmetic=arithmetic, interval_batch_size=interval_batch_size)[3],
        _published(arithmetic=arithmetic, interval_batch_size=0)[3],
        err_msg=f"arithmetic={arithmetic}, interval_batch_size={interval_batch_size}",
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_one_shot_owner_names_several_candidates(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """The owner comparison ranges over distinct identities, not a constant."""
    _skip_without_payload(arithmetic)
    owner = _published(arithmetic=arithmetic, interval_batch_size=0)[3]
    assert np.unique(owner[owner != NO_OWNER]).size > 1


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [0, 1, 2, 4, 7])
def test_an_owner_is_published_exactly_where_a_level_is(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """A node carries an owner if and only if it carries a finite value."""
    _skip_without_payload(arithmetic)
    value, _, _, owner = _published(
        arithmetic=arithmetic, interval_batch_size=interval_batch_size
    )
    np.testing.assert_array_equal(owner != NO_OWNER, np.isfinite(value))


@pytest.mark.parametrize("channel", range(len(_CHANNELS)), ids=_CHANNELS)
@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [1, 2, 4, 7])
def test_streamed_step_agrees_with_the_one_shot_merge_to_the_format_spacing(
    *,
    arithmetic: ComparisonArithmetic,
    interval_batch_size: int,
    channel: int,
) -> None:
    """Every partition names the one-shot level in each published channel."""
    _skip_without_payload(arithmetic)
    assert_agrees_to_ulp(
        got=_published(arithmetic=arithmetic, interval_batch_size=interval_batch_size)[
            channel
        ],
        expected=_published(arithmetic=arithmetic, interval_batch_size=0)[channel],
        n_ulp=_PARTITION_ULP,
        err_msg=(
            f"channel={_CHANNELS[channel]}, arithmetic={arithmetic}, "
            f"interval_batch_size={interval_batch_size}"
        ),
    )


def test_the_agreement_bound_rejects_a_fold_that_publishes_another_owner() -> None:
    """The instrument fires on the defect it guards against, in this run."""
    reference = _published(arithmetic="ordinary", interval_batch_size=0)[0]
    # A fold that hands a query to the wrong candidate moves its level by a finite
    # amount. The smallest gap between two distinct published levels stands in for
    # that, and it is orders of magnitude above the spacing budget above.
    distinct = np.unique(reference[np.isfinite(reference)])
    assert distinct.size > 1
    smallest_ownership_gap = float(np.min(np.diff(distinct)))
    mis_owned = np.where(
        np.isfinite(reference), reference + smallest_ownership_gap, reference
    )

    with pytest.raises(AssertionError, match="ULP, above the"):
        assert_agrees_to_ulp(got=mis_owned, expected=reference, n_ulp=_PARTITION_ULP)


def test_streaming_reader_is_called_with_fixed_width_blocks() -> None:
    """The direct production seam asks for blocks, never the full interval matrix."""
    value, marginal = _continuation()
    seen_shapes: list[tuple[int, ...]] = []

    def read(indices: IntND) -> tuple[FloatND, FloatND]:
        seen_shapes.append(indices.shape)
        return value[indices], marginal[indices]

    with jax.disable_jit():
        nbegm_per_interval_continuation_step_savings(
            **_geometry(),
            cont_value=None,
            cont_marginal=None,
            arithmetic="ordinary",
            interval_block_reader=read,
            interval_batch_size=2,
        )
    assert seen_shapes
    assert set(seen_shapes) == {(2,)}
