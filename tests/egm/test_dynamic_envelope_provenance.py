"""Original saving and candidate ownership survive exact-envelope emission."""

from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope import (
    _bracket_from_owned_row,
    get_bracket_finder,
    get_upper_envelope,
)
from _lcm.solution.dcegm import (
    DCEGM,
    EnvelopeConfig,
    ExactEnvelope,
    FUESEnvelope,
    LTMEnvelope,
    MSSEnvelope,
    RFCEnvelope,
)
from _lcm.typing import PytreeValue
from lcm import LinSpacedGrid
from lcm.typing import Float1D
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _solver(*, envelope: EnvelopeConfig | None = None) -> DCEGM:
    """Return the configured minimal solver for envelope construction."""
    return DCEGM(
        savings_grid=LinSpacedGrid(start=0.0, stop=2.0, n_points=3),
        envelope=ExactEnvelope() if envelope is None else envelope,
    )


class _EnvelopeCandidates(TypedDict):
    endog_grid: Float1D
    policy: Float1D
    value: Float1D
    savings: Float1D
    marginal_utility: Float1D


def _crossing_candidates() -> _EnvelopeCandidates:
    return {
        "endog_grid": jnp.array([0.0, 2.0, 0.0, 2.0]),
        "policy": jnp.array([10.0, 12.0, 20.0, 22.0]),
        "value": jnp.array([2.0, 0.0, 0.0, 2.0]),
        "savings": jnp.array([101.0, 105.0, 201.0, 209.0]),
        "marginal_utility": jnp.ones(4),
    }


def test_owned_crossing_retains_original_saving_and_two_sides() -> None:
    """Each side reads saving from its winning candidate link."""
    result = jax.jit(
        get_upper_envelope(solver=_solver(), n_refined=16, publish_provenance=True)
    )(**_crossing_candidates())
    assert int(result.n_kept) == 4
    np.testing.assert_array_almost_equal(
        result.grid[:4], [0, 1, 1, 2], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.savings[:4], [101, 103, 205, 209], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result.owner_id[:4], [0, 0, 1, 1])
    np.testing.assert_array_equal(result.side_id[:4], [1, -1, 1, -1])
    np.testing.assert_array_equal(result.link_valid[:3], [True, False, True])
    np.testing.assert_array_equal(result.source_left_index[:4], [0, 0, 2, 2])
    np.testing.assert_array_equal(result.source_right_index[:4], [1, 1, 3, 3])
    assert not np.any(
        np.asarray(result.savings[:4])
        == np.asarray(result.grid[:4] - result.policy[:4])
    )


@pytest.mark.parametrize(
    ("query", "saving", "owner"), [(0.5, 102.0, 0), (1.0, 205.0, 1), (1.5, 207.0, 1)]
)
def test_owned_bracket_uses_the_selected_side(
    *, query: float, saving: float, owner: int
) -> None:
    """The bracket's saving and ownership select the same side as its value."""
    result = jax.jit(
        get_bracket_finder(solver=_solver(), n_refined=16, publish_provenance=True)
    )(**_crossing_candidates(), x_query=jnp.asarray(query))
    assert bool(result.read_supported)
    assert int(result.owner_id) == owner
    np.testing.assert_array_almost_equal(
        result.savings, saving, decimal=DECIMAL_PRECISION
    )
    if query == 1.0:
        assert int(result.side_id) == 1
        np.testing.assert_array_almost_equal(
            result.bracket.lower_policy, 21.0, decimal=DECIMAL_PRECISION
        )


def test_owned_bracket_extends_the_first_owning_secant_below_support() -> None:
    """Below the first node, original saving uses the first valid owning secant."""
    result = jax.jit(
        get_bracket_finder(solver=_solver(), n_refined=16, publish_provenance=True)
    )(**_crossing_candidates(), x_query=jnp.asarray(-0.5))
    assert bool(result.read_supported)
    assert int(result.owner_id) == 0
    assert int(result.side_id) == 0
    assert bool(result.link_valid)
    np.testing.assert_array_almost_equal(
        result.savings, 100.0, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        [result.bracket.lower_grid, result.bracket.upper_grid],
        [0, 1],
        decimal=DECIMAL_PRECISION,
    )


def test_owned_bracket_clamps_to_last_original_saving_above_support() -> None:
    """Above the last node, the selected tuple is that owned endpoint."""
    result = jax.jit(
        get_bracket_finder(solver=_solver(), n_refined=16, publish_provenance=True)
    )(**_crossing_candidates(), x_query=jnp.asarray(2.5))
    assert bool(result.read_supported)
    assert int(result.owner_id) == 1
    assert int(result.side_id) == -1
    np.testing.assert_array_almost_equal(
        result.savings, 209.0, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        [result.bracket.lower_grid, result.bracket.upper_grid],
        [2, 2],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        [result.bracket.lower_policy, result.bracket.upper_policy],
        [22, 22],
        decimal=DECIMAL_PRECISION,
    )


def test_gap_is_closed_and_exact_endpoint_is_supported() -> None:
    """Absent candidate cells cannot acquire links through envelope compaction."""
    candidates = {
        "endog_grid": jnp.array([0.0, 1.0, jnp.nan, 3.0, 4.0]),
        "policy": jnp.array([10.0, 11.0, jnp.nan, 13.0, 14.0]),
        "value": jnp.array([0.0, 1.0, jnp.nan, 3.0, 4.0]),
        "savings": jnp.array([100.0, 102.0, jnp.nan, 300.0, 304.0]),
        "marginal_utility": jnp.ones(5),
    }
    result = jax.jit(
        get_upper_envelope(solver=_solver(), n_refined=16, publish_provenance=True)
    )(**candidates)
    assert int(result.n_kept) == 4
    np.testing.assert_array_almost_equal(
        result.grid[:4], [0, 1, 3, 4], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result.owner_id[:4], [0, 0, 1, 1])
    np.testing.assert_array_equal(result.link_valid[:3], [True, False, True])
    finder = jax.jit(
        get_bracket_finder(solver=_solver(), n_refined=16, publish_provenance=True)
    )
    endpoint = finder(**candidates, x_query=jnp.asarray(1.0))
    assert bool(endpoint.read_supported)
    np.testing.assert_array_almost_equal(
        endpoint.savings, 102.0, decimal=DECIMAL_PRECISION
    )
    missing = finder(**candidates, x_query=jnp.asarray(2.0))
    assert not bool(missing.read_supported)


def test_connected_edges_share_run_and_keep_legacy_numerics() -> None:
    """Adjacent candidate links in one run remain safe interpolation support."""
    candidates = {
        "endog_grid": jnp.array([0.0, 1.0, 2.0]),
        "policy": jnp.array([3.0, 4.0, 5.0]),
        "value": jnp.array([0.0, 2.0, 3.0]),
        "savings": jnp.array([17.0, 19.0, 23.0]),
        "marginal_utility": jnp.ones(3),
    }
    legacy = jax.jit(get_upper_envelope(solver=_solver(), n_refined=16))(**candidates)
    assert len(legacy) == 4
    owned = jax.jit(
        get_upper_envelope(solver=_solver(), n_refined=16, publish_provenance=True)
    )(**candidates)
    for actual, expected in zip(owned[:4], legacy, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(owned.owner_id[:3], [0, 0, 0])
    np.testing.assert_array_equal(owned.link_valid[:2], [True, True])


@pytest.mark.parametrize(
    "envelope", [FUESEnvelope(), RFCEnvelope(), LTMEnvelope(), MSSEnvelope()]
)
@pytest.mark.parametrize("factory", [get_upper_envelope, get_bracket_finder])
def test_other_backends_refuse_unavailable_owned_provenance(
    envelope: EnvelopeConfig, factory: Callable[..., Callable[..., PytreeValue]]
) -> None:
    """A backend without original candidate ownership refuses feature publication."""
    with pytest.raises(ValueError, match=r"provenance.*ExactEnvelope"):
        factory(
            solver=_solver(envelope=envelope), n_refined=16, publish_provenance=True
        )
    assert callable(factory(solver=_solver(envelope=envelope), n_refined=16))


def test_coincident_value_and_policy_keep_foreign_saving_owners() -> None:
    """A boundary with equal value and policy retains both saving owners."""
    candidates = _crossing_candidates()
    candidates["policy"] = jnp.ones(4)
    result = jax.jit(
        get_upper_envelope(solver=_solver(), n_refined=16, publish_provenance=True)
    )(**candidates)
    assert int(result.n_kept) == 4
    np.testing.assert_array_equal(result.owner_id[:4], [0, 0, 1, 1])
    np.testing.assert_array_equal(result.side_id[:4], [1, -1, 1, -1])
    np.testing.assert_array_almost_equal(
        result.savings[:4], [101, 103, 205, 209], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.policy[:4], [1, 1, 1, 1], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.value[:4], [2, 1, 1, 2], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result.link_valid[:3], [True, False, True])


def test_unreadable_original_saving_withholds_associated_value_and_policy() -> None:
    """A failed original-saving read cannot publish its value or policy alone."""
    candidates = _crossing_candidates()
    candidates["savings"] = jnp.full(4, jnp.nan)
    result = jax.jit(
        get_upper_envelope(solver=_solver(), n_refined=16, publish_provenance=True)
    )(**candidates)
    assert int(result.n_kept) == 4
    np.testing.assert_array_equal(np.isnan(result.savings[:4]), [True] * 4)
    np.testing.assert_array_equal(np.isnan(result.policy[:4]), [True] * 4)
    np.testing.assert_array_equal(np.isnan(result.value[:4]), [True] * 4)
    np.testing.assert_array_equal(result.link_valid[:3], [False] * 3)


def test_known_economic_value_does_not_invalidate_finite_owned_geometry() -> None:
    """Known economic minus infinity differs from unresolved associated arithmetic."""
    refined = get_upper_envelope(
        solver=_solver(), n_refined=16, publish_provenance=True
    )(**_crossing_candidates())
    refined = refined._replace(value=refined.value.at[3].set(-jnp.inf))
    result = _bracket_from_owned_row(refined=refined, x_query=jnp.asarray(0.5))
    assert bool(result.read_supported)
    assert not bool(getattr(result, "construction_invalid", jnp.asarray(a=False)))
