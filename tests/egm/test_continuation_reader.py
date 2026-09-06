"""A continuation payload answers value and marginal queries about itself.

A parent asks its target's reader what the continuation is worth at a query and
how it changes there, instead of interpolating the target's rows itself.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.continuation import ContinuationPayload
from _lcm.egm.carry import EGMCarry
from lcm.solver_api import (
    EGM_CONTINUATION,
    EGM_ENDOGENOUS_COORDINATE,
    ContinuationArtifact,
    ContinuationCapabilities,
    ContinuationReader,
)
from tests.conftest import DECIMAL_PRECISION


def _carry() -> EGMCarry:
    """A one-row carry whose value is `2 * R` with constant marginal `2`."""
    grid = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    return EGMCarry(
        endog_grid=grid,
        value=2.0 * grid,
        marginal_utility=jnp.full_like(grid, 2.0),
        taste_shock_scale=jnp.asarray(0.0),
    )


def test_an_egm_carry_is_a_continuation_reader() -> None:
    """The shipped continuation payload satisfies the public protocol."""
    assert isinstance(_carry(), ContinuationReader)


def test_a_carry_reports_value_and_a_marginal_in_its_own_coordinate() -> None:
    """The carry publishes a value everywhere and a marginal in resources."""
    assert _carry().capabilities == ContinuationCapabilities(
        value=True,
        marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE}),
        exact_candidate_identity=False,
        discontinuities=False,
    )


def test_a_carry_with_breakpoints_reports_discontinuity_information() -> None:
    """A payload carrying boundary locations says so in its capabilities."""
    carry = _carry()
    with_breaks = EGMCarry(
        endog_grid=carry.endog_grid,
        value=carry.value,
        marginal_utility=carry.marginal_utility,
        taste_shock_scale=carry.taste_shock_scale,
        breakpoints=jnp.asarray([1.5]),
    )

    assert with_breaks.capabilities.discontinuities


def test_value_at_interpolates_the_carrys_value_row() -> None:
    """The reader returns the row's value at an off-node query."""
    got = np.asarray(_carry().value_at(query=jnp.asarray(1.5)))

    aaae(got, 3.0, decimal=DECIMAL_PRECISION)


def test_marginal_at_returns_the_marginal_in_the_endogenous_coordinate() -> None:
    """The marginal at a query is the marginal row read at that query."""
    got = np.asarray(
        _carry().marginal_at(query=jnp.asarray(1.5), state=EGM_ENDOGENOUS_COORDINATE)
    )

    aaae(got, 2.0, decimal=DECIMAL_PRECISION)


def test_marginal_at_refuses_a_state_the_carry_does_not_carry() -> None:
    """A marginal in a state the payload never tabulated is refused."""
    with pytest.raises(ValueError, match="housing"):
        _carry().marginal_at(query=jnp.asarray(1.5), state="housing")


def test_leaves_addresses_every_published_row_by_its_field_path() -> None:
    """`leaves()` is the addressable content of the payload."""
    assert set(_carry().leaves()) == {
        ("endog_grid",),
        ("value",),
        ("marginal_utility",),
        ("taste_shock_scale",),
    }


def test_a_leaf_is_the_array_the_field_holds() -> None:
    """Addressing a leaf yields the payload's own array, not a copy of it."""
    carry = _carry()

    assert carry.leaves()[("value",)].tobytes() == carry.value.tobytes()


def test_the_continuation_payload_alias_still_names_the_artifact_protocol() -> None:
    """Adding a reader does not turn the rolled channel into a reader channel.

    The engine keeps storing a payload by its versioned key alone.
    """
    assert ContinuationPayload.__value__ is ContinuationArtifact


def test_a_carry_is_still_an_opaque_keyed_artifact() -> None:
    """The payload the engine rolls is identified by its versioned key."""
    assert _carry().artifact_key is EGM_CONTINUATION
