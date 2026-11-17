"""Conditional DC candidates precede current-shock integration and full tangents."""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.asset_row import publish_node_from_bracket
from _lcm.egm.upper_envelope.fues import QueryBracket
from lcm.typing import (
    ScalarFloat,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _zero_utility(consumption: ScalarFloat) -> ScalarFloat:
    return 0.0 * consumption


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_asset_publication_owns_corner_equality() -> None:
    """At equal values the literal corner wins and sets the closed-form action."""
    bracket = QueryBracket(
        lower_grid=jnp.asarray(1.0),
        upper_grid=jnp.asarray(2.0),
        lower_policy=jnp.asarray(0.5),
        upper_policy=jnp.asarray(1.0),
        lower_value=jnp.asarray(0.0),
        upper_value=jnp.asarray(0.0),
        first_grid=jnp.asarray(1.0),
        n_kept=jnp.int32(2),
    )
    value, policy = publish_node_from_bracket(
        bracket=bracket,
        n_pad=2,
        resources_at_node=jnp.asarray(1.5),
        borrowing_limit=jnp.asarray(0.25),
        utility_of_action=_zero_utility,
        discounted_expected_value_at_limit=jnp.asarray(0.0),
    )
    np.testing.assert_array_equal([value, policy], [0.0, 1.25])
