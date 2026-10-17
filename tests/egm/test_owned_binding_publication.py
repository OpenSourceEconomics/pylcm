"""Native publication flags retain the same envelope's candidate provenance."""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("case", ["mixed", "overflow", "invalid_policy"])
def test_native_binding_flags_preserve_every_owned_channel(
    *, case: Literal["mixed", "overflow", "invalid_policy"], compiled: bool
) -> None:
    """Only actual floor-source links receive flags beside unchanged owned rows."""
    grid = jnp.asarray([1.0, 2.0, 3.0])
    policy = jnp.asarray([1.0, 2.0, 2.0])
    value = jnp.asarray([0.0, 1.0, 1.5])
    saving = jnp.asarray([0.0, 0.0, 1.0])
    queries = jnp.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    np.testing.assert_array_equal(grid - policy, saving)
    np.testing.assert_array_equal(jnp.isfinite(grid), [True] * 3)
    np.testing.assert_array_equal(jnp.isfinite(value), [True] * 3)
    np.testing.assert_array_equal(jnp.isfinite(saving), [True] * 3)
    capacity = 2 if case == "overflow" else 8
    if case == "invalid_policy":
        policy = policy.at[1].set(jnp.nan)
        assert bool(jnp.isnan(policy[1]))
    refine = partial(
        refine_envelope_exact, n_refined=capacity, max_runs=2, cell_width=1
    )
    combined_reader = partial(refine, _binding_source_count=2)
    if compiled:
        combined_reader = jax.jit(combined_reader)
    combined = combined_reader(
        endog_grid=grid,
        policy=policy,
        value=value,
        savings=saving,
        _binding_queries=queries,
    )
    owned = refine(endog_grid=grid, policy=policy, value=value, savings=saving)
    binding = refine(
        endog_grid=grid,
        policy=policy,
        value=value,
        _binding_source_count=2,
        _binding_queries=queries,
    )
    for name in (
        "grid",
        "policy",
        "value",
        "n_kept",
        "savings",
        "owner_id",
        "side_id",
        "link_valid",
        "source_left_index",
        "source_right_index",
    ):
        np.testing.assert_array_equal(
            getattr(combined.owned, name), getattr(owned, name)
        )
    np.testing.assert_array_equal(combined.binding_query, binding.binding_query)
    expected = (
        [False, True, True, True, False, False] if case == "mixed" else [False] * 6
    )
    np.testing.assert_array_equal(combined.binding_query, expected)
    assert combined.binding_query.dtype == jnp.bool_
    assert combined.binding_query.shape == queries.shape
    if case == "overflow":
        assert int(combined.owned.n_kept) > capacity
    elif case == "invalid_policy":
        live = jnp.arange(capacity) < combined.owned.n_kept
        assert bool(jnp.any(live & ~jnp.isfinite(combined.owned.policy)))
    else:
        assert int(combined.owned.n_kept) == 3
        np.testing.assert_array_equal(combined.owned.source_left_index[:3], [0, 1, 1])
        np.testing.assert_array_equal(combined.owned.source_right_index[:3], [1, 2, 2])
