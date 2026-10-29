"""Read exact original NB resource-bank points with the native envelope comparator."""

import jax.numpy as jnp

from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from _lcm.egm.nbegm_resources import NBEGMResourcesBank
from _lcm.egm.nbegm_step import _publish_no_action_carry
from _lcm.egm.upper_envelope.query import (
    ComparisonArithmetic,
    envelope_at_query,
    published_owner,
)
from lcm.typing import ScalarFloat


def read_nbegm_resources_bank(
    *,
    bank: NBEGMResourcesBank,
    resources: ScalarFloat,
    arithmetic: ComparisonArithmetic,
) -> OwnedResourcesRead:
    """Select the associated original NB tuple with its original envelope rule."""
    published = envelope_at_query(
        endog_grid=bank.endog_grid,
        policy=bank.policy,
        value=bank.value,
        marginal=bank.marginal,
        segment_id=bank.segment_id,
        x_query=resources[None],
        arithmetic=arithmetic,
        candidate_savings=bank.savings,
        return_owner=True,
    )
    if len(published) != 5:  # noqa: PLR2004 - original value/policy/marginal/saving/owner ABI
        raise TypeError("Original NB resource bank lost its native selected owner.")
    value, policy, marginal, savings, owner = published
    value, marginal, policy = _publish_no_action_carry(
        value=value,
        marginal=marginal,
        policy=policy,
        coh=resources[None],
        lowest_savings=bank.borrowing_limit,
    )
    feasible = jnp.isfinite(value[0])
    numeric = (jnp.isfinite(value[0]) | jnp.isneginf(value[0])) & (
        ~feasible
        | (
            jnp.isfinite(policy[0])
            & jnp.isfinite(marginal[0])
            & jnp.isfinite(savings[0])
        )
    )
    # A query beyond the solve's resource support reads the outermost original
    # segments, as an ordinary node-state policy does off its grid.
    valid = (
        ~bank.construction_invalid
        & jnp.isfinite(resources)
        & numeric
        & (
            ~feasible
            | (
                (policy[0] > 0)
                & (savings[0] >= bank.borrowing_limit)
                & (savings[0] <= bank.upper_savings)
                & (owner[0] >= 0)
            )
        )
    )
    return OwnedResourcesRead(
        value=value[0],
        marginal=jnp.where(feasible, marginal[0], 0.0),
        policy=policy[0],
        savings=savings[0],
        native_owner=published_owner(value=value, stable_index=owner)[0],
        side_id=jnp.int32(0),
        corner=savings[0] == bank.borrowing_limit,
        feasible=feasible,
        valid=valid,
    )
