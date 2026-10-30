"""A DC-EGM utility reading the Euler state directly is refused by the envelope."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from lcm import Model
from lcm.exceptions import ModelInitializationError
from lcm.typing import ContinuousAction, ContinuousState
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution.test_current_iid_off_node_replay import _interior_declarations

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _state_utility(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> jax.Array:
    return jnp.log(consumption) + wealth


def test_public_dcegm_retains_its_original_state_utility_refusal() -> None:
    """Without a current shock the public constructor keeps the envelope refusal."""
    declarations = _interior_declarations()
    regimes = declarations["regimes"]
    child = replace(
        regimes["child"],
        functions={**regimes["child"].functions, "utility": _state_utility},
        shocks=(),
    )
    declarations["regimes"] = {**regimes, "child": child}
    declarations["shocks"] = {}
    with pytest.raises(ModelInitializationError, match="DCEGM envelope"):
        Model(**declarations)
