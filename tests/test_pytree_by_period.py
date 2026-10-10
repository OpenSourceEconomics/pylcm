from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.beartype_conf import INTERNAL_CONF
from _lcm.typing import PytreeByPeriod


@beartype(conf=INTERNAL_CONF)
def _take(*, tree: PytreeByPeriod) -> PytreeByPeriod:
    return tree


@pytest.mark.parametrize(
    "tree",
    [
        {3: {"wealth": jnp.ones(2)}},
        {3: {4: jnp.ones(2)}},
        {"wealth": jnp.ones(2)},
    ],
    ids=["period-over-names", "period-over-period", "names-only"],
)
def test_pytree_by_period_admits_period_levels_at_the_top(
    *, tree: PytreeByPeriod
) -> None:
    """A tree keyed by period at its top levels passes the runtime check."""
    assert _take(tree=tree) is tree


@pytest.mark.parametrize(
    "tree",
    ["wealth", {3: "wealth"}, {"regime": {0: jnp.ones(2)}}],
    ids=["string", "period-over-string", "period-below-a-name"],
)
def test_pytree_by_period_rejects_what_is_not_a_period_tree(
    *, tree: str | Mapping[int, str] | Mapping[str, Mapping[int, jax.Array]]
) -> None:
    """A string leaf, or a period level below a name level, fails the check."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _take(tree=tree)  # ty: ignore[invalid-argument-type]
