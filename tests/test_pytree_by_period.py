from collections.abc import Mapping

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
        ({0: jnp.ones(2)},),
        {"working": {0: jnp.ones(2)}},
        [{"working": {0: jnp.ones(2)}}],
        (({0: jnp.ones(2)},),),
    ],
    ids=[
        "period-over-names",
        "period-over-period",
        "names-only",
        "period-inside-a-tuple",
        "period-below-a-name",
        "list-of-names-over-period",
        "period-two-tuples-deep",
    ],
)
def test_pytree_by_period_admits_period_levels_at_any_depth(
    *, tree: PytreeByPeriod
) -> None:
    """A tree with name or period levels anywhere passes the runtime check."""
    assert _take(tree=tree) is tree


@pytest.mark.parametrize(
    "tree",
    ["wealth", {3: "wealth"}, {"wealth": "high"}, object()],
    ids=["string", "period-over-string", "name-over-string", "object"],
)
def test_pytree_by_period_rejects_what_is_not_a_value_tree(
    *,
    tree: str | Mapping[int, str] | Mapping[str, str] | object,  # noqa: PAN001 - includes a literal object to test rejection
) -> None:
    """A string leaf or an arbitrary object fails the check."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _take(tree=tree)  # ty: ignore[invalid-argument-type]
