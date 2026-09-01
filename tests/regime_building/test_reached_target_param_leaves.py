"""The target-parameter leaves ONE edge consumer reaches through the target DAG.

`_reached_target_param_leaves` seeds the ancestry-aware fences that reject an
edge whose gate or projection reaches a parameter the TARGET regime owns
(`_reject_target_function_params`). Six build-time fences read it, and a walk
that under-reaches makes all six silently pass, so the reached set is pinned
here against a hand-checked topology rather than only through the fences.

What the walk must return is the free (non-node, non-state) arguments of the
target-DAG closure reachable from that ONE consumer's own declared arguments:

- a chain of nodes contributes every link's free parameter;
- a diamond contributes each shared node's parameters exactly once;
- a node the consumer never reaches contributes nothing, even though it sits in
  the same pool -- unioning the whole pool's free arguments would reject valid
  topologies on a bare name collision;
- a seed naming no pool node at all (a source-declared parameter, or a target
  state) enters no graph and so contributes nothing;
- the target's own state names are engine-wired coordinates, not parameters,
  and are dropped wherever they appear.
"""

import functools

import jax.numpy as jnp
import pytest

from _lcm.regime_building.gated_edges import _reached_target_param_leaves
from lcm.typing import FloatND


def wage(*, human_capital: FloatND, wage_level: float) -> FloatND:
    """Chain head: reads a target state and one target-owned parameter."""
    return human_capital * wage_level


def labor_income(*, wage: FloatND, hours: float) -> FloatND:
    """Chain middle: reads the node above and one more target-owned parameter."""
    return wage * hours


def net_income(*, labor_income: FloatND, tax_rate: float) -> FloatND:
    """Chain tail, and one arm of the diamond below."""
    return labor_income * (1 - tax_rate)


def transfers(*, wage: FloatND, transfer_rate: float) -> FloatND:
    """The diamond's second arm: rejoins `wage` from the other side."""
    return wage * transfer_rate


def resources(*, net_income: FloatND, transfers: FloatND, wealth: FloatND) -> FloatND:
    """Diamond join: both arms plus a target state."""
    return net_income + transfers + wealth


def unreached_helper(unreached_param: float) -> FloatND:
    """A target helper in the same pool that no consumer below ever reaches."""
    return jnp.asarray(unreached_param)


DAG_POOL = {
    "wage": wage,
    "labor_income": labor_income,
    "net_income": net_income,
    "transfers": transfers,
    "resources": resources,
    "unreached_helper": unreached_helper,
}

STATE_NAMES = frozenset({"human_capital", "wealth"})


@pytest.mark.parametrize(
    ("seed_args", "expected"),
    [
        pytest.param(
            ["net_income"],
            frozenset({"wage_level", "hours", "tax_rate"}),
            id="chain-collects-every-link's-parameter",
        ),
        pytest.param(
            ["resources"],
            frozenset({"wage_level", "hours", "tax_rate", "transfer_rate"}),
            id="diamond-collects-both-arms-once",
        ),
        pytest.param(
            ["wage"],
            frozenset({"wage_level"}),
            id="a-seed-reaches-only-its-own-ancestors",
        ),
        pytest.param(
            ["source_own_param", "wealth", "human_capital"],
            frozenset(),
            id="seeds-naming-no-pool-node-reach-nothing",
        ),
        pytest.param(
            ["transfers", "source_own_param"],
            frozenset({"wage_level", "transfer_rate"}),
            id="a-non-pool-seed-does-not-widen-the-closure",
        ),
        pytest.param(
            [],
            frozenset(),
            id="no-seeds-reach-nothing",
        ),
        pytest.param(
            ["net_income", "transfers"],
            frozenset({"wage_level", "hours", "tax_rate", "transfer_rate"}),
            id="several-seeds-union-their-closures",
        ),
    ],
)
def test_reached_target_param_leaves_returns_the_consumer_s_closure(
    *,
    seed_args: list[str],
    expected: frozenset[str],
) -> None:
    """The walk returns exactly the free parameters of the seeds' ancestor closure."""
    got = _reached_target_param_leaves(
        dag_pool=DAG_POOL, seed_args=seed_args, state_names=STATE_NAMES
    )
    assert got == expected


def test_reached_target_param_leaves_excludes_an_unreached_pool_helper() -> None:
    """A pool helper outside the consumer's closure contributes no leaf.

    Its parameter is in the pool's overall free-argument union, so a walk that
    unioned the whole pool would report it and reject the topology.
    """
    got = _reached_target_param_leaves(
        dag_pool=DAG_POOL, seed_args=["resources"], state_names=STATE_NAMES
    )
    assert "unreached_param" not in got


def test_reached_target_param_leaves_ignores_a_keyword_bound_parameter() -> None:
    """A parameter already bound on a pool node is not a leaf the consumer reaches.

    The gate is compiled by concatenating the consumer with this pool, and DAG
    concatenation binds only a node's FREE arguments -- a keyword-bound one is
    not in the compiled signature, so it cannot be evaluated from the wrong
    namespace and must not trip the fence.
    """
    pool = {**DAG_POOL, "wage": functools.partial(wage, wage_level=2.0)}
    got = _reached_target_param_leaves(
        dag_pool=pool, seed_args=["labor_income"], state_names=STATE_NAMES
    )
    assert got == frozenset({"hours"})
