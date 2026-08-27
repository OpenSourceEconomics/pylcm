"""A subject's role is its own, carried row by row through the simulation.

A collective regime's stakeholders are the roles its members occupy, and which
role a row occupies decides where that row goes when the household ends: her
leg's fallback for her, his for him. Fixing one role for a whole `simulate()`
call makes a mixed cohort inexpressible and, worse, silently simulates every
divorced husband as his wife.

So the role is per row: seeded in the initial conditions, set from the branch a
gated edge actually takes, and published beside the states. Roles are integers
on the device with published string labels, drawn from one model-wide
vocabulary, so a model whose collective regimes name disjoint roles is
expressible and a role can be compared across regimes.

The dissolution miniature below empties the participation mask at `wage = 2`
alone, so the three subjects separate exactly: two stay married, one dissolves.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.simulate import simulate
from _lcm.utils.logging import get_logger
from lcm.exceptions import InvalidInitialConditionsError
from tests.regime_building.test_collective_regime_simulate import _solve_dissolution

# `NO_ROLE` is what a row in a singleton regime carries: it occupies no role in
# any household.
_NO_ROLE = -1


def _simulate_cohort(*, own_stakeholder, wages=(1.0, 2.0, 3.0)):
    """Simulate the dissolution miniature with one role per subject."""
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_dissolution()
    )
    initial_conditions = {
        "wage": jnp.asarray(wages),
        "age": jnp.zeros(len(wages)),
        "regime_id": jnp.array(
            [regime_names_to_ids["married"]] * len(wages), dtype=jnp.int32
        ),
    }
    if own_stakeholder is not None:
        initial_conditions["own_stakeholder"] = jnp.asarray(
            own_stakeholder, dtype=jnp.int32
        )
    return simulate(
        flat_params=flat_params,
        initial_conditions=MappingProxyType(initial_conditions),
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    ), solution


def _role_ids(regimes) -> dict[str, int]:
    """The model-wide role vocabulary, read off the processed regimes."""
    return dict(regimes["married"].stakeholder_names_to_ids)


def test_a_mixed_cohort_dissolves_each_row_into_its_own_single_regime() -> None:
    """The wife's row enters `single_f`, the husband's `single_m`, in one call.

    Only the `wage = 2` subject dissolves, and here that subject is the one
    carrying the husband's role, so `single_m` receives it and `single_f`
    receives nobody. The middle row is the discriminating one: a router that
    picked a leg by declaration order would send it to `single_f`.
    """
    _ages, regimes, *_rest = _solve_dissolution()
    roles = _role_ids(regimes)

    result, _solution = _simulate_cohort(
        own_stakeholder=[roles["f"], roles["m"], roles["f"]]
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["single_m"][1].in_regime),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        np.asarray(result.raw_results["single_f"][1].in_regime),
        [False, False, False],
    )


def test_the_same_cohort_with_the_roles_swapped_dissolves_the_other_way() -> None:
    """Giving the dissolving row the wife's role sends it to `single_f`.

    The gate is a property of the household, not of who is looking at it, so
    the same subject dissolves either way — only its destination moves. Without
    this control the result above could come from the cohort rather than from
    the role.
    """
    _ages, regimes, *_rest = _solve_dissolution()
    roles = _role_ids(regimes)

    result, _solution = _simulate_cohort(
        own_stakeholder=[roles["m"], roles["f"], roles["m"]]
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["single_f"][1].in_regime),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        np.asarray(result.raw_results["single_m"][1].in_regime),
        [False, False, False],
    )


def test_a_dissolved_row_stops_carrying_a_role() -> None:
    """Entering a singleton regime leaves the row occupying no role.

    A single person is nobody's stakeholder, so the published role is the
    no-role sentinel rather than the role the row held while married.
    """
    _ages, regimes, *_rest = _solve_dissolution()
    roles = _role_ids(regimes)

    result, _solution = _simulate_cohort(
        own_stakeholder=[roles["f"], roles["m"], roles["f"]]
    )

    published = result.to_dataframe(use_labels=False)
    dissolved = published.query("period == 1 and regime_name == 'single_m'")
    assert list(dissolved["own_stakeholder"]) == [_NO_ROLE]


def test_a_row_that_stays_married_keeps_its_role() -> None:
    """A household that does not dissolve leaves every member's role alone."""
    _ages, regimes, *_rest = _solve_dissolution()
    roles = _role_ids(regimes)

    result, _solution = _simulate_cohort(
        own_stakeholder=[roles["f"], roles["m"], roles["f"]]
    )

    published = result.to_dataframe(use_labels=False)
    married = published.query("period == 1 and regime_name == 'married_ir'")
    assert list(married.sort_values("subject_id")["own_stakeholder"]) == [
        roles["f"],
        roles["f"],
    ]


def test_a_collective_start_without_a_role_is_refused() -> None:
    """A cohort starting married must say which partner each row is.

    Picking one for it would simulate a whole population as one partner, which
    is a modelling decision and not the engine's to make.
    """
    with pytest.raises(InvalidInitialConditionsError, match="own_stakeholder"):
        _simulate_cohort(own_stakeholder=None)


def test_a_role_no_regime_declares_is_refused() -> None:
    """An out-of-vocabulary role code names the vocabulary it is not in."""
    with pytest.raises(InvalidInitialConditionsError, match="own_stakeholder"):
        _simulate_cohort(own_stakeholder=[7, 7, 7])


def test_the_published_role_column_carries_its_labels() -> None:
    """`to_dataframe()` names roles the way the model declared them."""
    _ages, regimes, *_rest = _solve_dissolution()
    roles = _role_ids(regimes)

    result, _solution = _simulate_cohort(
        own_stakeholder=[roles["f"], roles["m"], roles["f"]]
    )

    published = result.to_dataframe()
    assert set(published.query("period == 0")["own_stakeholder"]) == {"f", "m"}
