"""A liquid node whose budget affords no action publishes the infeasible carry.

Every one-period NB-EGM step publishes its value, marginal and policy on the
liquid grid. Where cash-on-hand net of the lowest savings node is not positive,
no consumption is affordable, so the node has no action at all. There the step
publishes the infeasible-carry contract:

- value `-inf`;
- policy NaN;
- marginal zero.

NaN in the value stays reserved for a node that affords an action no live
candidate brackets. The budget is `coh = liquid + intercept` on a liquid grid
starting at `0.1`: an intercept of `-1` leaves the two lowest nodes with
cash-on-hand `-0.9` and `-0.2`, and an intercept of `+1` is the control where
every node affords an action.

Every envelope reduction publishes NaN at a node where no winner is decided,
whether the reduction is the ordinary one, the certified one, or the streamed
interval fold; only the step knows whether that node affords an action. So the
carry is checked on every step and route, including the streamed per-interval
route, and a node that does afford an action but whose every candidate is
non-finite keeps NaN:

- a NaN continuation, where no candidate value exists;
- a felicity that overflows to `+inf`, which must not surface as a finite value
  or as the `-inf` of a node without an action.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_step import (
    nbegm_discrete_envelope_step,
    nbegm_multi_interval_step,
    nbegm_multi_interval_step_savings,
    nbegm_one_asset_step,
    nbegm_per_interval_continuation_step_savings,
    nbegm_recurring_jump_step,
    nbegm_unified_step,
    nbegm_unified_step_savings,
)
from _lcm.egm.preferences import Preferences
from _lcm.egm.upper_envelope.query import ComparisonArithmetic
from lcm.typing import Float1D, FloatND, IntND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution._crra_preferences import crra_preferences

_REQUIRES_KERNEL = pytest.mark.requires_exact_affine_kernel(
    reason=EXACT_KERNEL_SKIP_REASON
)

_UNAFFORDABLE_INTERCEPT = -1.0
_AFFORDABLE_INTERCEPT = 1.0
_N_LIQUID = 8
_UNAFFORDABLE_NODES = np.array([True, True] + [False] * (_N_LIQUID - 2))
_DISCOUNT = 0.95
_GROSS_RETURN = 1.03
_INCOME = 0.5
# The steps built around a jump take one at a liquid level above both
# unaffordable nodes, with the same budget on either side of it.
_CLIFF = 2.5
_SMOOTH = "smooth"
_NAN_CONTINUATION = "nan_continuation"
_OVERFLOWING_FELICITY = "overflowing_felicity"

type _Channels = tuple[FloatND, FloatND, FloatND]
type _Step = Callable[..., _Channels]


@dataclass(frozen=True, kw_only=True)
class _Grids:
    """Liquid and savings grids plus a smooth concave continuation on each."""

    liquid: Float1D
    """Liquid grid, starting at `0.1`."""

    savings: Float1D
    """Savings grid, starting at the no-save node `0`."""

    cont_value: Float1D
    """Savings-space expected continuation value."""

    cont_marginal: Float1D
    """Savings-space expected marginal continuation."""

    next_value: Float1D
    """Next period's value on the liquid grid."""

    next_marginal: Float1D
    """Next period's marginal value of liquid on the liquid grid."""

    next_liquid: Float1D
    """Where each savings node lands next period."""

    marginal_return: Float1D
    """The landing point's derivative with respect to savings."""


def _grids(*, specimen: str) -> _Grids:
    """Build the grids; `_NAN_CONTINUATION` makes both continuation values NaN."""
    liquid = jnp.linspace(0.1, 5.0, _N_LIQUID)
    savings = jnp.linspace(0.0, 5.0, 10)
    poison = jnp.nan if specimen == _NAN_CONTINUATION else 0.0
    return _Grids(
        liquid=liquid,
        savings=savings,
        cont_value=-1.0 / (1.0 + savings) + poison,
        cont_marginal=(1.0 + savings) ** -2.0,
        next_value=-1.0 / (_INCOME + liquid) + poison,
        next_marginal=(_INCOME + liquid) ** -2.0,
        next_liquid=_GROSS_RETURN * savings + _INCOME,
        marginal_return=jnp.full_like(savings, _GROSS_RETURN),
    )


def _preferences(*, specimen: str) -> Preferences:
    """CRRA(2), whose felicity is `+inf` everywhere for `_OVERFLOWING_FELICITY`."""
    crra = crra_preferences(crra=2.0)
    if specimen != _OVERFLOWING_FELICITY:
        return crra
    return Preferences(
        utility=lambda consumption: jnp.full_like(consumption, jnp.inf),
        marginal_utility=crra.marginal_utility,
        inverse_marginal_utility=crra.inverse_marginal_utility,
    )


def _multi_interval_step_savings(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)
    return nbegm_multi_interval_step_savings(
        cont_value=grids.cont_value,
        cont_marginal=grids.cont_marginal,
        liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=jnp.asarray(_DISCOUNT),
        preferences=_preferences(specimen=specimen),
        coh_slopes=jnp.ones(1),
        coh_intercepts=jnp.reshape(intercept, (1,)),
        breakpoints=jnp.zeros((0,)),
        arithmetic=arithmetic,
    )


def _per_interval_continuation_step_savings(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)
    value, marginal, policy = nbegm_per_interval_continuation_step_savings(
        cont_value=grids.cont_value[None, :],
        cont_marginal=grids.cont_marginal[None, :],
        liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=jnp.asarray(_DISCOUNT),
        preferences=_preferences(specimen=specimen),
        coh_slopes=jnp.ones(1),
        coh_intercepts=jnp.reshape(intercept, (1,)),
        breakpoints=jnp.zeros((0,)),
        arithmetic=arithmetic,
    )
    return value, marginal, policy


def _per_interval_continuation_step_savings_streamed(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)

    def read(interval_indices: IntND) -> tuple[FloatND, FloatND]:
        return (
            grids.cont_value[None, :][interval_indices],
            grids.cont_marginal[None, :][interval_indices],
        )

    value, marginal, policy = nbegm_per_interval_continuation_step_savings(
        cont_value=None,
        cont_marginal=None,
        liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=jnp.asarray(_DISCOUNT),
        preferences=_preferences(specimen=specimen),
        coh_slopes=jnp.ones(1),
        coh_intercepts=jnp.reshape(intercept, (1,)),
        breakpoints=jnp.zeros((0,)),
        arithmetic=arithmetic,
        interval_block_reader=read,
        interval_width=1,
    )
    return value, marginal, policy


def _unified_step_savings(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)
    return nbegm_unified_step_savings(
        cont_value=grids.cont_value,
        cont_marginal=grids.cont_marginal,
        liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=jnp.asarray(_DISCOUNT),
        preferences=_preferences(specimen=specimen),
        coh_slopes=jnp.ones(1),
        coh_intercepts=jnp.reshape(intercept, (1,)),
        breakpoints=jnp.zeros((0,)),
        jump_positions=(),
        arithmetic=arithmetic,
    )


def _multi_interval_step(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)
    return nbegm_multi_interval_step(
        next_value=grids.next_value,
        next_marginal=grids.next_marginal,
        liquid_grid=grids.liquid,
        next_liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=_DISCOUNT,
        preferences=_preferences(specimen=specimen),
        next_liquid=grids.next_liquid,
        marginal_return=grids.marginal_return,
        coh_slopes=jnp.ones(1),
        coh_intercepts=jnp.reshape(intercept, (1,)),
        breakpoints=jnp.zeros((0,)),
        arithmetic=arithmetic,
    )


def _unified_step(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)
    return nbegm_unified_step(
        next_value=grids.next_value,
        next_marginal=grids.next_marginal,
        liquid_grid=grids.liquid,
        next_liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=_DISCOUNT,
        preferences=_preferences(specimen=specimen),
        next_liquid=grids.next_liquid,
        marginal_return=grids.marginal_return,
        coh_slopes=jnp.ones(2),
        coh_intercepts=jnp.full((2,), intercept),
        breakpoints=jnp.asarray([_CLIFF]),
        jump_mask=(True,),
        arithmetic=arithmetic,
    )


def _recurring_jump_step(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)
    return nbegm_recurring_jump_step(
        next_value=grids.next_value,
        next_marginal=grids.next_marginal,
        liquid_grid=grids.liquid,
        next_liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=_DISCOUNT,
        preferences=_preferences(specimen=specimen),
        next_liquid=grids.next_liquid,
        marginal_return=grids.marginal_return,
        subsidy_levels=jnp.full((2,), intercept),
        jump_breakpoints=jnp.asarray([_CLIFF]),
        arithmetic=arithmetic,
    )


def _one_asset_step(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)
    return nbegm_one_asset_step(
        next_value=grids.next_value,
        next_marginal=grids.next_marginal,
        liquid_grid=grids.liquid,
        next_liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=_DISCOUNT,
        preferences=_preferences(specimen=specimen),
        next_liquid=grids.next_liquid,
        marginal_return=grids.marginal_return,
        subsidy_when=intercept,
        subsidy_otherwise=intercept,
        asset_limit=_CLIFF,
        equality_owner="otherwise",
        arithmetic=arithmetic,
    )


def _discrete_envelope_step(
    *, intercept: FloatND, arithmetic: ComparisonArithmetic, specimen: str
) -> _Channels:
    grids = _grids(specimen=specimen)
    value, marginal, policy, _ = nbegm_discrete_envelope_step(
        next_value=grids.next_value,
        next_marginal=grids.next_marginal,
        liquid_grid=grids.liquid,
        next_liquid_grid=grids.liquid,
        savings_grid=grids.savings,
        discount_factor=_DISCOUNT,
        preferences=_preferences(specimen=specimen),
        next_liquid=grids.next_liquid,
        marginal_return=grids.marginal_return,
        choices=(
            {
                "coh_slopes": jnp.ones(1),
                "coh_intercepts": jnp.reshape(intercept, (1,)),
                "breakpoints": jnp.zeros((0,)),
            },
        ),
        arithmetic=arithmetic,
    )
    return value, marginal, policy


_STEPS: dict[str, _Step] = {
    "multi_interval_step_savings": _multi_interval_step_savings,
    "per_interval_continuation_step_savings": _per_interval_continuation_step_savings,
    "per_interval_continuation_step_savings_streamed": (
        _per_interval_continuation_step_savings_streamed
    ),
    "unified_step_savings": _unified_step_savings,
    "multi_interval_step": _multi_interval_step,
    "unified_step": _unified_step,
    "recurring_jump_step": _recurring_jump_step,
    "one_asset_step": _one_asset_step,
    "discrete_envelope_step": _discrete_envelope_step,
}
_ARITHMETICS = (
    pytest.param("ordinary", id="ordinary"),
    pytest.param("certified", id="certified", marks=_REQUIRES_KERNEL),
)


def _solve(
    *,
    step: str,
    intercept: float,
    arithmetic: ComparisonArithmetic,
    specimen: str = _SMOOTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one step eagerly at a scalar budget intercept."""
    value, marginal, policy = _STEPS[step](
        intercept=jnp.asarray(intercept), arithmetic=arithmetic, specimen=specimen
    )
    return np.asarray(value), np.asarray(marginal), np.asarray(policy)


def _solve_transformed(
    *, step: str, transform: str, arithmetic: ComparisonArithmetic
) -> np.ndarray:
    """Return the value at the unaffordable intercept under a JAX transform.

    - `"jit"` traces the intercept, so the carry is decided on a traced budget;
    - `"vmap"` maps the step over the unaffordable and the affordable intercept
      together and keeps the unaffordable row.
    """
    step_func = _STEPS[step]

    def value_at(intercept: FloatND) -> FloatND:
        return step_func(intercept=intercept, arithmetic=arithmetic, specimen=_SMOOTH)[
            0
        ]

    if transform == "jit":
        return np.asarray(jax.jit(value_at)(jnp.asarray(_UNAFFORDABLE_INTERCEPT)))
    mapped = jax.vmap(value_at)
    intercepts = jnp.asarray([_UNAFFORDABLE_INTERCEPT, _AFFORDABLE_INTERCEPT])
    return np.asarray(mapped(intercepts))[0]


@pytest.mark.parametrize("arithmetic", _ARITHMETICS)
@pytest.mark.parametrize("step", tuple(_STEPS))
def test_unaffordable_node_publishes_minus_inf_value(
    *, step: str, arithmetic: ComparisonArithmetic
) -> None:
    """The value is `-inf` exactly where cash-on-hand affords no action."""
    value, _, _ = _solve(
        step=step, intercept=_UNAFFORDABLE_INTERCEPT, arithmetic=arithmetic
    )

    np.testing.assert_array_equal(np.isneginf(value), _UNAFFORDABLE_NODES)


@pytest.mark.parametrize("arithmetic", _ARITHMETICS)
@pytest.mark.parametrize("step", tuple(_STEPS))
def test_unaffordable_node_publishes_nan_policy(
    *, step: str, arithmetic: ComparisonArithmetic
) -> None:
    """The consumption policy is NaN at every node that affords no action."""
    _, _, policy = _solve(
        step=step, intercept=_UNAFFORDABLE_INTERCEPT, arithmetic=arithmetic
    )

    assert np.isnan(policy[_UNAFFORDABLE_NODES]).all()


@pytest.mark.parametrize("arithmetic", _ARITHMETICS)
@pytest.mark.parametrize("step", tuple(_STEPS))
def test_unaffordable_node_publishes_zero_marginal(
    *, step: str, arithmetic: ComparisonArithmetic
) -> None:
    """The marginal value of liquid is zero at every node that affords no action."""
    _, marginal, _ = _solve(
        step=step, intercept=_UNAFFORDABLE_INTERCEPT, arithmetic=arithmetic
    )

    np.testing.assert_array_equal(marginal[_UNAFFORDABLE_NODES], 0.0)


@pytest.mark.parametrize("arithmetic", _ARITHMETICS)
@pytest.mark.parametrize("step", tuple(_STEPS))
def test_affordable_nodes_beside_unaffordable_ones_stay_finite(
    *, step: str, arithmetic: ComparisonArithmetic
) -> None:
    """Every node with positive cash-on-hand publishes a finite value."""
    value, _, _ = _solve(
        step=step, intercept=_UNAFFORDABLE_INTERCEPT, arithmetic=arithmetic
    )

    assert np.isfinite(value[~_UNAFFORDABLE_NODES]).all()


@pytest.mark.parametrize("arithmetic", _ARITHMETICS)
@pytest.mark.parametrize("step", tuple(_STEPS))
def test_budget_affording_an_action_everywhere_stays_finite(
    *, step: str, arithmetic: ComparisonArithmetic
) -> None:
    """With positive cash-on-hand at every node, every published value is finite."""
    value, _, _ = _solve(
        step=step, intercept=_AFFORDABLE_INTERCEPT, arithmetic=arithmetic
    )

    assert np.isfinite(value).all()


@pytest.mark.parametrize("transform", ["jit", "vmap"])
@pytest.mark.parametrize("step", tuple(_STEPS))
def test_unaffordable_node_publishes_minus_inf_value_under_transforms(
    *, step: str, transform: str
) -> None:
    """The `-inf` carry is decided on a traced budget under `jit` and `vmap`."""
    value = _solve_transformed(step=step, transform=transform, arithmetic="ordinary")

    np.testing.assert_array_equal(np.isneginf(value), _UNAFFORDABLE_NODES)


@pytest.mark.parametrize("specimen", [_NAN_CONTINUATION, _OVERFLOWING_FELICITY])
@pytest.mark.parametrize("arithmetic", _ARITHMETICS)
@pytest.mark.parametrize("step", tuple(_STEPS))
def test_affordable_node_without_a_finite_candidate_publishes_nan(
    *, step: str, arithmetic: ComparisonArithmetic, specimen: str
) -> None:
    """A node that affords an action but has no finite candidate publishes NaN,
    never `-inf` and never a finite value."""
    value, _, _ = _solve(
        step=step,
        intercept=_AFFORDABLE_INTERCEPT,
        arithmetic=arithmetic,
        specimen=specimen,
    )

    assert np.isnan(value).all()
