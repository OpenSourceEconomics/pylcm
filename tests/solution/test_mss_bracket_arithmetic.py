"""The asset-row bracket obeys the arithmetic the MSS envelope was given.

A regime solved per exogenous asset node reads the refined envelope at one query
per node through the bracket finder rather than through the full row. Both routes
build the same refined row, so the arithmetic the regime selects with
`MSSEnvelope(arithmetic=...)` has to reach both: the bracket route locates the
bracket in the row produced under that same selection, the ordinary arithmetic
never requests the exact-affine payload, and the certified arithmetic never
settles a comparison in the working format.
"""

import functools
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.comparison_arithmetic import ComparisonArithmetic
from _lcm.egm.upper_envelope import get_bracket_finder, get_upper_envelope, mss
from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, GridSearch, MSSEnvelope
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt
from lcm_examples.iskhakov_et_al_2017 import dead
from tests.conftest import EXACT_KERNEL_SKIP_REASON

requires_exact_kernel = pytest.mark.requires_exact_affine_kernel(
    reason=EXACT_KERNEL_SKIP_REASON
)
ARITHMETICS = [
    pytest.param("certified", marks=requires_exact_kernel),
    "ordinary",
]
TRANSFORMS = ["eager", "jit", "jit_vmap"]
N_REFINED = 20


def _solver(*, arithmetic: ComparisonArithmetic) -> DCEGM:
    return DCEGM(
        savings_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=4),
        envelope=MSSEnvelope(arithmetic=arithmetic),
    )


def _candidates() -> dict[str, jax.Array]:
    """Two branches whose values differ by one representable step at the handover.

    The working format cannot separate the branches at the crossing, so the row
    the ordinary arithmetic publishes differs from the certified one, and a
    bracket read in the wrong row publishes a different policy.
    """
    dtype = jnp.asarray(0.0).dtype
    above = np.nextafter(dtype.type(1), dtype.type(np.inf))
    return {
        "endog_grid": jnp.asarray([50, 58, 51, 55], dtype=dtype),
        "policy": jnp.asarray([49, 49, 39, 39], dtype=dtype),
        "value": jnp.asarray([1, above, 1, above], dtype=dtype),
        "marginal_utility": jnp.ones(4, dtype=dtype),
        "savings": jnp.arange(4, dtype=dtype),
    }


def _evaluate(
    *,
    function: Callable[..., tuple[jax.Array, ...]],
    kwargs: Mapping[str, jax.Array],
    transform: str,
) -> tuple[jax.Array, ...]:
    """Call `function` eagerly, under `jit`, or batched under `jit(vmap)`.

    The batched call returns a plain tuple from inside the map, the way the
    per-node solve consumes a bracket without ever returning the bracket struct
    out of the batch.
    """
    if transform == "eager":
        return tuple(function(**kwargs))
    if transform == "jit":
        return tuple(jax.jit(function)(**kwargs))
    batch = {key: jnp.stack([value, value]) for key, value in kwargs.items()}
    batched = jax.jit(jax.vmap(lambda **inputs: tuple(function(**inputs))))(**batch)
    return tuple(channel[0] for channel in batched)


def _bracket_read_off_the_row(
    *, row: tuple[jax.Array, ...], query: float
) -> list[np.ndarray]:
    """Locate the query bracket in a published row without a production helper."""
    grid, policy, value, kept = map(np.asarray, row)
    n_live = int(np.isfinite(grid).sum())
    search_grid = np.where(np.isnan(grid), np.inf, grid)
    upper = int(
        np.clip(
            np.searchsorted(search_grid, query, side="right"), 1, max(n_live - 1, 1)
        )
    )
    lower = upper - 1
    return [
        grid[lower],
        grid[upper],
        policy[lower],
        policy[upper],
        value[lower],
        value[upper],
        grid[0],
        kept,
    ]


@pytest.mark.parametrize("arithmetic", ARITHMETICS)
@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("query", [51.5, 52.5])
def test_the_bracket_is_read_in_the_row_of_the_selected_arithmetic(
    *, arithmetic: ComparisonArithmetic, transform: str, query: float
) -> None:
    """The bracket route publishes the pair the same arithmetic's full row holds."""
    solver = _solver(arithmetic=arithmetic)
    kwargs = _candidates()
    row = _evaluate(
        function=get_upper_envelope(solver=solver, n_refined=N_REFINED),
        kwargs=kwargs,
        transform=transform,
    )
    bracket = _evaluate(
        function=get_bracket_finder(solver=solver, n_refined=N_REFINED),
        kwargs={**kwargs, "x_query": jnp.asarray(query)},
        transform=transform,
    )
    np.testing.assert_array_equal(
        np.asarray(bracket), np.asarray(_bracket_read_off_the_row(row=row, query=query))
    )


class _PayloadRequestedError(RuntimeError):
    """Raised in place of every exact-affine entry point the envelope could reach."""


def _refuse_payload(*args: object, **kwargs: object) -> None:
    del args, kwargs
    msg = "the exact-affine payload was requested"
    raise _PayloadRequestedError(msg)


def _without_the_exact_payload() -> AbstractContextManager[object]:
    return patch.multiple(
        mss,
        certified_margin_sign=_refuse_payload,
        exact_affine_handover=_refuse_payload,
        exact_affine_read=_refuse_payload,
        exact_query_winner_batched=_refuse_payload,
    )


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_the_ordinary_bracket_requests_no_exact_payload(transform: str) -> None:
    """An ordinary bracket is located without any exact-affine entry point."""
    finder = get_bracket_finder(solver=_solver(arithmetic="ordinary"), n_refined=20)
    kwargs = {**_candidates(), "x_query": jnp.asarray(51.5)}
    jax.clear_caches()
    try:
        with _without_the_exact_payload():
            bracket = _evaluate(function=finder, kwargs=kwargs, transform=transform)
        assert all(np.isfinite(np.asarray(channel)).all() for channel in bracket)
    finally:
        jax.clear_caches()


@requires_exact_kernel
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_the_certified_bracket_requests_the_exact_payload(transform: str) -> None:
    """A certified bracket reaches the exact-affine payload rather than downgrading."""
    finder = get_bracket_finder(solver=_solver(arithmetic="certified"), n_refined=20)
    kwargs = {**_candidates(), "x_query": jnp.asarray(51.5)}
    jax.clear_caches()
    try:
        with (
            _without_the_exact_payload(),
            pytest.raises(_PayloadRequestedError, match="payload was requested"),
        ):
            _evaluate(function=finder, kwargs=kwargs, transform=transform)
    finally:
        jax.clear_caches()


# An asset-row model: the survival probability reads the Euler state `wealth`, so
# the regime is solved per exogenous asset node and every node's value is read
# through the bracket route. The oracle is a dense-grid brute-force solve of a
# mathematically equivalent specification.

N_PERIODS = 4
BAND_START = 30.0
BAND_WIDTH = 20.0
SURVIVAL_LOW = 0.55
SURVIVAL_HIGH = 0.95
RATE_OF_RETURN = 0.04
LABOR_INCOME = 5.0
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=160)
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=120.0, n_points=4000)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 149) ** 3 for i in range(150)))
# Lowest wealth nodes excluded from the comparison: there the brute solver leans on
# consumption choices near its grid start, where log utility curves hardest.
N_BRUTE_UNSTABLE_NODES = 16


@categorical(ordered=False)
class AssetRowRegimeId:
    working_life: ScalarInt
    dead: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def resources(*, wealth: ContinuousState, rate_of_return: float) -> FloatND:
    return wealth * (1.0 + rate_of_return)


def savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth_dcegm(savings: FloatND) -> ContinuousState:
    return savings + LABOR_INCOME


def next_wealth_brute(
    *, resources: FloatND, consumption: ContinuousAction
) -> ContinuousState:
    return resources - consumption + LABOR_INCOME


def budget_constraint(*, consumption: ContinuousAction, resources: FloatND) -> BoolND:
    return consumption <= resources


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    t = jnp.clip((wealth - BAND_START) / BAND_WIDTH, 0.0, 1.0)
    smoothstep = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * smoothstep


def stay_prob(*, wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_of_wealth(wealth))


def death_prob(*, wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return 1.0 - stay_prob(wealth=wealth, age=age, final_age_alive=final_age_alive)


def _active(age: int) -> bool:
    return age < 40 + (N_PERIODS - 1) * 10


def _params() -> dict:
    return {
        "discount_factor": 0.95,
        "final_age_alive": 40 + (N_PERIODS - 2) * 10,
        "rate_of_return": RATE_OF_RETURN,
    }


@functools.cache
def _asset_row_model(*, arithmetic: ComparisonArithmetic | None) -> Model:
    """The asset-row model under an MSS arithmetic, or its brute-force twin."""
    is_dcegm = arithmetic is not None
    regime_type = ConsumptionSavingsRegime if is_dcegm else UserRegime
    working_life = regime_type(
        transition={
            "working_life": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions=(
            {
                "utility": utility,
                "resources": resources,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            }
            if is_dcegm
            else {"utility": utility, "resources": resources}
        ),
        solver=(
            DCEGM(
                savings_grid=SAVINGS_GRID,
                n_constrained_points=64,
                envelope=MSSEnvelope(arithmetic=arithmetic),
            )
            if is_dcegm
            else GridSearch()
        ),
        **(
            {
                "liquid": LiquidMargin(
                    state="wealth",
                    action="consumption",
                    resources="resources",
                    post_decision_state="savings",
                )
            }
            if is_dcegm
            else {}
        ),
    )
    return Model(
        regimes={"working_life": working_life, "dead": dead},
        ages=AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y"),
        regime_id_class=AssetRowRegimeId,
    )


def _working_life_values(model: Model) -> dict[int, np.ndarray]:
    values = model.solve(params=_params(), log_level="debug").values
    return {
        period: np.asarray(values[period]["working_life"])
        for period in sorted(values)[:-1]
    }


@pytest.mark.parametrize("arithmetic", ARITHMETICS)
def test_an_asset_row_solve_under_either_arithmetic_matches_brute_force(
    arithmetic: ComparisonArithmetic,
) -> None:
    """Every asset node's value, read through the bracket route, matches brute force."""
    brute = _working_life_values(_asset_row_model(arithmetic=None))
    solved = _working_life_values(_asset_row_model(arithmetic=arithmetic))
    for period, brute_V in brute.items():
        np.testing.assert_allclose(
            solved[period][..., N_BRUTE_UNSTABLE_NODES:],
            brute_V[..., N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


def test_an_ordinary_asset_row_solve_requests_no_exact_payload() -> None:
    """A whole ordinary asset-row solve completes without the exact-affine payload."""
    model = _asset_row_model(arithmetic="ordinary")
    jax.clear_caches()
    try:
        with _without_the_exact_payload():
            values = _working_life_values(model)
        assert all(np.isfinite(V).all() for V in values.values())
    finally:
        jax.clear_caches()


@requires_exact_kernel
def test_a_certified_asset_row_solve_requests_the_exact_payload() -> None:
    """A certified asset-row solve reaches the exact-affine payload at every node."""
    model = _asset_row_model(arithmetic="certified")
    jax.clear_caches()
    try:
        with (
            _without_the_exact_payload(),
            pytest.raises(_PayloadRequestedError, match="payload was requested"),
        ):
            _working_life_values(model)
    finally:
        jax.clear_caches()
