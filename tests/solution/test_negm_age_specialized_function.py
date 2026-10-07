"""NEGM regimes honour `AgeSpecializedFunction` in their outer helpers per period.

An age-specialized function is a *different function* at each age, so a period's
kernel has to be built from that period's concrete function. NEGM reads two helpers
from the regime's function pool before the inner DC-EGM builder ever runs — the
no-adjustment map (`outer_continuous.no_adjustment`) and the adjustment
cost (`liquid.resources.cost`) — and both must be that period's own.

The oracle is the last age at which the NEGM regime is active. Its value depends on
exactly two things: its own economics and a continuation into the terminal regime,
which no age specialization touches. So the age-specialized solve must reproduce,
*exactly*, a plain solve whose helper is the concrete function `build(age)` returns
at that age. Resolving the specialization at some other age moves the value by a
finite amount, which makes equality the right instrument rather than a tolerance.

The same oracle isolates a NEGM *child's* keeper resources. An age-invariant
`early` regime moves into a `late` regime whose no-adjustment map depreciates the
durable by age, and `late` is active only at its last age. The parent's read of
`late` composes `late`'s resources with that keeper, so `early`'s solution at the
period before must equal the one with `late` pinned to the keeper of `late`'s own
age — and differ from one whose read alone uses the keeper of `early`'s age.

The parent's read is also checked directly. With a second root that makes `late`
active at the two ages before `dead`, `late`'s first active age is no longer the
age `early` reads, so the resources map in `early`'s built keeper continuation plan
must carry the keeper of `late`'s age at the following period, value and gradient
alike.
"""

import functools
from dataclasses import replace
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import regime_introspection
from _lcm.execution.core_program import core_program_graph
from lcm import (
    AgeGrid,
    AgeRange,
    AgeSpecializedFunction,
    DeterministicTransition,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    Transition,
    categorical,
    outer_unchanged,
)
from lcm.solver_api import EGM_CONTINUATION, ArtifactRef, ResultRetention
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.test_models import negm_kinked_toy
from tests.test_models.negm_kinked_toy import (
    N_PERIODS,
    NEGM_SOLVER,
    RegimeId,
    build_dead_regime,
    credited,
    inverse_marginal_utility,
    liquid_savings,
    new_durable,
    next_regime,
    next_wealth,
    resources_before_outer_cost,
    utility,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_MIN_AGE = 20
_AGE_STEP = 5
_FINAL_AGE_ALIVE = _MIN_AGE + (N_PERIODS - 2) * _AGE_STEP
_PARAMS = {"discount_factor": 0.95, "alive": {}}


def _make_depreciating_keep(age: float):
    """`keep_illiquid` with the no-adjustment stock depreciating by age."""
    retained = 1.0 - 0.02 * (age - _MIN_AGE)

    def keep_illiquid_at_age(illiquid: ContinuousState) -> FloatND:
        return retained * illiquid

    return keep_illiquid_at_age


def _make_penalised_credited(age: float):
    """`credited` with the withdrawal penalty drifting by age."""
    penalty = 0.10 + 0.02 * (age - _MIN_AGE)

    def credited_at_age(
        *, illiquid: ContinuousState, new_durable: ContinuousState
    ) -> FloatND:
        investment = new_durable - illiquid
        return jnp.where(investment < 0.0, (1.0 - penalty) * investment, investment)

    return credited_at_age


_HELPERS = {
    "keep_illiquid": _make_depreciating_keep,
    "credited": _make_penalised_credited,
}


def _build_model(*, helper_name: str, override) -> Model:
    """The kinked NEGM toy with one outer helper optionally replaced.

    `override` is the value to bind to `helper_name` — an `AgeSpecializedFunction`,
    a concrete function pinned to one age, or `None` to keep the model's own
    age-invariant helper.
    """
    functions = {
        "utility": utility,
        "new_durable": new_durable,
        "resources_before_outer_cost": resources_before_outer_cost,
        "liquid_savings": liquid_savings,
        "credited": credited,
        "inverse_marginal_utility": inverse_marginal_utility,
    }
    no_adjustment = outer_unchanged
    if override is not None:
        functions[helper_name] = override
        if helper_name == "keep_illiquid":
            no_adjustment = "keep_illiquid"
    alive_law = DeterministicTransition(func=next_regime)
    alive = NestedConsumptionSavingsRegime(
        states={
            "wealth": negm_kinked_toy.WEALTH_GRID,
            "illiquid": negm_kinked_toy.ILLIQUID_GRID,
        },
        state_transitions={
            "wealth": next_wealth,
            "illiquid": negm_kinked_toy.durable_transition,
        },
        actions={
            "consumption": negm_kinked_toy.CONSUMPTION_GRID,
            "illiquid_investment": negm_kinked_toy.ILLIQUID_INVESTMENT_GRID,
        },
        functions=functions,
        solver=replace(NEGM_SOLVER),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources=NetOfAdjustmentCost(
                output="resources",
                before_cost="resources_before_outer_cost",
                cost="credited",
            ),
            post_decision_state="liquid_savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state="new_durable",
            no_adjustment=no_adjustment,
        ),
    )
    return Model(
        regimes={"alive": alive, "dead": build_dead_regime()},
        edges={
            "alive": Transition(
                targets={
                    "alive": AgeRange(exclusive_stop=_FINAL_AGE_ALIVE),
                    "dead": AgeRange(exclusive_stop=_FINAL_AGE_ALIVE + _AGE_STEP),
                },
                law=alive_law,
            )
        },
        regime_id_class=RegimeId,
        ages=AgeGrid(
            start=_MIN_AGE,
            inclusive_stop=_MIN_AGE + (N_PERIODS - 1) * _AGE_STEP,
            step=f"{_AGE_STEP}Y",
        ),
        fixed_params={"final_age_alive": _FINAL_AGE_ALIVE},
        initial_nodes={20: "alive"},
    )


def _specialized(helper_name: str) -> AgeSpecializedFunction:
    build = _HELPERS[helper_name]
    return AgeSpecializedFunction(build=build, signature=lambda age: age)


@pytest.mark.parametrize("helper_name", ["keep_illiquid", "credited"])
def test_the_last_covered_age_uses_that_ages_own_outer_helper(helper_name):
    """The last covered age's value equals a plain solve pinned to that age's helper.

    The last covered period's value depends on its own economics and on a
    continuation into the terminal regime, which no age specialization touches. So
    the age-specialized solve must reproduce, *exactly*, a plain solve whose helper
    is the concrete function `build(age)` returns at that age.
    """
    build = _HELPERS[helper_name]
    specialized = (
        _build_model(helper_name=helper_name, override=_specialized(helper_name))
        .solve(params=_PARAMS, log_level="debug")
        .values
    )
    last_active = max(
        period for period, regimes in specialized.items() if "alive" in regimes
    )
    pinned = (
        _build_model(
            helper_name=helper_name,
            override=build(_MIN_AGE + last_active * _AGE_STEP),
        )
        .solve(params=_PARAMS, log_level="debug")
        .values
    )

    expected = np.asarray(pinned[last_active]["alive"])
    got = np.asarray(specialized[last_active]["alive"])
    np.testing.assert_array_equal(np.isneginf(got), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_almost_equal(
        got[finite], expected[finite], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("helper_name", ["keep_illiquid", "credited"])
def test_an_age_specialized_outer_helper_moves_the_negm_solution(helper_name):
    """The drifting helper changes the NEGM value function.

    Without this, the agreement test above would still pass if the specialization
    were ignored in both solves in the same way.
    """
    drifting = (
        _build_model(helper_name=helper_name, override=_specialized(helper_name))
        .solve(params=_PARAMS, log_level="debug")
        .values
    )
    flat = (
        _build_model(helper_name=helper_name, override=None)
        .solve(params=_PARAMS, log_level="debug")
        .values
    )

    moved = [
        period
        for period in drifting
        if "alive" in drifting[period]
        and not np.allclose(
            np.asarray(drifting[period]["alive"]),
            np.asarray(flat[period]["alive"]),
            equal_nan=True,
        )
    ]
    assert moved, "the age-specialized outer helper left every period unchanged"


@pytest.mark.parametrize("helper_name", ["keep_illiquid", "credited"])
def test_ages_sharing_one_signature_resolve_to_one_concrete_helper(helper_name):
    """A signature constant across ages makes every age use one concrete helper.

    `signature(age)` is the sharing key and a correctness precondition: ages with
    equal signatures share one compiled program, so an equal signature must imply an
    identical resolved closure. A specialization that declares one signature for
    every age therefore has to behave exactly like a plain solve pinned to the single
    helper that group resolves to — the first active age's.

    This is the companion to the per-age agreement test above. That one pins that
    distinct signatures are *not* collapsed; this one pins that equal signatures *are*
    shared rather than silently re-resolved per period.
    """
    build = _HELPERS[helper_name]
    constant_signature = (
        _build_model(
            helper_name=helper_name,
            override=AgeSpecializedFunction(build=build, signature=lambda _age: 0),
        )
        .solve(params=_PARAMS, log_level="debug")
        .values
    )
    pinned = (
        _build_model(helper_name=helper_name, override=build(_MIN_AGE))
        .solve(params=_PARAMS, log_level="debug")
        .values
    )

    for period, regimes in pinned.items():
        if "alive" not in regimes:
            continue
        expected = np.asarray(regimes["alive"])
        got = np.asarray(constant_signature[period]["alive"])
        finite = np.isfinite(expected)
        np.testing.assert_array_almost_equal(
            got[finite], expected[finite], decimal=DECIMAL_PRECISION
        )


@categorical(ordered=False)
class _EarlyLateId:
    early: ScalarInt
    late: ScalarInt
    dead: ScalarInt


_LATE_AGE = _MIN_AGE + 2 * _AGE_STEP
_EARLY_LAST_PERIOD = 1
_EARLY_LATE_PARAMS = {"discount_factor": 0.95, "early": {}, "late": {}}


def _negm_regime(*, keep_illiquid: Any) -> NestedConsumptionSavingsRegime:
    """The kinked NEGM regime with `keep_illiquid` as its no-adjustment map.

    `keep_illiquid=None` keeps the durable unchanged without adjusting.
    """
    functions = {
        "utility": utility,
        "new_durable": new_durable,
        "resources_before_outer_cost": resources_before_outer_cost,
        "liquid_savings": liquid_savings,
        "credited": credited,
        "inverse_marginal_utility": inverse_marginal_utility,
    }
    no_adjustment = outer_unchanged
    if keep_illiquid is not None:
        functions["keep_illiquid"] = keep_illiquid
        no_adjustment = "keep_illiquid"
    return NestedConsumptionSavingsRegime(
        states={
            "wealth": negm_kinked_toy.WEALTH_GRID,
            "illiquid": negm_kinked_toy.ILLIQUID_GRID,
        },
        state_transitions={
            "wealth": next_wealth,
            "illiquid": negm_kinked_toy.durable_transition,
        },
        actions={
            "consumption": negm_kinked_toy.CONSUMPTION_GRID,
            "illiquid_investment": negm_kinked_toy.ILLIQUID_INVESTMENT_GRID,
        },
        functions=functions,
        solver=replace(NEGM_SOLVER),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources=NetOfAdjustmentCost(
                output="resources",
                before_cost="resources_before_outer_cost",
                cost="credited",
            ),
            post_decision_state="liquid_savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state="new_durable",
            no_adjustment=no_adjustment,
        ),
    )


def _early_late_model(*, late_keep: Any, late_start_age: int = _LATE_AGE) -> Model:
    """`early` at the first two ages, then `late` until the third, then `dead`.

    `late_keep` is `late`'s no-adjustment map: an `AgeSpecializedFunction` or a
    concrete function pinned to one age. `late_start_age` below `_LATE_AGE` adds a
    root in `late` at that age, so `late` is active from there through `_LATE_AGE`.
    """
    initial_nodes: list[tuple[object, str]] = [(_MIN_AGE, "early")]
    late_edges: dict[str, Any] = {"dead": _LATE_AGE}
    if late_start_age < _LATE_AGE:
        initial_nodes.append((late_start_age, "late"))
        late_edges["late"] = AgeRange(start=late_start_age, exclusive_stop=_LATE_AGE)
    return Model(
        edges={
            "early": {"early": _MIN_AGE, "late": _MIN_AGE + _AGE_STEP},
            "late": late_edges,
        },
        regimes={
            "early": _negm_regime(
                keep_illiquid=None,
            ),
            "late": _negm_regime(
                keep_illiquid=late_keep,
            ),
            "dead": build_dead_regime(),
        },
        regime_id_class=_EarlyLateId,
        ages=AgeGrid(
            start=_MIN_AGE,
            inclusive_stop=_MIN_AGE + (N_PERIODS - 1) * _AGE_STEP,
            step=f"{_AGE_STEP}Y",
        ),
        initial_nodes=initial_nodes,
    )


def _early_continuation(*, late_keep: Any, read_keep: Any = None) -> Any:
    """`early`'s published continuation at its last period, before `late`.

    `read_keep`, if given, replaces `late`'s keeper in the parent's read of `late`
    only; `late`'s own solve keeps `late_keep`.
    """
    with pytest.MonkeyPatch.context() as patch:
        if read_keep is not None:
            patch.setattr(
                regime_introspection,
                "_keeper_no_adjustment_function",
                _with_read_keep(read_keep),
            )
        result = _early_late_model(late_keep=late_keep).solve(
            params=_EARLY_LATE_PARAMS,
            log_level="off",
            retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
        )
    return result.retained_continuations[
        ArtifactRef(period=_EARLY_LAST_PERIOD, regime="early", key=EGM_CONTINUATION)
    ]


def _with_read_keep(read_keep: Any) -> Any:
    """The parent-read keeper builder with a declared keeper swapped for `read_keep`."""
    build = regime_introspection._keeper_no_adjustment_function

    def build_with_read_keep(*, no_adjustment_func: Any, **kwargs: Any) -> Any:
        return build(
            no_adjustment_func=None if no_adjustment_func is None else read_keep,
            **kwargs,
        )

    return build_with_read_keep


@pytest.fixture(scope="module")
def early_continuations() -> dict[str, Any]:
    """`early`'s continuation with `late`'s keeper specialized and pinned."""
    return {
        "specialized": _early_continuation(late_keep=_specialized("keep_illiquid")),
        "pinned-to-child-age": _early_continuation(
            late_keep=_make_depreciating_keep(_LATE_AGE)
        ),
        "read-at-source-age": _early_continuation(
            late_keep=_make_depreciating_keep(_LATE_AGE),
            read_keep=_make_depreciating_keep(_LATE_AGE - _AGE_STEP),
        ),
    }


@pytest.mark.parametrize("field", ["endog_grid", "value", "marginal_utility"])
def test_parent_reads_a_negm_childs_keeper_at_the_childs_age(
    *, early_continuations: dict[str, Any], field: str
) -> None:
    """The parent's endogenous grid, value and marginal match the child-age pin.

    `late` is solved only at its last age, so its own solution is the same under
    the specialized keeper and the keeper pinned to that age; `early`, reading
    `late`'s keeper resources the period before, must then agree exactly too.
    """
    got = np.asarray(getattr(early_continuations["specialized"], field))
    expected = np.asarray(getattr(early_continuations["pinned-to-child-age"], field))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=10.0**-DECIMAL_PRECISION)


def test_parent_value_moves_when_the_childs_keeper_is_the_sources_age(
    *, early_continuations: dict[str, Any]
) -> None:
    """Reading `late`'s keeper at `early`'s age changes `early`'s value.

    `late` publishes its continuation on its keeper's cash-on-hand axis,
    `wealth + 5 - credited(illiquid, keep(illiquid))`. Its own keeper retains
    `0.8` of the durable, which puts a durable `z` at cash `wealth + 5 + 0.18 z`;
    a read with `early`'s keeper (retaining `0.9`) queries `wealth + 5 + 0.09 z`,
    `0.09 z` lower on a value strictly increasing in cash. Every cell whose
    continuation carries a positive durable into `late` therefore moves.

    `late`'s own solve keeps its own keeper in both solves, so its published
    carry is common to the two and only the parent's query differs. Changing the
    keeper of `late`'s solve as well would change that carry together with the
    query, and raw parent values from two such models need not separate.
    """
    specialized = np.asarray(early_continuations["specialized"].value)
    source_age = np.asarray(early_continuations["read-at-source-age"].value)
    assert not np.allclose(specialized, source_age, equal_nan=True)


_WEALTH_AT_READ = 3.0
_DURABLE_AT_READ = 4.0


def _keeper_resources_slope(age: int) -> Fraction:
    """`d resources / d illiquid` of the keeper map at `age`, on a withdrawal.

    The keeper retains `1 - (age - 20) / 50` of the durable, and the withdrawn
    share returns `9 / 10` of its value, so keeping lifts resources by
    `(9 / 10) (age - 20) / 50` per unit durable.
    """
    return Fraction(9, 10) * Fraction(age - _MIN_AGE, 50)


def _keeper_resources_and_gradient(age: int) -> np.ndarray:
    """`(R, dR/dwealth, dR/dilliquid)` of the keeper map at `age`, at the read point.

    `R = wealth + 5 + slope(age) * illiquid` at `(wealth, illiquid) = (3, 4)`: at
    age 30 that is `(218/25, 1, 9/50)`, at age 25 `(209/25, 1, 9/100)`.
    """
    slope = _keeper_resources_slope(age)
    resources = (
        Fraction(_WEALTH_AT_READ)
        + Fraction(negm_kinked_toy.LABOUR_INCOME)
        + slope * Fraction(_DURABLE_AT_READ)
    )
    return np.asarray([float(resources), 1.0, float(slope)])


def _early_reads_of_late(*, model: Model) -> Any:
    """The read of `late` in `early`'s built keeper continuation plan at age 25."""
    assert {_EARLY_LAST_PERIOD, _EARLY_LAST_PERIOD + 1} <= set(
        model._regimes["late"].solution.period_kernels
    )
    kernel: Any = model._regimes["early"].solution.period_kernels[_EARLY_LAST_PERIOD]
    step: Any = core_program_graph(kernel=kernel.keeper_kernel)["replay"].function
    seen: set[int] = set()
    while not hasattr(step, "pieces"):
        assert id(step) not in seen, "the keeper replay wraps no EGM step"
        seen.add(id(step))
        step = step.func if isinstance(step, functools.partial) else step.__wrapped__
    return step.pieces.continuation_plan.child_reads["late"]


# `late`'s keeper per variant; `late` is active at the source age and the next.
_LATE_KEEPS = {
    "specialized": lambda: _specialized("keep_illiquid"),
    "pinned-to-child-age": lambda: _make_depreciating_keep(_LATE_AGE),
    "pinned-to-source-age": lambda: _make_depreciating_keep(_LATE_AGE - _AGE_STEP),
}

# The age whose keeper each variant's read must carry.
_READ_AGE = {
    "specialized": _LATE_AGE,
    "pinned-to-child-age": _LATE_AGE,
    "pinned-to-source-age": _LATE_AGE - _AGE_STEP,
}


@pytest.fixture(scope="module", params=tuple(_LATE_KEEPS))
def late_read(request: pytest.FixtureRequest) -> tuple[str, Any]:
    """A variant name and `early`'s read of `late` in the two-root model."""
    model = _early_late_model(
        late_keep=_LATE_KEEPS[request.param](),
        late_start_age=_LATE_AGE - _AGE_STEP,
    )
    return request.param, _early_reads_of_late(model=model)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_parent_reads_the_keeper_resources_of_the_childs_age(
    *, late_read: tuple[str, Any], jit: bool
) -> None:
    """`early`'s read of `late` carries the keeper of `late`'s age, not its first.

    `late` is active at 25 and 30 and `early` at 25 reads `late` at 30. The
    specialized keeper and the keeper pinned to 30 must both give the age-30
    resources map, `(R, dR/dwealth, dR/dilliquid) = (8.72, 1, 0.18)` at
    `(wealth, illiquid) = (3, 4)`; the keeper pinned to 25 gives
    `(8.36, 1, 0.09)`. A specialized read resolved at `late`'s first active age
    would return the age-25 triple and fail.
    """
    variant, read = late_read
    dtype = jnp.asarray(1.0).dtype

    def resources(point: FloatND) -> FloatND:
        return read.resources_func(wealth=point[0], illiquid=point[1])

    evaluate = jax.value_and_grad(resources)
    if jit:
        evaluate = jax.jit(evaluate)
    value, gradient = evaluate(
        jnp.asarray([_WEALTH_AT_READ, _DURABLE_AT_READ], dtype=dtype)
    )
    got = np.concatenate([np.asarray(value).reshape(1), np.asarray(gradient)])
    expected = _keeper_resources_and_gradient(_READ_AGE[variant]).astype(dtype)
    np.testing.assert_allclose(
        got, expected, rtol=0.0, atol=16 * float(np.finfo(dtype).eps)
    )


def test_keeper_resources_of_the_two_late_ages_differ_beyond_rounding() -> None:
    """The age-30 and age-25 keeper maps differ by 0.36 in R and 0.09 in dR/dz."""
    gap = _keeper_resources_and_gradient(_LATE_AGE) - _keeper_resources_and_gradient(
        _LATE_AGE - _AGE_STEP
    )
    np.testing.assert_allclose(gap, [0.36, 0.0, 0.09], rtol=0.0, atol=1e-12)
