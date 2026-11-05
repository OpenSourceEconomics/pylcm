"""Collective current shocks select and route one common policy before storage."""

from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from _lcm.policy_queries import _PolicyQueryKernel
from _lcm.simulation.random_address import (
    RandomAddress,
    draw_window_keys,
    encode_subject_row,
)
from _lcm.solution import backward_induction as backward
from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    ParetoObjective,
    Periods,
    Regime,
    categorical,
)
from lcm.exceptions import ExecutionPlanningError, RegimeInitializationError
from lcm.shocks import Discrete, Normal, ShockGroup, UserNodes
from lcm.solver_api import SolutionResult
from lcm.typing import FloatND, IntND, ScalarInt
from tests.conftest import DECIMAL_PRECISION
from tests.solution import test_exit_reference_household as reference
from tests.solution import test_public_exit_queries as public

pytestmark = pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="both"
)


@categorical(ordered=False)
class _Regimes:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


@categorical(ordered=False)
class _HouseholdRegimes:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt
    couple_terminal: ScalarInt


@categorical(ordered=False)
class _WageShock:
    low: ScalarInt
    high: ScalarInt


def _wage_value(wage_node_code: IntND) -> FloatND:
    return 2.0 * wage_node_code - 1.0


def _female(*, action: IntND, eps: FloatND, wealth: FloatND) -> FloatND:
    return wealth + jnp.where(action == 0, 5.0 + 5 * eps, 6.0)


def _male(*, action: IntND, eps: FloatND, wealth: FloatND) -> FloatND:
    return wealth + jnp.where(action == 0, 5.0 - 5 * eps, 2.0)


def _outside() -> float:
    return 4.0


def _unavailable() -> bool:
    return False


def _empty() -> bool:
    return False


def _model(
    *,
    exits: bool,
    enable_jit: bool,
    width: int,
    zero_weight: bool = False,
    available: bool = True,
    empty: bool = False,
    budget: int | None = None,
    reverse: bool = False,
) -> Model:
    regimes = {
        "couple": Regime(
            states={"wealth": LinSpacedGrid(start=0, stop=2, n_points=3)},
            actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
            functions={
                "utility": CollectiveUtility(
                    utilities={"m": _male, "f": _female}
                    if reverse
                    else {"f": _female, "m": _male},
                    objective=ParetoObjective(weights={"f": 1.0, "m": 0.0})
                    if zero_weight
                    else None,
                )
            },
            shocks=("innovation",),
            exits={
                "divorce": Exit(
                    targets={
                        "f": ExitTarget(regime="single_f", projection={}),
                        "m": ExitTarget(regime="single_m", projection={}),
                    },
                    rule="unilateral",
                    available=None if available else _unavailable,
                )
            }
            if exits
            else {},
            constraints={"ordinary": _empty} if empty else {},
        ),
        "single_f": Regime(functions={"utility": _outside}),
        "single_m": Regime(functions={"utility": _outside}),
    }
    return Model(
        regimes=dict(reversed(tuple(regimes.items()))) if reverse else regimes,
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=1)},
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0], [1.0], [1.0]]),
                    weights=np.asarray([0.5, 0.125, 0.375]),
                ),
            )
        },
        n_periods=1,
        regime_id_class=_Regimes,
        initial_nodes=InitialNodes(by_period={0: ("couple",)}),
        edges={},
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(
            axis_widths={"query_cell": width},
            device_memory_bytes=budget,
        ),
    )


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("exits", [False, True])
def test_common_policy_is_routed_before_current_node_integration(
    *, exits: bool, enable_jit: bool, width: int
) -> None:
    """Every node vetoes stay although its averaged tuple (5,5) participates."""
    q = reference._q(f=[Fraction(0), Fraction(6)], m=[Fraction(10), Fraction(2)])
    outside: dict[str, Fraction | float] = {"f": Fraction(4), "m": Fraction(4)}
    assert (
        reference.unilateral(
            q=q, feasible=reference._all(2), v_exit=outside, weight=reference.HALF
        )
        == 2
    )
    assert (
        reference.unilateral(
            q=reference._q(f=[Fraction(5), Fraction(6)], m=[Fraction(5), Fraction(2)]),
            feasible=reference._all(2),
            v_exit=outside,
            weight=reference.HALF,
        )
        == 0
    )
    model = _model(exits=exits, enable_jit=enable_jit, width=width)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    values = result.values[0]["couple"]
    assert values.shape == (3, 2)
    assert model._regimes["couple"].solution.state_names == ("wealth",)
    assert_array_almost_equal(
        values,
        [[4.0, 4.0], [4.0, 4.0], [8.0, 4.0]]
        if exits
        else [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],
        decimal=DECIMAL_PRECISION,
    )


def _single_f(*, experience: IntND, work: IntND, z: FloatND) -> FloatND:
    return 1 + work * (2 + experience + z) + 2.5 * (1 - work)


def _single_m(*, experience: IntND, work: IntND, z: FloatND) -> FloatND:
    return 1 + work * (3 + experience + z) + 2.5 * (1 - work)


def _consumption(
    *,
    experience_f: IntND,
    experience_m: IntND,
    work_f: IntND,
    work_m: IntND,
    z: FloatND,
) -> FloatND:
    return 1 + work_f * (2 + experience_f + z) + work_m * (3 + experience_m + z)


def _couple_f(*, consumption: FloatND, work_f: IntND) -> FloatND:
    return consumption / 2 + 2.5 * (1 - work_f) + 1


def _couple_m(*, consumption: FloatND, work_m: IntND) -> FloatND:
    return consumption / 2 + 2.5 * (1 - work_m) + 1


def _experience(experience: IntND) -> IntND:
    return experience


def _experience_f(experience_f: IntND) -> IntND:
    return experience_f


def _experience_m(experience_m: IntND) -> IntND:
    return experience_m


def _next_experience(*, experience: IntND, work: IntND) -> IntND:
    return jnp.minimum(experience + work, 2)


def _next_f(*, experience_f: IntND, work_f: IntND) -> IntND:
    return jnp.minimum(experience_f + work_f, 2)


def _next_m(*, experience_m: IntND, work_m: IntND) -> IntND:
    return jnp.minimum(experience_m + work_m, 2)


def _household_model() -> Model:
    experience = DiscreteGrid(values=jnp.arange(3))
    work = DiscreteGrid(values=jnp.arange(2))
    singles = {
        f"single_{member}": Regime(
            states={"experience": experience},
            actions={"work": work},
            functions={
                "utility": _single_f if member == "f" else _single_m,
                "z": _wage_value,
            },
            shocks=("wage",),
            state_transitions={"experience": _next_experience},
            exits={
                "marry": Exit(
                    targets={
                        "self": ExitTarget(
                            regime="couple",
                            role=member,
                            projection={
                                "experience_f": _experience,
                                "experience_m": _experience,
                            },
                        )
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_m" if member == "f" else "single_f",
                            role="m" if member == "f" else "f",
                            projection={"experience": _experience},
                        )
                    ),
                )
            },
        )
        for member in ("f", "m")
    }
    regimes = {
        **singles,
        "couple": Regime(
            states={"experience_f": experience, "experience_m": experience},
            actions={"work_f": work, "work_m": work},
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _couple_f, "m": _couple_m}
                ),
                "consumption": _consumption,
                "z": _wage_value,
            },
            state_transitions={"experience_f": _next_f, "experience_m": _next_m},
            shocks=("wage",),
            exits={
                "divorce": Exit(
                    targets={
                        "f": ExitTarget(
                            regime="single_f",
                            projection={"experience": _experience_f},
                        ),
                        "m": ExitTarget(
                            regime="single_m",
                            projection={"experience": _experience_m},
                        ),
                    },
                    rule="unilateral",
                )
            },
        ),
    }
    for name, regime in tuple(regimes.items()):
        regimes[f"{name}_terminal"] = replace(
            regime,
            state_transitions={},
            exits={
                key: replace(
                    declaration,
                    targets={
                        member: replace(target, regime=f"{target.regime}_terminal")
                        for member, target in declaration.targets.items()
                    },
                    rule=replace(
                        declaration.rule,
                        counterparty=replace(
                            declaration.rule.counterparty,
                            regime=f"{declaration.rule.counterparty.regime}_terminal",
                        ),
                    )
                    if isinstance(declaration.rule, Consent)
                    else declaration.rule,
                )
                for key, declaration in regime.exits.items()
            },
        )
    return Model(
        regimes=regimes,
        shocks={
            "wage": ShockGroup(
                shocks={"wage_node": Discrete(categories=_WageShock, probs=(0.5, 0.5))},
            )
        },
        n_periods=2,
        regime_id_class=_HouseholdRegimes,
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        edges={
            name: {f"{name}_terminal": Periods(values=(0,))}
            for name in ("couple", "single_f", "single_m")
        },
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def test_full_household_nonterminal_collective_anchor() -> None:
    """Full routed terminal values feed each member's next-period continuation."""
    model = _household_model()
    expected = reference.solve(replace(reference.Params(), n_periods=2))
    solution = model.solve(
        params={"discount_factor": 0.9}, log_level="off", runtime_checks=False
    )
    assert expected.V.couple["f"][0, 0, 0] == Fraction(351, 40)
    assert_array_almost_equal(
        solution.values[0]["couple"][0, 0],
        [float(Fraction(351, 40))] * 2,
        decimal=DECIMAL_PRECISION,
    )
    for period in range(2):
        for member in ("f", "m"):
            assert_array_almost_equal(
                solution.values[period][
                    f"single_{member}{'_terminal' if period else ''}"
                ],
                [float(expected.V.single[member][period, e]) for e in range(3)],
                decimal=DECIMAL_PRECISION,
            )
        names = model._regimes["couple"].solution.state_names
        target = np.empty((3, 3, 2))
        for ef in range(3):
            for em in range(3):
                coordinates = tuple(
                    {"experience_f": ef, "experience_m": em}[name] for name in names
                )
                target[coordinates] = [
                    float(expected.V.couple[member][period, ef, em])
                    for member in ("f", "m")
                ]
        assert_array_almost_equal(
            solution.values[period]["couple_terminal" if period else "couple"],
            target,
            decimal=DECIMAL_PRECISION,
        )
    _check_household_replay(model=model, solution=solution, expected=expected)


def _check_household_replay(
    *, model: Model, solution: SolutionResult, expected: reference.Solution
) -> None:
    """Choose at addressed finite-law draws and advance the final actor."""
    size = 16
    frame = model.simulate(
        params={"discount_factor": 0.9},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(size, jnp.int32),
            "regime_id": jnp.asarray(
                [model.regime_names_to_ids[name] for name in ("single_f", "couple")]
                * 8,
                jnp.int32,
            ),
            "own_stakeholder": jnp.asarray(
                [-1, model.stakeholder_names_to_ids["m"]] * 8, jnp.int32
            ),
            "experience": jnp.zeros(size, jnp.int32),
            "experience_f": jnp.asarray([0, 1] * 8, jnp.int32),
            "experience_m": jnp.asarray([0, 2] * 8, jnp.int32),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    subjects = [("single", "f", 0, 0), ("couple", "m", 2, 1)] * 8
    moves = set()
    for period in range(2):
        keys = draw_window_keys(
            key=jax.random.key(17),
            address_words=np.asarray(
                RandomAddress(kind="shock", name="wage_node").words(), np.uint32
            ),
            subject_start=encode_subject_row(row=0),
            last_real_subject=encode_subject_row(row=size - 1),
            subject_count=size,
            period=period,
        )
        shocks = np.where(np.asarray(jax.vmap(jax.random.uniform)(keys)) >= 0.5, 1, -1)
        for subject in range(size):
            regime, sex, own, partner = subjects[subject]
            node = replace(
                reference.Params(),
                n_periods=2,
                wage_shock_nodes=(Fraction(float(shocks[subject])),),
            )
            ef, em = (own, partner) if sex == "f" else (partner, own)
            if regime == "single":
                choice, _ = reference.single_decision(
                    p=node,
                    V=expected.V,
                    t=period,
                    sex=sex,
                    e=own,
                    z=0,
                    offer=1,
                    partner=own,
                    job_offer=1,
                    theta=0,
                )
                if choice == reference.MARRY:
                    regime, partner, ef, em = "couple", own, own, own
                    moves.add("marry")
            else:
                choice, _ = reference.couple_decision(
                    p=node, V=expected.V, t=period, ef=ef, em=em, z=0, theta=0
                )
                if choice == reference.DIVORCE:
                    regime = "single"
                    moves.add("divorce")
            row = frame.loc[
                (frame["period"] == period) & (frame["subject_id"] == subject)
            ].iloc[0]
            acting = "couple" if regime == "couple" else f"single_{sex}"
            assert row["regime_name"] == (f"{acting}_terminal" if period else acting), (
                period,
                subject,
                subjects[subject],
                int(shocks[subject]),
            )
            assert row["wage_node"] == (shocks[subject] + 1) // 2
            if regime == "couple":
                policy = reference.couple_participating(
                    p=node, V=expected.V, t=period, ef=ef, em=em, z=0, theta=0
                )
                work_f, work_m = reference.COUPLE_ALTERNATIVES[policy.choice]
                np.testing.assert_array_equal(
                    [row["work_f"], row["work_m"]], [work_f, work_m]
                )
                np.testing.assert_array_equal(
                    [row["experience_f"], row["experience_m"]], [ef, em]
                )
                assert_array_almost_equal(
                    [row["value_f"], row["value_m"]],
                    [float(policy.values[member]) for member in ("f", "m")],
                    decimal=DECIMAL_PRECISION,
                )
                work_own, work_partner = (
                    (work_f, work_m) if sex == "f" else (work_m, work_f)
                )
                own, partner = min(own + work_own, 2), min(partner + work_partner, 2)
            else:
                policy = reference.single_stay(
                    p=node, V=expected.V, t=period, sex=sex, e=own, z=0, job_offer=1
                )
                assert row["work"] == policy.choice
                assert row["experience"] == own
                assert_array_almost_equal(
                    row["value"], float(policy.values[sex]), decimal=DECIMAL_PRECISION
                )
                own = min(own + policy.choice, 2)
            subjects[subject] = regime, sex, own, partner
    assert moves == {"marry", "divorce"}


def _hidden_target(*, action: IntND, x: FloatND, y: FloatND) -> FloatND:
    return 2.0 + action * (x + y)


def _hidden_source(eps: FloatND) -> FloatND:
    return -10.0 + eps


def _observing_source(*, eps: FloatND, x: FloatND) -> FloatND:
    return -10.0 + eps + x


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("correlated", [False, True])
def test_collective_prices_hidden_target_policy_with_original_joint_law(
    *, observed: bool, correlated: bool
) -> None:
    """Caller observations preserve independent hidden rows or refuse conditioning."""
    joint = ShockGroup(
        shocks={"x": Normal(mu=0, sigma=1), "y": Normal(mu=0, sigma=1)},
        **({"correlation": ((1.0, 0.5), (0.5, 1.0))} if correlated else {}),
        nodes=UserNodes(
            nodes=np.asarray([[-1.0, -1.0], [1.0, 1.0]]), weights=np.asarray([0.5, 0.5])
        ),
    )
    model = Model(
        regimes={
            "couple": Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities=dict.fromkeys(
                            ("f", "m"),
                            _observing_source if observed else _hidden_source,
                        )
                    )
                },
                shocks=("innovation", "joint") if observed else ("innovation",),
                exits={
                    "divorce": Exit(
                        targets={
                            member: ExitTarget(regime=f"single_{member}", projection={})
                            for member in ("f", "m")
                        },
                        rule="unilateral",
                    )
                },
            ),
            **{
                f"single_{member}": Regime(
                    actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                    functions={"utility": _hidden_target},
                    shocks=("joint",),
                )
                for member in ("f", "m")
            },
        },
        shocks={
            "joint": joint,
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=1)},
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0], [1.0], [0.0]]),
                    weights=np.asarray([0.25, 0.25, 0.5]),
                ),
            ),
        },
        n_periods=1,
        regime_id_class=_Regimes,
        initial_nodes=InitialNodes(by_period={0: ("couple",)}),
        edges={},
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    if observed and correlated:
        with pytest.raises(ExecutionPlanningError, match="unresolved numerical status"):
            model.solve(params={}, log_level="off", runtime_checks=False)
        return
    expected = (
        2.5 if observed else 2.0 + (1.5 + np.sqrt(0.75)) / 2 if correlated else 3.0
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["couple"], [expected, expected], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_zero_weight_member_keeps_its_veto_and_empty_stay_can_exit(
    *, available: bool, empty: bool
) -> None:
    """Zero objective weight never removes a member's participation threshold."""
    model = _model(
        exits=True,
        enable_jit=True,
        width=1,
        zero_weight=True,
        available=available,
        empty=empty,
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    expected = (
        [[4.0, 4.0]] * 3
        if available and empty
        else [[4.0, 4.0], [4.0, 4.0], [8.0, 4.0]]
        if available
        else [[-np.inf, -np.inf]] * 3
        if empty
        else [[8.0, 1.0], [9.0, 2.0], [10.0, 3.0]]
    )
    assert_array_almost_equal(
        solution.values[0]["couple"], expected, decimal=DECIMAL_PRECISION
    )


def test_collective_current_shocks_replay_same_original_subject_and_role() -> None:
    """Fresh continuous draws route members to their own singleton destinations."""
    model = _model(exits=True, enable_jit=True, width=1)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    ids = model.regime_names_to_ids
    roles = model.stakeholder_names_to_ids
    frame = (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(8, jnp.int32),
                "wealth": jnp.zeros(8),
                "regime_id": jnp.full(8, ids["couple"], jnp.int32),
                "own_stakeholder": jnp.asarray([roles["f"], roles["m"]] * 4, jnp.int32),
            },
            seed=17,
            log_level="off",
            runtime_checks=False,
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
    )
    keys = draw_window_keys(
        key=jax.random.key(17),
        address_words=np.asarray(
            RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
        ),
        subject_start=encode_subject_row(row=0),
        last_real_subject=encode_subject_row(row=7),
        subject_count=8,
        period=0,
    )
    eps = np.asarray(jax.vmap(jax.random.normal)(keys))
    move = np.abs(eps) > 0.2
    assert move.any()
    expected_regime = np.where(move, np.asarray(["single_f", "single_m"] * 4), "couple")
    np.testing.assert_array_equal(frame["regime_name"], expected_regime)
    np.testing.assert_array_equal(frame["subject_id"], np.arange(8))
    assert_array_almost_equal(frame.loc[move, "value"], 4.0, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(
        frame.loc[~move, "value_f"], 5 + 5 * eps[~move], decimal=DECIMAL_PRECISION
    )
    assert_array_almost_equal(
        frame.loc[~move, "value_m"], 5 - 5 * eps[~move], decimal=DECIMAL_PRECISION
    )


def test_collective_native_budget_includes_member_banks() -> None:
    """The native budget includes the member payload before any query executes."""
    budget = 2**26
    model = _model(exits=True, enable_jit=True, width=1, budget=budget)
    with patch.object(
        backward, "_log_core_plan_record", wraps=backward._log_core_plan_record
    ) as observed:
        solution = model.solve(params={}, log_level="debug", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["couple"],
        [[4.0, 4.0], [4.0, 4.0], [8.0, 4.0]],
        decimal=DECIMAL_PRECISION,
    )
    kernel = model._regimes["couple"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    banks = kernel.programs["_initialize"].requirements.host_buffers["query_banks"]
    assert isinstance(banks, jax.ShapeDtypeStruct)
    assert banks.shape == (2, 3, 2)
    member_bank_bytes = (
        2 * 3 * 2 * np.dtype(solution.values[0]["couple"].dtype).itemsize
    )
    records = [call.kwargs for call in observed.call_args_list]
    assert records
    for record in records:
        assert record["resident_bytes"] >= member_bank_bytes
        assert record["memory"].reservation_bytes + record["resident_bytes"] <= budget
    refused = _model(exits=True, enable_jit=True, width=1, budget=member_bank_bytes - 1)
    kernel = refused._regimes["couple"].solution.period_kernels[0]
    with (
        patch.object(
            type(kernel),
            "__call__",
            side_effect=AssertionError("An unadmitted query must not execute."),
        ),
        pytest.raises(ExecutionPlanningError, match="budget"),
    ):
        refused.solve(params={}, log_level="off", runtime_checks=False)


def _female_offer(
    *, action: IntND, wealth: FloatND, married_last_period: IntND, offer_f_code: IntND
) -> FloatND:
    return 20.0 - 14.0 * action + offer_f_code + 0 * wealth + 0 * married_last_period


def _male_offer(
    *, action: IntND, wealth: FloatND, married_last_period: IntND, offer_m_code: IntND
) -> FloatND:
    return 4.0 + 2.0 * action + offer_m_code + 0 * wealth + 0 * married_last_period


@pytest.mark.parametrize("observed", [False, True])
def test_formation_shares_both_current_offers_at_mapped_marital_history(
    *, observed: bool
) -> None:
    """Newlyweds price lag-zero participation with both spouses' current offers."""
    base = public._household(newlywed=True, enable_jit=True)
    source = base.user_regimes["single_f"]
    regimes = {
        **base.user_regimes,
        "single_f": replace(
            source,
            exits={
                "marry": replace(
                    source.exits["marry"],
                    observes=("offer_f", "offer_m") if observed else ("offer_f",),
                )
            },
        ),
        "couple": replace(
            base.user_regimes["couple"],
            shocks=("offers",),
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _female_offer, "m": _male_offer}
                )
            },
        ),
    }

    def build() -> Model:
        return Model(
            regimes=regimes,
            shocks={
                "offers": ShockGroup(
                    shocks={
                        name: Discrete(categories=public._Offer, probs=(0.5, 0.5))
                        for name in ("offer_f", "offer_m")
                    }
                )
            },
            n_periods=1,
            regime_id_class=public._Household,
            initial_nodes=base.initial_nodes,
            edges={},
            fixed_params=base.fixed_params,
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )

    if not observed:
        with pytest.raises(
            RegimeInitializationError, match="unobserved decision information"
        ):
            build()
        return
    model = build()
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["single_f"], 5.75, decimal=DECIMAL_PRECISION
    )
    names = model._regimes["couple"].solution.state_names
    by_lag = np.moveaxis(
        np.asarray(solution.values[0]["couple"]), names.index("married_last_period"), 0
    )
    assert_array_almost_equal(
        by_lag[0],
        np.broadcast_to([13.5, 5.5], by_lag[0].shape),
        decimal=DECIMAL_PRECISION,
    )
    assert_array_almost_equal(
        by_lag[1],
        np.broadcast_to([20.5, 4.5], by_lag[1].shape),
        decimal=DECIMAL_PRECISION,
    )


def test_collective_current_policy_preserves_names_under_reversed_declarations() -> (
    None
):
    """Reordering regimes and members keeps named values and one joint action."""
    model = _model(exits=True, enable_jit=True, width=3, reverse=True)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    members = model._regimes["couple"].stakeholders
    assert members == ("m", "f")
    for member, expected in (("f", [4.0, 4.0, 8.0]), ("m", [4.0, 4.0, 4.0])):
        assert_array_almost_equal(
            solution.values[0]["couple"][:, members.index(member)],
            expected,
            decimal=DECIMAL_PRECISION,
        )
    frame = (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(8, jnp.int32),
                "wealth": jnp.full(8, 2.0),
                "regime_id": jnp.full(
                    8, model.regime_names_to_ids["couple"], jnp.int32
                ),
                "own_stakeholder": jnp.asarray(
                    [model.stakeholder_names_to_ids[member] for member in ("f", "m")]
                    * 4,
                    jnp.int32,
                ),
            },
            seed=17,
            log_level="off",
            runtime_checks=False,
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
    )
    keys = draw_window_keys(
        key=jax.random.key(17),
        address_words=np.asarray(
            RandomAddress(kind="shock", name="eps").words(), np.uint32
        ),
        subject_start=encode_subject_row(row=0),
        last_real_subject=encode_subject_row(row=7),
        subject_count=8,
        period=0,
    )
    eps = np.asarray(jax.vmap(jax.random.normal)(keys))
    common_action = np.abs(eps) > 0.6
    np.testing.assert_array_equal(frame["regime_name"], "couple")
    np.testing.assert_array_equal(frame["action"], common_action.astype(np.int32))
    assert_array_almost_equal(
        frame["value_f"],
        np.where(common_action, 8.0, 7 + 5 * eps),
        decimal=DECIMAL_PRECISION,
    )
    assert_array_almost_equal(
        frame["value_m"],
        np.where(common_action, 4.0, 7 - 5 * eps),
        decimal=DECIMAL_PRECISION,
    )
