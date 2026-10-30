"""Participation pricing reads later participation dates from stored surfaces.

A participation reader rebuilds the target's action problem exactly at its landing. That
action problem's own participation checks, one date later, read the stored participation
surface of that edge: each branch value and each operand of the participation check,
tabulated on the target's grid, interpolated separately at the landing and decided again
there. So each period's query program prices one participation level, whatever the
number of later participation dates; on grid nodes the values are those of reading every
later participation check at its landing, and between nodes a member whose value jumps
where the decision switches is priced on the right side of it.
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm import policy_queries
from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    ParticipationCheck,
    ParticipationRead,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    load_solution,
)
from lcm.exceptions import SolutionIntegrityError
from lcm.solver_api import SolutionResult
from lcm.transition import StochasticTransition, Transition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    UserParamsNode,
)
from lcm_examples import collective_household as household
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED
from tests.solution import _discrete_household as discrete

_FIXTURES = Path(__file__).parents[1] / "data" / "participation_pricing_depth"


def _nested_values(name: str) -> Mapping[int, Mapping[str, np.ndarray]]:
    """Values captured by pricing each later participation check at its landing."""
    path = _FIXTURES / f"{name}_x{int(X64_ENABLED)}.json"
    stored = json.loads(path.read_text())["values"]
    return {
        int(period): {regime: np.asarray(value) for regime, value in regimes.items()}
        for period, regimes in stored.items()
    }


def _query_flops(
    *, monkeypatch: pytest.MonkeyPatch, n_periods: int
) -> dict[tuple[str, int], float]:
    """Compiled flops of each period's query chunk loop, by regime and period."""
    flops: dict[tuple[str, int], float] = {}
    original = policy_queries._PolicyQueryKernel.__call__

    def measured(self, **kwargs):
        cost = kwargs["compiled_cores"]["_chunks"].compiled.cost_analysis()
        cost = cost[0] if isinstance(cost, list) else cost
        flops[self.root_context.regime_name, self.period] = cost["flops"]
        return original(self, **kwargs)

    model = household.get_model(
        n_periods=n_periods,
        wealth_n_points=3,
        consumption_n_points=2,
        execution_config=ExecutionConfig(axis_widths={"query_cell": 1}),
    )
    # Scoped, so a second measurement wraps the original kernel, not this one.
    with monkeypatch.context() as patch:
        patch.setattr(policy_queries._PolicyQueryKernel, "__call__", measured)
        model.solve(params=household.get_params(), log_level="off")
    return flops


def test_query_programs_do_not_grow_with_remaining_participation_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More later participation dates keep the query program flat; one costs a constant.

    Flatness is the guard: at five periods, period 1 prices two later participation
    dates and period 0 three, and both programs must cost the same up to 1.1x.
    Nesting would multiply the program by the target's actions times its reads
    and lottery nodes for every extra date.

    The constant: at four periods, period 1 prices one participation date and period 0
    two, so period 0 adds one stored-surface read to each rebuilt action problem.
    That read interpolates every channel of the surface (the couple's branches
    and decision operands, eight channels) and takes the participation check again
    at the landing, which keeps it under 4x the one-date program.
    """
    deep = _query_flops(monkeypatch=monkeypatch, n_periods=5)
    shallow = _query_flops(monkeypatch=monkeypatch, n_periods=4)
    for regime in ("couple", "single_f", "single_m"):
        assert deep[regime, 0] <= 1.1 * deep[regime, 1], (regime, deep)
        one_date, two_dates = shallow[regime, 1], shallow[regime, 0]
        assert two_dates <= 4 * one_date, (regime, one_date, two_dates)


def test_query_program_cost_is_the_same_for_every_earlier_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once every target is reachable, one more later participation date costs nothing.

    At five periods, period 1 prices two later participation dates and period 0 three.
    Each rebuilt action problem reads the next date's stored surface, so both
    programs do the same work.
    """
    flops = _query_flops(monkeypatch=monkeypatch, n_periods=5)
    for regime in ("couple", "single_f", "single_m"):
        assert flops[regime, 0] == pytest.approx(flops[regime, 1], rel=0.05), regime


def test_grid_aligned_participation_price_like_reads_at_every_landing() -> None:
    """Where every landing is a node, a stored surface reads the exact value there."""
    model = discrete.get_model(n_periods=4)
    result = model.solve(params=discrete.get_params(), log_level="off")
    expected = _nested_values("discrete_np4_w4")
    assert set(result.values) == set(expected)
    for period, regimes in expected.items():
        assert set(result.values[period]) == set(regimes)
        for regime, value in regimes.items():
            aaae(
                np.asarray(result.values[period][regime]),
                value,
                decimal=DECIMAL_PRECISION,
            )


def test_periods_pricing_one_participation_date_keep_their_values() -> None:
    """Off-grid landings change values only where a second participation date is priced.

    With four periods only period 0 prices two participation dates; every later
    period's value is the one an exact read at every landing gives.
    """
    model = household.get_model(n_periods=4, wealth_n_points=4, consumption_n_points=3)
    result = model.solve(params=household.get_params(), log_level="off")
    expected = _nested_values("household_np4_w4_c3")
    for period in (1, 2, 3):
        for regime, value in expected[period].items():
            aaae(
                np.asarray(result.values[period][regime]),
                value,
                decimal=DECIMAL_PRECISION,
            )


def test_a_saved_solution_replays_its_stored_participation_surfaces(
    tmp_path: Path,
) -> None:
    """A loaded archive simulates exactly like the solution that wrote it."""
    model = household.get_model(n_periods=4, wealth_n_points=4, consumption_n_points=3)
    params = household.get_params()
    solution = model.solve(params=params, log_level="off")
    loaded = load_solution(path=solution.save(path=tmp_path / "solution.h5"))

    def simulate(solved: SolutionResult) -> pd.DataFrame:
        return model.simulate(
            params=params,
            initial_conditions=household.get_initial_conditions(
                n_subjects=16, model=model
            ),
            solution=solved,
            log_level="off",
            seed=0,
        ).to_dataframe()

    pd.testing.assert_frame_equal(simulate(loaded), simulate(solution))


# A couple whose wealth drifts up by a fixed amount each period, whatever it
# consumes, re-checks at every landing whether to stay together. At the last
# check (age 2) she stays only where the household is worth at least her own
# single value; he always prefers the household, so where she leaves he is
# overruled and his value jumps. Singles' and the terminal values are linear in
# wealth and the drift lands every period-0 node at the midpoint of a cell at the
# last check, so the only error a solve can carry is how the stored surface of
# that check is read between its nodes.
_LOW, _HIGH = 4.0, 20.0
_BETA = 0.9
_TERMINAL_SLOPE = 0.1
_PREMIUM_M = 0.5
_SINGLE = {"f": (0.3, -2.08), "m": (0.05, -2.0)}
_LAST_CHECK_AGE = 2


def _her_consent_at_the_last_check(
    *, V_target_f: FloatND, V_alone_f: FloatND, age: FloatND
) -> BoolND:
    return (age < _LAST_CHECK_AGE) | (V_target_f >= V_alone_f)


def _drift(*, wealth: ContinuousState, drift: float) -> ContinuousState:
    return wealth + drift


def _identity(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _utility_f(
    *, wealth: ContinuousState, consumption_share: ContinuousAction
) -> FloatND:
    return jnp.log(0.5 * consumption_share * wealth)


def _utility_m(
    *, wealth: ContinuousState, consumption_share: ContinuousAction
) -> FloatND:
    return jnp.log(0.5 * consumption_share * wealth) + _PREMIUM_M


def _linear_utility(*, slope: float, level: float) -> Callable[..., FloatND]:
    def utility(
        *, wealth: ContinuousState, consumption_share: ContinuousAction
    ) -> FloatND:
        return slope * wealth + level + jnp.log(consumption_share)

    return utility


def _overruled_partner_model(*, n_points: int) -> Model:
    """Four periods; the couple's checks land at ages 1 and 2."""
    wealth = LinSpacedGrid(start=_LOW, stop=_HIGH, n_points=n_points)
    share = LinSpacedGrid(start=0.5, stop=1.0, n_points=2)
    probability = household._transition_probabilities(last_age=3)

    def law(*, stays: str, terminal: str) -> ByAge:
        return household._until_last_age(
            {
                stays: StochasticTransition(func=probability["stays_married"]),
                terminal: StochasticTransition(func=probability["reaches_last_age"]),
            },
            terminal=terminal,
            last_age=3,
        )

    def regime(
        *, utility: Callable[..., FloatND] | CollectiveUtility, transitions: bool
    ) -> Regime:
        return Regime(
            states={"wealth": wealth},
            state_transitions={"wealth": _drift} if transitions else {},
            actions={"consumption_share": share},
            functions={"utility": utility},
        )

    def single(*, member: str, transitions: bool) -> Regime:
        slope, level = _SINGLE[member]
        return regime(
            utility=_linear_utility(slope=slope, level=level), transitions=transitions
        )

    check = ParticipationCheck(
        predicate=_her_consent_at_the_last_check,
        reads={
            "V_target_f": ParticipationRead(
                regime="couple",
                stakeholder="f",
                kind="Participating",
                projection={"wealth": _identity},
            ),
            "V_alone_f": ParticipationRead(
                regime="single_f", kind="Raw", projection={"wealth": _identity}
            ),
        },
        routes={
            member: StakeholderRoute(
                target_stakeholder=member,
                fallback=ProjectedRegimeValue(
                    regime=f"single_{member}", projection={"wealth": _identity}
                ),
            )
            for member in ("f", "m")
        },
    )
    terminal = _linear_utility(slope=_TERMINAL_SLOPE, level=0.0)
    return Model(
        edges={
            "couple": Transition(
                law=law(stays="couple", terminal="couple_terminal"),
                participation={"couple": check},
            ),
            "single_f": Transition(
                law=law(stays="single_f", terminal="single_f_terminal")
            ),
            "single_m": Transition(
                law=law(stays="single_m", terminal="single_m_terminal")
            ),
        },
        regimes={
            "couple": regime(
                utility=CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m}),
                transitions=True,
            ),
            "couple_terminal": regime(
                utility=CollectiveUtility(utilities={"f": terminal, "m": terminal}),
                transitions=False,
            ),
            "single_f": single(member="f", transitions=True),
            "single_m": single(member="m", transitions=True),
            "single_f_terminal": single(member="f", transitions=False),
            "single_m_terminal": single(member="m", transitions=False),
        },
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=household.RegimeId,
        initial_nodes={0: ("couple", "single_f", "single_m")},
    )


def _overruled_partner_params(*, drift: float) -> dict[str, UserParamsNode]:
    moving = {"koopmans_aggregator": {"discount_factor": _BETA}}
    return {
        "couple": {**moving, "next_wealth": {"drift": drift}},
        "couple_terminal": {},
        "single_f": {**moving, "next_wealth": {"drift": drift}},
        "single_m": {**moving, "next_wealth": {"drift": drift}},
        "single_f_terminal": {},
        "single_m_terminal": {},
    }


def _exact_period_0_values(*, wealth: np.ndarray, drift: float) -> np.ndarray:
    """The couple's period-0 values, every check decided at its exact landing."""

    def couple_flow(*, member: str, y: np.ndarray) -> np.ndarray:
        return np.log(0.5 * y) + (_PREMIUM_M if member == "m" else 0.0)

    def single_value(*, member: str, y: np.ndarray) -> np.ndarray:
        slope, level = _SINGLE[member]
        return slope * y + level + _BETA * (slope * (y + drift) + level)

    def last_check(*, member: str, y: np.ndarray) -> np.ndarray:
        stays = {
            other: couple_flow(member=other, y=y)
            + _BETA * _TERMINAL_SLOPE * (y + drift)
            for other in ("f", "m")
        }
        leaves = single_value(member=member, y=y)
        return np.where(
            stays["f"] >= single_value(member="f", y=y), stays[member], leaves
        )

    return np.stack(
        [
            couple_flow(member=member, y=wealth)
            + _BETA * couple_flow(member=member, y=wealth + drift)
            + _BETA**2 * last_check(member=member, y=wealth + 2 * drift)
            for member in ("f", "m")
        ],
        axis=-1,
    )


def _last_check_switch(*, drift: float) -> float:
    """Wealth at which she starts to leave at the last check, by bisection."""

    def leaves(y: float) -> bool:
        stays = np.log(0.5 * y) + _BETA * _TERMINAL_SLOPE * (y + drift)
        slope, level = _SINGLE["f"]
        return bool(stays < slope * y + level + _BETA * (slope * (y + drift) + level))

    low, high = _LOW, _HIGH
    assert not leaves(low)
    assert leaves(high)
    for _ in range(100):
        middle = 0.5 * (low + high)
        low, high = (low, middle) if leaves(middle) else (middle, high)
    return 0.5 * (low + high)


@pytest.mark.parametrize("n_points", [5, 9, 17])
def test_an_overruled_partner_is_priced_on_the_right_side_of_the_switch(
    n_points: int,
) -> None:
    """Between nodes, the decision is taken again on interpolated branches.

    Her decision switches inside one cell of the last check's grid, and his
    value jumps there by several utils. Each period-0 node lands at the midpoint
    of a cell at that check. Reading a stored value after the decision would
    average his two branches across the switch; reading the branches and
    deciding at the landing leaves only their own interpolation error, second
    order in the node spacing.
    """
    h = (_HIGH - _LOW) / (n_points - 1)
    drift = h / 4
    switch = _last_check_switch(drift=drift)
    midpoints = _LOW + h * (np.arange(n_points - 1) + 0.5)
    # The witness lands at a midpoint of the cell holding the switch, but not on
    # the switch itself, where any read is within the switch location's error.
    assert ((midpoints - h / 2 < switch) & (switch < midpoints + h / 2)).any()
    assert np.abs(midpoints - switch).min() > h / 8

    model = _overruled_partner_model(n_points=n_points)
    result = model.solve(params=_overruled_partner_params(drift=drift), log_level="off")
    wealth = np.linspace(_LOW, _HIGH, n_points)
    expected = _exact_period_0_values(wealth=wealth, drift=drift)
    error = np.abs(np.asarray(result.values[0]["couple"]) - expected)
    # Linear interpolation of the log branch: h**2 / 8 * max |(log y)''| = h**2 / 128
    # on [4, 20], times the two-period discount; the bound keeps a factor of three.
    assert error.max() <= 3 * _BETA**2 * h**2 / 128, error


def test_an_archive_of_another_surface_layout_is_refused(tmp_path: Path) -> None:
    """A loaded surface must hold the branches this release reads, or none at all."""
    solution = _overruled_partner_model(n_points=5).solve(
        params=_overruled_partner_params(drift=1.0), log_level="off"
    )
    path = solution.save(path=tmp_path / "solution.h5")
    with h5py.File(path, "r+") as archive:
        manifest = json.loads(bytes(archive["manifest"][()]))
        assert manifest["participation_surfaces"]
        for entry in manifest["participation_surfaces"]:
            entry["layout"] = "value-after-decision"
        encoded = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        del archive["manifest"]
        dataset = archive.create_dataset(
            "manifest", data=np.frombuffer(encoded, dtype=np.uint8)
        )
        dataset.attrs["sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(SolutionIntegrityError, match="layout"):
        load_solution(path=path)
