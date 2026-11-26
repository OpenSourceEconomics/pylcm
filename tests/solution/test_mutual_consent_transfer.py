"""Composition action rules at the joint optimum, and a transfer before separation.

Every model has one period, so a stay value is the flow utility of the household's
choice and an outside value is the destination's flow utility plus the adjustment.
Each expected number is written down by hand from the rule, or solved exactly in
rational arithmetic from the stayer's indifference equation; none is computed by a
second implementation of the engine's search.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm import composition_queries, policy_queries
from _lcm.solution.fingerprint import _semantic_fingerprint
from lcm import (
    CollectiveUtility,
    CompositionAction,
    Consent,
    Counterparty,
    Destination,
    DiscreteGrid,
    ExecutionConfig,
    ExtremeValueTasteShocks,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    MutualConsent,
    ParetoObjective,
    PCOnActions,
    PCOnWeight,
    Regime,
    Transfer,
    categorical,
)
from lcm.composition_actions import CompositionRule
from lcm.exceptions import ExecutionPlanningError, RegimeInitializationError
from lcm.typing import BoolND, FloatND, IntND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, assert_agrees_to_ulp


@categorical(ordered=False)
class _Household:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


# Each member's outside value, `(f, m)`, wherever the composition action is available.
_OUTSIDE = (1.0, 2.0)

# Stay values `(f, m)` per cell and action. Equal Pareto weights pick the action
# with the larger sum; action 1 is never the optimum but passes some members.
_STAY = (
    ((3.0, 4.0), (2.0, 3.0)),  # 0: nobody refuses at the optimum
    ((6.0, 1.0), (2.0, 3.0)),  # 1: m refuses at the optimum; action 1 keeps both
    ((0.0, 1.0), (4.0, -5.0)),  # 2: both refuse; action 1 keeps f
    ((1.0, 5.0), (2.0, 3.0)),  # 3: f is indifferent at the optimum
    ((1.0, 2.0), (0.0, 2.5)),  # 4: both indifferent; action 1 strictly keeps m
    ((6.0, 1.0), (2.0, 3.0)),  # 5: as 1, but the composition action is not available
    ((6.0, 1.0), (2.0, 3.0)),  # 6: no action is feasible
)
_UNAVAILABLE = 5
_INFEASIBLE = 6
# Simulation starts no agent where no action is feasible, so it covers the cells
# before the last.
_SIMULATED = _INFEASIBLE

_SEPARATE = _OUTSIDE

# Hand-derived published values `(f, m)` per cell, by comparison and tie.
_EXPECTED = {
    ("each_action", "stay"): (
        (3, 4), (2, 3), _SEPARATE, (1, 5), (1, 2), (6, 1), _SEPARATE,
    ),
    ("optimum", "stay"): (
        (3, 4), _SEPARATE, _SEPARATE, (1, 5), (1, 2), (6, 1), _SEPARATE,
    ),
    ("each_action", "leave"): (
        (3, 4), (2, 3), _SEPARATE, (2, 3), _SEPARATE, (6, 1), _SEPARATE,
    ),
    ("optimum", "leave"): (
        (3, 4), _SEPARATE, _SEPARATE, _SEPARATE, _SEPARATE, (6, 1), _SEPARATE,
    ),
}  # fmt: skip

# Whether the pair separates in each cell under the optimum comparison. Cell 4 under
# `tie="stay"` publishes the outside values while staying, so only the route
# tells the two apart.
_SEPARATES = {
    "stay": (False, True, True, False, False, False, True),
    "leave": (False, True, True, True, True, False, True),
}


@dataclass(frozen=True, kw_only=True)
class _StayValue:
    member: int

    def __call__(self, *, case: IntND, choice: IntND) -> FloatND:
        return jnp.asarray(_STAY)[case, choice, self.member]


@dataclass(frozen=True, kw_only=True)
class _Constant:
    value: float

    def __call__(self) -> float:
        return self.value


def _feasible(*, case: IntND) -> BoolND:
    return case != _INFEASIBLE


def _available(*, case: IntND) -> BoolND:
    return case != _UNAVAILABLE


def _comparison_model(*, rule: CompositionRule) -> Model:
    return Model(
        regimes={
            "couple": Regime(
                states={"case": DiscreteGrid(values=jnp.arange(len(_STAY)))},
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                constraints={"feasible": _feasible},
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            member: _StayValue(member=index)
                            for index, member in enumerate(("f", "m"))
                        }
                    )
                },
                composition_action=CompositionAction(
                    rule=rule,
                    destinations={
                        member: Destination(regime=f"single_{member}", projection={})
                        for member in ("f", "m")
                    },
                    available=_available,
                ),
            ),
            **{
                f"single_{member}": Regime(
                    functions={"utility": _Constant(value=value)}
                )
                for member, value in zip(("f", "m"), _OUTSIDE, strict=True)
            },
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: "couple"}),
        regime_id_class=_Household,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _simulate_every_cell(*, model: Model, n_cells: int) -> pd.DataFrame:
    """Simulate one agent per cell and member, ordered by cell, then f before m."""
    solution = model.solve(params={}, log_level="off")
    roles = [model.member_names_to_ids[member] for member in ("f", "m")]
    return (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(2 * n_cells, dtype=jnp.int32),
                "regime_id": jnp.full(
                    2 * n_cells, model.regime_names_to_ids["couple"], dtype=jnp.int32
                ),
                "own_member": jnp.tile(jnp.asarray(roles, dtype=jnp.int32), n_cells),
                "case": jnp.repeat(jnp.arange(n_cells, dtype=jnp.int32), 2),
            },
            seed=6271,
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("agent_id")
        .reset_index(drop=True)
    )


def _own_values(frame: pd.DataFrame) -> np.ndarray:
    """Each agent's reported value: its member's couple value, or its single value."""
    own = np.where(frame.index % 2 == 0, frame["value_f"], frame["value_m"])
    return np.where(frame["regime_name"] == "couple", own, frame["value"])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("compares", ["each_action", "optimum"])
def test_comparison_point_decides_the_published_values(
    *, compares: Literal["each_action", "optimum"], tie: Literal["stay", "leave"]
) -> None:
    """Each cell publishes the hand-derived stay or outside values of its rule."""
    rule = PCOnActions if compares == "each_action" else PCOnWeight
    model = _comparison_model(rule=rule(tie=tie))
    solution = model.solve(params={}, log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["couple"],
        np.asarray(_EXPECTED[compares, tie], dtype=float),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
def test_optimum_comparison_routes_simulated_members_as_the_solve_prices_them(
    *, tie: Literal["stay", "leave"]
) -> None:
    """Separating cells land in the singles; every agent reports the solve's value."""
    model = _comparison_model(rule=PCOnWeight(tie=tie))
    frame = _simulate_every_cell(model=model, n_cells=_SIMULATED)
    separates = np.repeat(np.asarray(_SEPARATES[tie][:_SIMULATED]), 2)
    expected_regimes = np.where(
        separates, np.tile(["single_f", "single_m"], _SIMULATED), "couple"
    )
    assert list(frame["regime_name"]) == list(expected_regimes)
    np.testing.assert_array_almost_equal(
        _own_values(frame),
        np.asarray(_EXPECTED["optimum", tie][:_SIMULATED], dtype=float).reshape(-1),
        decimal=DECIMAL_PRECISION,
    )


def test_optimum_comparison_stays_at_the_optimal_action() -> None:
    """A pair that stays acts at the household optimum, never at a passing action."""
    model = _comparison_model(rule=PCOnWeight())
    frame = _simulate_every_cell(model=model, n_cells=_SIMULATED)
    staying = frame["regime_name"] == "couple"
    np.testing.assert_array_equal(frame.loc[staying, "choice"], 0)


def test_the_comparison_point_of_the_rule_is_part_of_the_durable_identity() -> None:
    """The two rules of unilateral divorce have different solution identities."""
    assert _semantic_fingerprint(PCOnActions()) != _semantic_fingerprint(PCOnWeight())


@categorical(ordered=False)
class _Singleton:
    source: ScalarInt
    target: ScalarInt


def _choice(*, choice: IntND) -> FloatND:
    return jnp.asarray(choice, dtype=float)


def _taste_single(
    *, composition_action_declaration: CompositionAction | None
) -> Regime:
    return Regime(
        actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
        functions={"utility": _choice},
        taste_shocks=ExtremeValueTasteShocks(),
        composition_action=composition_action_declaration,
    )


def _taste_couple(
    *, composition_action_declaration: CompositionAction | None
) -> Regime:
    return Regime(
        actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _choice, "m": _choice},
                objective=ParetoObjective(weights={"f": 1.0, "m": 0.0}),
            )
        },
        taste_shocks=ExtremeValueTasteShocks(),
        composition_action=composition_action_declaration,
    )


_PLAIN_SINGLE = Regime(functions={"utility": _Constant(value=0.0)})
_PLAIN_COUPLE = Regime(
    functions={
        "utility": CollectiveUtility(
            utilities={"f": _Constant(value=0.0), "m": _Constant(value=0.0)}
        )
    }
)

# Each case: the regime that has taste shocks and a composition action, and the three
# regimes.
_TASTE_SHOCKS_WITH_OWN_COMPOSITION_ACTION = {
    "single_dissolution": (
        "single_f",
        {
            "single_f": _taste_single(
                composition_action_declaration=CompositionAction(
                    destinations={
                        "self": Destination(regime="single_m", projection={})
                    },
                    rule=PCOnActions(),
                )
            ),
            "single_m": _PLAIN_SINGLE,
            "couple": _PLAIN_COUPLE,
        },
    ),
    "single_formation": (
        "single_f",
        {
            "single_f": _taste_single(
                composition_action_declaration=CompositionAction(
                    destinations={
                        "self": Destination(regime="couple", role="f", projection={})
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_m", role="m", projection={}
                        )
                    ),
                )
            ),
            "single_m": _PLAIN_SINGLE,
            "couple": _PLAIN_COUPLE,
        },
    ),
    "household_dissolution": (
        "couple",
        {
            "couple": _taste_couple(
                composition_action_declaration=CompositionAction(
                    destinations={
                        "f": Destination(regime="single_f", projection={}),
                        "m": Destination(regime="single_m", projection={}),
                    },
                    rule=PCOnActions(),
                )
            ),
            "single_f": _PLAIN_SINGLE,
            "single_m": _PLAIN_SINGLE,
        },
    ),
}


@pytest.mark.parametrize("case", list(_TASTE_SHOCKS_WITH_OWN_COMPOSITION_ACTION))
def test_a_regime_with_taste_shocks_refuses_its_own_composition_action(
    case: str,
) -> None:
    """A regime with taste shocks declares neither a composition action nor a formation.

    The message names the regime and both ways out: drop its taste shocks, or
    declare the composition action on a regime without them.
    """
    source, regimes = _TASTE_SHOCKS_WITH_OWN_COMPOSITION_ACTION[case]
    with pytest.raises(
        RegimeInitializationError,
        match=(
            f"Regime '{source}' has taste shocks and a composition action.*cannot "
            "have its own composition action or formation.*drop `taste_shocks` on "
            "this regime, or declare the composition action on a regime without "
            "taste shocks"
        ),
    ):
        Model(
            regimes=regimes,
            n_periods=1,
            regime_id_class=_Household,
            initial_nodes=InitialNodes(by_period={0: source}),
            edges={},
            fixed_params={source: {"taste_shocks": {"scale": 1.0}}},
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )


# Transfer cells. The couple has no action, so its stay values are `W`. At
# transfer `s` the outside values are `O_f(s) = A_f + 4 s` and
# `O_m(s) = A_m - 4 s`, so increasing `s` favours f, the first member, and one
# candidate step moves each outside value by one.
_POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Each cell is the stay values `W`, the intercepts `A` and the default transfer.
_SETTLE_CELLS = (
    ((5.0, 5.0), (0.0, 0.0), 0.5),  # 0: nobody refuses at the default
    ((0.0, 0.0), (1.0, 3.0), 0.375),  # 1: both refuse at a default between nodes
    ((2.5, 0.5), (0.0, 4.0), 0.375),  # 2: m refuses; f accepts first at 0.75
    ((2.5, 1.5), (0.0, 4.0), 0.375),  # 3: at 0.75 m no longer gains
    ((4.5, 0.5), (0.0, 4.0), 0.375),  # 4: no candidate makes f accept
    ((-0.5, 3.5), (0.0, 4.0), 0.375),  # 5: f refuses; m accepts first at 0
    ((-0.5, 3.0), (0.0, 4.0), 0.375),  # 6: at 0.25 the stayer m is indifferent
    ((1.0, 2.8), (0.0, 4.0), 0.375),  # 7: at 0.25 f is indifferent
    (
        (0.0, 0.0),
        (1.0, 3.0),
        0.375,
    ),  # 8: as 1, but the composition action is not available
    ((-10.0, -10.0), (1.0, 3.0), 1.5),  # 9: both refuse; the default is clipped
)
_SETTLE_UNAVAILABLE = 8

# Hand-derived transfer at which each cell separates, None where the pair stays.
# In cell 6 the stayer m has `O_m(0.25) = 3 = W_m`: under `tie="stay"` it passes
# that candidate over and accepts the next one, 0, where `O_m = 4 > 3` and the
# leaver f gains, `O_f = 0 > -0.5`; under `tie="leave"` it accepts 0.25, where f
# gains, `O_f = 1 >= -0.5`.
_SETTLED_AT = {
    "stay": (None, 0.375, 0.75, None, None, 0.0, 0.0, None, None, 1.0),
    "leave": (None, 0.375, 0.75, None, None, 0.0, 0.25, 0.25, None, 1.0),
}


def _settled_values(*, tie: Literal["stay", "leave"]) -> np.ndarray:
    """The published `(f, m)` values: `O(s*)` where the pair separates, else `W`."""
    return np.asarray(
        [
            inside
            if point is None
            else (intercept[0] + 4.0 * point, intercept[1] - 4.0 * point)
            for (inside, intercept, _), point in zip(
                _SETTLE_CELLS, _SETTLED_AT[tie], strict=True
            )
        ]
    )


@dataclass(frozen=True, kw_only=True)
class _CellValue:
    values: tuple[float, ...]

    def __call__(self, *, case: IntND) -> FloatND:
        return jnp.asarray(self.values)[case]


@dataclass(frozen=True, kw_only=True)
class _Divided:
    """`A_i + 4 share` for f and `A_i - 4 share` for m."""

    intercepts: tuple[float, ...]
    slope: float

    def __call__(self, *, case: IntND, share: FloatND) -> FloatND:
        return jnp.asarray(self.intercepts)[case] + self.slope * share


def _divided(member: int) -> _Divided:
    return _Divided(
        intercepts=tuple(cell[1][member] for cell in _SETTLE_CELLS),
        slope=4.0 if member == 0 else -4.0,
    )


def _settle_available(*, case: IntND) -> BoolND:
    return case != _SETTLE_UNAVAILABLE


def _assets(*, assets: FloatND) -> FloatND:
    return assets


def _transfer(
    *,
    name: str = "share",
    candidates: LinSpacedGrid | DiscreteGrid | None = None,
) -> Transfer:
    return Transfer(
        name=name,
        candidates=LinSpacedGrid(start=0.0, stop=1.0, n_points=len(_POINTS))
        if candidates is None
        else candidates,
        default=_CellValue(values=tuple(cell[2] for cell in _SETTLE_CELLS)),
    )


def _transfer_model(
    *,
    division: Literal["transfer", "assets"],
    tie: Literal["stay", "leave"] = "stay",
    transfer: Transfer | None = None,
) -> Model:
    """Two members who settle through a transfer or through a share of assets."""
    transfer = _transfer() if transfer is None else transfer
    singles = {
        f"single_{member}": Regime(
            states={"assets": LinSpacedGrid(start=-16.0, stop=16.0, n_points=33)},
            functions={"utility": _assets},
        )
        if division == "assets"
        else Regime(functions={"utility": _Constant(value=0.0)})
        for member in ("f", "m")
    }
    return Model(
        regimes={
            "couple": Regime(
                states={"case": DiscreteGrid(values=jnp.arange(len(_SETTLE_CELLS)))},
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            member: _CellValue(
                                values=tuple(cell[0][index] for cell in _SETTLE_CELLS)
                            )
                            for index, member in enumerate(("f", "m"))
                        }
                    )
                },
                composition_action=CompositionAction(
                    rule=MutualConsent(tie=tie, transfer=transfer),
                    destinations={
                        member: Destination(
                            regime=f"single_{member}",
                            projection={"assets": _divided(index)}
                            if division == "assets"
                            else {},
                            utility_adjustment=None
                            if division == "assets"
                            else _divided(index),
                        )
                        for index, member in enumerate(("f", "m"))
                    },
                    available=_settle_available,
                ),
            ),
            **singles,
        },
        n_periods=1,
        edges={},
        # The singles start agents too, so they are solved and outside values
        # landing on their grid are read from stored tables.
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        regime_id_class=_Household,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("division", ["transfer", "assets"])
def test_transfer_publishes_the_hand_derived_values(
    *, division: Literal["transfer", "assets"], tie: Literal["stay", "leave"]
) -> None:
    """Each cell publishes `W` where the pair stays and `O(s*)` where it separates."""
    model = _transfer_model(division=division, tie=tie)
    np.testing.assert_array_almost_equal(
        model.solve(params={}, log_level="off").values[0]["couple"],
        _settled_values(tie=tie),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("division", ["transfer", "assets"])
def test_transfer_lands_simulated_members_at_the_selected_transfer(
    *, division: Literal["transfer", "assets"], tie: Literal["stay", "leave"]
) -> None:
    """Separating members land at `s*` and report the solve's value there."""
    model = _transfer_model(division=division, tie=tie)
    frame = _simulate_every_cell(model=model, n_cells=len(_SETTLE_CELLS))
    settled = np.repeat(
        np.asarray([point is not None for point in _SETTLED_AT[tie]]), 2
    )
    expected_regimes = np.where(
        settled, np.tile(["single_f", "single_m"], len(_SETTLE_CELLS)), "couple"
    )
    assert list(frame["regime_name"]) == list(expected_regimes)
    np.testing.assert_array_almost_equal(
        _own_values(frame),
        _settled_values(tie=tie).reshape(-1),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_transfer_divides_the_assets_movers_land_with() -> None:
    """A mover's assets are its projection evaluated at the selected transfer."""
    frame = _simulate_every_cell(
        model=_transfer_model(division="assets"), n_cells=len(_SETTLE_CELLS)
    )
    moved = frame["regime_name"] != "couple"
    np.testing.assert_array_almost_equal(
        frame.loc[moved, "assets"],
        _settled_values(tie="stay").reshape(-1)[moved.to_numpy()],
        decimal=DECIMAL_PRECISION,
    )


def _solve_with_stored_reads(
    *, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, set[str]]:
    """Solve the assets model with stored or point outside reads, and its tables."""
    with monkeypatch.context() as patch:
        patch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
        model = _transfer_model(division="assets")
        tables = {
            name
            for regime in model._regimes.values()
            for kernel in regime.solution.period_kernels.values()
            if isinstance(kernel, policy_queries._PolicyQueryKernel)
            for name in kernel.programs
            if name.startswith("_outside:")
        }
        values = np.asarray(model.solve(params={}, log_level="off").values[0]["couple"])
    return values, tables


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_transfer_values_agree_with_and_without_stored_outside_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates on the singles' grid read a table; the values match point reads."""
    stored, tables = _solve_with_stored_reads(stored=True, monkeypatch=monkeypatch)
    point, no_tables = _solve_with_stored_reads(stored=False, monkeypatch=monkeypatch)
    assert tables
    assert not no_tables
    assert_agrees_to_ulp(
        got=stored,
        expected=point,
        n_ulp=16,
        operand_magnitude=float(np.max(np.abs(point))),
    )


_INTERCEPTS = tuple(cell[1] for cell in _SETTLE_CELLS)


@dataclass(frozen=True, kw_only=True)
class _Poisoned:
    """`_divided(member)`, but NaN at one candidate in the listed cells."""

    member: int
    cells: tuple[int, ...]
    point: float

    def __call__(self, *, case: IntND, share: FloatND) -> FloatND:
        slope = 4.0 if self.member == 0 else -4.0
        # Candidates are a quarter apart, so the band holds exactly one of them.
        poisoned = jnp.isin(case, jnp.asarray(self.cells)) & (
            jnp.abs(share - self.point) < 0.125
        )
        return jnp.where(
            poisoned,
            jnp.nan,
            jnp.asarray(_INTERCEPTS)[case, self.member] + slope * share,
        )


def _with_poison(poison: _Poisoned) -> Callable[[Regime], Regime]:
    member = ("f", "m")[poison.member]

    def change(regime: Regime) -> Regime:
        declaration = regime.composition_action
        assert declaration is not None
        return replace(
            regime,
            composition_action=replace(
                declaration,
                destinations={
                    **declaration.destinations,
                    member: replace(
                        declaration.destinations[member], utility_adjustment=poison
                    ),
                },
            ),
        )

    return change


# Cells in which a NaN at transfer 1.0 enters no comparison: nobody refuses (0), both
# refuse at a default below 1.0 (1), the stayer f accepts nearer (2, 3), the stayer m
# looks below the default (5, 6, 7), or the composition action is not available (8).
_UNCOMPARED_AT_ONE = (0, 1, 2, 3, 5, 6, 7, 8)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    "poison",
    [
        _Poisoned(member=0, cells=_UNCOMPARED_AT_ONE, point=1.0),
        _Poisoned(member=1, cells=_UNCOMPARED_AT_ONE, point=1.0),
        # The leaver m's value matters at the selected transfer only.
        _Poisoned(member=1, cells=(2,), point=0.5),
        _Poisoned(member=1, cells=(4,), point=1.0),
    ],
    ids=("f-at-1", "m-at-1", "leaver-before-selected", "leaver-none-accepted"),
)
def test_transfer_ignores_a_nan_no_comparison_reads(*, poison: _Poisoned) -> None:
    """A NaN outside value the selection never reads leaves every published value."""
    model = _model_with(regime_change=_with_poison(poison))
    np.testing.assert_array_almost_equal(
        model.solve(params={}, log_level="off").values[0]["couple"],
        _settled_values(tie="stay"),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_transfer_lands_members_past_a_nan_no_comparison_reads() -> None:
    """Simulated members separate where the selection puts them despite the NaN."""
    poison = _Poisoned(member=0, cells=_UNCOMPARED_AT_ONE, point=1.0)
    frame = _simulate_every_cell(
        model=_model_with(regime_change=_with_poison(poison)),
        n_cells=len(_SETTLE_CELLS),
    )
    np.testing.assert_array_almost_equal(
        _own_values(frame),
        _settled_values(tie="stay").reshape(-1),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize(
    "poison",
    [
        _Poisoned(member=0, cells=(2,), point=0.75),
        _Poisoned(member=1, cells=(2,), point=0.75),
        _Poisoned(member=0, cells=(2,), point=0.5),
        _Poisoned(member=0, cells=(4,), point=1.0),
        _Poisoned(member=0, cells=(9,), point=1.0),
    ],
    ids=(
        "stayer-at-selected",
        "leaver-at-selected",
        "stayer-before-selected",
        "stayer-none-accepted",
        "at-default",
    ),
)
def test_transfer_refuses_a_nan_the_selection_reads(*, poison: _Poisoned) -> None:
    """A NaN at the selected transfer, or one the stayer compared, is unresolved."""
    model = _model_with(regime_change=_with_poison(poison))
    with pytest.raises(ExecutionPlanningError, match="unresolved numerical status"):
        model.solve(params={}, log_level="off")


def _model_with(*, regime_change: Callable[[Regime], Regime]) -> Model:
    base = _transfer_model(division="transfer")
    return Model(
        regimes={
            **base.user_regimes,
            "couple": regime_change(base.user_regimes["couple"]),
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: "couple"}),
        regime_id_class=_Household,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _with_rule(rule: MutualConsent) -> Callable[[Regime], Regime]:
    def change(regime: Regime) -> Regime:
        declaration = regime.composition_action
        assert declaration is not None
        return replace(regime, composition_action=replace(declaration, rule=rule))

    return change


def _without_reads(regime: Regime) -> Regime:
    declaration = regime.composition_action
    assert declaration is not None
    return replace(
        regime,
        composition_action=replace(
            declaration,
            destinations={
                member: replace(target, utility_adjustment=_Constant(value=0.0))
                for member, target in declaration.destinations.items()
            },
        ),
    )


@pytest.mark.parametrize(
    ("regime_change", "match"),
    [
        (_without_reads, "read by no destination projection"),
        (_with_rule(MutualConsent(transfer=_transfer(name="case"))), "name of a state"),
        (
            _with_rule(MutualConsent(transfer=_transfer(name="utility"))),
            "name of a state",
        ),
        (
            _with_rule(MutualConsent(transfer=_transfer(name="utility_f"))),
            "name of a state",
        ),
    ],
    ids=("unread", "state", "function", "engine_function"),
)
def test_transfer_refuses_an_unread_or_taken_name(
    *, regime_change: Callable[[Regime], Regime], match: str
) -> None:
    """A transfer must be read on leaving and cannot shadow the regime's own names."""
    with pytest.raises(RegimeInitializationError, match=match):
        _model_with(regime_change=regime_change)


def test_transfer_refuses_a_singleton_regime() -> None:
    """Moving a division toward the member who stays needs two members."""
    with pytest.raises(RegimeInitializationError, match="two collective"):
        Model(
            regimes={
                "source": Regime(
                    states={
                        "case": DiscreteGrid(values=jnp.arange(len(_SETTLE_CELLS)))
                    },
                    functions={"utility": _CellValue(values=(0.0,) * 10)},
                    composition_action=CompositionAction(
                        destinations={
                            "self": Destination(
                                regime="target",
                                projection={},
                                utility_adjustment=_divided(0),
                            )
                        },
                        rule=MutualConsent(transfer=_transfer()),
                    ),
                ),
                "target": Regime(functions={"utility": _Constant(value=0.0)}),
            },
            n_periods=1,
            regime_id_class=_Singleton,
            initial_nodes=InitialNodes(by_period={0: "source"}),
            edges={},
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )


@pytest.mark.parametrize(
    "candidates",
    [DiscreteGrid(values=jnp.asarray([0, 1])), IrregSpacedGrid(n_points=3)],
    ids=("discrete", "runtime-points"),
)
def test_transfer_refuses_candidates_without_fixed_ordered_points(
    *, candidates: DiscreteGrid | IrregSpacedGrid
) -> None:
    """Candidates are an ordered continuous grid that is known at declaration."""
    with pytest.raises(RegimeInitializationError, match="Transfer candidates"):
        Transfer(name="share", candidates=candidates, default=_Constant(value=0.5))


def test_mutual_consent_requires_a_transfer() -> None:
    """Mutual consent always settles before the pair separates."""
    with pytest.raises(TypeError, match="transfer"):
        MutualConsent()  # ty: ignore[missing-argument] - the refused declaration


# Cells of the indifferent-stayer family: a name giving the signs of `(W_f - O_f(d), W_m
# - O_m(d))`, the stay values `W`, the default `d` and whether the composition action is
# available. Outside values are `O_f(s) = 4 s` and `O_m(s) = 4 - 4 s`, so every number
# below is dyadic and equality at the default is exact at both precisions. A `0` names a
# member exactly indifferent at the default.
_ROOT_INTERCEPTS = (Fraction(0), Fraction(4))
_ROOT_SLOPES = (Fraction(4), Fraction(-4))
_ROOT_CELLS = (
    ("++", (2.5, 3.5), 0.375, True),
    ("+0", (2.5, 2.5), 0.375, True),
    ("+-", (2.5, 0.5), 0.375, True),
    ("0+", (1.5, 3.5), 0.375, True),
    ("00", (1.5, 2.5), 0.375, True),
    ("0-:shift", (1.5, 0.5), 0.375, True),
    ("0-:route", (1.5, 2.2), 0.375, True),
    ("0-:default_on_a_node", (2.0, 1.0), 0.5, True),
    ("0-:default_at_the_upper_end", (4.0, -1.0), 1.0, True),
    ("0-:unavailable", (1.5, 0.5), 0.375, False),
    ("-+", (-0.5, 3.5), 0.375, True),
    ("-0:shift", (0.5, 2.5), 0.375, True),
    ("-0:route", (1.0, 2.5), 0.375, True),
    ("-0:default_at_the_lower_end", (-1.0, 4.0), 0.0, True),
    ("--", (1.0, 2.0), 0.375, True),
)
_ROOT_FRACTION_POINTS = tuple(Fraction(point) for point in _POINTS)

type _RootCell = tuple[str, tuple[float, float], float, bool]


def _exact_outside(transfer: Fraction) -> tuple[Fraction, Fraction]:
    return (
        _ROOT_INTERCEPTS[0] + _ROOT_SLOPES[0] * transfer,
        _ROOT_INTERCEPTS[1] + _ROOT_SLOPES[1] * transfer,
    )


def _exact_transfer(
    *, cell: _RootCell, tie: Literal["stay", "leave"]
) -> Fraction | None:
    """The transfer at which the pair separates, None where it stays.

    The stayer's share solves its indifference equation `O_p(r) = W_p` exactly. The
    stayer accepts the first candidate beyond the root, or at the root under
    `tie="leave"`; the pair separates there if the leaver prefers that to staying.
    """
    _, stay_values, default_value, available = cell
    if not available:
        return None
    inside = tuple(Fraction(value) for value in stay_values)
    default = Fraction(default_value)
    surplus = tuple(w - o for w, o in zip(inside, _exact_outside(default), strict=True))
    willing = tuple(s >= 0 if tie == "stay" else s > 0 for s in surplus)
    if all(willing):
        return None
    if not any(willing):
        return default
    stayer = willing.index(True)
    leaver = 1 - stayer
    root = (inside[stayer] - _ROOT_INTERCEPTS[stayer]) / _ROOT_SLOPES[stayer]
    # A willing stayer's root is at or beyond the default, so are these candidates.
    beyond = [
        point
        for point in _ROOT_FRACTION_POINTS
        if (point > root if stayer == 0 else point < root)
        or (point == root and tie == "leave")
    ]
    if not beyond:
        return None
    transfer = min(beyond) if stayer == 0 else max(beyond)
    gain = _exact_outside(transfer)[leaver] - inside[leaver]
    return transfer if (gain > 0 if tie == "stay" else gain >= 0) else None


def _published(
    *, cells: tuple[_RootCell, ...], transfers: tuple[Fraction | None, ...]
) -> np.ndarray:
    """`O(s*)` where the pair separates at `s*`, else `W`, as `(f, m)` rows."""
    return np.asarray(
        [
            [
                float(value)
                for value in (cell[1] if transfer is None else _exact_outside(transfer))
            ]
            for cell, transfer in zip(cells, transfers, strict=True)
        ]
    )


@dataclass(frozen=True, kw_only=True)
class _CellFlag:
    values: tuple[bool, ...]

    def __call__(self, *, case: IntND) -> BoolND:
        return jnp.asarray(self.values)[case]


# Single f has a feasible action only from these assets on.
_FLOOR_F = 2.0


def _solvent(*, assets: FloatND) -> BoolND:
    return assets >= _FLOOR_F


def _root_model(
    *,
    cells: tuple[_RootCell, ...],
    division: Literal["transfer", "assets"],
    tie: Literal["stay", "leave"],
    floor_f: bool = False,
) -> Model:
    """Two members settling `_ROOT_CELLS`-style cells; `floor_f` bounds f's assets."""
    n_cells = len(cells)
    divided = {
        member: _Divided(
            intercepts=(float(_ROOT_INTERCEPTS[index]),) * n_cells,
            slope=float(_ROOT_SLOPES[index]),
        )
        for index, member in enumerate(("f", "m"))
    }
    singles = {
        f"single_{member}": Regime(
            states={"assets": LinSpacedGrid(start=-16.0, stop=16.0, n_points=33)},
            functions={"utility": _assets},
            constraints={"solvent": _solvent} if floor_f and member == "f" else {},
        )
        if division == "assets"
        else Regime(functions={"utility": _Constant(value=0.0)})
        for member in ("f", "m")
    }
    return Model(
        regimes={
            "couple": Regime(
                states={"case": DiscreteGrid(values=jnp.arange(n_cells))},
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            member: _CellValue(
                                values=tuple(cell[1][index] for cell in cells)
                            )
                            for index, member in enumerate(("f", "m"))
                        }
                    )
                },
                composition_action=CompositionAction(
                    rule=MutualConsent(
                        tie=tie,
                        transfer=Transfer(
                            name="share",
                            candidates=LinSpacedGrid(
                                start=0.0, stop=1.0, n_points=len(_POINTS)
                            ),
                            default=_CellValue(values=tuple(cell[2] for cell in cells)),
                        ),
                    ),
                    destinations={
                        member: Destination(
                            regime=f"single_{member}",
                            projection={"assets": divided[member]}
                            if division == "assets"
                            else {},
                            utility_adjustment=None
                            if division == "assets"
                            else divided[member],
                        )
                        for member in ("f", "m")
                    },
                    available=_CellFlag(values=tuple(cell[3] for cell in cells)),
                ),
            ),
            **singles,
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        regime_id_class=_Household,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _check_simulated_transfers(
    *,
    model: Model,
    cells: tuple[_RootCell, ...],
    transfers: tuple[Fraction | None, ...],
    division: Literal["transfer", "assets"],
) -> None:
    """Movers land in the singles at `s*` and report `O(s*)`; stayers report `W`."""
    frame = _simulate_every_cell(model=model, n_cells=len(cells))
    separates = np.repeat(np.asarray([s is not None for s in transfers], dtype=bool), 2)
    assert list(frame["regime_name"]) == list(
        np.where(separates, np.tile(["single_f", "single_m"], len(cells)), "couple")
    )
    published = _published(cells=cells, transfers=transfers).reshape(-1)
    np.testing.assert_array_almost_equal(
        _own_values(frame), published, decimal=DECIMAL_PRECISION
    )
    if division == "assets":
        np.testing.assert_array_almost_equal(
            frame.loc[separates, "assets"],
            published[separates],
            decimal=DECIMAL_PRECISION,
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("division", ["transfer", "assets"])
def test_transfer_publishes_the_exact_root_of_the_stayers_indifference(
    *, division: Literal["transfer", "assets"], tie: Literal["stay", "leave"]
) -> None:
    """No stayer accepts the default; one indifferent there takes the next candidate.

    Under `tie="stay"` a member exactly indifferent at the default is the stayer when
    the other refuses, and accepts the first candidate beyond the default on its
    side. Under `tie="leave"` an indifferent member refuses, so both do.
    """
    model = _root_model(cells=_ROOT_CELLS, division=division, tie=tie)
    transfers = tuple(_exact_transfer(cell=cell, tie=tie) for cell in _ROOT_CELLS)
    np.testing.assert_array_almost_equal(
        model.solve(params={}, log_level="off").values[0]["couple"],
        _published(cells=_ROOT_CELLS, transfers=transfers),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("division", ["transfer", "assets"])
def test_transfer_lands_members_of_an_indifferent_stayer(
    *, division: Literal["transfer", "assets"], tie: Literal["stay", "leave"]
) -> None:
    """Simulated members separate, land and are paid at the solve's transfer."""
    _check_simulated_transfers(
        model=_root_model(cells=_ROOT_CELLS, division=division, tie=tie),
        cells=_ROOT_CELLS,
        transfers=tuple(_exact_transfer(cell=cell, tie=tie) for cell in _ROOT_CELLS),
        division=division,
    )


def test_exact_transfers_of_the_indifferent_stayer_cells() -> None:
    """The rational oracle gives the hand-derived transfer of every cell.

    With `O(s) = (4 s, 4 - 4 s)`, a default of 0.375 gives `O(d) = (1.5, 2.5)`. Under
    `tie="leave"` an indifferent member refuses, so every `0-` and `-0` cell has both
    refuse and separates at the default. Under `tie="stay"` the indifferent member is
    the stayer and never accepts the default, where `O = W` is no gain:

    - `0-:shift`, `W = (1.5, 0.5)`: f accepts 0.5, `2 > 1.5`; m gains, `2 > 0.5`.
    - `0-:route`, `W = (1.5, 2.2)`: f accepts 0.5; m does not gain, `2 < 2.2`: stay.
    - `0-:default_on_a_node`, `W = (2, 1)`, `d = 0.5`: f accepts 0.75, `3 > 2`; m is
      indifferent there, `1 = 1`: stay.
    - `0-:default_at_the_upper_end`, `W = (4, -1)`, `d = 1`: no candidate above: stay.
    - `-0:shift`, `W = (0.5, 2.5)`: m accepts 0.25, `3 > 2.5`; f gains, `1 > 0.5`.
    - `-0:route`, `W = (1, 2.5)`: m accepts 0.25; f is indifferent, `1 = 1`: stay.
    - `-0:default_at_the_lower_end`, `W = (-1, 4)`, `d = 0`: no candidate below: stay.
    """
    stay = {cell[0]: _exact_transfer(cell=cell, tie="stay") for cell in _ROOT_CELLS}
    leave = {cell[0]: _exact_transfer(cell=cell, tie="leave") for cell in _ROOT_CELLS}
    at_default = {cell[0]: Fraction(cell[2]) for cell in _ROOT_CELLS}
    assert stay == {
        "++": None,
        "+0": None,
        "+-": Fraction(3, 4),
        "0+": None,
        "00": None,
        "0-:shift": Fraction(1, 2),
        "0-:route": None,
        "0-:default_on_a_node": None,
        "0-:default_at_the_upper_end": None,
        "0-:unavailable": None,
        "-+": Fraction(0),
        "-0:shift": Fraction(1, 4),
        "-0:route": None,
        "-0:default_at_the_lower_end": None,
        "--": at_default["--"],
    }
    assert leave == {
        "++": None,
        "+0": None,
        "+-": Fraction(3, 4),
        "0+": None,
        "00": at_default["00"],
        **{
            name: at_default[name]
            for name in (
                "0-:shift",
                "0-:route",
                "0-:default_on_a_node",
                "0-:default_at_the_upper_end",
                "-0:shift",
                "-0:route",
                "-0:default_at_the_lower_end",
                "--",
            )
        },
        "0-:unavailable": None,
        "-+": Fraction(0),
    }


# With `_FLOOR_F` f's destination is unsupported at a default of 0.25 (f lands with
# 1) and supported at 0.5, where f accepts, `2 > 1`, and m, who refuses at the
# default, gains, `2 > 0.5`.
_FLOOR_CELLS = (("0-:default_unsupported", (1.0, 0.5), 0.25, True),)
_FLOOR_SETTLED_AT = (Fraction(1, 2),)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_transfer_searches_the_candidates_past_an_unsupported_default() -> None:
    """The stayer's search of the candidates does not need a supported default."""
    model = _root_model(cells=_FLOOR_CELLS, division="assets", tie="stay", floor_f=True)
    np.testing.assert_array_almost_equal(
        model.solve(params={}, log_level="off").values[0]["couple"],
        _published(cells=_FLOOR_CELLS, transfers=_FLOOR_SETTLED_AT),
        decimal=DECIMAL_PRECISION,
    )
    _check_simulated_transfers(
        model=model,
        cells=_FLOOR_CELLS,
        transfers=_FLOOR_SETTLED_AT,
        division="assets",
    )


# Stay values `(f, m)` per cell at the one feasible action. m's destination has
# no feasible action at his landing state, and his stay value is `-inf`.
_UNDEFINED_STAY = ((3.0, -np.inf), (0.0, -np.inf))


def _undefined_stay_f(*, case: IntND, choice: IntND) -> FloatND:
    return jnp.asarray(_UNDEFINED_STAY)[case, 0] + 0.0 * choice


def _undefined_stay_m(*, case: IntND, choice: IntND) -> FloatND:
    return jnp.asarray(_UNDEFINED_STAY)[case, 1] + 0.0 * choice


def _only_the_first_action(*, choice: IntND) -> BoolND:
    return choice == 0


def _below_the_grid(*, case: IntND) -> FloatND:
    return -1.0 + 0.0 * case


def _nonnegative_assets(*, assets: FloatND) -> BoolND:
    return assets >= 0.0


def _own_assets(*, assets: FloatND, c: IntND) -> FloatND:
    return assets + 0.0 * c


def _paid(*, t: FloatND) -> FloatND:
    return t


def _no_default(*, case: IntND) -> FloatND:
    return 0.0 * case


_UNDEFINED_DESTINATION_RULES = {
    "each action": (PCOnActions(tie="leave"), [[3.0, -np.inf], [-np.inf, -np.inf]]),
    "optimum": (
        PCOnWeight(tie="leave"),
        [[3.0, -np.inf], [-np.inf, -np.inf]],
    ),
    "transfer": (
        MutualConsent(
            tie="leave",
            transfer=Transfer(
                name="t",
                candidates=LinSpacedGrid(start=0.0, stop=4.0, n_points=5),
                default=_no_default,
            ),
        ),
        [[3.0, -np.inf], [0.0, -np.inf]],
    ),
}


@pytest.mark.parametrize("rule", list(_UNDEFINED_DESTINATION_RULES))
def test_a_member_with_an_undefined_destination_accepts_staying_under_leave(
    *, rule: str
) -> None:
    """m's destination has no feasible action, so he accepts staying under "leave".

    Specification, section 5 [A 6]: such a member's outside value is `-inf` and
    he accepts every feasible stay action, also under `tie="leave"` and also
    where his stay value is `-inf`. In cell 0 f gains from staying, 3 against
    her 1, so the pair stays. In cell 1 f is at 0 against her 1: she refuses,
    and the pair cannot separate, which leaves no value under the unilateral
    rules; under the transfer m is the stayer, accepts no candidate and the
    pair stays.
    """
    declaration, expected = _UNDEFINED_DESTINATION_RULES[rule]
    settles = isinstance(declaration, MutualConsent)
    model = Model(
        regimes={
            "couple": Regime(
                states={"case": DiscreteGrid(values=jnp.arange(2))},
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                constraints={"only_the_first_action": _only_the_first_action},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _undefined_stay_f, "m": _undefined_stay_m}
                    )
                },
                composition_action=CompositionAction(
                    rule=declaration,
                    destinations={
                        "f": Destination(
                            regime="single_f",
                            projection={},
                            utility_adjustment=_paid if settles else None,
                        ),
                        "m": Destination(
                            regime="single_m", projection={"assets": _below_the_grid}
                        ),
                    },
                ),
            ),
            "single_f": Regime(functions={"utility": _Constant(value=1.0)}),
            "single_m": Regime(
                states={"assets": LinSpacedGrid(start=-2.0, stop=2.0, n_points=5)},
                actions={"c": DiscreteGrid(values=jnp.asarray([0, 1]))},
                constraints={"solvent": _nonnegative_assets},
                functions={"utility": _own_assets},
            ),
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        regime_id_class=_Household,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )

    solution = model.solve(params={}, log_level="off")

    assert np.asarray(solution.values[0]["couple"]).tolist() == expected
