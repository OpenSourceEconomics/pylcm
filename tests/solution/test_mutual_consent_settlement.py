"""Exit rules compared at the household optimum, and a settlement before separation.

Every model has one period, so a stay value is the flow utility of the household's
choice and an outside value is the destination's flow utility plus the adjustment.
Each expected number is written down by hand from the rule, not computed by a second
implementation.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm import exit_queries, policy_queries
from _lcm.solution.fingerprint import _semantic_fingerprint
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    ExtremeValueTasteShocks,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    MutualConsent,
    Regime,
    Renegotiate,
    Settlement,
    Unilateral,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.exits import ExitRule
from lcm.typing import BoolND, FloatND, IntND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, assert_agrees_to_ulp


@categorical(ordered=False)
class _Household:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


# Each member's outside value, `(f, m)`, wherever the exit is available.
_OUTSIDE = (1.0, 2.0)

# Stay values `(f, m)` per cell and action. Equal Pareto weights pick the action
# with the larger sum; action 1 is never the optimum but passes some members.
_STAY = (
    ((3.0, 4.0), (2.0, 3.0)),  # 0: nobody refuses at the optimum
    ((6.0, 1.0), (2.0, 3.0)),  # 1: m refuses at the optimum; action 1 keeps both
    ((0.0, 1.0), (4.0, -5.0)),  # 2: both refuse; action 1 keeps f
    ((1.0, 5.0), (2.0, 3.0)),  # 3: f is indifferent at the optimum
    ((1.0, 2.0), (0.0, 2.5)),  # 4: both indifferent; action 1 strictly keeps m
    ((6.0, 1.0), (2.0, 3.0)),  # 5: as 1, but the exit is not available
    ((6.0, 1.0), (2.0, 3.0)),  # 6: no action is feasible
)
_UNAVAILABLE = 5
_INFEASIBLE = 6
# Simulation starts no subject where no action is feasible, so it covers the cells
# before the last.
_SIMULATED = _INFEASIBLE

_SEPARATE = _OUTSIDE

# Hand-derived published values `(f, m)` per cell, by rule, comparison and tie.
_EXPECTED = {
    ("unilateral", "each_action", "stay"): (
        (3, 4), (2, 3), _SEPARATE, (1, 5), (1, 2), (6, 1), _SEPARATE,
    ),
    ("unilateral", "optimum", "stay"): (
        (3, 4), _SEPARATE, _SEPARATE, (1, 5), (1, 2), (6, 1), _SEPARATE,
    ),
    ("mutual", "each_action", "stay"): (
        (3, 4), (6, 1), (4, -5), (1, 5), (1, 2), (6, 1), _SEPARATE,
    ),
    ("mutual", "optimum", "stay"): (
        (3, 4), (6, 1), _SEPARATE, (1, 5), (1, 2), (6, 1), _SEPARATE,
    ),
    ("unilateral", "each_action", "leave"): (
        (3, 4), (2, 3), _SEPARATE, (2, 3), _SEPARATE, (6, 1), _SEPARATE,
    ),
    ("unilateral", "optimum", "leave"): (
        (3, 4), _SEPARATE, _SEPARATE, _SEPARATE, _SEPARATE, (6, 1), _SEPARATE,
    ),
    ("mutual", "each_action", "leave"): (
        (3, 4), (6, 1), (4, -5), (1, 5), (0, 2.5), (6, 1), _SEPARATE,
    ),
    ("mutual", "optimum", "leave"): (
        (3, 4), (6, 1), _SEPARATE, (1, 5), _SEPARATE, (6, 1), _SEPARATE,
    ),
}  # fmt: skip

# Whether the pair separates in each cell under the optimum comparison. Cell 4 under
# `tie="stay"` publishes the outside values while staying, so only the route
# tells the two apart.
_SEPARATES = {
    ("unilateral", "stay"): (False, True, True, False, False, False, True),
    ("mutual", "stay"): (False, False, True, False, False, False, True),
    ("unilateral", "leave"): (False, True, True, True, True, False, True),
    ("mutual", "leave"): (False, False, True, False, True, False, True),
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


def _rule(
    *,
    rule: Literal["unilateral", "mutual"],
    compares: Literal["each_action", "optimum"] | None,
    tie: Literal["stay", "leave"],
) -> ExitRule:
    if rule == "unilateral":
        return Unilateral(tie=tie, compares=compares)
    return MutualConsent(tie=tie, compares=compares)


def _comparison_model(*, rule: ExitRule) -> Model:
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
                exit=Exit(
                    rule=rule,
                    targets={
                        member: ExitTarget(regime=f"single_{member}", projection={})
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
    """Simulate one subject per cell and member, ordered by cell, then f before m."""
    solution = model.solve(params={}, log_level="off")
    roles = [model.stakeholder_names_to_ids[member] for member in ("f", "m")]
    return (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(2 * n_cells, dtype=jnp.int32),
                "regime_id": jnp.full(
                    2 * n_cells, model.regime_names_to_ids["couple"], dtype=jnp.int32
                ),
                "own_stakeholder": jnp.tile(
                    jnp.asarray(roles, dtype=jnp.int32), n_cells
                ),
                "case": jnp.repeat(jnp.arange(n_cells, dtype=jnp.int32), 2),
            },
            seed=6271,
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
        .reset_index(drop=True)
    )


def _own_values(frame: pd.DataFrame) -> np.ndarray:
    """Each subject's reported value: its member's couple value, or its single value."""
    own = np.where(frame.index % 2 == 0, frame["value_f"], frame["value_m"])
    return np.where(frame["regime_name"] == "couple", own, frame["value"])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("compares", ["each_action", "optimum"])
@pytest.mark.parametrize("rule", ["unilateral", "mutual"])
def test_comparison_point_decides_the_published_values(
    *,
    rule: Literal["unilateral", "mutual"],
    compares: Literal["each_action", "optimum"],
    tie: Literal["stay", "leave"],
) -> None:
    """Each cell publishes the hand-derived stay or outside values of its rule."""
    model = _comparison_model(rule=_rule(rule=rule, compares=compares, tie=tie))
    solution = model.solve(params={}, log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["couple"],
        np.asarray(_EXPECTED[rule, compares, tie], dtype=float),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("rule", ["unilateral", "mutual"])
def test_compares_none_without_adjustment_is_each_action(
    *, rule: Literal["unilateral", "mutual"]
) -> None:
    """An undeclared comparison point keeps the per-action rule."""
    model = _comparison_model(rule=_rule(rule=rule, compares=None, tie="stay"))
    np.testing.assert_array_almost_equal(
        model.solve(params={}, log_level="off").values[0]["couple"],
        np.asarray(_EXPECTED[rule, "each_action", "stay"], dtype=float),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("rule", ["unilateral", "mutual"])
def test_optimum_comparison_routes_simulated_members_as_the_solve_prices_them(
    *, rule: Literal["unilateral", "mutual"], tie: Literal["stay", "leave"]
) -> None:
    """Separating cells land in the singles; every subject reports the solve's value."""
    model = _comparison_model(rule=_rule(rule=rule, compares="optimum", tie=tie))
    frame = _simulate_every_cell(model=model, n_cells=_SIMULATED)
    separates = np.repeat(np.asarray(_SEPARATES[rule, tie][:_SIMULATED]), 2)
    expected_regimes = np.where(
        separates, np.tile(["single_f", "single_m"], _SIMULATED), "couple"
    )
    assert list(frame["regime_name"]) == list(expected_regimes)
    np.testing.assert_array_almost_equal(
        _own_values(frame),
        np.asarray(_EXPECTED[rule, "optimum", tie][:_SIMULATED], dtype=float).reshape(
            -1
        ),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("rule", ["unilateral", "mutual"])
def test_optimum_comparison_stays_at_the_optimal_action(
    *, rule: Literal["unilateral", "mutual"]
) -> None:
    """A pair that stays acts at the household optimum, never at a passing action."""
    model = _comparison_model(rule=_rule(rule=rule, compares="optimum", tie="stay"))
    frame = _simulate_every_cell(model=model, n_cells=_SIMULATED)
    staying = frame["regime_name"] == "couple"
    np.testing.assert_array_equal(frame.loc[staying, "choice"], 0)


# Semantic fingerprints of these rules as declared before `compares` and `settle`
# existed.
_DURABLE_FINGERPRINTS = (
    (
        Unilateral(),
        "75063a53a29d490928e0bff21b58763b0623947b4371a5dc7f07507bdb587288",
    ),
    (
        Unilateral(tie="leave"),
        "7337f3697d2cd718a5629c9eb44928b47d6379adc26a6e38166c3677f6bc8ee5",
    ),
    (
        MutualConsent(),
        "0e895bc87e7f2a7f84438b4b43b49c9dcfaee3003333fa15211999051b57d17a",
    ),
    (
        MutualConsent(tie="leave"),
        "59cb3cb5a3f7fdb4ef9ab6ccd89777da157a60b31234ab04eac52957b3ad0154",
    ),
    (
        Unilateral(renegotiate=Renegotiate(state="mu")),
        "4b8cddfe7f3b1db2a56d82f0c2735d6a4328ec0e06815e8ae29f34250aad7e2e",
    ),
)


@pytest.mark.parametrize(
    ("rule", "fingerprint"),
    _DURABLE_FINGERPRINTS,
    ids=("unilateral", "unilateral-leave", "mutual", "mutual-leave", "renegotiate"),
)
def test_rules_without_the_new_fields_keep_their_durable_fingerprint(
    *, rule: ExitRule, fingerprint: str
) -> None:
    """A rule leaving `compares` and `settle` unset hashes as before they existed."""
    assert _semantic_fingerprint(rule) == fingerprint


@pytest.mark.parametrize("rule", ["unilateral", "mutual"])
def test_explicit_comparison_point_is_part_of_the_durable_identity(
    *, rule: Literal["unilateral", "mutual"]
) -> None:
    """Declaring where the rule compares changes the solution identity."""
    default = _rule(rule=rule, compares=None, tie="stay")
    assert _semantic_fingerprint(default) != _semantic_fingerprint(
        replace(default, compares="optimum")
    )


@categorical(ordered=False)
class _Singleton:
    source: ScalarInt
    target: ScalarInt


def _choice(*, choice: IntND) -> FloatND:
    return jnp.asarray(choice, dtype=float)


def test_optimum_comparison_refuses_taste_shocks() -> None:
    """A household optimum under taste shocks is a draw, not a value to compare."""
    with pytest.raises(RegimeInitializationError, match="taste shocks"):
        Model(
            regimes={
                "source": Regime(
                    actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                    functions={"utility": _choice},
                    taste_shocks=ExtremeValueTasteShocks(),
                    exit=Exit(
                        targets={"self": ExitTarget(regime="target", projection={})},
                        rule=Unilateral(compares="optimum"),
                    ),
                ),
                "target": Regime(functions={"utility": _Constant(value=0.0)}),
            },
            n_periods=1,
            regime_id_class=_Singleton,
            initial_nodes=InitialNodes(by_period={0: "source"}),
            edges={},
            fixed_params={"source": {"taste_shocks": {"scale": 1.0}}},
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )


def test_renegotiation_refuses_a_per_action_comparison() -> None:
    """An adjustment before the exit is decided at the optimum only."""
    with pytest.raises(RegimeInitializationError, match="each_action"):
        Unilateral(compares="each_action", renegotiate=Renegotiate(state="mu"))


# Settlement cells. The couple has no action, so its stay values are `W`. At
# settlement `s` the outside values are `O_f(s) = A_f + 4 s` and
# `O_m(s) = A_m - 4 s`, so increasing `s` favours f, the first stakeholder, and one
# candidate step moves each outside value by one.
_POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Each cell is the stay values `W`, the intercepts `A` and the default settlement.
_SETTLE_CELLS = (
    ((5.0, 5.0), (0.0, 0.0), 0.5),  # 0: nobody refuses at the default
    ((0.0, 0.0), (1.0, 3.0), 0.375),  # 1: both refuse at a default between nodes
    ((2.5, 0.5), (0.0, 4.0), 0.375),  # 2: m refuses; f accepts first at 0.75
    ((2.5, 1.5), (0.0, 4.0), 0.375),  # 3: at 0.75 m no longer gains
    ((4.5, 0.5), (0.0, 4.0), 0.375),  # 4: no candidate makes f accept
    ((-0.5, 3.5), (0.0, 4.0), 0.375),  # 5: f refuses; m accepts first at 0
    ((-0.5, 3.0), (0.0, 4.0), 0.375),  # 6: m accepts at equality at 0.25
    ((1.0, 2.8), (0.0, 4.0), 0.375),  # 7: at 0.25 f is indifferent
    ((0.0, 0.0), (1.0, 3.0), 0.375),  # 8: as 1, but the exit is not available
    ((-10.0, -10.0), (1.0, 3.0), 1.5),  # 9: both refuse; the default is clipped
)
_SETTLE_UNAVAILABLE = 8

# Hand-derived settlement at which each cell separates, None where the pair stays.
_SETTLED_AT = {
    "stay": (None, 0.375, 0.75, None, None, 0.0, 0.25, None, None, 1.0),
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


def _settlement(
    *,
    name: str = "share",
    candidates: LinSpacedGrid | DiscreteGrid | None = None,
) -> Settlement:
    return Settlement(
        name=name,
        candidates=LinSpacedGrid(start=0.0, stop=1.0, n_points=len(_POINTS))
        if candidates is None
        else candidates,
        default=_CellValue(values=tuple(cell[2] for cell in _SETTLE_CELLS)),
    )


def _settlement_model(
    *,
    division: Literal["transfer", "assets"],
    tie: Literal["stay", "leave"] = "stay",
    settlement: Settlement | None = None,
) -> Model:
    """Two members who settle through a transfer or through a share of assets."""
    settle = _settlement() if settlement is None else settlement
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
                exit=Exit(
                    rule=MutualConsent(tie=tie, settle=settle),
                    targets={
                        member: ExitTarget(
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
        # The singles start subjects too, so they are solved and outside values
        # landing on their grid are read from stored tables.
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        regime_id_class=_Household,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("division", ["transfer", "assets"])
def test_settlement_publishes_the_hand_derived_values(
    *, division: Literal["transfer", "assets"], tie: Literal["stay", "leave"]
) -> None:
    """Each cell publishes `W` where the pair stays and `O(s*)` where it separates."""
    model = _settlement_model(division=division, tie=tie)
    np.testing.assert_array_almost_equal(
        model.solve(params={}, log_level="off").values[0]["couple"],
        _settled_values(tie=tie),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize("division", ["transfer", "assets"])
def test_settlement_lands_simulated_members_at_the_selected_settlement(
    *, division: Literal["transfer", "assets"], tie: Literal["stay", "leave"]
) -> None:
    """Separating members land at `s*` and report the solve's value there."""
    model = _settlement_model(division=division, tie=tie)
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
def test_settlement_divides_the_assets_movers_land_with() -> None:
    """A mover's assets are its projection evaluated at the selected settlement."""
    frame = _simulate_every_cell(
        model=_settlement_model(division="assets"), n_cells=len(_SETTLE_CELLS)
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
        patch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
        model = _settlement_model(division="assets")
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
def test_settlement_values_agree_with_and_without_stored_outside_reads(
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


def _model_with(*, regime_change: Callable[[Regime], Regime]) -> Model:
    base = _settlement_model(division="transfer")
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
        declaration = regime.exit
        assert declaration is not None
        return replace(regime, exit=replace(declaration, rule=rule))

    return change


def _without_reads(regime: Regime) -> Regime:
    declaration = regime.exit
    assert declaration is not None
    return replace(
        regime,
        exit=replace(
            declaration,
            targets={
                member: replace(target, utility_adjustment=_Constant(value=0.0))
                for member, target in declaration.targets.items()
            },
        ),
    )


@pytest.mark.parametrize(
    ("regime_change", "match"),
    [
        (_without_reads, "read by no exit-target"),
        (_with_rule(MutualConsent(settle=_settlement(name="case"))), "name of a state"),
        (
            _with_rule(MutualConsent(settle=_settlement(name="utility"))),
            "name of a state",
        ),
        (
            _with_rule(MutualConsent(settle=_settlement(name="utility_f"))),
            "name of a state",
        ),
    ],
    ids=("unread", "state", "function", "engine_function"),
)
def test_settlement_refuses_an_unread_or_taken_name(
    *, regime_change: Callable[[Regime], Regime], match: str
) -> None:
    """A settlement must be read on exit and cannot shadow the regime's own names."""
    with pytest.raises(RegimeInitializationError, match=match):
        _model_with(regime_change=regime_change)


def test_settlement_refuses_a_singleton_regime() -> None:
    """Moving a division toward the member who stays needs two members."""
    with pytest.raises(RegimeInitializationError, match="two collective"):
        Model(
            regimes={
                "source": Regime(
                    states={
                        "case": DiscreteGrid(values=jnp.arange(len(_SETTLE_CELLS)))
                    },
                    functions={"utility": _CellValue(values=(0.0,) * 10)},
                    exit=Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection={},
                                utility_adjustment=_divided(0),
                            )
                        },
                        rule=MutualConsent(settle=_settlement()),
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
def test_settlement_refuses_candidates_without_fixed_ordered_points(
    *, candidates: DiscreteGrid | IrregSpacedGrid
) -> None:
    """Candidates are an ordered continuous grid known when the exit is declared."""
    with pytest.raises(RegimeInitializationError, match="Settlement candidates"):
        Settlement(name="share", candidates=candidates, default=_Constant(value=0.5))


def test_settlement_refuses_a_per_action_comparison() -> None:
    """A settlement before the exit is decided at the optimum only."""
    with pytest.raises(RegimeInitializationError, match="each_action"):
        MutualConsent(compares="each_action", settle=_settlement())


def test_settlement_is_part_of_the_durable_identity() -> None:
    """Declaring a settlement changes the rule's fingerprint."""
    assert _semantic_fingerprint(MutualConsent()) != _semantic_fingerprint(
        MutualConsent(settle=_settlement())
    )
