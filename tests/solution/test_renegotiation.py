"""Discrete renegotiation follows a literal directional walk of stay policies."""

import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import which
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

import lcm
import tests.conftest
from lcm import (
    ByPeriod,
    CollectiveUtility,
    CompositionAction,
    Destination,
    DiscreteGrid,
    ExecutionConfig,
    Gate,
    InitialNodes,
    LinSpacedGrid,
    MemberRoute,
    Model,
    ParetoObjective,
    PCOnActions,
    PCOnWeight,
    Periods,
    Phased,
    ProjectedRegimeValue,
    Regime,
    SimulationResult,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    RegimeInitializationError,
    ShockDeclarationError,
    UnsupportedOperationError,
)
from lcm.typing import BoolND, FloatND, IntND, ScalarInt, StateName


@categorical(ordered=False)
class _Household:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


@categorical(ordered=False)
class _DynamicHousehold:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class _Weight:
    low: ScalarInt
    middle_low: ScalarInt
    middle_high: ScalarInt
    high: ScalarInt


@dataclass(frozen=True, kw_only=True)
class _Payoff:
    values: tuple[int, ...]

    def __call__(self, *, mu: IntND, choice: IntND) -> FloatND:
        return jnp.asarray(self.values, dtype=float)[mu] - 3.0 * (choice - mu) ** 2


def _outside() -> float:
    return 0.0


def _reference_node(
    *,
    values: tuple[tuple[int, int], ...],
    incumbent: int,
    tie: Literal["stay", "leave"] = "stay",
) -> int | None:
    """Walk integer payoffs, without JAX, engine helpers or a global search."""
    willing = tuple(
        tuple(value >= 0 if tie == "stay" else value > 0 for value in pair)
        for pair in values
    )
    wife, husband = willing[incumbent]
    if wife and husband:
        return incumbent
    if not wife and not husband:
        return None
    step = 1 if not wife else -1
    candidate = incumbent + step
    while 0 <= candidate < len(values):
        wife, husband = willing[candidate]
        if (step == 1 and not husband) or (step == -1 and not wife):
            return None
        if wife and husband:
            return candidate
        candidate += step
    return None


def _model(
    *,
    values: tuple[tuple[int, int], ...],
    tie: Literal["stay", "leave"] = "stay",
    n_periods: int = 1,
    execution_config: ExecutionConfig | None = None,
    mu_law: Callable[..., IntND] | None = None,
) -> Model:
    return Model(
        regimes={
            "couple": Regime(
                states={"mu": DiscreteGrid(_Weight)},
                actions={"choice": DiscreteGrid(_Weight)},
                state_transitions={
                    "mu": fixed_transition("mu") if mu_law is None else mu_law
                }
                if n_periods > 1
                else {},
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            member: _Payoff(
                                values=tuple(pair[index] for pair in values)
                            )
                            for index, member in enumerate(("f", "m"))
                        }
                    )
                },
                composition_action=CompositionAction(
                    rule=PCOnWeight(state="mu", tie=tie),
                    destinations={
                        member: Destination(regime=f"single_{member}", projection={})
                        for member in ("f", "m")
                    },
                ),
            ),
            "single_f": Regime(functions={"utility": _outside}),
            "single_m": Regime(functions={"utility": _outside}),
            **(
                {
                    "done": Regime(
                        functions={
                            "utility": CollectiveUtility(
                                utilities={"f": _outside, "m": _outside}
                            )
                        }
                    )
                }
                if n_periods > 1
                else {}
            ),
        },
        n_periods=n_periods + 1 if n_periods > 1 else n_periods,
        edges={
            "couple": {
                "couple": Periods(values=tuple(range(n_periods - 1))),
                "done": Periods(values=(n_periods - 1,)),
            }
        }
        if n_periods > 1
        else {},
        initial_nodes=InitialNodes(by_period={0: "couple"}),
        regime_id_class=_DynamicHousehold if n_periods > 1 else _Household,
        execution_config=execution_config or ExecutionConfig(device_memory_bytes=None),
    )


_VALUES = ((-3, 4), (-1, 3), (1, 2), (2, 1))


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("tie", ["stay", "leave"])
@pytest.mark.parametrize(
    "values",
    [
        _VALUES,
        ((1, 2), (2, 1), (3, -1), (4, -3)),
        ((-1, -1), (1, 1), (2, 2), (-2, -2)),
        ((-2, 3), (-1, -1), (2, 2), (3, 1)),
        ((-2, 3), (-1, 2), (-1, 1), (-1, 1)),
        ((1, 1), (0, 2), (2, 0), (1, 1)),
    ],
)
def test_renegotiation_prices_the_first_mutually_willing_grid_policy(
    *, values: tuple[tuple[int, int], ...], tie: Literal["stay", "leave"]
) -> None:
    """Values use the selected node's optimum, including directional vetoes and ties."""
    nodes = [
        _reference_node(values=values, incumbent=index, tie=tie) for index in range(4)
    ]
    expected = np.asarray(
        [values[index] if index is not None else (0, 0) for index in nodes]
    )
    model = _model(values=values, tie=tie)
    solution = model.solve(params={}, log_level="off")
    np.testing.assert_array_equal(solution.values[0]["couple"], expected)


def test_renegotiation_carries_selected_state_and_its_action_to_the_next_period() -> (
    None
):
    """A refused incumbent changes before acting and remains changed next period."""
    model = _model(values=_VALUES, n_periods=2)
    params = {"discount_factor": 0.0}
    solution = model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(4, dtype=jnp.int32),
            "regime_id": jnp.full(
                4, model.regime_names_to_ids["couple"], dtype=jnp.int32
            ),
            "own_member": jnp.full(4, model.member_names_to_ids["f"], dtype=jnp.int32),
            "mu": jnp.arange(4, dtype=jnp.int32),
        },
        seed=659,
        log_level="off",
    )
    frame = result.to_dataframe(use_categorical_codes=True).sort_values(
        ["period", "agent_id"]
    )
    np.testing.assert_array_equal(
        frame.loc[frame["regime_name"] == "couple", ["mu", "choice"]],
        np.tile(np.asarray([[2, 2], [2, 2], [2, 2], [3, 3]]), (2, 1)),
    )


@pytest.mark.parametrize("state", ["missing", "choice"])
def test_renegotiation_refuses_a_name_that_is_not_a_state(*, state: StateName) -> None:
    """The adjustment cannot silently index an action or an absent coordinate."""
    base = _model(values=_VALUES)
    couple = base.user_regimes["couple"]
    declaration = couple.composition_action
    assert isinstance(declaration, CompositionAction)
    changed = replace(declaration, rule=PCOnWeight(state=state))
    with pytest.raises(RegimeInitializationError, match="declared discrete grid state"):
        Model(
            regimes={
                **base.user_regimes,
                "couple": replace(couple, composition_action=changed),
            },
            n_periods=1,
            edges={},
            initial_nodes=InitialNodes(by_period={0: "couple"}),
            regime_id_class=_Household,
        )


@pytest.mark.parametrize("continuous", [False, True])
def test_renegotiation_refuses_an_unordered_or_continuous_state(
    *, continuous: bool
) -> None:
    """A direction is meaningful only on the declared ordered discrete domain."""
    base = _model(values=_VALUES)
    couple = base.user_regimes["couple"]
    grid = (
        LinSpacedGrid(start=0, stop=3, n_points=4)
        if continuous
        else DiscreteGrid(_Household)
    )
    with pytest.raises(
        RegimeInitializationError, match=r"discrete grid state|ordered discrete state"
    ):
        Model(
            regimes={
                **base.user_regimes,
                "couple": replace(couple, states={"mu": grid}),
            },
            n_periods=1,
            edges={},
            initial_nodes=InitialNodes(by_period={0: "couple"}),
            regime_id_class=_Household,
        )


def _separation_simulation(*, execution_config: ExecutionConfig) -> SimulationResult:
    """Invented actors cover a veto before a later island, staying and a move down."""
    model = _model(
        values=((-2, 3), (-1, -1), (2, 2), (3, -1)), execution_config=execution_config
    )
    return model.simulate(
        params={},
        solution=model.solve(params={}, log_level="off"),
        initial_conditions={
            "period": jnp.zeros(7, dtype=jnp.int32),
            "regime_id": jnp.full(
                7, model.regime_names_to_ids["couple"], dtype=jnp.int32
            ),
            "own_member": jnp.asarray(
                [
                    model.member_names_to_ids[member]
                    for member in ("f", "m", "f", "m", "f", "m", "f")
                ],
                dtype=jnp.int32,
            ),
            "mu": jnp.asarray([0, 1, 2, 3, 0, 2, 3], dtype=jnp.int32),
        },
        seed=659,
        log_level="off",
    )


def check_agent_sharded_renegotiation() -> None:
    """Check independent branch anchors and every native output under partitioning."""
    expected = _separation_simulation(
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=None)
    )
    frame = expected.to_dataframe(use_categorical_codes=True).sort_values("agent_id")
    np.testing.assert_array_equal(
        frame["regime_name"],
        ["single_f", "single_m", "couple", "couple", "single_f", "couple", "couple"],
    )
    np.testing.assert_array_equal(
        frame.loc[frame["regime_name"] == "couple", ["mu", "choice"]],
        np.full((4, 2), 2),
    )
    actual = _separation_simulation(
        execution_config=ExecutionConfig(
            devices=(0, 1, 2, 3),
            simulation_sharding="agents",
            device_memory_bytes=None,
        )
    )
    assert actual.n_agents == expected.n_agents == 7
    _assert_same_raw_results(actual=actual, expected=expected)
    expected_gate = _gated_outside_simulation(
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=None)
    )
    actual_gate = _gated_outside_simulation(
        execution_config=ExecutionConfig(
            devices=(0, 1, 2, 3),
            simulation_sharding="agents",
            device_memory_bytes=None,
        )
    )
    gate_frame = expected_gate.to_dataframe(use_categorical_codes=True)
    current = gate_frame.loc[gate_frame["period"] == 0].sort_values("agent_id")
    np.testing.assert_array_equal(
        current["regime_name"],
        ["couple", "outside", "couple", "outside", "couple", "outside", "couple"],
    )
    np.testing.assert_array_equal(
        current.loc[current["regime_name"] == "couple", ["mu", "choice"]],
        [[1, 1], [2, 2], [2, 2], [1, 1]],
    )
    _assert_same_raw_results(actual=actual_gate, expected=expected_gate)


def _assert_same_raw_results(
    *, actual: SimulationResult, expected: SimulationResult
) -> None:
    import jax  # noqa: PLC0415

    # Compare every field and leaf with the same mapping-container representation.
    actual_tree, expected_tree = (
        {
            name: {
                period: replace(data, shocks=dict(data.shocks))
                for period, data in periods.items()
            }
            for name, periods in result.raw_results.items()
        }
        for result in (actual, expected)
    )
    assert jax.tree.structure(actual_tree) == jax.tree.structure(expected_tree)
    for got, want in zip(
        jax.tree.leaves(actual_tree),
        jax.tree.leaves(expected_tree),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)


def test_renegotiation_agent_sharding_preserves_state_actions_and_separation() -> None:
    """The selected state stays agent-local, with padding and split separations."""
    code = (
        "import jax; jax.config.update('jax_num_cpu_devices', 4); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "from tests.solution.test_renegotiation import "
        f"{check_agent_sharded_renegotiation.__name__}; "
        f"{check_agent_sharded_renegotiation.__name__}()"
    )
    pixi = which("pixi")
    assert pixi is not None, "The isolated topology check requires the Pixi runner."
    result = subprocess.run(  # noqa: S603 - fixed source on the locked project runner
        [pixi, "run", "--locked", "-e", "tests-cpu", "python", "-c", code],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_budgeted_composition_action_replay_reports_the_unbudgeted_raw_results() -> (
    None
):
    """A device budget leaves every raw result and its state order unchanged."""
    expected = _gated_outside_simulation(
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=None)
    )
    actual = _gated_outside_simulation(
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=2**30)
    )
    _assert_same_raw_results(actual=actual, expected=expected)


def _reset_weight(*, mu: IntND) -> IntND:
    return jnp.zeros_like(mu)


def test_renegotiation_refuses_a_law_that_discards_the_selected_node() -> None:
    """The next weight must be the selected node, not a conflicting user reset."""
    with pytest.raises(RegimeInitializationError, match="fixed_transition"):
        _model(values=_VALUES, n_periods=2, mu_law=_reset_weight)


def test_policy_lookup_refuses_to_bypass_current_renegotiation() -> None:
    """An unrouted action lookup cannot report the incumbent Raw choice as Full."""
    model = _model(values=_VALUES)
    solution = model.solve(params={}, log_level="off")
    with pytest.raises(
        UnsupportedOperationError, match="PCOnWeight with a Pareto weight state"
    ):
        model.lookup_policy(
            params={},
            solution=solution,
            period=0,
            regime_name="couple",
            states={"mu": jnp.arange(4, dtype=jnp.int32)},
            use_categorical_codes=True,
        )


def test_renegotiation_is_part_of_the_durable_solution_identity() -> None:
    """Opting into adjustment cannot share the identity of ordinary dissolution."""
    adjusted = _model(values=_VALUES)
    couple = adjusted.user_regimes["couple"]
    declaration = couple.composition_action
    assert isinstance(declaration, CompositionAction)
    ordinary = Model(
        regimes={
            **adjusted.user_regimes,
            "couple": replace(
                couple, composition_action=replace(declaration, rule=PCOnActions())
            ),
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: "couple"}),
        regime_id_class=_Household,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    assert adjusted.solve(params={}, log_level="off").metadata.model_fingerprint != (
        ordinary.solve(params={}, log_level="off").metadata.model_fingerprint
    )


def _female_tradeoff(*, choice: IntND) -> FloatND:
    return jnp.where(choice == 0, 3.0, 0.0)


def _male_tradeoff(*, choice: IntND) -> FloatND:
    return jnp.where(choice == 0, -1.0, 2.0)


def _state_weight(*, mu: IntND) -> FloatND:
    return 2.0 ** (1 - jnp.asarray(mu, dtype=float))


def test_renegotiation_adjusts_after_the_state_weighted_raw_choice() -> None:
    """A feasible alternative action at the incumbent does not replace Raw policy."""
    base = _model(values=_VALUES)
    couple = base.user_regimes["couple"]
    model = Model(
        regimes={
            **base.user_regimes,
            "couple": replace(
                couple,
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _female_tradeoff, "m": _male_tradeoff},
                        objective=lcm.ParetoObjective(
                            weights={"f": 1.0, "m": _state_weight}
                        ),
                    )
                },
            ),
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: "couple"}),
        regime_id_class=_Household,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    # At `mu = 1` the weights are (1,1), both actions tie and Raw takes `f`'s best,
    # index 0, worth (3,-1). `m` refuses, so the node moves down to `mu = 0`, whose
    # weights (1,2) choose index 1, worth (0,2); both then accept.
    result = model.simulate(
        params={},
        solution=model.solve(params={}, log_level="off"),
        initial_conditions={
            "period": jnp.asarray([0], dtype=jnp.int32),
            "regime_id": jnp.asarray(
                [model.regime_names_to_ids["couple"]], dtype=jnp.int32
            ),
            "own_member": jnp.asarray(
                [model.member_names_to_ids["f"]], dtype=jnp.int32
            ),
            "mu": jnp.asarray([1], dtype=jnp.int32),
        },
        seed=659,
        log_level="off",
    )
    frame = result.to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(
        frame[["mu", "choice", "value_f", "value_m"]], [[0, 1, 0, 2]]
    )


@categorical(ordered=False)
class _GatedHousehold:
    couple: ScalarInt
    outside: ScalarInt
    landing: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt
    done: ScalarInt


def _closed_gate() -> BoolND:
    return jnp.asarray(a=False)


def _unreachable_landing() -> float:
    return 99.0


def _female_retirement(*, assets: FloatND) -> FloatND:
    return assets / 2


def _male_retirement(*, assets: FloatND) -> FloatND:
    return 2 * assets


def _low_assets_gate(*, assets: FloatND) -> BoolND:
    return assets < 3


def _open_female(*, assets: FloatND) -> FloatND:
    return assets / 2 + 1


def _open_male(*, assets: FloatND) -> FloatND:
    return 2 * assets - 1


def _gated_outside_model(
    *, pointwise: bool, execution_config: ExecutionConfig | None = None
) -> Model:
    """Build an outside query with separate personal future domains."""
    assets = LinSpacedGrid(start=2.0, stop=4.0, n_points=2)
    return Model(
        regimes={
            "couple": Regime(
                states={"assets": assets, "mu": DiscreteGrid(_Weight)},
                actions={"choice": DiscreteGrid(_Weight)},
                state_transitions={
                    "assets": fixed_transition("assets"),
                    "mu": fixed_transition("mu"),
                },
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            "f": _Payoff(values=(0, 2, 3, 4)),
                            "m": _Payoff(values=(5, 4, 3, 2)),
                        }
                    )
                },
                composition_action=CompositionAction(
                    rule=PCOnWeight(state="mu"),
                    destinations={
                        member: Destination(
                            regime="outside",
                            role=member,
                            projection={"assets": fixed_transition("assets")},
                        )
                        for member in ("f", "m")
                    },
                ),
            ),
            "outside": Regime(
                states={"assets": assets},
                state_transitions={"assets": fixed_transition("assets")},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _outside, "m": _outside}
                    )
                },
            ),
            "landing": Regime(
                states={"assets": assets},
                functions={
                    "utility": CollectiveUtility(
                        utilities=(
                            {
                                "aux": _unreachable_landing,
                                "m": _open_male,
                                "f": _open_female,
                            }
                            if pointwise
                            else {
                                "f": _unreachable_landing,
                                "m": _unreachable_landing,
                            }
                        )
                    ),
                    "landing_assets": fixed_transition("assets"),
                },
                output_only_functions=("landing_assets",),
            ),
            "single_f": Regime(
                states={"assets": assets},
                functions={"utility": _female_retirement},
            ),
            "single_m": Regime(
                states={"assets": assets},
                functions={"utility": _male_retirement},
            ),
            "done": Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _outside, "m": _outside}
                    )
                }
            ),
        },
        edges={
            "couple": {"done": Periods(values=(0,))},
            "outside": Transition(
                law=ByPeriod(cases={0: "landing"}),
                gates={
                    "landing": Gate(
                        predicate=_low_assets_gate if pointwise else _closed_gate,
                        routes={
                            member: MemberRoute(
                                target_member=member,
                                fallback=ProjectedRegimeValue(
                                    regime=f"single_{member}",
                                    projection={"assets": fixed_transition("assets")},
                                ),
                            )
                            for member in ("f", "m")
                        },
                    )
                },
            ),
        },
        initial_nodes=InitialNodes(by_period={0: "couple"}),
        n_periods=2,
        regime_id_class=_GatedHousehold,
        execution_config=execution_config or ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.parametrize("pointwise", [False, True])
def test_renegotiation_outside_option_keeps_future_personal_gate_values(
    *, pointwise: bool
) -> None:
    """The outside benchmark reads future personal domains through its own gate."""
    model = _gated_outside_model(pointwise=pointwise)
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    couple_values = solution.value_array(step=0, regime="couple")
    np.testing.assert_array_equal(
        couple_values.sel(assets=2.0),
        [[2, 4], [2, 4], [3, 3], [3, 3]] if pointwise else [[2, 4]] * 4,
    )
    np.testing.assert_array_equal(couple_values.sel(assets=4.0), [[2, 8]] * 4)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.full(
                2, model.regime_names_to_ids["couple"], dtype=jnp.int32
            ),
            "own_member": jnp.full(2, model.member_names_to_ids["f"], dtype=jnp.int32),
            "assets": jnp.asarray([2.0, 4.0]),
            "mu": jnp.zeros(2, dtype=jnp.int32),
        },
        seed=659,
        log_level="off",
    )
    frame = result.to_dataframe(use_categorical_codes=True)
    current = frame.loc[frame["period"] == 0].sort_values("agent_id")
    np.testing.assert_array_equal(current["regime_name"], ["couple", "outside"])
    np.testing.assert_array_equal(
        current.loc[current["regime_name"] == "couple", ["mu", "choice"]],
        [[1, 1]],
    )
    future = frame.loc[frame["regime_name"] == "single_f"]
    np.testing.assert_array_equal(future["value"], [2.0])


def _gated_outside_simulation(*, execution_config: ExecutionConfig) -> SimulationResult:
    model = _gated_outside_model(pointwise=True, execution_config=execution_config)
    params = {"discount_factor": 1.0}
    return model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions={
            "period": jnp.zeros(7, dtype=jnp.int32),
            "regime_id": jnp.full(
                7, model.regime_names_to_ids["couple"], dtype=jnp.int32
            ),
            "own_member": jnp.asarray(
                [
                    model.member_names_to_ids[member]
                    for member in ("f", "m", "f", "m", "f", "m", "f")
                ],
                dtype=jnp.int32,
            ),
            "assets": jnp.asarray([2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0]),
            "mu": jnp.asarray([0, 1, 2, 3, 3, 0, 1], dtype=jnp.int32),
        },
        seed=659,
        log_level="off",
    )


@categorical(ordered=False)
class Household:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


@categorical(ordered=True)
class OneNode:
    only: ScalarInt


@categorical(ordered=True)
class TwoNodes:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=True)
class FourNodes:
    low: ScalarInt
    middle_low: ScalarInt
    middle_high: ScalarInt
    high: ScalarInt


@dataclass(frozen=True, kw_only=True)
class TableUtility:
    values: tuple[tuple[float, ...], ...]
    overflow: bool = False

    def __call__(self, *, mu: IntND, choice: IntND) -> FloatND:
        value = jnp.asarray(self.values, dtype=float)[mu, choice]
        return value * jnp.exp(jnp.asarray(1000.0)) if self.overflow else value


@dataclass(frozen=True, kw_only=True)
class Feasibility:
    values: tuple[tuple[bool, ...], ...]

    def __call__(self, *, mu: IntND, choice: IntND) -> BoolND:
        return jnp.asarray(self.values, dtype=bool)[mu, choice]


@dataclass(frozen=True, kw_only=True)
class Availability:
    value: bool

    def __call__(self) -> BoolND:
        return jnp.asarray(self.value)


def outside() -> float:
    return 0.0


@dataclass(frozen=True, kw_only=True)
class NativeCase:
    name: str
    n: int
    bad: int
    kind: str
    placement: str = "incumbent"
    reflected: bool = False
    tie: Literal["stay", "leave"] = "stay"
    active: bool = True

    @property
    def undefined(self) -> bool:
        return self.kind in {"undefined", "undefined_alt", "overflow"}


CASES = (
    NativeCase(name="one_finite_expression_weak", n=1, bad=0, kind="overflow"),
    NativeCase(
        name="one_finite_expression_strict", n=1, bad=0, kind="overflow", tie="leave"
    ),
    NativeCase(name="one_incumbent_weak", n=1, bad=0, kind="undefined"),
    NativeCase(
        name="one_roles_strict_inactive",
        n=1,
        bad=0,
        kind="undefined",
        reflected=True,
        tie="leave",
        active=False,
    ),
    NativeCase(
        name="two_incumbent_finite_alternative",
        n=2,
        bad=0,
        kind="undefined_alt",
        tie="leave",
    ),
    NativeCase(
        name="two_scanned_roles_finite_alternative",
        n=2,
        bad=1,
        kind="undefined_alt",
        placement="scanned",
        reflected=True,
    ),
    NativeCase(
        name="four_scanned_last", n=4, bad=3, kind="undefined", placement="scanned"
    ),
    NativeCase(
        name="four_unused_required_raw",
        n=4,
        bad=3,
        kind="undefined",
        placement="unused",
        reflected=True,
    ),
    NativeCase(
        name="four_scanned_strict_inactive",
        n=4,
        bad=3,
        kind="undefined",
        placement="scanned",
        tie="leave",
        active=False,
    ),
    NativeCase(
        name="four_incumbent_roles_finite_alternative",
        n=4,
        bad=0,
        kind="undefined_alt",
        reflected=True,
        tie="leave",
    ),
    NativeCase(name="one_infeasible_poison_control", n=1, bad=0, kind="infeasible_alt"),
    NativeCase(
        name="four_infeasible_roles_control",
        n=4,
        bad=3,
        kind="infeasible_alt",
        placement="scanned",
        reflected=True,
        tie="leave",
    ),
    NativeCase(name="two_zero_first_weight_control", n=2, bad=0, kind="zero_first"),
    NativeCase(
        name="four_zero_second_weight_roles_inactive",
        n=4,
        bad=2,
        kind="zero_second",
        reflected=True,
        tie="leave",
        active=False,
    ),
    NativeCase(name="one_finite_weak_tie_control", n=1, bad=0, kind="finite"),
    NativeCase(
        name="two_finite_strict_roles_tie_control",
        n=2,
        bad=0,
        kind="finite",
        reflected=True,
        tie="leave",
    ),
    NativeCase(
        name="four_finite_inactive_control", n=4, bad=0, kind="finite", active=False
    ),
    NativeCase(
        name="two_empty_raw_node_control",
        n=2,
        bad=0,
        kind="empty",
        reflected=True,
        tie="leave",
    ),
)


type Pair = tuple[float, float]
type Tables = tuple[tuple[Pair, Pair], ...]


def case_inputs(
    *, case: NativeCase
) -> tuple[Tables, tuple[tuple[bool, bool], ...], tuple[str, str], Pair | None]:
    tables = []
    masks = []
    for index in range(case.n):
        first = (
            (-2.0, 3.0)
            if case.placement == "scanned" and index < case.bad
            else (2.0, 2.0)
        )
        second = (-20.0, -20.0)
        feasible = (True, True)
        if index == case.bad:
            first = (
                (0.0, 0.0) if case.kind == "finite" else (float("inf"), -float("inf"))
            )
            if case.kind == "overflow":
                first = (1.0, -1.0)
            if case.kind in {"undefined_alt", "infeasible_alt"}:
                second = (5.0, 5.0)
            if case.kind == "infeasible_alt":
                feasible = (False, True)
            if case.kind == "empty":
                feasible = (False, False)
            if case.kind == "zero_first":
                # Both candidates have a defined -inf objective; member
                # infinities remain visible instead of losing to the finite alt.
                second = first
        tables.append((first, second))
        masks.append(feasible)
    weights = (
        (0.0, 1.0)
        if case.kind == "zero_first"
        else ((1.0, 0.0) if case.kind == "zero_second" else None)
    )
    roles = ("f", "m")
    if case.reflected:
        roles = ("m", "f")
        tables = [
            ((row[0][1], row[0][0]), (row[1][1], row[1][0])) for row in reversed(tables)
        ]
        masks.reverse()
        weights = None if weights is None else (weights[1], weights[0])
    return tuple(tables), tuple(masks), roles, weights


def build_model(
    *,
    case: NativeCase,
    replay: bool = False,
    enable_jit: bool = True,
    execution_config: ExecutionConfig | None = None,
) -> Model:
    tables, masks, roles, weights = case_inputs(case=case)
    utilities = {
        member: Phased(
            solve=TableUtility(values=tuple((2.0, 2.0) for _ in tables)),
            simulate=TableUtility(
                values=tuple(tuple(pair[i] for pair in row) for row in tables),
                overflow=case.kind == "overflow",
            ),
        )
        if replay
        else TableUtility(
            values=tuple(tuple(pair[i] for pair in row) for row in tables),
            overflow=case.kind == "overflow",
        )
        for i, member in enumerate(roles)
    }
    objective = (
        None
        if weights is None
        else ParetoObjective(weights=dict(zip(roles, weights, strict=True)))
    )
    collective = (
        CollectiveUtility(utilities=utilities)
        if objective is None
        else (CollectiveUtility(utilities=utilities, objective=objective))
    )
    return Model(
        regimes={
            "couple": Regime(
                states={
                    "mu": DiscreteGrid({1: OneNode, 2: TwoNodes, 4: FourNodes}[case.n])
                },
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": collective},
                constraints={"feasible_choice": Feasibility(values=tuple(masks))},
                composition_action=CompositionAction(
                    rule=PCOnWeight(state="mu", tie=case.tie),
                    available=Availability(value=case.active),
                    destinations={
                        member: Destination(regime=f"single_{member}", projection={})
                        for member in roles
                    },
                ),
            ),
            "single_f": Regime(functions={"utility": outside}),
            "single_m": Regime(functions={"utility": outside}),
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: "couple"}),
        regime_id_class=Household,
        enable_jit=enable_jit,
        execution_config=execution_config or ExecutionConfig(device_memory_bytes=None),
    )


def reference_stay_values(*, case: NativeCase) -> tuple[tuple[float, ...], ...]:
    """Enumerate every feasible extended-real scalarization before selection."""
    tables, masks, _, weights = case_inputs(case=case)
    weights = (0.5, 0.5) if weights is None else weights
    raw = []
    for row, mask in zip(tables, masks, strict=True):
        feasible = []
        for pair, admitted in zip(row, mask, strict=True):
            if not admitted:
                continue
            if case.kind == "overflow":
                assert all(math.isfinite(value) for value in pair)
                action_pair = tuple(math.copysign(math.inf, value) for value in pair)
            else:
                action_pair = pair
            objective = sum(
                0.0 if weight == 0 else weight * value
                for weight, value in zip(weights, action_pair, strict=True)
            )
            if math.isnan(objective):
                raise ValueError("undefined feasible household objective")
            feasible.append((objective, action_pair))
        raw.append(
            max(feasible, key=lambda item: item[0])[1]
            if feasible
            else (-math.inf, -math.inf)
        )
    return tuple(raw)


def expected_values(*, case: NativeCase) -> NDArray[np.float64]:
    """Walk each incumbent's scalar-reference policies against a fixed outside."""
    raw = reference_stay_values(case=case)
    result = []
    for incumbent, pair in enumerate(raw):
        if not case.active:
            result.append(pair)
            continue

        def willing(value: float) -> bool:
            return value >= 0 if case.tie == "stay" else value > 0

        direction = 1 if not willing(pair[0]) else -1
        selected = (0.0, 0.0)
        candidate = incumbent
        while 0 <= candidate < len(raw):
            candidate_pair = raw[candidate]
            if all(willing(value) for value in candidate_pair):
                selected = candidate_pair
                break
            if not any(willing(value) for value in candidate_pair) or not willing(
                candidate_pair[1 if direction == 1 else 0]
            ):
                break
            candidate += direction
        result.append(selected)
    return np.asarray(result, dtype=np.float64)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_feasible_objective_integrity_survives_composition_action_replacement(
    *, case: NativeCase
) -> None:
    """Undefined feasible candidates fail; extended-real controls retain values."""
    model = build_model(case=case)
    if case.undefined:
        with pytest.raises(ValueError, match="undefined feasible"):
            expected_values(case=case)
        with pytest.raises(ExecutionPlanningError, match="unresolved numerical status"):
            model.solve(params={}, log_level="off", runtime_checks=True)
    else:
        expected = expected_values(case=case)
        result = model.solve(params={}, log_level="off", runtime_checks=True)
        np.testing.assert_array_equal(result.values[0]["couple"], expected)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_phase_specific_undefined_objective_fails_before_routing(
    *, replay: bool, enable_jit: bool
) -> None:
    """A valid solve does not excuse an undefined simulate-phase Raw objective."""
    case = NativeCase(
        name="phase_integrity", n=2, bad=1, kind="undefined_alt", placement="scanned"
    )
    model = build_model(case=case, replay=replay, enable_jit=enable_jit)
    if not replay:
        with pytest.raises(ExecutionPlanningError, match="unresolved numerical status"):
            model.solve(params={}, log_level="off", runtime_checks=True)
        return
    solution = model.solve(params={}, log_level="off", runtime_checks=True)
    np.testing.assert_array_equal(solution.values[0]["couple"], np.full((2, 2), 2.0))
    with pytest.raises(ShockDeclarationError, match="query is invalid or unresolved"):
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(7, dtype=jnp.int32),
                "regime_id": jnp.full(
                    7, model.regime_names_to_ids["couple"], dtype=jnp.int32
                ),
                "own_member": jnp.asarray(
                    [
                        model.member_names_to_ids[member]
                        for member in ("f", "m", "f", "m", "f", "m", "f")
                    ],
                    dtype=jnp.int32,
                ),
                "mu": jnp.asarray([0, 1, 0, 1, 0, 1, 0], dtype=jnp.int32),
            },
            seed=659,
            log_level="off",
            runtime_checks=True,
        )


def check_agent_sharded_objective_integrity() -> None:
    """Padded personal gate routes preserve values and reject undefined queries."""
    config = ExecutionConfig(
        devices=(0, 1, 2, 3), simulation_sharding="agents", device_memory_bytes=None
    )
    base = _gated_outside_model(pointwise=True, execution_config=config)
    couple = base.user_regimes["couple"]
    utility = couple.functions["utility"]
    assert isinstance(utility, CollectiveUtility)
    model = Model(
        regimes={
            **base.user_regimes,
            "couple": replace(
                couple,
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            member: Phased(
                                solve=utility.utilities[member],
                                simulate=TableUtility(
                                    values=tuple(
                                        (
                                            (math.inf if member == "f" else -math.inf),
                                            2.0,
                                            2.0,
                                            2.0,
                                        )
                                        for _ in range(4)
                                    )
                                ),
                            )
                            for member in ("f", "m")
                        }
                    )
                },
            ),
        },
        edges=base.edges,
        initial_nodes=InitialNodes(by_period={0: "couple"}),
        n_periods=2,
        regime_id_class=_GatedHousehold,
        execution_config=config,
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off", runtime_checks=True)
    baseline = base.solve(params=params, log_level="off", runtime_checks=True)
    np.testing.assert_array_equal(
        solution.values[0]["couple"], baseline.values[0]["couple"]
    )
    with pytest.raises(ShockDeclarationError, match="query is invalid or unresolved"):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(7, dtype=jnp.int32),
                "regime_id": jnp.full(
                    7, model.regime_names_to_ids["couple"], dtype=jnp.int32
                ),
                "own_member": jnp.asarray(
                    [
                        model.member_names_to_ids[member]
                        for member in ("f", "m", "f", "m", "f", "m", "f")
                    ],
                    dtype=jnp.int32,
                ),
                "assets": jnp.asarray([2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0]),
                "mu": jnp.asarray([0, 1, 2, 3, 3, 0, 1], dtype=jnp.int32),
            },
            seed=659,
            log_level="off",
            runtime_checks=True,
        )
    expected = _gated_outside_simulation(
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=None)
    )
    actual = _gated_outside_simulation(execution_config=config)
    _assert_same_raw_results(actual=actual, expected=expected)


def test_undefined_objective_cannot_escape_padded_agent_replay() -> None:
    """Both personal outside routes retain numerical integrity under partitioning."""
    code = (
        "import jax; jax.config.update('jax_num_cpu_devices', 4); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "from tests.solution.test_renegotiation import "
        f"{check_agent_sharded_objective_integrity.__name__}; "
        f"{check_agent_sharded_objective_integrity.__name__}()"
    )
    pixi = which("pixi")
    assert pixi is not None, "The isolated topology check requires the Pixi runner."
    result = subprocess.run(  # noqa: S603 - fixed source on the locked project runner
        [pixi, "run", "--locked", "-e", "tests-cpu", "python", "-c", code],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
