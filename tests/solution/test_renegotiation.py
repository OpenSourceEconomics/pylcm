"""Discrete renegotiation follows a literal directional walk of stay policies."""

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import which
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

import lcm
import tests.conftest
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Periods,
    Regime,
    SimulationResult,
    Unilateral,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError, UnsupportedOperationError
from lcm.typing import FloatND, IntND, ScalarInt, StateName


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
    renegotiate = getattr(lcm, "Renegotiate", None)
    assert renegotiate is not None, "The public exit rule must declare Renegotiate."
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
                exit=Exit(
                    rule=Unilateral(tie=tie, renegotiate=renegotiate(state="mu")),
                    targets={
                        member: ExitTarget(regime=f"single_{member}", projection={})
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
            "own_stakeholder": jnp.full(
                4, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
            "mu": jnp.arange(4, dtype=jnp.int32),
        },
        seed=659,
        log_level="off",
    )
    frame = result.to_dataframe(use_categorical_codes=True).sort_values(
        ["period", "subject_id"]
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
    declaration = couple.exit
    assert isinstance(declaration, Exit)
    changed = replace(
        declaration, rule=Unilateral(renegotiate=lcm.Renegotiate(state=state))
    )
    with pytest.raises(RegimeInitializationError, match="declared discrete grid state"):
        Model(
            regimes={**base.user_regimes, "couple": replace(couple, exit=changed)},
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
            "own_stakeholder": jnp.asarray(
                [
                    model.stakeholder_names_to_ids[member]
                    for member in ("f", "m", "f", "m", "f", "m", "f")
                ],
                dtype=jnp.int32,
            ),
            "mu": jnp.asarray([0, 1, 2, 3, 0, 2, 3], dtype=jnp.int32),
        },
        seed=659,
        log_level="off",
    )


def check_subject_sharded_renegotiation() -> None:
    """Check independent branch anchors and every native output under partitioning."""
    import jax  # noqa: PLC0415

    expected = _separation_simulation(
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=None)
    )
    frame = expected.to_dataframe(use_categorical_codes=True).sort_values("subject_id")
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
            simulation_sharding="subjects",
            device_memory_bytes=None,
        )
    )
    assert actual.n_subjects == expected.n_subjects == 7
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


def test_renegotiation_subject_sharding_preserves_state_actions_and_separation() -> (
    None
):
    """The selected state stays subject-local, including padding and split exits."""
    code = (
        "import jax; jax.config.update('jax_num_cpu_devices', 4); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "from tests.solution.test_renegotiation import "
        f"{check_subject_sharded_renegotiation.__name__}; "
        f"{check_subject_sharded_renegotiation.__name__}()"
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
    with pytest.raises(UnsupportedOperationError, match="Renegotiating exits"):
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
    declaration = couple.exit
    assert isinstance(declaration, Exit)
    ordinary = Model(
        regimes={
            **adjusted.user_regimes,
            "couple": replace(couple, exit=replace(declaration, rule=Unilateral())),
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
    return jnp.where(choice == 0, -1.0, 2.0)


def _male_tradeoff(*, choice: IntND) -> FloatND:
    return jnp.where(choice == 0, 3.0, 0.0)


def _state_weight(*, mu: IntND) -> FloatND:
    return jnp.asarray(mu + 1, dtype=float)


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
                            weights={"f": _state_weight, "m": 1.0}
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
    # At weights (1,1), both actions tie and Raw chooses index 0, worth (-1,3).
    # The next node's weights (2,1) choose index 1, worth (2,0); both then accept.
    result = model.simulate(
        params={},
        solution=model.solve(params={}, log_level="off"),
        initial_conditions={
            "period": jnp.asarray([0], dtype=jnp.int32),
            "regime_id": jnp.asarray(
                [model.regime_names_to_ids["couple"]], dtype=jnp.int32
            ),
            "own_stakeholder": jnp.asarray(
                [model.stakeholder_names_to_ids["f"]], dtype=jnp.int32
            ),
            "mu": jnp.asarray([0], dtype=jnp.int32),
        },
        seed=659,
        log_level="off",
    )
    frame = result.to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(
        frame[["mu", "choice", "value_f", "value_m"]], [[1, 1, 2, 0]]
    )
