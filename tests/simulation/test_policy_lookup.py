"""The public per-period policy lookup on a solved model."""

import json
import logging
from collections.abc import Callable, Mapping
from functools import partial
from math import ceil, log, prod
from pathlib import Path
from typing import NotRequired, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
import lcm.solvers
from _lcm.execution.execution_plan import visible_device_pool_limits
from _lcm.typing import JSONValue
from lcm import AgeGrid, DiscreteGrid, Model, PolicyLookup, Regime, categorical
from lcm.exceptions import ExecutionPlanningError, InvalidSimulationInputError
from lcm.persistence import load_solution, save_solution
from lcm.solver_api import SolutionResult, SolutionSource, ValueStore
from lcm.typing import (
    ActionName,
    DiscreteAction,
    FloatND,
    Intermediate,
    RegimeName,
    ScalarInt,
    StateName,
    UserParams,
)
from lcm_examples.mortality import LaborSupply
from tests.test_models.deterministic.discrete import get_model as get_discrete_model
from tests.test_models.deterministic.discrete import (
    get_params as get_discrete_params,
)
from tests.test_models.deterministic.regression import (
    DEFAULT_CONSUMPTION_GRID,
    RegimeId,
    dead,
    get_model,
    get_params,
)
from tests.test_models.nbegm_multi_discrete_toy import build_model as build_shock_model
from tests.test_models.nbegm_multi_discrete_toy import (
    build_params as build_shock_params,
)

N_PERIODS = 5
LAST_ALIVE_PERIOD = N_PERIODS - 2
WEALTH = jnp.array([5.0, 20.0, 40.0, 70.0])

type Solved = tuple[Model, UserParams, SolutionResult]


class _LookupOptions(TypedDict, total=False):
    action_grids: Mapping[ActionName, jax.Array] | None
    keep_intermediates: tuple[Intermediate, ...]


class _LookupArguments(_LookupOptions):
    period: int
    regime_name: NotRequired[RegimeName]
    states: Mapping[StateName, jax.Array]


@pytest.fixture(scope="module")
def solved() -> Solved:
    model = get_model(n_periods=N_PERIODS)
    params = get_params(n_periods=N_PERIODS)
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


def _lookup(solved: Solved, **kwargs: Unpack[_LookupArguments]) -> PolicyLookup:
    model, params, solution = solved
    kwargs.setdefault("regime_name", "working_life")
    return model.lookup_policy(
        params=params,
        solution=solution,
        **kwargs,
    )


def test_lookup_policy_returns_a_policy_lookup(solved: Solved) -> None:
    got = _lookup(solved, period=0, states={"wealth": WEALTH})
    assert isinstance(got, PolicyLookup)


def test_lookup_policy_last_alive_period_consumes_largest_feasible_grid_point(
    solved: Solved,
) -> None:
    """With a zero continuation, retiring and eating all feasible wealth is best."""
    got = _lookup(solved, period=LAST_ALIVE_PERIOD, states={"wealth": WEALTH})
    assert got.actions is not None
    grid = np.asarray(DEFAULT_CONSUMPTION_GRID.to_jax())
    expected_c = np.array([grid[grid <= w].max() for w in np.asarray(WEALTH)])
    np.testing.assert_allclose(got.actions["consumption"], expected_c, rtol=1e-6)


def test_lookup_policy_last_alive_period_value_is_log_consumption(
    solved: Solved,
) -> None:
    got = _lookup(solved, period=LAST_ALIVE_PERIOD, states={"wealth": WEALTH})
    grid = np.asarray(DEFAULT_CONSUMPTION_GRID.to_jax())
    expected_v = np.log([grid[grid <= w].max() for w in np.asarray(WEALTH)])
    np.testing.assert_allclose(got.value, expected_v, rtol=1e-5)


def test_lookup_policy_last_alive_period_retires(solved: Solved) -> None:
    got = _lookup(solved, period=LAST_ALIVE_PERIOD, states={"wealth": WEALTH})
    assert got.actions is not None
    np.testing.assert_array_equal(got.actions["labor_supply"], LaborSupply.retire)


@pytest.mark.parametrize("column", ["consumption", "labor_supply", "value"])
def test_lookup_policy_equals_what_simulate_records(
    *, solved: Solved, column: str
) -> None:
    """At every simulated (period, subject), the lookup reproduces simulate's row."""
    model, params, solution = solved
    df = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": WEALTH,
            "age": jnp.full(WEALTH.shape, 18.0),
            "regime_id": jnp.full(WEALTH.shape, RegimeId.working_life),
        },
        log_level="off",
        seed=0,
    ).to_dataframe(use_categorical_codes=True)
    alive = df.query("regime_name == 'working_life'")
    got, expected = [], []
    for period, rows in alive.groupby("period"):
        lookup = _lookup(
            solved,
            period=int(cast("int", period)),
            states={"wealth": jnp.asarray(rows["wealth"].to_numpy())},
        )
        assert lookup.actions is not None
        values = lookup.value if column == "value" else lookup.actions[column]
        got.append(np.asarray(values, dtype=float))
        expected.append(rows[column].to_numpy(dtype=float))
    np.testing.assert_allclose(np.concatenate(got), np.concatenate(expected))


@pytest.mark.parametrize("branch", [LaborSupply.work, LaborSupply.retire])
def test_lookup_policy_restricted_action_grid_fixes_the_branch(
    *, solved: Solved, branch: ScalarInt
) -> None:
    got = _lookup(
        solved,
        period=0,
        states={"wealth": WEALTH},
        action_grids={"labor_supply": jnp.array([branch], dtype=jnp.int32)},
    )
    assert got.actions is not None
    np.testing.assert_array_equal(got.actions["labor_supply"], branch)


def test_lookup_policy_max_over_branches_equals_unconditional_value(
    solved: Solved,
) -> None:
    branch_values = [
        _lookup(
            solved,
            period=0,
            states={"wealth": WEALTH},
            action_grids={"labor_supply": jnp.array([code], dtype=jnp.int32)},
        ).value
        for code in (LaborSupply.work, LaborSupply.retire)
    ]
    unconditional = _lookup(solved, period=0, states={"wealth": WEALTH}).value
    np.testing.assert_allclose(
        np.maximum(branch_values[0], branch_values[1]), unconditional
    )


@pytest.mark.parametrize("period", [1, N_PERIODS - 1])
def test_lookup_policy_without_states_or_actions_returns_one_row_of_value(
    *, solved: Solved, period: int
) -> None:
    """`dead` has neither states nor actions: its empty query is one row of zero."""
    got = _lookup(solved, regime_name="dead", period=period, states={})
    assert (got.value.shape, np.asarray(got.value).tolist()) == ((1,), [0.0])


@categorical(ordered=False)
class StatelessRegimeId:
    choose: ScalarInt
    dead: ScalarInt


def utility_of_labor_supply(labor_supply: DiscreteAction) -> FloatND:
    return jnp.where(labor_supply == LaborSupply.work, 1.0, 0.25)


@pytest.fixture(scope="module")
def solved_stateless() -> Solved:
    """`choose` has one action and no states; it is followed by the terminal `dead`."""
    model = Model(
        regimes={
            "choose": Regime(
                actions={"labor_supply": DiscreteGrid(category_class=LaborSupply)},
                functions={"utility": utility_of_labor_supply},
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=StatelessRegimeId,
        initial_nodes={0: "choose"},
        edges={"choose": {"dead": 0}},
    )
    params = {"discount_factor": 0.5}
    return model, params, model.solve(params=params, log_level="off")


def _lookup_stateless(
    solved_stateless: Solved, **kwargs: Unpack[_LookupOptions]
) -> PolicyLookup:
    model, params, solution = solved_stateless
    return model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="choose",
        states={},
        **kwargs,
    )


def test_lookup_policy_without_states_returns_one_row_of_actions_and_value(
    solved_stateless: Solved,
) -> None:
    """Working (utility 1) beats retiring (1/4); `dead` adds a zero continuation."""
    got = _lookup_stateless(solved_stateless)
    assert got.actions is not None
    assert (
        np.asarray(got.actions["labor_supply"]).tolist(),
        np.asarray(got.value).tolist(),
    ) == ([LaborSupply.work], [1.0])


def test_lookup_policy_without_states_returns_one_row_of_action_values(
    solved_stateless: Solved,
) -> None:
    """`Q` holds each labor supply's utility in grid order, all feasible, in one row."""
    got = _lookup_stateless(solved_stateless, keep_intermediates=("Q", "F"))
    assert (np.asarray(got.Q).tolist(), np.asarray(got.F).tolist()) == (
        [[1.0, 0.25]],
        [[True, True]],
    )


def test_lookup_policy_rejects_an_unknown_regime(solved: Solved) -> None:
    with pytest.raises(InvalidSimulationInputError, match="retired"):
        _lookup(solved, regime_name="retired", period=0, states={"wealth": WEALTH})


def test_lookup_policy_rejects_a_period_the_regime_is_not_active_in(
    solved: Solved,
) -> None:
    with pytest.raises(InvalidSimulationInputError, match="period 4"):
        _lookup(solved, period=N_PERIODS - 1, states={"wealth": WEALTH})


def test_lookup_policy_rejects_missing_state(solved: Solved) -> None:
    with pytest.raises(InvalidSimulationInputError, match="wealth"):
        _lookup(solved, period=0, states={})


def test_lookup_policy_rejects_a_discrete_state_code_off_the_grid() -> None:
    model = get_discrete_model(n_periods=4)
    params = get_discrete_params(n_periods=4)
    solution = model.solve(params=params, log_level="off")
    with pytest.raises(InvalidSimulationInputError, match="wealth"):
        model.lookup_policy(
            params=params,
            solution=solution,
            period=0,
            regime_name="working_life",
            states={"wealth": jnp.array([0, 7], dtype=jnp.int32)},
        )


def test_lookup_policy_rejects_fractional_discrete_state_codes() -> None:
    """Codes between the categories `0`, `1`, `2` are refused, not truncated."""
    model = get_discrete_model(n_periods=4)
    params = get_discrete_params(n_periods=4)
    solution = model.solve(params=params, log_level="off")
    with pytest.raises(ValueError, match="wealth"):
        model.lookup_policy(
            params=params,
            solution=solution,
            period=0,
            regime_name="working_life",
            states={"wealth": jnp.array([0.5, 1.5])},
        )


def test_lookup_policy_takes_discrete_state_codes_of_any_integer_dtype() -> None:
    """Codes built as the default integer dtype give the rows of `int32` codes."""
    model = get_discrete_model(n_periods=4)
    params = get_discrete_params(n_periods=4)
    solution = model.solve(params=params, log_level="off")

    def rows(codes: jax.Array) -> dict[str, list[int | float]]:
        got = model.lookup_policy(
            params=params,
            solution=solution,
            period=0,
            regime_name="working_life",
            states={"wealth": codes},
        )
        assert got.actions is not None
        return {
            "value": np.asarray(got.value).tolist(),
            **{name: np.asarray(a).tolist() for name, a in got.actions.items()},
        }

    assert rows(jnp.asarray([0, 1])) == rows(jnp.asarray([0, 1], dtype=jnp.int32))


def test_lookup_policy_rejects_an_unknown_action_grid(solved: Solved) -> None:
    with pytest.raises(InvalidSimulationInputError, match="leisure"):
        _lookup(
            solved,
            period=0,
            states={"wealth": WEALTH},
            action_grids={"leisure": jnp.array([0])},
        )


def test_lookup_policy_rejects_params_the_solution_was_not_solved_with(
    solved: Solved,
) -> None:
    model, _, solution = solved
    with pytest.raises(InvalidSimulationInputError, match="params_fingerprint"):
        model.lookup_policy(
            params=get_params(n_periods=N_PERIODS, discount_factor=0.5),
            solution=solution,
            period=0,
            regime_name="working_life",
            states={"wealth": WEALTH},
        )


def test_lookup_policy_above_range_continuous_state_equals_simulate(
    solved: Solved,
) -> None:
    """A continuous state above its grid is extrapolated as simulate does."""
    model, params, solution = solved
    wealth = jnp.array([450.0, 600.0])
    df = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": wealth,
            "age": jnp.full(wealth.shape, 18.0),
            "regime_id": jnp.full(wealth.shape, RegimeId.working_life),
        },
        log_level="off",
        seed=0,
    ).to_dataframe(use_categorical_codes=True)
    first = df.query("period == 0")
    got = _lookup(solved, period=0, states={"wealth": wealth})
    assert got.actions is not None
    np.testing.assert_allclose(
        np.column_stack([got.actions["consumption"], got.value]),
        first[["consumption", "value"]].to_numpy(dtype=float),
    )


@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_state_grid_scales_shock_nodes_with_runtime_sigma(sigma: float) -> None:
    """Gauss-Hermite income nodes are the unit-sigma nodes times the runtime sigma."""
    model = build_shock_model()
    params = build_shock_params()
    unit = model.state_grid(params=params, regime_name="alive", state_name="income")
    alive = params["alive"]
    assert isinstance(alive, dict)
    alive["income"] = {"mu": 0.0, "sigma": sigma}
    got = model.state_grid(params=params, regime_name="alive", state_name="income")
    np.testing.assert_allclose(got, sigma * np.asarray(unit), rtol=1e-6)


def test_state_grid_rejects_an_unknown_state(solved: Solved) -> None:
    model, params, _ = solved
    with pytest.raises(InvalidSimulationInputError, match="income"):
        model.state_grid(params=params, regime_name="working_life", state_name="income")


def test_solution_result_is_public_in_lcm_solvers(solved: Solved) -> None:
    _, _, solution = solved
    assert isinstance(solution, lcm.solvers.SolutionResult)


def test_state_names_is_the_value_function_axis_order(solved: Solved) -> None:
    model, _, _ = solved
    assert model.state_names(regime_name="working_life") == ("wealth",)


def test_value_array_indexed_in_state_names_order_matches_lookup(
    solved: Solved,
) -> None:
    """V at grid node k of each state, in `state_names` order, is the lookup's V."""
    model, params, solution = solved
    nodes = {
        name: model.state_grid(
            params=params, regime_name="working_life", state_name=name
        )
        for name in model.state_names(regime_name="working_life")
    }
    index = (3,)
    V = solution.values[0]["working_life"]
    got = _lookup(
        solved,
        period=0,
        states={
            name: nodes[name][i : i + 1] for name, i in zip(nodes, index, strict=True)
        },
    )
    np.testing.assert_allclose(got.value, V[index], rtol=1e-6)


@pytest.fixture(scope="module")
def solved_two_states() -> Solved:
    model = build_shock_model()
    params = build_shock_params()
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


def _nodes_in_state_names_order(
    *, model: Model, params: UserParams
) -> dict[StateName, jax.Array]:
    return {
        name: model.state_grid(params=params, regime_name="alive", state_name=name)
        for name in model.state_names(regime_name="alive")
    }


def test_value_array_shape_follows_state_names_order(solved_two_states: Solved) -> None:
    """Two states on grids of different sizes: V's shape is their sizes in order."""
    model, params, solution = solved_two_states
    nodes = _nodes_in_state_names_order(model=model, params=params)
    shape = tuple(len(grid) for grid in nodes.values())
    assert solution.values[0]["alive"].shape == shape != shape[::-1]


def test_value_array_indexed_in_state_names_order_matches_lookup_two_states(
    solved_two_states: Solved,
) -> None:
    model, params, solution = solved_two_states
    nodes = _nodes_in_state_names_order(model=model, params=params)
    index = (2, 1)
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={
            name: nodes[name][i : i + 1] for name, i in zip(nodes, index, strict=True)
        },
    )
    V = solution.values[0]["alive"]
    np.testing.assert_allclose(got.value, V[index], rtol=1e-5)
    assert not np.isclose(V[index], V[index[::-1]], rtol=1e-5)


def test_lookup_policy_under_a_device_budget_has_analytic_last_decision_policy() -> (
    None
):
    """A budgeted model looks up the analytic final-decision policy and value."""
    grid = lcm.LinSpacedGrid(start=1, stop=3, n_points=3)
    model = get_model(
        n_periods=3,
        wealth_grid=grid,
        consumption_grid=grid,
        execution_config=lcm.ExecutionConfig(device_memory_bytes=2**30),
    )
    params = get_params(n_periods=3)
    solution = model.solve(params=params, log_level="off")
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=1,
        regime_name="working_life",
        states={"wealth": jnp.array([2.0])},
    )
    assert got.actions is not None
    # All successors are dead with utility zero; c <= wealth leaves c in {1, 2}
    # and work subtracts 0.5, so the optimum is c = 2, retire, value log(2).
    np.testing.assert_allclose(
        [got.actions["consumption"][0], got.actions["labor_supply"][0], got.value[0]],
        [2.0, 1.0, log(2)],
        rtol=1e-5,
    )


@pytest.mark.requires(device="gpu")
@pytest.mark.coverage(backends=("gpu-small",), precisions="both")
def test_restored_budgeted_lookup_has_analytic_last_decision_policy(
    *,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    record_property: Callable[[str, JSONValue], None],
) -> None:
    """Restored public lookup owns its live inputs under the real device budget."""
    assert jax.default_backend() == "gpu"
    dtype = np.dtype("float64" if jax.config.x64_enabled else "float32")
    execution = lcm.ExecutionConfig(
        axis_widths={"subject": 2000},
        device_memory_bytes="device",
        device_memory_headroom_fraction=0.15,
    )
    build = partial(
        get_model,
        n_periods=3,
        wealth_grid=lcm.LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=lcm.LinSpacedGrid(start=1, stop=3, n_points=3),
        execution_config=execution,
    )
    with caplog.at_level(logging.INFO):
        producer = build()
        consumer = build()

    devices = {device.id: device for device in jax.devices()}
    assert producer.execution_devices == consumer.execution_devices
    limits = {}
    for device_id in consumer.execution_devices:
        limit = visible_device_pool_limits(devices=(devices[device_id],))[device_id]
        assert limit is not None
        limits[device_id] = limit
    effective = min(limit - ceil(0.15 * limit) for limit in limits.values())
    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Device-memory budget:")
    ]
    assert len(summaries) == 2
    assert all(
        "derived from the device pool limit (default)" in message
        and "headroom fraction 0.15;" in message
        and message.endswith(f"effective {effective} bytes.")
        for message in summaries
    )
    record_property(
        "lookup_lane",
        json.dumps(
            {
                "backend": jax.default_backend(),
                "dtype": dtype.name,
                "device_pool_limits": limits,
                "effective_budget_bytes": effective,
                "requested_subject_width": 2000,
            },
            sort_keys=True,
        ),
    )

    params = get_params(n_periods=3)
    solved = producer.solve(params=params, log_level="off")
    path = save_solution(solution=solved, path=tmp_path / "tiny.solution")
    restored = load_solution(path=path, verify_checksums=True)
    before = restored.metadata
    assert before.source is SolutionSource.PERSISTED
    assert before.model_fingerprint == solved.metadata.model_fingerprint
    assert before.params_fingerprint == solved.metadata.params_fingerprint
    schema = before.value_schemas[1, "working_life"]
    assert (schema.shape, schema.axis_names, schema.dtype) == (
        (3,),
        ("wealth",),
        dtype.name,
    )
    record_property(
        "lookup_metadata_before",
        json.dumps(
            {
                "model_fingerprint": before.model_fingerprint,
                "params_fingerprint": before.params_fingerprint,
                "source": before.source.value,
                "dtype": schema.dtype,
            },
            sort_keys=True,
        ),
    )
    states = {"wealth": jnp.array([2.0], dtype=dtype)}
    try:
        with pytest.raises(InvalidSimulationInputError, match="params_fingerprint"):
            consumer.lookup_policy(
                params=get_params(n_periods=3, discount_factor=0.5),
                solution=restored,
                period=1,
                regime_name="working_life",
                states=states,
            )
        result = consumer.lookup_policy(
            params=params,
            solution=restored,
            period=1,
            regime_name="working_life",
            states=states,
        )
        assert result.actions is not None
        jax.block_until_ready((result.value, dict(result.actions)))
        # At age 19 all successors are dead with utility zero. The constraint
        # c <= wealth leaves c in {1, 2}; work subtracts 0.5. Thus these literals
        # follow from the economic functions, not from archived solution arrays.
        np.testing.assert_array_equal(result.actions["consumption"], [2.0])
        np.testing.assert_array_equal(result.actions["labor_supply"], [1])
        np.testing.assert_allclose(
            result.value, [log(2)], rtol=1e-12 if dtype.itemsize == 8 else 1e-5
        )
        assert result.value.dtype == dtype
        assert result.actions["consumption"].dtype == dtype
        assert result.actions["labor_supply"].dtype == np.dtype("int32")
    finally:
        after = restored.metadata
        record_property(
            "lookup_metadata_after",
            json.dumps(
                {
                    "model_fingerprint": after.model_fingerprint,
                    "params_fingerprint": after.params_fingerprint,
                    "source": after.source.value,
                    "dtype": after.value_schemas[1, "working_life"].dtype,
                },
                sort_keys=True,
            ),
        )
        record_property(
            "lookup_device_stats_after",
            json.dumps(
                {
                    device_id: devices[device_id].memory_stats()
                    for device_id in consumer.execution_devices
                },
                sort_keys=True,
            ),
        )
        assert after == before


_OWNER_PERIODS = 16
_OWNER_POINTS = 32768


def _restored_owner_fixture(
    *, tmp_path: Path, archive_state: StateName
) -> tuple[SolutionResult, UserParams, Callable[[int | None], Model], int]:
    """Solve unbudgeted, save, release the producer and reload the archive.

    Returns the restored result, its parameters, a budgeted-consumer factory and
    the total value payload `S` the restored result advertises.
    """
    build = partial(
        get_model,
        n_periods=_OWNER_PERIODS,
        wealth_grid=lcm.LinSpacedGrid(start=1, stop=3, n_points=_OWNER_POINTS),
        consumption_grid=lcm.LinSpacedGrid(start=1, stop=3, n_points=3),
    )
    params = get_params(n_periods=_OWNER_PERIODS)
    solved = build().solve(params=params, log_level="off")
    payload = sum(
        prod(schema.shape) * np.dtype(schema.dtype).itemsize
        for schema in solved.metadata.value_schemas.values()
    )
    path = save_solution(solution=solved, path=tmp_path / "values.solution")
    del solved
    restored = load_solution(path=path, verify_checksums=True)
    if archive_state == "warm":
        assert isinstance(restored.values, ValueStore)
        jax.block_until_ready(restored.values.materialize())

    def build_budgeted(budget: int | None) -> Model:
        return build(
            execution_config=lcm.ExecutionConfig(
                device_memory_bytes=budget, device_memory_headroom_fraction=0.0
            )
        )

    return restored, params, build_budgeted, payload


def _final_decision_lookup(
    *, model: Model, params: UserParams, restored: SolutionResult
) -> PolicyLookup:
    result = model.lookup_policy(
        params=params,
        solution=restored,
        period=_OWNER_PERIODS - 2,
        regime_name="working_life",
        states={"wealth": jnp.array([2.0])},
    )
    assert result.actions is not None
    jax.block_until_ready((result.value, dict(result.actions)))
    return result


@pytest.mark.parametrize("archive_state", ["cold", "warm"])
def test_restored_budgeted_lookup_refuses_when_cache_and_view_exceed_budget(
    *, tmp_path: Path, archive_state: StateName
) -> None:
    """A budget below archive cache plus resolved view refuses the lookup.

    The restored result keeps its archive cache (`S`) while lookup resolves a
    detached value view (`S`). Under a `3S/2` budget both owners cannot coexist,
    so the public call must refuse rather than return.
    """
    restored, params, build_budgeted, payload = _restored_owner_fixture(
        tmp_path=tmp_path, archive_state=archive_state
    )
    consumer = build_budgeted(3 * payload // 2)
    with pytest.raises(ExecutionPlanningError):
        _final_decision_lookup(model=consumer, params=params, restored=restored)


def test_restored_budgeted_lookup_refuses_after_other_consumers_add_views(
    tmp_path: Path,
) -> None:
    """Views that other models leave on the result count against a later lookup.

    Let `S` be the restored value payload, at least fifteen wealth-grid-sized
    arrays (one per working-life decision period). At its peak the cold first
    call holds the archive cache, one transient copy and the resolved view of
    every value array (at most `3S`) plus the model's own grids and the lookup
    workspace, which together stay below one further `S`. So a `4S` budget
    admits it, and it returns the analytic policy. Afterwards the result keeps
    the cache and that view (`2S`). Three unbudgeted consumers of the same
    result each retain one more resolved view, so the repeat call by
    the budgeted consumer starts with at least `5S > 4S` held and must refuse.
    """
    restored, params, build_budgeted, payload = _restored_owner_fixture(
        tmp_path=tmp_path, archive_state="cold"
    )
    consumer = build_budgeted(4 * payload)
    first = _final_decision_lookup(model=consumer, params=params, restored=restored)
    assert first.actions is not None
    np.testing.assert_allclose(
        [
            first.actions["consumption"][0],
            first.actions["labor_supply"][0],
            first.value[0],
        ],
        [2.0, 1.0, log(2)],
        rtol=1e-5,
    )
    for _ in range(3):
        _final_decision_lookup(
            model=build_budgeted(None), params=params, restored=restored
        )
    with pytest.raises(ExecutionPlanningError):
        _final_decision_lookup(model=consumer, params=params, restored=restored)


@pytest.mark.parametrize("archive_state", ["cold", "warm"])
def test_restored_lookup_under_a_generous_budget_has_analytic_policy_on_repeat(
    *, tmp_path: Path, archive_state: StateName
) -> None:
    """A budget far above every retained owner returns the analytic policy twice.

    All successors of the final decision are dead with utility zero; c <= wealth
    leaves c in {1, 2} and work subtracts 0.5, so c = 2, retire, value log(2).
    """
    restored, params, build_budgeted, payload = _restored_owner_fixture(
        tmp_path=tmp_path, archive_state=archive_state
    )
    consumer = build_budgeted(32 * payload)
    got = [
        _final_decision_lookup(model=consumer, params=params, restored=restored)
        for _ in range(2)
    ]
    rows = []
    for r in got:
        assert r.actions is not None
        rows.append(
            [r.actions["consumption"][0], r.actions["labor_supply"][0], r.value[0]]
        )
    np.testing.assert_allclose(rows, [[2.0, 1.0, log(2)]] * 2, rtol=1e-5)
