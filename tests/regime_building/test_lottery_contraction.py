"""Lottery contraction sums a pure-index draw into the next-period value once.

The source regime `alive` carries wealth, a labor choice and five drawn states. Its
only target `final` values all of them. Each draw exercises one eligibility rule:

- `z` has weights that read only `z` itself and a parameter, and nothing else reads
  the draw, so it is contracted;
- `income` is a Rouwenhorst process whose weights read only its own node, so it is
  contracted on its node index;
- `q` has weights that also read the current `z`, a second state;
- `w` has weights that read the `labor` action;
- `y` is read by next-period wealth, so it moves the interpolation point.

The last three stay enumerated as node axes of the continuation. With `final_z`
set to `"cross"`, `final` keeps two categories of `z` against the source's three, so
the contracted table is indexed by the source's grid.

The oracle for the values is the literal one-step Bellman operator: an explicit
loop over every joint node, interpolating the terminal value linearly in wealth.
It shares nothing with the engine but the model's primitives.
"""

import inspect
import itertools
from types import MappingProxyType
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype

from _lcm.regime_building.lottery_contraction import (
    LotteryContraction,
    LotteryContractionSource,
    plan_lottery_contraction,
)
from _lcm.regime_building.Q_and_F import draw_dependencies_by_law
from _lcm.regime_building.V import create_v_interpolation_info
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    RouwenhorstAR1Process,
    StochasticTransition,
    categorical,
)
from lcm.regime import Regime
from lcm.solvers import GridSearch
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    RegimeName,
    ScalarInt,
    UserParamsNode,
)
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.processes import get_model as get_process_model


@categorical(ordered=False)
class _Z:
    z0: ScalarInt
    z1: ScalarInt
    z2: ScalarInt


@categorical(ordered=False)
class _ZFinal:
    z0: ScalarInt
    z1: ScalarInt


@categorical(ordered=False)
class _Binary:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    final: ScalarInt


type _FinalZ = Literal["same", "cross"]

_WEALTH = (0.0, 4.0, 5)
_CONSUMPTION = (0.0, 1.5, 6)
_DISCOUNT_FACTOR = 0.9


type _IncomeKind = Literal["markov", "iid"]


def _income_process(
    *, kind: _IncomeKind = "markov"
) -> RouwenhorstAR1Process | NormalIIDProcess:
    if kind == "iid":
        return NormalIIDProcess(n_points=3, mu=0.0, sigma=0.2, gauss_hermite=True)
    return RouwenhorstAR1Process(n_points=3, rho=0.7, sigma=0.2, mu=0.0)


_Z_ROWS = MappingProxyType(
    {
        "same": np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]]),
        "cross": np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]]),
    }
)
_BINARY_ROWS = np.array([[0.8, 0.2], [0.35, 0.65]])


# Checked at call time, so the contracted law must hand the weights the state's own
# integer code type, as it does inside the cell.
@beartype
def _z_probs(*, z: DiscreteState, z_rows: FloatND) -> FloatND:
    return z_rows[z]


def _q_probs_reading_z(*, q: DiscreteState, z: DiscreteState) -> FloatND:
    return jnp.asarray(_BINARY_ROWS)[(q + z) % 2]


def _w_probs_reading_labor(*, w: DiscreteState, labor: DiscreteAction) -> FloatND:
    return jnp.asarray(_BINARY_ROWS)[(w + labor) % 2]


def _y_probs(*, y: DiscreteState) -> FloatND:
    return jnp.asarray(_BINARY_ROWS)[y]


def _next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor: DiscreteAction,
    next_y: DiscreteState,
) -> ContinuousState:
    return wealth - consumption + 0.5 + 0.5 * labor + 0.25 * next_y


def _alive_utility(
    *, consumption: ContinuousAction, labor: DiscreteAction, income: ContinuousState
) -> FloatND:
    return jnp.log1p(consumption) - 0.1 * labor + 0.05 * income


def _final_utility(
    *,
    wealth: ContinuousState,
    z: DiscreteState,
    income: ContinuousState,
    q: DiscreteState,
    w: DiscreteState,
    y: DiscreteState,
) -> FloatND:
    return (
        jnp.sqrt(wealth + 2.0) * (1.0 + 0.3 * z + 0.4 * income)
        + 0.2 * q
        + 0.1 * w
        + 0.05 * y
    )


def _model(
    *, final_z: _FinalZ, solver: GridSearch, income_kind: _IncomeKind = "markov"
) -> Model:
    shared = {
        "wealth": LinSpacedGrid(start=_WEALTH[0], stop=_WEALTH[1], n_points=_WEALTH[2]),
        "income": _income_process(kind=income_kind),
        "q": DiscreteGrid(category_class=_Binary),
        "w": DiscreteGrid(category_class=_Binary),
        "y": DiscreteGrid(category_class=_Binary),
    }
    alive = Regime(
        states={**shared, "z": DiscreteGrid(category_class=_Z)},
        state_transitions={
            "wealth": _next_wealth,
            "z": {"final": StochasticTransition(func=_z_probs)},
            "q": StochasticTransition(func=_q_probs_reading_z),
            "w": StochasticTransition(func=_w_probs_reading_labor),
            "y": StochasticTransition(func=_y_probs),
        },
        actions={
            "consumption": LinSpacedGrid(
                start=_CONSUMPTION[0], stop=_CONSUMPTION[1], n_points=_CONSUMPTION[2]
            ),
            "labor": DiscreteGrid(category_class=_Binary),
        },
        functions={"utility": _alive_utility},
        solver=solver,
    )
    final = Regime(
        states={
            **shared,
            "z": DiscreteGrid(category_class=_Z if final_z == "same" else _ZFinal),
        },
        functions={"utility": _final_utility},
    )
    return Model(
        regimes={"alive": alive, "final": final},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"alive": {"final": 0}},
        initial_nodes=((0, "alive"),),
    )


def _params(*, final_z: _FinalZ) -> dict[str, UserParamsNode]:
    return {
        "alive": {
            "koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR},
            "final": {"next_z": {"z_rows": jnp.asarray(_Z_ROWS[final_z])}},
        },
        "final": {},
    }


def _planned_contraction(
    *, model: Model, source_name: RegimeName, target_name: RegimeName
) -> LotteryContraction | None:
    """Plan the contraction from one regime toward another as the solve does."""
    source = model._regimes[source_name].solution
    info = create_v_interpolation_info(user_regime=model.user_regimes[target_name])
    bundle = source.transitions[target_name]
    plan = source.transition_plans[target_name]
    read_draws = frozenset(
        draw
        for draws in draw_dependencies_by_law(
            bundle=bundle,
            functions=source.functions,
            stochastic_names=tuple(name for name in bundle if plan.is_lottery(name)),
        ).values()
        for draw in draws
    )
    return plan_lottery_contraction(
        target_regime_name=target_name,
        bundle=bundle,
        functions=source.functions,
        transition_plans=source.transition_plans,
        v_interpolation_info=info,
        received_state_names=info.state_names,
        read_draws=read_draws,
        source=LotteryContractionSource(
            state_grids=MappingProxyType(
                {name: source.grids[name] for name in source.state_names}
            ),
            action_names=frozenset(source.action_names),
        ),
    )


def _planned_lotteries(*, final_z: _FinalZ) -> dict[str, int]:
    """Return each contracted state with the length of its table axis."""
    contraction = _planned_contraction(
        model=_model(final_z=final_z, solver=GridSearch()),
        source_name="alive",
        target_name="final",
    )
    assert contraction is not None
    return {
        lottery.state_name: len(lottery.source_nodes)
        for lottery in contraction.lotteries
    }


@pytest.mark.parametrize("final_z", ["same", "cross"])
def test_plan_lottery_contraction_selects_draws_that_only_index_the_value(
    *, final_z: _FinalZ
) -> None:
    """Only `z` and the process are contracted; `q`, `w` and `y` stay node axes."""
    assert set(_planned_lotteries(final_z=final_z)) == {"z", "income"}


def test_plan_lottery_contraction_indexes_the_table_by_the_source_grid() -> None:
    """Against a two-category target, the table has one row per source category."""
    assert _planned_lotteries(final_z="cross")["z"] == 3


def _solved_alive_values(
    *, final_z: _FinalZ, contract: bool, income_kind: _IncomeKind = "markov"
) -> np.ndarray:
    model = _model(
        final_z=final_z,
        solver=GridSearch(lottery_contraction=contract),
        income_kind=income_kind,
    )
    solution = model.solve(params=_params(final_z=final_z), log_level="off")
    names = model.state_names(regime_name="alive")
    return np.transpose(
        np.asarray(solution.values[0]["alive"]),
        [names.index(name) for name in ("wealth", "z", "income", "q", "w", "y")],
    )


def _interpolate_in_wealth(*, values: np.ndarray, wealth: np.ndarray) -> np.ndarray:
    """Linearly interpolate `values` on the wealth grid, extrapolating linearly."""
    start, stop, n_points = _WEALTH
    coordinate = (wealth - start) / ((stop - start) / (n_points - 1))
    lower = np.clip(np.floor(coordinate), 0, n_points - 2).astype(int)
    upper_weight = coordinate - lower
    return (1.0 - upper_weight) * values[lower] + upper_weight * values[lower + 1]


def _oracle(*, final_z: _FinalZ, income_kind: _IncomeKind = "markov") -> np.ndarray:
    """Return `alive`'s period-0 value, indexed `(wealth, z, income, q, w, y)`."""
    process = _income_process(kind=income_kind)
    income_nodes = np.asarray(process.to_jax())
    income_rows = np.asarray(process.get_transition_probs())
    z_rows = _Z_ROWS[final_z]
    n_z_next = z_rows.shape[1]
    wealth_grid = np.linspace(*_WEALTH)
    consumption_grid = np.linspace(*_CONSUMPTION)
    result = np.empty((_WEALTH[2], 3, 3, 2, 2, 2))
    for z, income, q, w, y in itertools.product(
        range(3), range(3), range(2), range(2), range(2)
    ):
        best = np.full(_WEALTH[2], -np.inf)
        for consumption, labor in itertools.product(consumption_grid, range(2)):
            expectation = np.zeros(_WEALTH[2])
            for z_next, income_next, q_next, w_next, y_next in itertools.product(
                range(n_z_next), range(3), range(2), range(2), range(2)
            ):
                probability = (
                    z_rows[z, z_next]
                    * income_rows[income, income_next]
                    * _BINARY_ROWS[(q + z) % 2, q_next]
                    * _BINARY_ROWS[(w + labor) % 2, w_next]
                    * _BINARY_ROWS[y, y_next]
                )
                final_values = np.sqrt(wealth_grid + 2.0) * (
                    1.0 + 0.3 * z_next + 0.4 * income_nodes[income_next]
                ) + (0.2 * q_next + 0.1 * w_next + 0.05 * y_next)
                next_wealth = (
                    wealth_grid - consumption + 0.5 + 0.5 * labor + 0.25 * y_next
                )
                expectation += probability * _interpolate_in_wealth(
                    values=final_values, wealth=next_wealth
                )
            action_value = (
                np.log1p(consumption)
                - 0.1 * labor
                + 0.05 * income_nodes[income]
                + _DISCOUNT_FACTOR * expectation
            )
            best = np.maximum(best, action_value)
        result[:, z, income, q, w, y] = best
    return result


@pytest.mark.parametrize("income_kind", ["markov", "iid"])
@pytest.mark.parametrize("final_z", ["same", "cross"])
@pytest.mark.parametrize("contract", [True, False])
def test_solve_matches_the_full_node_product_oracle(
    *, final_z: _FinalZ, contract: bool, income_kind: _IncomeKind
) -> None:
    """With and without contraction, the value is the literal joint expectation."""
    np.testing.assert_allclose(
        _solved_alive_values(
            final_z=final_z, contract=contract, income_kind=income_kind
        ),
        _oracle(final_z=final_z, income_kind=income_kind),
        rtol=10.0**-DECIMAL_PRECISION,
    )


def test_contracted_iid_draw_does_not_read_the_current_state() -> None:
    """An IID draw's table rows coincide, so its law reads no current state."""
    model = _model(final_z="same", solver=GridSearch(), income_kind="iid")
    contraction = _planned_contraction(
        model=model, source_name="alive", target_name="final"
    )
    assert contraction is not None
    bundle = contraction.rewrite_bundle(
        bundle=model._regimes["alive"].solution.transitions["final"]
    )
    assert tuple(inspect.signature(bundle["next_income"]).parameters) == ()


@pytest.mark.parametrize("distribution_type", ["tauchen", "rouwenhorst", "normal"])
def test_plan_lottery_contraction_keeps_a_process_with_runtime_nodes(
    *, distribution_type: Literal["tauchen", "rouwenhorst", "normal"]
) -> None:
    """A process whose nodes come from runtime parameters keeps its node axis."""
    contraction = _planned_contraction(
        model=get_process_model(n_periods=3, distribution_type=distribution_type),
        source_name="alive",
        target_name="alive",
    )
    contracted = (
        set()
        if contraction is None
        else {lottery.state_name for lottery in contraction.lotteries}
    )
    assert "income" not in contracted
