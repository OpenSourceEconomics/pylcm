"""An unconditioned IID process enters the continuation without its current value.

Every row of an IID transition matrix is the same distribution, so the expected
continuation `E[V'(x', shock')]` does not depend on the current `shock`. The
continuation must then not read `shock`, so GridSearch computes it once per remaining
cell and broadcasts it along `shock` rather than batching it along that axis.

A process whose rows differ, a Tauchen AR(1), keeps the read: its next-period law
does depend on the current value.

The values are checked against the same model written with the shock as an explicit
Markov state whose transition returns the process's row. That route draws on no
process weight or next-index function at all: the row is a literal array the
transition returns, and utility reads the node through a code lookup.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution import grid_search
from _lcm.typing import QAndFFunction
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Regime,
    StochasticTransition,
    TauchenAR1Process,
    categorical,
)
from lcm.solvers import CELL_AXIS
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    StateName,
)

_N_X = 5
_N_AGES = 4
_DENSE = "dense"
_STREAMED = "streamed"
_SINGLETON = "singleton"
_COLLECTIVE = "collective"


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    terminal: ScalarInt


@categorical(ordered=True)
class Work:
    rest: ScalarInt
    work: ScalarInt


@categorical(ordered=True)
class OneCode:
    c0: ScalarInt


@categorical(ordered=True)
class TwoCodes:
    c0: ScalarInt
    c1: ScalarInt


@categorical(ordered=True)
class ThreeCodes:
    c0: ScalarInt
    c1: ScalarInt
    c2: ScalarInt


_CODES = {1: OneCode, 2: TwoCodes, 3: ThreeCodes}


def _process(*, family: str, n_nodes: int) -> NormalIIDProcess | TauchenAR1Process:
    """Return the shock process of one family with `n_nodes` nodes."""
    if family == "iid":
        return NormalIIDProcess(n_points=n_nodes, gauss_hermite=True, mu=0.0, sigma=0.5)
    return TauchenAR1Process(
        n_points=n_nodes, gauss_hermite=False, rho=0.8, sigma=0.5, mu=0.0, n_std=2.0
    )


def _next_x(*, x: ContinuousState, work: DiscreteAction) -> ContinuousState:
    """Wealth moves one grid step up when working and one down otherwise."""
    return jnp.clip(x + 2.0 * work - 1.0, 0.0, _N_X - 1.0)


def _utility(
    *, x: ContinuousState, shock_value: FloatND, work: DiscreteAction
) -> FloatND:
    return jnp.log1p(x) + (shock_value - 0.2) * work


def _utility_m(*, x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.sqrt(x) - 0.1 * work


def _terminal_utility(x: ContinuousState) -> FloatND:
    return jnp.log1p(x)


def _model(
    *, kind: str, family: str, n_nodes: int, dispatch: str, as_markov: bool
) -> Model:
    """Build the model, its shock a process or the equivalent Markov state."""
    process = _process(family=family, n_nodes=n_nodes)
    if as_markov:
        nodes = process.get_gridpoints()
        rows = process.get_transition_probs()
        shock = DiscreteGrid(category_class=_CODES[n_nodes])

        def next_shock(shock: DiscreteState) -> FloatND:
            return rows[shock]

        transitions = {"shock": StochasticTransition(func=next_shock)}

        def shock_value(shock: DiscreteState) -> FloatND:
            return nodes[shock]

    else:
        shock = process
        transitions = {}

        def shock_value(shock: ContinuousState) -> FloatND:
            return shock

    utility = (
        _utility
        if kind == _SINGLETON
        else CollectiveUtility(utilities={"f": _utility, "m": _utility_m})
    )
    terminal_utility = (
        _terminal_utility
        if kind == _SINGLETON
        else CollectiveUtility(
            utilities={"f": _terminal_utility, "m": _terminal_utility}
        )
    )
    alive = Regime(
        states={
            "x": LinSpacedGrid(start=0.0, stop=_N_X - 1.0, n_points=_N_X),
            "shock": shock,
        },
        state_transitions={"x": _next_x, **transitions},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": utility, "shock_value": shock_value},
    )
    terminal = Regime(
        states={"x": LinSpacedGrid(start=0.0, stop=_N_X - 1.0, n_points=_N_X)},
        functions={"utility": terminal_utility},
    )
    width = _N_X * n_nodes if dispatch == _DENSE else 2 * n_nodes
    return Model(
        regimes={"alive": alive, "terminal": terminal},
        ages=AgeGrid(start=0, inclusive_stop=_N_AGES, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
        edges={
            "alive": {"alive": tuple(range(_N_AGES - 1)), "terminal": (_N_AGES - 1,)}
        },
        execution_config=ExecutionConfig(axis_widths={CELL_AXIS: {"alive": width}}),
    )


@functools.cache
def _solve(
    *, kind: str, family: str, n_nodes: int, dispatch: str, as_markov: bool
) -> tuple[np.ndarray, frozenset[str] | None, tuple[StateName, ...]]:
    """Solve one case; return the first-period alive values, the continuation reads
    of the alive kernel, and the states GridSearch broadcasts past it."""
    reads: list[frozenset[str] | None] = []
    selections: list[tuple[StateName, ...]] = []
    select = grid_search._continuation_unread_state_names

    def observed_select(
        *, Q_and_F: QAndFFunction, inner_state_names: tuple[StateName, ...]
    ) -> tuple[StateName, ...]:
        selected = select(Q_and_F=Q_and_F, inner_state_names=inner_state_names)
        if "shock" in inner_state_names:
            reads.append(getattr(Q_and_F, "continuation_reads", None))
            selections.append(selected)
        return selected

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            grid_search, "_continuation_unread_state_names", observed_select
        )
        model = _model(
            kind=kind,
            family=family,
            n_nodes=n_nodes,
            dispatch=dispatch,
            as_markov=as_markov,
        )
        params = {
            "alive": {"koopmans_aggregator": {"discount_factor": 0.95}},
            "terminal": {},
        }
        solution = model.solve(params=params, log_level="off")
    return np.asarray(solution.values[0]["alive"]), reads[0], selections[0]


def test_iid_process_is_not_read_by_the_singleton_continuation() -> None:
    """The singleton continuation reads `x` but not the IID `shock`."""
    reads = _solve(
        kind=_SINGLETON, family="iid", n_nodes=2, dispatch=_DENSE, as_markov=False
    )[1]

    assert (reads or frozenset()) & {"x", "shock"} == {"x"}


def test_grid_search_broadcasts_the_iid_process_past_the_continuation() -> None:
    """GridSearch maps the IID `shock` outside the flat cell of the singleton."""
    selected = _solve(
        kind=_SINGLETON, family="iid", n_nodes=2, dispatch=_DENSE, as_markov=False
    )[2]

    assert selected == ("shock",)


def test_ar1_process_is_read_by_the_continuation() -> None:
    """A Tauchen AR(1) process's next-period law depends on its current value, so
    the continuation reads it."""
    reads = _solve(
        kind=_SINGLETON, family="ar1", n_nodes=2, dispatch=_DENSE, as_markov=False
    )[1]

    assert "shock" in (reads or ())


def test_ar1_rows_differ() -> None:
    """The AR(1) control has rows that differ, so reading the first row alone, as
    an IID process may, would price it under the wrong law."""
    rows = np.asarray(
        _process(family="ar1", n_nodes=3).get_transition_probs(), dtype=float
    )

    assert np.abs(rows - rows[0]).max() > 0.1


_VALUE_CASES = tuple(
    (kind, family, n_nodes, dispatch)
    for kind in (_SINGLETON, _COLLECTIVE)
    for dispatch in (_DENSE, _STREAMED)
    for family, n_nodes in (("iid", 1), ("iid", 2), ("iid", 3), ("ar1", 2), ("ar1", 3))
)


@pytest.mark.parametrize(
    ("kind", "family", "n_nodes", "dispatch"),
    _VALUE_CASES,
    ids=["-".join(map(str, case)) for case in _VALUE_CASES],
)
def test_process_values_equal_the_explicit_markov_state(
    *, kind: str, family: str, n_nodes: int, dispatch: str
) -> None:
    """The model solves to the values of the same model with the shock written as
    a Markov state whose transition returns the process's transition row."""
    process = _solve(
        kind=kind, family=family, n_nodes=n_nodes, dispatch=dispatch, as_markov=False
    )[0]
    markov = _solve(
        kind=kind, family=family, n_nodes=n_nodes, dispatch=dispatch, as_markov=True
    )[0]
    rtol = 1e-12 if process.dtype == np.float64 else 1e-5

    np.testing.assert_allclose(process, markov, rtol=rtol, atol=0.0)
