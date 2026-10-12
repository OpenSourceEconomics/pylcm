"""A fixed zero probability is pruned whatever numeric carrier returns it.

The source regime `source` moves to the terminal regime `sink` with probability
one and to `unused` with a fixed probability of exactly zero. A Python float, a
JAX array, a NumPy scalar and a zero-dimensional NumPy array all state that zero,
so each removes the edge `source -> unused` before the model's demand is derived.
When `unused` is itself a nonterminal regime, a retained edge would demand it at
the final age, where it has no problem to solve, and construction would fail.
"""

from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from _lcm.regime_building.fixed_regime_support import _prune_cells
from lcm import AgeGrid, InitialNodes, Model, categorical
from lcm.regime import Regime
from lcm.transition import StochasticTransition, Transition
from lcm.typing import FloatND, ScalarInt

type _Probability = Array | float | np.generic | NDArray[np.generic]
type _ProbabilityCell = Callable[[], _Probability]


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    sink: ScalarInt
    unused: ScalarInt


def _utility() -> FloatND:
    return jnp.asarray(0.0)


def _one() -> FloatND:
    return jnp.asarray(1.0)


def _python_zero() -> float:
    return 0.0


def _jax_zero() -> FloatND:
    return jnp.asarray(0.0)


def _numpy_float32_zero() -> np.float32:
    return np.float32(0.0)


def _numpy_0d_float32_zero() -> NDArray[np.float32]:
    return np.asarray(0.0, dtype=np.float32)


_ZERO_CELLS = [
    pytest.param(_python_zero, id="python-float"),
    pytest.param(_jax_zero, id="jax-array"),
    pytest.param(_numpy_float32_zero, id="numpy-float32"),
    pytest.param(_numpy_0d_float32_zero, id="numpy-0d-float32"),
]


def _model(*, zero_cell: _ProbabilityCell, unused_is_nonterminal: bool) -> Model:
    edges: dict[str, Transition | dict[str, int]] = {
        "source": Transition(
            law={
                "sink": StochasticTransition(func=_one),
                # The public signature names a JAX result; the runtime also
                # accepts the host carriers under test.
                "unused": StochasticTransition(func=zero_cell),  # ty: ignore[invalid-argument-type] -- a host carrier on purpose
            }
        )
    }
    if unused_is_nonterminal:
        edges["unused"] = {"sink": 25}
    return Model(
        regimes={
            "source": Regime(functions={"utility": _utility}),
            "sink": Regime(functions={"utility": _utility}),
            "unused": Regime(functions={"utility": _utility}),
        },
        ages=AgeGrid(start=25, inclusive_stop=26, step="Y"),
        regime_id_class=_RegimeId,
        edges=edges,
        initial_nodes=InitialNodes(by_age={25: "source"}),
    )


@pytest.mark.parametrize("zero_cell", _ZERO_CELLS)
def test_fixed_zero_edge_to_a_nonterminal_regime_leaves_a_valid_model(
    *, zero_cell: _ProbabilityCell
) -> None:
    """`unused` is never demanded at the final age, so the model constructs."""
    model = _model(zero_cell=zero_cell, unused_is_nonterminal=True)

    assert model.graph.visited_nodes == {(25, "source"), (26, "sink")}


@pytest.mark.parametrize("zero_cell", _ZERO_CELLS)
def test_fixed_zero_edge_to_a_terminal_regime_is_pruned_from_the_live_path(
    *, zero_cell: _ProbabilityCell
) -> None:
    """The zero edge is recorded as pruned, and only the live path is visited."""
    model = _model(zero_cell=zero_cell, unused_is_nonterminal=False)

    assert model.graph.pruned_edges["simulate"].get((25, "source", "unused")) == (
        "fixed_zero_probability"
    )
    assert model.graph.visited_nodes == {(25, "source"), (26, "sink")}


def _constant(value: _Probability) -> _ProbabilityCell:
    def probability() -> _Probability:
        return value

    return probability


# Each fixed value and whether it proves the edge has zero probability. A tiny
# positive host value stays positive even where a lower-precision device dtype would
# round it to zero.
@pytest.mark.parametrize(
    ("value", "pruned"),
    [
        pytest.param(np.float32(0), True, id="numpy-float32-zero"),
        pytest.param(np.array(0, dtype=np.float32), True, id="numpy-0d-float32-zero"),
        pytest.param(np.float32(0.25), False, id="numpy-float32-positive"),
        pytest.param(np.float64(2.0**-200), False, id="float64-below-float32-range"),
    ],
)
def test_only_an_exact_zero_removes_a_fixed_host_cell(
    *, value: _Probability, pruned: bool
) -> None:
    """The retained cells, in authored order, are exactly the unproven ones."""
    law = {
        "sink": StochasticTransition(func=_one),
        "unused": StochasticTransition(func=_constant(value)),  # ty: ignore[invalid-argument-type] -- a host carrier on purpose
    }

    retained = _prune_cells(
        law=law,
        side="simulate",
        regime_name="source",
        regime=Regime(functions={"utility": _utility}),
        fixed_flat=MappingProxyType({}),
        consumed=set(),
        protected=frozenset(),
    )

    assert tuple(retained) == (("sink",) if pruned else ("sink", "unused"))
