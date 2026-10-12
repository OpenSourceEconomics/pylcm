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
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from _lcm.regime_building.fixed_regime_support import _prune_cells
from lcm import AgeGrid, InitialNodes, Model, categorical
from lcm.phased import Phased
from lcm.regime import Regime
from lcm.transition import StochasticTransition, Transition
from lcm.typing import FloatND, ScalarInt

type _Probability = Array | float | int | np.generic | NDArray[np.generic]
type _ProbabilityCell = Callable[[], _Probability]
type _Side = Literal["solve", "simulate"]

_FINAL_DEMAND = frozenset({(25, "source"), (26, "sink")})


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


@pytest.mark.parametrize("phase", ["solve", "simulate"])
@pytest.mark.parametrize("zero_cell", _ZERO_CELLS)
def test_fixed_zero_edge_to_a_terminal_regime_is_pruned(
    *, zero_cell: _ProbabilityCell, phase: str
) -> None:
    """The zero edge is recorded as pruned for a fixed zero probability."""
    model = _model(zero_cell=zero_cell, unused_is_nonterminal=False)

    assert model.graph.pruned_edges[phase].get((25, "source", "unused")) == (
        "fixed_zero_probability"
    )


@pytest.mark.parametrize("zero_cell", _ZERO_CELLS)
def test_fixed_zero_edge_to_a_terminal_regime_demands_only_the_live_path(
    *, zero_cell: _ProbabilityCell
) -> None:
    """Only `source` at 25 and `sink` at 26 remain visited."""
    model = _model(zero_cell=zero_cell, unused_is_nonterminal=False)

    assert model.graph.visited_nodes == _FINAL_DEMAND


@pytest.mark.parametrize("zero_cell", _ZERO_CELLS)
def test_fixed_zero_edge_to_a_nonterminal_regime_leaves_a_valid_model(
    *, zero_cell: _ProbabilityCell
) -> None:
    """`unused` is never demanded at the final age, so the model constructs."""
    model = _model(zero_cell=zero_cell, unused_is_nonterminal=True)

    assert model.graph.visited_nodes == _FINAL_DEMAND


def _constant(value: _Probability) -> _ProbabilityCell:
    def probability() -> _Probability:
        return value

    return probability


# Each fixed value and whether it proves the edge has zero probability. Only a finite
# numeric scalar equal to zero does; a tiny positive host value stays positive even
# where a lower-precision device dtype would round it to zero.
_PROBABILITY_CASES = [
    pytest.param(0.0, True, id="python-float-zero"),
    pytest.param(-0.0, True, id="python-negative-zero"),
    pytest.param(0, True, id="python-int-zero"),
    pytest.param(jnp.asarray(0.0), True, id="jax-zero"),
    pytest.param(np.float32(0), True, id="numpy-float32-zero"),
    pytest.param(np.float32(-0.0), True, id="numpy-float32-negative-zero"),
    pytest.param(np.int64(0), True, id="numpy-int64-zero"),
    pytest.param(np.uint64(0), True, id="numpy-uint64-zero"),
    pytest.param(np.array(0, dtype=np.float32), True, id="numpy-0d-float32-zero"),
    pytest.param(np.array(0, dtype=np.int64), True, id="numpy-0d-int64-zero"),
    pytest.param(np.float32(0.25), False, id="numpy-positive"),
    pytest.param(np.float32(-0.25), False, id="numpy-negative"),
    pytest.param(np.float32(2.0**-149), False, id="smallest-float32-positive"),
    pytest.param(np.float64(2.0**-200), False, id="float64-below-float32-range"),
    pytest.param(np.float32(np.nan), False, id="numpy-nan"),
    pytest.param(np.float32(np.inf), False, id="numpy-inf"),
    pytest.param(np.array([0], dtype=np.float32), False, id="numpy-length-one"),
    pytest.param(np.array([], dtype=np.float32), False, id="numpy-empty"),
    pytest.param(np.False_, False, id="numpy-bool"),
    pytest.param(np.complex64(0), False, id="numpy-complex"),
]


@pytest.mark.parametrize("side", ["solve", "simulate"])
@pytest.mark.parametrize(("value", "pruned"), _PROBABILITY_CASES)
def test_only_an_exact_finite_scalar_zero_removes_a_fixed_cell(
    *, value: _Probability, pruned: bool, side: _Side
) -> None:
    """The retained cells, in authored order, are exactly the unproven ones."""
    law = {
        "sink": StochasticTransition(func=_one),
        "unused": StochasticTransition(func=_constant(value)),  # ty: ignore[invalid-argument-type] -- a host carrier on purpose
    }

    retained = _prune_cells(
        law=law,
        side=side,
        regime_name="source",
        regime=Regime(functions={"utility": _utility}),
        fixed_flat=MappingProxyType({}),
        consumed=set(),
        protected=frozenset(),
    )

    assert tuple(retained) == (("sink",) if pruned else ("sink", "unused"))


@pytest.mark.parametrize("side", ["solve", "simulate"])
@pytest.mark.parametrize(
    "zero",
    [np.float32(0), np.array(0, dtype=np.float32)],
    ids=["numpy-float32", "numpy-0d-float32"],
)
def test_an_all_zero_row_of_host_zeros_is_kept_for_validation(
    *, zero: _Probability, side: _Side
) -> None:
    """A row with no positive mass stays whole for the probability validator."""
    law = {
        "left": StochasticTransition(func=_jax_zero),
        "right": StochasticTransition(func=_constant(zero)),  # ty: ignore[invalid-argument-type] -- a host carrier on purpose
    }

    retained = _prune_cells(
        law=law,
        side=side,
        regime_name="source",
        regime=Regime(functions={"utility": _utility}),
        fixed_flat=MappingProxyType({}),
        consumed=set(),
        protected=frozenset(),
    )

    assert tuple(retained) == ("left", "right")


def _probability_from_helper(fixed_support: _Probability) -> _Probability:
    return fixed_support


@pytest.mark.parametrize(("side", "pruned"), [("solve", True), ("simulate", False)])
def test_a_host_zero_from_a_phased_helper_prunes_only_its_phase(
    *, side: _Side, pruned: bool
) -> None:
    """A helper returning a NumPy zero for solve and a one for simulate."""
    regime = Regime(
        functions={
            "utility": _utility,
            "fixed_support": Phased(
                solve=_constant(np.float32(0)), simulate=_constant(np.float32(1))
            ),
        }
    )
    law = {
        "sink": StochasticTransition(func=_one),
        "unused": StochasticTransition(func=_probability_from_helper),  # ty: ignore[invalid-argument-type] -- a host carrier on purpose
    }

    retained = _prune_cells(
        law=law,
        side=side,
        regime_name="source",
        regime=regime,
        fixed_flat=MappingProxyType({}),
        consumed=set(),
        protected=frozenset(),
    )

    assert tuple(retained) == (("sink",) if pruned else ("sink", "unused"))
