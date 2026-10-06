"""A fixed-zero edge may carry a joint transition or a per-target state law.

The source `source` moves to `low`, `mid` or `high` with probabilities read from
fixed parameters. `low` is worth 2, `mid` is worth its wealth, which the edge law
sets to 4, and `high` is worth its wealth, drawn from a joint lottery that puts
1/4 on wealth 1 and 3/4 on wealth 3, so `high` is worth 2.5 in expectation. An
edge whose probability is fixed at exactly zero is removed from the effective
graph together with the laws that hand states across it; the declared topology
keeps it. A nonzero fixed probability keeps the edge and its value.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    JointTransition,
    LinSpacedGrid,
    Model,
    Phased,
    StochasticTransition,
    categorical,
)
from lcm.regime import Regime
from lcm.typing import ContinuousState, FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    low: ScalarInt
    mid: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _AgeRegimeId:
    source: ScalarInt
    low: ScalarInt
    high: ScalarInt


def _low_mass(*, p_mid: ScalarFloat, p_high: ScalarFloat) -> FloatND:
    return 1.0 - p_mid - p_high


def _mid_mass(*, p_mid: ScalarFloat) -> FloatND:
    return jnp.asarray(p_mid)


def _high_mass(*, p_high: ScalarFloat) -> FloatND:
    return jnp.asarray(p_high)


def _joint_probabilities(*, tilt: ScalarFloat) -> FloatND:
    return jnp.asarray([0.25, 0.75]) * tilt


def _joint_wealth(*, match: dict[str, FloatND]) -> ContinuousState:
    return match["wealth"]


def _ordinary_high_wealth() -> ContinuousState:
    return jnp.asarray(2.5)


def _mid_wealth() -> ContinuousState:
    return jnp.asarray(4.0)


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _two() -> FloatND:
    return jnp.asarray(2.0)


def _wealth_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _wealth_grid() -> LinSpacedGrid:
    return LinSpacedGrid(start=1.0, stop=4.0, n_points=7)


def _model(
    *, joint: bool, fixed_params: dict, simulate_high_mass: bool = False
) -> Model:
    transitions = {
        "low": StochasticTransition(func=_low_mass),
        "mid": StochasticTransition(func=_mid_mass),
        "high": StochasticTransition(func=_high_mass),
    }
    source = Regime(
        regime_transitions=(
            Phased(
                solve=transitions,
                simulate={
                    "low": StochasticTransition(func=lambda: jnp.asarray(0.5)),
                    "mid": StochasticTransition(func=lambda: jnp.asarray(0.0)),
                    "high": StochasticTransition(func=lambda: jnp.asarray(0.5)),
                },
            )
            if simulate_high_mass
            else transitions
        ),
        state_transitions={
            "wealth": {"mid": _mid_wealth}
            | ({} if joint else {"high": _ordinary_high_wealth})
        },
        joint_transitions=(
            {
                "high": {
                    "match": JointTransition(
                        support_size=2,
                        support={"wealth": jnp.asarray([1.0, 3.0])},
                        probabilities=_joint_probabilities,
                        outputs={"wealth": _joint_wealth},
                    )
                }
            }
            if joint
            else {}
        ),
        functions={"utility": _zero},
    )
    regimes = {
        "source": source,
        "low": Regime(regime_transitions=None, functions={"utility": _two}),
        "mid": Regime(
            regime_transitions=None,
            states={"wealth": _wealth_grid()},
            functions={"utility": _wealth_utility},
        ),
        "high": Regime(
            regime_transitions=None,
            states={"wealth": _wealth_grid()},
            functions={"utility": _wealth_utility},
        ),
    }
    return Model(
        regimes=regimes,
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"source": {"low": 0, "mid": 0, "high": 0}},
        initial_nodes=((0, "source"),),
        fixed_params=fixed_params,
        enable_jit=False,
    )


def _fixed(*, joint: bool, p_mid: float, p_high: float) -> dict:
    return {"p_mid": p_mid, "p_high": p_high} | ({"tilt": 1.0} if joint else {})


def _reference_value(*, p_mid: Fraction, p_high: Fraction) -> Fraction:
    """Enumerate the finite paths of the two-period model exactly."""
    high_value = Fraction(1, 4) * 1 + Fraction(3, 4) * 3
    return (1 - p_mid - p_high) * 2 + p_mid * 4 + p_high * high_value


def _source_value(*, model: Model, params: dict) -> np.ndarray:
    result = model.solve(params={"discount_factor": 1.0, **params}, log_level="off")
    return np.asarray(result.values[0]["source"])


def test_fixed_zero_edge_with_joint_transition_constructs_and_values_exactly() -> None:
    """A joint-lottery edge fixed at zero probability leaves the value at exactly 2."""
    model = _model(joint=True, fixed_params=_fixed(joint=True, p_mid=0.0, p_high=0.0))
    assert float(_source_value(model=model, params={})) == float(
        _reference_value(p_mid=Fraction(0), p_high=Fraction(0))
    )


@pytest.mark.parametrize("joint", [False, True])
def test_several_fixed_zero_edges_leave_only_the_positive_edge(*, joint: bool) -> None:
    """Every edge fixed at zero is removed from both phases' effective graphs."""
    model = _model(joint=joint, fixed_params=_fixed(joint=joint, p_mid=0.0, p_high=0.0))
    assert (
        model.graph.solution.targets(period=0, source="source"),
        model.graph.simulation.targets(period=0, source="source"),
        dict(model.graph.pruned_edges["solve"]),
        set(model.graph.edges.solve["source"]),
    ) == (
        ("low",),
        ("low",),
        {
            (0, "source", "mid"): "fixed_zero_probability",
            (0, "source", "high"): "fixed_zero_probability",
        },
        {"low", "mid", "high"},
    )


@pytest.mark.parametrize("joint", [False, True])
def test_several_fixed_zero_edges_value_only_the_positive_edge(*, joint: bool) -> None:
    """Removing several zero edges leaves the exactly enumerated value."""
    model = _model(joint=joint, fixed_params=_fixed(joint=joint, p_mid=0.0, p_high=0.0))
    assert float(_source_value(model=model, params={})) == float(
        _reference_value(p_mid=Fraction(0), p_high=Fraction(0))
    )


def test_fixed_key_read_only_by_a_removed_joint_kernel_is_consumed() -> None:
    """A fixed parameter read only by a removed joint lottery is not unknown."""
    model = _model(joint=True, fixed_params=_fixed(joint=True, p_mid=0.0, p_high=0.0))
    assert "tilt" not in repr(model.get_params_template())


@pytest.mark.parametrize("joint", [False, True])
def test_nonzero_fixed_probability_keeps_the_edge(*, joint: bool) -> None:
    """A positive fixed probability keeps the edge and prices its lottery."""
    model = _model(
        joint=joint, fixed_params=_fixed(joint=joint, p_mid=0.25, p_high=0.5)
    )
    assert set(model.graph.solution.targets(period=0, source="source")) == {
        "low",
        "mid",
        "high",
    }
    np.testing.assert_allclose(
        _source_value(model=model, params={}),
        float(_reference_value(p_mid=Fraction(1, 4), p_high=Fraction(1, 2))),
        rtol=1e-6,
    )


def test_joint_edge_zero_in_one_phase_only_stays_declared_in_both() -> None:
    """An edge zero only in the solve law keeps its joint lottery for simulation."""
    model = _model(
        joint=True,
        fixed_params=_fixed(joint=True, p_mid=0.0, p_high=0.0),
        simulate_high_mass=True,
    )
    assert (
        "high" in model.graph.simulation.targets(period=0, source="source"),
        float(_source_value(model=model, params={})),
    ) == (True, 2.0)


def test_joint_edge_zero_at_one_age_only_keeps_its_lottery_at_the_other() -> None:
    """A joint edge fixed at zero at age 0 is removed there and priced at age 1."""
    source = Regime(
        regime_transitions=ByAge(
            cases={
                0: {
                    "low": StochasticTransition(func=lambda p_high: 1.0 - p_high),
                    "high": StochasticTransition(func=_high_mass),
                },
                1: {
                    "low": StochasticTransition(func=lambda: jnp.asarray(0.5)),
                    "high": StochasticTransition(func=lambda: jnp.asarray(0.5)),
                },
            }
        ),
        joint_transitions={
            "high": {
                "match": JointTransition(
                    support_size=2,
                    support={"wealth": jnp.asarray([1.0, 3.0])},
                    probabilities=_joint_probabilities,
                    outputs={"wealth": _joint_wealth},
                )
            }
        },
        functions={"utility": _zero},
    )
    model = Model(
        regimes={
            "source": source,
            "low": Regime(regime_transitions=None, functions={"utility": _two}),
            "high": Regime(
                regime_transitions=None,
                states={"wealth": _wealth_grid()},
                functions={"utility": _wealth_utility},
            ),
        },
        regime_id_class=_AgeRegimeId,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"source": {"low": (0, 1), "high": (0, 1)}},
        initial_nodes=((0, "source"), (1, "source")),
        fixed_params={"p_high": 0.0, "tilt": 1.0},
        enable_jit=False,
    )
    values = model.solve(params={"discount_factor": 1.0}, log_level="off").values
    assert (
        dict(model.graph.pruned_edges["solve"]),
        float(values[0]["source"]),
        float(values[1]["source"]),
    ) == ({(0, "source", "high"): "fixed_zero_probability"}, 2.0, 2.25)
