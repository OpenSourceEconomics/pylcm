"""Target scope and state-cell ownership for joint transitions."""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.regime_building.canonicalize import canonicalize_regimes
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.phases import PhasedRegimeSpec
from lcm import (
    JointTransition,
    LinearAggregator,
    LinearExpectation,
    LinSpacedGrid,
    MarkovTransition,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.typing import FloatND


def _probability() -> FloatND:
    return jnp.asarray(1.0)


def _joint_probabilities() -> FloatND:
    return jnp.asarray([0.4, 0.6])


def _next_wealth(match: dict[str, FloatND]) -> FloatND:
    return match["wealth"]


def _joint(*, output: str = "wealth") -> JointTransition:
    return JointTransition(
        support_size=2,
        support={"wealth": jnp.asarray([1.0, 2.0])},
        probabilities=_joint_probabilities,
        outputs={output: _next_wealth},
    )


def _specs(
    *,
    joint_transitions: dict[str, dict[str, JointTransition]],
    state_transitions: dict[str, object] | None = None,
) -> MappingProxyType[str, PhasedRegimeSpec]:
    source = Regime(
        transition={"target": MarkovTransition(_probability)},
        functions={"utility": lambda: jnp.asarray(0.0)},
        state_transitions=state_transitions or {},  # ty: ignore[invalid-argument-type]
        joint_transitions=joint_transitions,
    )
    target = Regime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
        functions={"utility": lambda wealth: wealth},
    )
    finalized = finalize_regimes(
        user_regimes={"source": source, "target": target},
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    return canonicalize_regimes(user_regimes=finalized)


def test_joint_kernel_survives_canonicalization_on_its_explicit_target() -> None:
    """Canonical phase specs retain one immutable kernel on the declared edge."""
    kernel = _joint()
    specs = _specs(joint_transitions={"target": {"match": kernel}})

    assert specs["source"].solution.joint_transitions["target"]["match"] is kernel
    assert specs["source"].simulation.joint_transitions["target"]["match"] is kernel


def test_joint_output_claims_cell_before_bare_law_broadcast() -> None:
    """A bare ordinary law broadcasts only into cells unclaimed by a joint output."""
    specs = _specs(
        joint_transitions={"target": {"match": _joint()}},
        state_transitions={"wealth": lambda wealth: wealth},
    )

    assert "wealth" not in specs["source"].solution.state_transitions


def test_explicit_ordinary_law_cannot_collide_with_joint_output() -> None:
    """Two explicit producers for one target-state cell are rejected."""
    with pytest.raises(ModelInitializationError, match=r"multiple producers.*wealth"):
        _specs(
            joint_transitions={"target": {"match": _joint()}},
            state_transitions={
                "wealth": {"target": lambda wealth: wealth},
            },
        )


def test_two_joint_kernels_cannot_claim_the_same_output_cell() -> None:
    """Distinct lotteries cannot both own one target-state output."""
    with pytest.raises(ModelInitializationError, match=r"multiple producers.*wealth"):
        _specs(
            joint_transitions={
                "target": {"first": _joint(), "second": _joint()},
            }
        )


def test_joint_output_must_be_a_state_of_its_target() -> None:
    """A joint output is a genuine target state, not a derived or latent value."""
    with pytest.raises(ModelInitializationError, match=r"output.*health.*target state"):
        _specs(joint_transitions={"target": {"match": _joint(output="health")}})
