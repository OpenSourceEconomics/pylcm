"""Public declarations for correlated target-state transitions."""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from lcm import JointTransition, LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import FloatND


def _probabilities() -> FloatND:
    return jnp.asarray([0.4, 0.6])


def _next_wealth(match: dict[str, FloatND]) -> FloatND:
    return match["wealth"]


def _utility(wealth: float) -> FloatND:
    return jnp.asarray(wealth)


def _next_regime() -> FloatND:
    return jnp.asarray(0, dtype=jnp.int32)


def test_joint_transition_is_an_edge_owned_public_declaration() -> None:
    """A regime can declare one correlated kernel for a named target edge."""
    transition = JointTransition(
        support_size=2,
        support={"wealth": jnp.asarray([1.0, 2.0])},
        probabilities=_probabilities,
        outputs={"wealth": _next_wealth},
    )

    regime = Regime(
        transition=_next_regime,
        states={"wealth": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
        state_transitions={"wealth": lambda wealth: wealth},
        functions={"utility": _utility},
        joint_transitions={"couple": {"match": transition}},
    )

    assert regime.joint_transitions["couple"]["match"] is transition
    assert isinstance(regime.joint_transitions, MappingProxyType)
    assert isinstance(regime.joint_transitions["couple"], MappingProxyType)
    assert isinstance(transition.outputs, MappingProxyType)
    assert isinstance(transition.support, MappingProxyType)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "support_size": 0,
                "support": {"wealth": jnp.asarray([])},
                "probabilities": _probabilities,
                "outputs": {"wealth": _next_wealth},
            },
            "support_size",
        ),
        (
            {
                "support_size": 2,
                "support": {"wealth": jnp.asarray([1.0, 2.0])},
                "probabilities": _probabilities,
                "outputs": {},
            },
            "output",
        ),
    ],
)
def test_joint_transition_rejects_locally_invalid_declarations(
    *, kwargs: dict[str, object], match: str
) -> None:
    """A joint kernel requires a nonempty finite support and output mapping."""
    with pytest.raises(RegimeInitializationError, match=match):
        JointTransition(**kwargs)  # ty: ignore[invalid-argument-type]


def test_literal_joint_support_rejects_nonfinite_values() -> None:
    """Every literal support leaf contains only finite values."""
    with pytest.raises(RegimeInitializationError, match="finite"):
        JointTransition(
            support_size=2,
            support={"wealth": jnp.asarray([1.0, jnp.nan])},
            probabilities=_probabilities,
            outputs={"wealth": _next_wealth},
        )
