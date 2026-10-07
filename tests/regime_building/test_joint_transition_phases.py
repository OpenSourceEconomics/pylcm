"""Phase and edge-scope grammar for joint transitions."""

import jax.numpy as jnp
import pytest

from _lcm.regime_building.phases import PhasedRegimeSpec, normalize_regime_phases
from _lcm.regime_law import RegimeLaw, bind_regime_law
from _lcm.user_regime_validation import (
    fail_if_a_joint_target_is_unreachable,
    validate_regime_law,
)
from lcm import (
    AgeGrid,
    JointTransition,
    Model,
    Phased,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import FloatND, ScalarInt


def _probabilities() -> FloatND:
    return jnp.asarray([0.25, 0.75])


def _output(match: dict[str, FloatND]) -> FloatND:
    return match["value"]


def _kernel(*, output_name: str = "wealth", dtype: str = "float") -> JointTransition:
    values = (
        jnp.asarray([1, 2], dtype=jnp.int32)
        if dtype == "int"
        else jnp.asarray([1.0, 2.0])
    )
    return JointTransition(
        support_size=2,
        support={"value": values},
        probabilities=_probabilities,
        outputs={output_name: _output},
    )


def _next_regime() -> FloatND:
    return jnp.asarray(0, dtype=jnp.int32)


def _regime(
    *, joint_transitions: object, law: object = _next_regime
) -> tuple[Regime, RegimeLaw]:
    """A regime validated under `law`, returned with the bound law."""
    regime = Regime(
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions=joint_transitions,  # ty: ignore[invalid-argument-type]
    )
    bound = bind_regime_law(law)
    validate_regime_law(regime, law=bound)
    return regime, bound


def _phases(*, joint_transitions: object) -> PhasedRegimeSpec:
    regime, law = _regime(joint_transitions=joint_transitions)
    return normalize_regime_phases(regime, law=law)


def test_bare_joint_transition_broadcasts_to_both_phases() -> None:
    """A bare joint kernel is shared by solution and simulation."""
    kernel = _kernel()
    phases = _phases(joint_transitions={"couple": {"match": kernel}})

    assert phases.solution.joint_transitions["couple"]["match"] is kernel
    assert phases.simulation.joint_transitions["couple"]["match"] is kernel


def test_phased_joint_transition_resolves_whole_kernel_per_phase() -> None:
    """`Phased` wraps the whole kernel and is resolved at the phase boundary."""
    solve = _kernel()
    simulate = _kernel()
    phases = _phases(
        joint_transitions={"couple": {"match": Phased(solve=solve, simulate=simulate)}}
    )

    assert phases.solution.joint_transitions["couple"]["match"] is solve
    assert phases.simulation.joint_transitions["couple"]["match"] is simulate


@pytest.mark.parametrize(
    ("simulate", "match"),
    [
        (
            JointTransition(
                support_size=2,
                support={"value": jnp.asarray([1.0, 2.0])},
                probabilities=_probabilities,
                outputs={"health": _output},
            ),
            "output-state key",
        ),
        (
            JointTransition(
                support_size=3,
                support={"value": jnp.asarray([1.0, 2.0, 3.0])},
                probabilities=lambda: jnp.asarray([0.2, 0.3, 0.5]),
                outputs={"wealth": _output},
            ),
            "support_size",
        ),
        (_kernel(dtype="int"), "dtype"),
    ],
)
def test_phased_joint_transition_requires_a_static_support_schema(
    *, simulate: JointTransition, match: str
) -> None:
    """Phase variants keep identical outputs and support shape/dtype contracts."""
    with pytest.raises(RegimeInitializationError, match=match):
        _regime(
            joint_transitions={
                "couple": {
                    "match": Phased(solve=_kernel(), simulate=simulate),
                }
            }
        )


def test_joint_node_name_cannot_collide_with_source_function() -> None:
    """A transition-local node cannot shadow a source DAG producer."""
    with pytest.raises(RegimeInitializationError, match=r"node name.*match.*collides"):
        validate_regime_law(
            Regime(
                functions={
                    "utility": lambda: jnp.asarray(0.0),
                    "match": lambda: jnp.asarray(1.0),
                },
                joint_transitions={"couple": {"match": _kernel()}},
            ),
            law=bind_regime_law(_next_regime),
        )


def test_joint_node_name_cannot_use_a_reserved_transition_prefix() -> None:
    """Internal transition prefixes remain unavailable to joint node names."""
    with pytest.raises(RegimeInitializationError, match=r"reserved.*next_"):
        _regime(joint_transitions={"couple": {"next_wealth": _kernel()}})


def test_joint_transition_target_must_be_declared_reachable() -> None:
    """A joint kernel cannot name a target its source has no edge into."""
    regime = Regime(
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={"couple": {"match": _kernel()}},
    )
    with pytest.raises(
        RegimeInitializationError,
        match=r"'source'.*reachable under the declared edges \['single'\]; "
        r"got 'couple'",
    ):
        fail_if_a_joint_target_is_unreachable(
            user_regimes={"source": regime},
            targets_by_regime={"source": frozenset({"single"})},
        )


@categorical(ordered=False)
class _JointRegimeId:
    source: ScalarInt
    single: ScalarInt
    couple: ScalarInt


def test_model_rejects_a_joint_target_outside_the_declared_edges() -> None:
    """A source whose only edge leads to `single` cannot own a kernel into `couple`.

    The source has a single destination, so its law is that destination's name;
    the edges, not the law's form, decide which targets a kernel may name.
    """
    source = Regime(
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={"couple": {"match": _kernel()}},
    )
    with pytest.raises(
        RegimeInitializationError,
        match=r"'source'.*reachable under the declared edges \['single'\]; "
        r"got 'couple'",
    ):
        Model(
            regimes={
                "source": source,
                "single": Regime(functions={"utility": lambda: jnp.asarray(0.0)}),
                "couple": Regime(functions={"utility": lambda: jnp.asarray(0.0)}),
            },
            edges={"source": {"single": 0}},
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_JointRegimeId,
            initial_nodes={0: "source"},
        )


def test_terminal_regime_cannot_declare_joint_transition() -> None:
    """A terminal regime has no target edge on which to own a joint kernel."""
    with pytest.raises(RegimeInitializationError, match=r"Terminal.*joint_transitions"):
        _regime(law=None, joint_transitions={"couple": {"match": _kernel()}})


def test_literal_support_leading_axis_matches_support_size() -> None:
    """Every literal support leaf has the declared leading node dimension."""
    with pytest.raises(RegimeInitializationError, match="leading axis"):
        JointTransition(
            support_size=2,
            support={"value": jnp.asarray([1.0, 2.0, 3.0])},
            probabilities=_probabilities,
            outputs={"wealth": _output},
        )
