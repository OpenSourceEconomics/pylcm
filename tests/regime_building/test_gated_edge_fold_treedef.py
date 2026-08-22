"""The same-period value mapping a gated edge's fold reads has one shape.

Backward induction and forward simulation share one jitted fold per edge, and
each assembles the same-period value mapping that fold reads from the arrays it
has in hand. Whether a dissolution flag was supplied is a property of the
caller, so it must not reach the fold's pytree: a key present on one side and
absent on the other traces the same fold twice, on grids whose size is the
model's.

A gate that consumes the flag is a different matter — there the absent flag is
not something to stand in for, and the mapping refuses to build.
"""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest

from _lcm.regime_building.gated_edges import (
    D_KEY_SUFFIX,
    ResolvedEdgeLeg,
    ResolvedGatedEdge,
    build_same_period_mapping_for_fold,
)
from _lcm.regime_building.Q_and_F import ResolvedSamePeriodRef
from lcm.typing import BoolND, ContinuousState, FloatND

# The collective target's value: three wage nodes by two stakeholders. Its
# dissolution flag lives on the state axes alone, so it is one axis shorter.
_TARGET_V = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
_TARGET_D = jnp.asarray([False, True, False])

# The fallback regime's value, on its own single wage axis.
_FALLBACK_V = jnp.asarray([0.5, 1.5, 2.5])


def test_same_period_mapping_carries_the_flag_key_when_no_flag_is_supplied():
    """The mapping's key set is the same with and without a supplied flag."""
    with_flag = _build_mapping(gate=_consent_gate, supply_flag=True)
    without_flag = _build_mapping(gate=_consent_gate, supply_flag=False)

    assert set(without_flag) == set(with_flag)


def test_same_period_mapping_has_one_treedef_whether_or_not_a_flag_is_supplied():
    """Both mappings flatten to the same pytree structure, so the fold traces once."""
    with_flag = _build_mapping(gate=_consent_gate, supply_flag=True)
    without_flag = _build_mapping(gate=_consent_gate, supply_flag=False)

    assert jax.tree_util.tree_structure(without_flag) == jax.tree_util.tree_structure(
        with_flag
    )


def test_stand_in_flag_has_the_shape_and_dtype_of_a_supplied_one():
    """The stand-in flag is shaped like the flag the target would publish.

    A jitted fold is traced per argument shape as well as per pytree structure,
    so a stand-in of some other shape would retrace the fold just as a missing
    key would.
    """
    key = f"married_terminal{D_KEY_SUFFIX}"
    supplied = _build_mapping(gate=_consent_gate, supply_flag=True)[key]
    stand_in = _build_mapping(gate=_consent_gate, supply_flag=False)[key]

    assert (stand_in.shape, stand_in.dtype) == (supplied.shape, supplied.dtype)


def test_gate_reading_the_flag_without_one_supplied_is_refused():
    """A gate consuming `D_target` with no flag supplied names how to supply it.

    Standing in for the flag here would answer "nobody dissolved" to a gate
    asking who did, so the mapping refuses to build instead.
    """
    with pytest.raises(
        NotImplementedError, match="period_to_regime_to_dissolution_flags"
    ):
        _build_mapping(gate=_dissolution_gate, supply_flag=False)


def _build_mapping(*, gate, supply_flag: bool) -> MappingProxyType:
    """Assemble the fold's same-period mapping for a one-leg collective edge.

    Args:
        gate: The edge's gate predicate.
        supply_flag: Whether the target regime published a dissolution flag.

    Returns:
        Immutable mapping of regime name to the arrays the fold reads.

    """
    edge = ResolvedGatedEdge(
        target="married_terminal",
        gate=gate,
        gate_refs={},
        legs=(
            ResolvedEdgeLeg(
                source_stakeholder=None,
                target_component_index=0,
                fallback=ResolvedSamePeriodRef(
                    regime="single_f",
                    projection={"wage": _identity_wage},
                    stakeholder_index=None,
                ),
            ),
        ),
        reference_regimes=("single_f",),
    )
    return build_same_period_mapping_for_fold(
        edge=edge,
        period_solution=MappingProxyType(
            {"married_terminal": _TARGET_V, "single_f": _FALLBACK_V}
        ),
        period_dissolution_flags=MappingProxyType(
            {"married_terminal": _TARGET_D} if supply_flag else {}
        ),
    )


def _consent_gate(V_target_f: FloatND, V_target_m: FloatND) -> BoolND:
    """Consent holds where both partners are worth more together than apart."""
    return (V_target_f > 0.0) & (V_target_m > 0.0)


def _dissolution_gate(D_target: BoolND) -> BoolND:
    """The couple continues exactly where the household did not dissolve."""
    return ~D_target


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    """A partner keeps the household wage on entering her own regime."""
    return wage
