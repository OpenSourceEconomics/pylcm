"""A period's simulated value array has one row per subject, and at most a role axis.

`PeriodRegimeSimulationData.V_arr` is rank-polymorphic because a collective
regime publishes a value per stakeholder: `(n_subjects,)` for a singleton
regime, `(n_subjects, n_stakeholders)` for a collective one. A rank-polymorphic
annotation accepts any rank, so the shape agreement that used to be a property
of the annotation is asserted here instead — every other field of the record is
per-subject, and a `V_arr` that disagrees with them silently misaligns every
value in the simulated frame.
"""

import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.engine import PeriodRegimeSimulationData

_N_SUBJECTS = 4
_N_STAKEHOLDERS = 2


def _period_data(*, V_arr: jnp.ndarray) -> PeriodRegimeSimulationData:
    """One period's record for `_N_SUBJECTS` subjects, carrying `V_arr`."""
    column = jnp.arange(_N_SUBJECTS, dtype=jnp.zeros(()).dtype)
    return PeriodRegimeSimulationData(
        V_arr=V_arr,
        actions=MappingProxyType({"consumption": column}),
        states=MappingProxyType({"wealth": column}),
        in_regime=jnp.arange(_N_SUBJECTS) % 2 == 0,
        nested_policy_fallback=jnp.zeros(_N_SUBJECTS, dtype=bool),
    )


def test_a_singleton_regimes_value_array_is_one_value_per_subject():
    """A rank-1 `V_arr` with one entry per subject is accepted."""
    data = _period_data(V_arr=jnp.zeros(_N_SUBJECTS))
    assert data.V_arr.shape == (_N_SUBJECTS,)


def test_a_collective_regimes_value_array_carries_a_trailing_role_axis():
    """A rank-2 `V_arr` of subjects by stakeholders is accepted."""
    data = _period_data(V_arr=jnp.zeros((_N_SUBJECTS, _N_STAKEHOLDERS)))
    assert data.V_arr.shape == (_N_SUBJECTS, _N_STAKEHOLDERS)


def test_a_value_array_with_more_than_one_trailing_axis_is_rejected():
    """A rank-3 `V_arr` names no axis a simulated record has."""
    with pytest.raises(ValueError, match="V_arr"):
        _period_data(V_arr=jnp.zeros((_N_SUBJECTS, _N_STAKEHOLDERS, 3)))


def test_a_value_array_whose_leading_axis_is_not_subjects_is_rejected():
    """`V_arr`'s first axis must match the per-subject `in_regime` mask."""
    with pytest.raises(ValueError, match="V_arr"):
        _period_data(V_arr=jnp.zeros(_N_SUBJECTS + 1))


def test_a_scalar_value_array_is_rejected():
    """A 0-d `V_arr` carries no subject axis at all."""
    with pytest.raises(ValueError, match="V_arr"):
        _period_data(V_arr=jnp.zeros(()))


def test_the_record_still_populates_every_declared_field():
    """The guard reads only fields the dataclass declares, so it cannot go stale."""
    names = {field.name for field in dataclasses.fields(PeriodRegimeSimulationData)}
    assert {"V_arr", "in_regime"} <= names
