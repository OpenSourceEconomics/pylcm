from jax import Array

from lcm.interfaces import InternalRegime
from lcm.state_action_space import create_state_action_space
from lcm.typing import Int1D, RegimeName


def get_regime_name_to_id_mapping(internal_regimes):
    return {
        regime.name: i
        for regime, i in zip(
            internal_regimes.values(), range(len(internal_regimes)), strict=False
        )
    }


def create_regime_state_action_space(
    regime_name: RegimeName,
    states: dict[str, Array],
    internal_regime: InternalRegime,
    subjects_in_regime: Int1D,
    *,
    is_last_period: bool,
):
    if is_last_period:
        query = "is_state and enters_concurrent_valuation"
    else:
        query = "is_state and (enters_concurrent_valuation | enters_transition)"

    # Create state action space with current subjects
    states_for_state_action_space = {
        state_name: states[f"{regime_name}__{state_name}"][subjects_in_regime]
        for state_name in internal_regime.variable_info.query(query).index
    }
    return create_state_action_space(
        variable_info=internal_regime.variable_info,
        grids=internal_regime.grids,
        states=states_for_state_action_space,
        is_last_period=is_last_period,
    )
