"""Helper module for regime class initialization and utilities."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax

from lcm.dispatchers import vmap_1d
from lcm.input_processing.util import get_grids, get_variable_info
from lcm.interfaces import (
    InternalFunctions,
    StateActionSpace,
    StateSpaceInfo,
    Target,
    TerminalNonTerminal,
)
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.next_state import get_next_state_function
from lcm.Q_and_F import get_Q_and_F
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)

if TYPE_CHECKING:
    from jax import Array

    from lcm.regime import Regime
    from lcm.typing import (
        ArgmaxQOverAFunction,
        MaxQOverAFunction,
        NextStateSimulationFunction,
        QAndFFunction,
    )


def build_state_space_infos(regime: Regime) -> TerminalNonTerminal[StateSpaceInfo]:
    terminal_ssi = create_state_space_info(
        regime=regime,
        is_last_period=True,
    )

    non_terminal_ssi = create_state_space_info(
        regime=regime,
        is_last_period=False,
    )

    return TerminalNonTerminal(terminal=terminal_ssi, non_terminal=non_terminal_ssi)


def build_state_action_spaces(
    regime: Regime,
) -> TerminalNonTerminal[StateActionSpace]:
    variable_info = get_variable_info(regime)
    grids = get_grids(regime)

    terminal_sas = create_state_action_space(
        variable_info=variable_info,
        grids=grids,
        is_last_period=True,
    )

    non_terminal_sas = create_state_action_space(
        variable_info=variable_info,
        grids=grids,
        is_last_period=False,
    )

    return TerminalNonTerminal(terminal=terminal_sas, non_terminal=non_terminal_sas)


def build_Q_and_F_functions(
    regime: Regime,
    internal_functions: InternalFunctions,
) -> TerminalNonTerminal[QAndFFunction]:
    state_space_infos = build_state_space_infos(regime)

    # Create Q and F functions
    Q_and_F_terminal: QAndFFunction = get_Q_and_F(
        regime=regime,
        internal_functions=internal_functions,
        next_state_space_info=state_space_infos.terminal,
        is_last_period=True,
    )
    Q_and_F_non_terminal: QAndFFunction = get_Q_and_F(
        regime=regime,
        internal_functions=internal_functions,
        next_state_space_info=state_space_infos.non_terminal,
        is_last_period=False,
    )
    return TerminalNonTerminal(
        terminal=Q_and_F_terminal, non_terminal=Q_and_F_non_terminal
    )


def build_max_Q_over_a_functions(
    regime: Regime,
    Q_and_F_functions: TerminalNonTerminal[QAndFFunction],
    *,
    enable_jit: bool,
) -> TerminalNonTerminal[MaxQOverAFunction]:
    state_action_spaces = build_state_action_spaces(regime)

    max_Q_over_a_functions = {}

    for is_terminal in (True, False):
        state_action_space = state_action_spaces(is_terminal=is_terminal)

        max_Q_over_a = get_max_Q_over_a(
            Q_and_F=Q_and_F_functions(is_terminal=is_terminal),
            actions_names=state_action_space.actions_names,
            states_names=state_action_space.states_names,
        )

        if enable_jit:
            max_Q_over_a = jax.jit(max_Q_over_a)

        max_Q_over_a_functions[is_terminal] = max_Q_over_a

    return TerminalNonTerminal(
        terminal=max_Q_over_a_functions[True],
        non_terminal=max_Q_over_a_functions[False],
    )


def build_argmax_and_max_Q_over_a_functions(
    regime: Regime,
    Q_and_F_functions: TerminalNonTerminal[QAndFFunction],
    *,
    enable_jit: bool,
) -> TerminalNonTerminal[ArgmaxQOverAFunction]:
    state_action_spaces = build_state_action_spaces(regime)

    argmax_and_max_Q_over_a_functions = {}

    for is_terminal in (True, False):
        state_action_space = state_action_spaces(is_terminal=is_terminal)

        argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F_functions(is_terminal=is_terminal),
            actions_names=state_action_space.actions_names,
        )

        if enable_jit:
            argmax_and_max_Q_over_a = jax.jit(argmax_and_max_Q_over_a)

        argmax_and_max_Q_over_a_functions[is_terminal] = argmax_and_max_Q_over_a

    return TerminalNonTerminal(
        terminal=argmax_and_max_Q_over_a_functions[True],
        non_terminal=argmax_and_max_Q_over_a_functions[False],
    )


def build_next_state_simulation_functions(
    regime: Regime,
    internal_functions: InternalFunctions,
    grids: dict[str, Array],
    *,
    enable_jit: bool,
) -> NextStateSimulationFunction:
    state_action_spaces = build_state_action_spaces(regime)
    next_state = get_next_state_function(
        internal_functions=internal_functions,
        grids=grids,
        next_states=state_action_spaces.non_terminal.states_names,
        target=Target.SIMULATE,
    )
    signature = inspect.signature(next_state)
    parameters = list(signature.parameters)

    next_state_vmapped = vmap_1d(
        func=next_state,
        variables=tuple(
            parameter
            for parameter in parameters
            if parameter not in ["period", "params"]
        ),
    )
    return jax.jit(next_state_vmapped) if enable_jit else next_state_vmapped
