"""Lower the actual forward families from retained metadata, without populations."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from dags.tree import qname_from_tree_path

from _lcm.dtypes import canonical_float_dtype
from _lcm.engine import Regime, StateActionSpace
from _lcm.execution.core_program import CoreProgram
from _lcm.grids import DiscreteGrid
from _lcm.regime_building.Q_and_F import SAME_PERIOD_PARAMS_ARG, SAME_PERIOD_V_ARG
from _lcm.simulation.operand_placement import subject_operand_sharding
from _lcm.simulation.program_arguments import decision_arguments, transition_arguments
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.typing import FlatParams
from lcm.ages import AgeGrid
from lcm.exceptions import ExecutionPlanningError


@dataclass(frozen=True, kw_only=True)
class AbstractSimulationProfile:
    """Actual code plus abstract inputs; output shape comes from compiled.out_info."""

    executable: jax.stages.Compiled
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        """Reject caller owners and snapshot only their immutable descriptors."""
        if any(
            not isinstance(leaf, jax.ShapeDtypeStruct)
            for leaf in jax.tree.leaves(self.arguments)
        ):
            raise ExecutionPlanningError("Forward profiles require abstract arguments.")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@dataclass(frozen=True, kw_only=True)
class ForwardProgramProfile(AbstractSimulationProfile):
    """A declared core and its separately profiled action decoder when present."""

    action_decoder: AbstractSimulationProfile | None = None


def profile_forward_programs(
    *,
    runtime: SimulationRuntime,
    regimes: Mapping[str, Regime],
    flat_params: FlatParams,
    base_spaces: Mapping[str, StateActionSpace],
    values: Mapping[int, Mapping[str, jax.Array]],
    ages: AgeGrid,
    n_subjects: int,
    widths: Mapping[str, int],
    ordinary_key: jax.ShapeDtypeStruct,
    taste_key: jax.ShapeDtypeStruct | None,
) -> Mapping[tuple[int, str, str], ForwardProgramProfile]:
    """Prepare canonical entry descriptors for focused core-level profiling.

    Complete chunks call profile_forward_unit with their evolving carrier instead:
    this entry-only convenience does not model state updates between programs.
    """
    subject = subject_operand_sharding(devices=runtime.subject_devices)
    profiles = {}
    for name, regime in regimes.items():
        columns = {
            state: jax.ShapeDtypeStruct(
                (n_subjects,),
                np.dtype(
                    jnp.int32
                    if isinstance(regime.simulation.grids[state], DiscreteGrid)
                    else canonical_float_dtype()
                ),
                sharding=subject,
            )
            for state in regime.simulation.state_names
        }
        for period in regime.simulation.programs.decision:
            unit = profile_forward_unit(
                runtime=runtime,
                regime=regime,
                name=name,
                period=period,
                flat_params=flat_params,
                base=base_spaces[name],
                values=values,
                ages=ages,
                n_subjects=n_subjects,
                widths=widths,
                columns=columns,
                ordinary_key=ordinary_key,
                taste_key=taste_key,
            )
            profiles.update(
                {(period, name, family): profile for family, profile in unit.items()}
            )
    return MappingProxyType(profiles)


def profile_forward_unit(
    *,
    runtime: SimulationRuntime,
    regime: Regime,
    name: str,
    period: int,
    flat_params: FlatParams,
    base: StateActionSpace,
    values: Mapping[int, Mapping[str, jax.Array]],
    ages: AgeGrid,
    n_subjects: int,
    widths: Mapping[str, int],
    columns: Mapping[str, jax.ShapeDtypeStruct],
    ordinary_key: jax.ShapeDtypeStruct,
    taste_key: jax.ShapeDtypeStruct | None,
) -> Mapping[str, ForwardProgramProfile]:
    """Bind one actual current carrier to its decision, transition and route.

    All three families consume the pre-advance source states. The caller advances
    its complete carrier from the real merge out_info before profiling the next
    regime, including a successor later in this same period.
    """
    # The dispatch module imports the complete chunk-profile constructor.
    from _lcm.simulation.simulate import _lookup_values_from_indices  # noqa: PLC0415

    for descriptor in (ordinary_key, taste_key):
        if descriptor is not None and (
            descriptor.shape != ()
            or descriptor.sharding is None
            or not jax.dtypes.issubdtype(descriptor.dtype, jax.dtypes.prng_key)
        ):
            raise ExecutionPlanningError(
                "Forward key profiles require placed scalar PRNG descriptors."
            )
    if (
        regime.gated_edges
        or regime.simulation.replay_route.policy_applicable
        or regime.simulation.external_replay_route is not None
    ):
        raise ExecutionPlanningError(
            "Forward profiles require compiled decision routes."
        )
    if set(columns) != set(regime.simulation.state_names) or any(
        not isinstance(leaf, jax.ShapeDtypeStruct)
        or leaf.shape != (n_subjects,)
        or leaf.sharding is None
        for leaf in columns.values()
    ):
        raise ExecutionPlanningError(
            "Forward unit profiles require the complete placed current state carrier."
        )
    devices = runtime.subject_devices
    subject = subject_operand_sharding(devices=devices)
    shared = simulation_value_sharding(stored_sharding=subject, devices=devices)
    current = {
        state: _placed_abstract(leaf=leaf, sharding=subject)
        for state, leaf in columns.items()
    }
    states = {state: current[state] for state in base.states}
    carried = {state: current[state] for state in regime.simulation.carried_grids}
    params = _shared_tree(tree=flat_params[name], devices=devices)
    age = jax.ShapeDtypeStruct((), ages.values.dtype, sharding=shared)
    period_value = jax.ShapeDtypeStruct((), np.dtype(np.int32), sharding=shared)
    next_values = (
        MappingProxyType(
            {
                target: values[period + 1][target]
                for target in regime.solution.reachability.targets(
                    period=period, source=name
                )
            }
        )
        if period + 1 < ages.n_periods
        else MappingProxyType({})
    )
    references = {}
    if regime.same_period_ref_regimes:
        references[SAME_PERIOD_V_ARG] = MappingProxyType(
            {ref: values[period][ref] for ref in regime.same_period_ref_regimes}
        )
        references[SAME_PERIOD_PARAMS_ARG] = MappingProxyType(
            {ref: flat_params[ref] for ref in regime.same_period_ref_regimes}
        )
    subject_key = jax.ShapeDtypeStruct(
        (n_subjects,), ordinary_key.dtype, sharding=subject
    )
    decision_key = jax.ShapeDtypeStruct(
        (n_subjects,),
        (ordinary_key if taste_key is None else taste_key).dtype,
        sharding=subject,
    )
    arguments = decision_arguments(
        states=states,
        discrete_actions=_shared_tree(tree=base.discrete_actions, devices=devices),
        continuous_actions=_shared_tree(tree=base.continuous_actions, devices=devices),
        taste_keys={"taste_shock_key": decision_key} if regime.has_taste_shocks else {},
        next_values=_shared_tree(tree=next_values, devices=devices),
        references=_shared_tree(tree=references, devices=devices),
        params=params,
        period=period_value,
        age=age,
    )
    decision = regime.simulation.programs.decision[period]
    decision_profile = _profile_program(
        runtime=runtime,
        program=decision,
        arguments=arguments,
        period=period,
        n_subjects=n_subjects,
        widths=widths,
    )
    indices, _ = decision_profile.executable.out_info
    if regime.stakeholders is not None and not states:
        indices = jax.ShapeDtypeStruct((n_subjects,), indices.dtype, sharding=subject)
    decoder_arguments = {
        "flat_indices": _placed_abstract(leaf=indices, sharding=subject),
        "grids": _shared_tree(tree=base.actions, devices=devices),
    }
    decoded = runtime.operations.prepare_abstract(
        function=_lookup_values_from_indices,
        arguments=decoder_arguments,
        subject_arg_names=("flat_indices",),
        devices=devices,
    )
    profiles = {
        "decision": replace(
            decision_profile,
            action_decoder=AbstractSimulationProfile(
                executable=decoded.executable, arguments=decoder_arguments
            ),
        )
    }
    actions = decoded.executable.out_info
    for family in ("transition", "route"):
        program = getattr(regime.simulation.programs, family).get(period)
        if program is None:
            continue
        keys = (
            _stochastic_keys(regime=regime, key=subject_key)
            if family == "transition"
            else {}
        )
        arguments = transition_arguments(
            states=states,
            carried=carried,
            actions=actions,
            keys=keys,
            period=period_value,
            age=age,
            params=params,
        )
        profiles[family] = _profile_program(
            runtime=runtime,
            program=program,
            arguments=arguments,
            period=period,
            n_subjects=n_subjects,
            widths=widths,
        )
    return MappingProxyType(profiles)


def _profile_program(
    *,
    runtime: SimulationRuntime,
    program: CoreProgram,
    arguments: Mapping[str, object],
    period: int,
    n_subjects: int,
    widths: Mapping[str, int],
) -> ForwardProgramProfile:
    """Resolve the same declared axes as dispatch at this concrete chunk extent."""
    concrete_widths = {
        axis.name: min(
            widths.get(
                axis.name, n_subjects if axis.name == "subject" else axis.extent
            ),
            n_subjects if axis.name == "subject" else axis.extent,
        )
        for axis in program.requirements.axes
        if axis.name != "subject" or n_subjects > 1
    }
    compiled = runtime.prepare_abstract(
        program=program,
        arguments=arguments,
        period=period,
        n_subjects=n_subjects,
        widths=concrete_widths,
    )
    return ForwardProgramProfile(
        executable=cast("jax.stages.Compiled", compiled.executable), arguments=arguments
    )


def _stochastic_keys(
    *, regime: Regime, key: jax.ShapeDtypeStruct
) -> Mapping[str, jax.ShapeDtypeStruct]:
    """Use the same canonical stochastic transition key names as runtime."""
    names = sorted(
        qname_from_tree_path((target, name))
        for target, bundle in regime.simulation.transitions.items()
        for name in bundle
        if regime.simulation.transition_plans[target].is_lottery(name)
    )
    return {f"key_{name}": key for name in names}


def _shared_tree(
    *, tree: Mapping[str, object], devices: tuple[jax.Device, ...]
) -> Mapping[str, object]:
    """Project actual retained metadata to the same shared destination layout."""

    def abstract(leaf: object) -> jax.ShapeDtypeStruct:
        if not isinstance(leaf, jax.Array):
            raise ExecutionPlanningError(
                "Forward retained inputs must be canonical JAX arrays."
            )
        return _placed_abstract(
            leaf=leaf,
            sharding=simulation_value_sharding(
                stored_sharding=leaf.sharding, devices=devices
            ),
        )

    return jax.tree.map(abstract, tree)


def _placed_abstract(
    *, leaf: jax.Array | jax.ShapeDtypeStruct, sharding: jax.sharding.Sharding
) -> jax.ShapeDtypeStruct:
    """Keep shape, dtype and weak type while declaring the required placement."""
    return jax.ShapeDtypeStruct(
        leaf.shape, leaf.dtype, sharding=sharding, weak_type=leaf.weak_type
    )
