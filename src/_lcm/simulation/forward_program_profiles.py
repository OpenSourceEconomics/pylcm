"""Lower the actual forward families from retained metadata, without populations."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from dags.tree import qname_from_tree_path

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.engine import Regime, StateActionSpace
from _lcm.execution.core_program import CoreProgram
from _lcm.execution.workspace_planning import CompilerMemoryReservation
from _lcm.grids import DiscreteGrid
from _lcm.params.edges import regime_kernel_params
from _lcm.regime_building.Q_and_F import (
    EDGE_REF_PARAMS_ARG,
    EDGE_REF_V_ARG,
    SAME_PERIOD_PARAMS_ARG,
    SAME_PERIOD_V_ARG,
)
from _lcm.simulation.operand_placement import subject_operand_sharding
from _lcm.simulation.policy_programs import ReplayPayload
from _lcm.simulation.program_arguments import (
    decision_arguments,
    gate_fold_arguments,
    policy_prepare_arguments,
    policy_rank_arguments,
    transition_arguments,
)
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.subject_groups import type_local_template
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.solution.backward_induction import CompilationWave, _states_for_period
from _lcm.time import TimeAxis
from _lcm.typing import (
    FlatParams,
    PytreeValue,
    QualifiedName,
    ShapeDtypePytree,
    SimulationPolicy,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import ActionName, ReferenceName, RegimeName

# Output descriptors of the finite ranking stage: the chosen actions, the value
# they attain, and the nested-policy fallback flag.
type _FiniteDecisionOutput = tuple[
    Mapping[ActionName, jax.ShapeDtypeStruct],
    jax.ShapeDtypeStruct,
    jax.ShapeDtypeStruct,
]


@dataclass(frozen=True, kw_only=True)
class AbstractSimulationProfile:
    """Actual code plus abstract inputs; output shape comes from compiled.out_info.

    `memory` is the exact executable's compiler report, already read once at
    compilation by the runtime's own executable cache
    (`CompiledSimulationProgram.memory` / `_ProfiledOperation.memory`). It is
    carried here so chunk-profile construction adopts it instead of rereading
    it from `executable` on every candidate.
    """

    executable: jax.stages.Compiled
    arguments: Mapping[ReferenceName, ShapeDtypePytree]
    memory: CompilerMemoryReservation

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
    flags: Mapping[int, Mapping[str, jax.Array]] = MappingProxyType({}),
    ages: TimeAxis,
    n_subjects: int,
    widths: Mapping[str, int],
    ordinary_key: jax.ShapeDtypeStruct,
    taste_key: jax.ShapeDtypeStruct | None,
    policies: Mapping[int, Mapping[RegimeName, SimulationPolicy]] | None = None,
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
                regimes=regimes,
                regime=regime,
                name=name,
                period=period,
                flat_params=flat_params,
                base=base_spaces[name],
                base_spaces=base_spaces,
                values=values,
                flags=flags,
                ages=ages,
                n_subjects=n_subjects,
                widths=widths,
                columns=columns,
                ordinary_key=ordinary_key,
                taste_key=taste_key,
                policy=(policies or {}).get(period, {}).get(name),
            )
            profiles.update(
                {(period, name, family): profile for family, profile in unit.items()}
            )
    return MappingProxyType(profiles)


def profile_forward_unit(  # noqa: C901, PLR0912, PLR0915
    *,
    runtime: SimulationRuntime,
    regimes: Mapping[str, Regime],
    regime: Regime,
    name: str,
    period: int,
    flat_params: FlatParams,
    base: StateActionSpace,
    base_spaces: Mapping[str, StateActionSpace],
    values: Mapping[int, Mapping[str, jax.Array]],
    flags: Mapping[int, Mapping[str, jax.Array]],
    ages: TimeAxis,
    n_subjects: int,
    widths: Mapping[str, int],
    columns: Mapping[str, jax.ShapeDtypeStruct],
    ordinary_key: jax.ShapeDtypeStruct,
    taste_key: jax.ShapeDtypeStruct | None,
    policy: SimulationPolicy | None = None,
    wave: CompilationWave | None = None,
) -> Mapping[str, ForwardProgramProfile]:
    """Bind one actual current carrier to its decision, transition and route.

    All three families consume the pre-advance source states. The caller advances
    its complete carrier from the real merge out_info before profiling the next
    regime, including a successor later in this same period.

    With a `wave`, every program the cache lacks is lowered into it instead of
    compiled here, each program's inputs following from the lowered outputs of
    the one before; the wave compiles and publishes them to the runtime's cache,
    and no profile is returned.
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
        regime.simulation.replay_route.policy_applicable
        and regime.simulation.replay_route.consumer_route != "nnbegm_finite"
    ) or regime.simulation.external_replay_route is not None:
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
    params = _shared_tree(
        tree=regime_kernel_params(flat_params, regime_name=name), devices=devices
    )
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
    profiles: dict[str, ForwardProgramProfile] = {}
    if period in regime.simulation.programs.gate_fold:
        edge_values = MappingProxyType(
            {
                target: MappingProxyType(
                    {
                        reference: values[period + 1][reference]
                        for reference in dict.fromkeys(
                            (target, *edge.reference_regimes)
                        )
                    }
                )
                for target, edge in regime.gated_edges.items()
                if target in values.get(period + 1, {})
            }
        )
        edge_flags = MappingProxyType(
            {
                target: flags[period + 1][target]
                for target in edge_values
                if target in flags.get(period + 1, {})
            }
        )
        fold_arguments = {
            **gate_fold_arguments(
                edge_values=_shared_tree(tree=edge_values, devices=devices),
                edge_flags=_shared_tree(tree=edge_flags, devices=devices),
                flat_params=_shared_tree(tree=flat_params, devices=devices),
                fold_age=age,
            ),
            "next_regime_to_V_arr": _shared_tree(tree=next_values, devices=devices),
            "target_states_by_target": _shared_tree(
                tree=MappingProxyType(
                    {
                        target: _states_for_period(
                            regime=regimes[target],
                            state_action_space=base_spaces[target],
                            period=period + 1,
                        )
                        for target in edge_values
                    }
                ),
                devices=devices,
            ),
        }
        folded = _prepare_program(
            runtime=runtime,
            program=regime.simulation.programs.gate_fold[period],
            arguments=fold_arguments,
            period=period,
            n_subjects=n_subjects,
            widths=widths,
            family="gate_fold",
            regime_name=name,
            profiles=profiles,
            wave=wave,
            feeds_forward=True,
        )
        next_values = MappingProxyType(
            {**next_values, **cast("Mapping[RegimeName, ShapeDtypePytree]", folded)}
        )
    references = {}
    if regime.same_period_ref_regimes:
        references[SAME_PERIOD_V_ARG] = MappingProxyType(
            {ref: values[period][ref] for ref in regime.same_period_ref_regimes}
        )
        references[SAME_PERIOD_PARAMS_ARG] = MappingProxyType(
            {
                ref: regime_kernel_params(flat_params, regime_name=ref)
                for ref in regime.same_period_ref_regimes
            }
        )
    edge_references = regime.simulation.edge_reference_regimes_by_period.get(period)
    if edge_references is not None:
        references[EDGE_REF_V_ARG] = MappingProxyType(
            {ref: values[period + 1][ref] for ref in edge_references}
        )
        references[EDGE_REF_PARAMS_ARG] = MappingProxyType(
            {
                ref: regime_kernel_params(flat_params, regime_name=ref)
                for ref in edge_references
            }
        )
    subject_key = jax.ShapeDtypeStruct(
        (n_subjects,), ordinary_key.dtype, sharding=subject
    )
    decision_key = jax.ShapeDtypeStruct(
        (n_subjects,),
        (ordinary_key if taste_key is None else taste_key).dtype,
        sharding=subject,
    )
    if regime.simulation.replay_route.consumer_route == "nnbegm_finite":
        actions, _, _ = cast(
            "_FiniteDecisionOutput",
            _profile_finite_decision(
                runtime=runtime,
                regime=regime,
                name=name,
                profiles=profiles,
                wave=wave,
                period=period,
                n_subjects=n_subjects,
                widths=widths,
                policy=policy,
                states=current,
                canonical_states=states,
                params=params,
                age=age,
                next_values=_shared_tree(tree=next_values, devices=devices),
                references=_shared_tree(tree=references, devices=devices),
            ),
        )
    else:
        # A grouped decision reads each continuation carrying the grouping
        # state through one code's block, which has that axis removed.
        grouping = regime.simulation.programs.grouping
        decision_values = MappingProxyType(
            {
                target: type_local_template(
                    route=grouping,
                    regime=target,
                    leaf=cast("jax.Array | jax.ShapeDtypeStruct", leaf),
                )
                for target, leaf in next_values.items()
            }
        )
        arguments = decision_arguments(
            states=states,
            discrete_actions=_shared_tree(tree=base.discrete_actions, devices=devices),
            continuous_actions=_shared_tree(
                tree=base.continuous_actions, devices=devices
            ),
            taste_keys={"taste_shock_key": decision_key}
            if regime.has_taste_shocks
            else {},
            next_values=_shared_tree(tree=decision_values, devices=devices),
            references=_shared_tree(tree=references, devices=devices),
            params=params,
            period=period_value,
            age=age,
        )
        indices, _ = cast(
            "tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]",
            _prepare_program(
                runtime=runtime,
                program=regime.simulation.programs.forward_decision[period],
                arguments=arguments,
                period=period,
                n_subjects=n_subjects,
                widths=widths,
                family="decision",
                regime_name=name,
                profiles=profiles,
                wave=wave,
            ),
        )
        if not states and indices.ndim == 0:
            indices = jax.ShapeDtypeStruct(
                (n_subjects,), indices.dtype, sharding=subject
            )
        decoder_arguments = {
            "flat_indices": _placed_abstract(
                leaf=indices, sharding=subject if indices.ndim else shared
            ),
            "grids": _shared_tree(tree=base.actions, devices=devices),
        }
        subject_arg_names = ("flat_indices",) if indices.ndim else ()
        if wave is None:
            decoded = runtime.operations.prepare_abstract(
                function=_lookup_values_from_indices,
                arguments=decoder_arguments,
                subject_arg_names=subject_arg_names,
                devices=devices,
            )
            profiles["decision"] = replace(
                profiles["decision"],
                action_decoder=AbstractSimulationProfile(
                    executable=decoded.executable,
                    arguments=decoder_arguments,
                    memory=decoded.memory,
                ),
            )
            actions = decoded.executable.out_info
        else:
            actions = cast(
                "Mapping[ActionName, ShapeDtypePytree]",
                runtime.operations.lower_abstract(
                    function=_lookup_values_from_indices,
                    arguments=decoder_arguments,
                    subject_arg_names=subject_arg_names,
                    devices=devices,
                    wave=wave,
                    label=f"{name} action decoder (period {period})",
                ),
            )
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
        _prepare_program(
            runtime=runtime,
            program=program,
            arguments=arguments,
            period=period,
            n_subjects=n_subjects,
            widths=widths,
            family=family,
            regime_name=name,
            profiles=profiles,
            wave=wave,
        )
    return MappingProxyType(profiles)


def _profile_finite_decision(
    *,
    runtime: SimulationRuntime,
    regime: Regime,
    name: str,
    profiles: dict[str, ForwardProgramProfile],
    wave: CompilationWave | None,
    period: int,
    n_subjects: int,
    widths: Mapping[str, int],
    policy: SimulationPolicy | None,
    states: Mapping[str, jax.ShapeDtypeStruct],
    canonical_states: Mapping[str, jax.ShapeDtypeStruct],
    params: Mapping[QualifiedName, ShapeDtypePytree],
    age: jax.ShapeDtypeStruct,
    next_values: Mapping[RegimeName, ShapeDtypePytree],
    references: Mapping[ReferenceName, ShapeDtypePytree],
) -> ShapeDtypePytree:
    """Use actual payload metadata and the preparation stage's bank schema.

    Returns:
        The ranking stage's output descriptors.

    """
    programs = regime.simulation.programs
    if (
        not isinstance(policy, NNBEGMSimPolicy)
        or period not in programs.policy_prepare
        or period not in programs.policy_rank
        or programs.decision[period] is not programs.policy_rank[period]
    ):
        raise ExecutionPlanningError(
            "Finite profiles require their declared policy stages."
        )
    payload = ReplayPayload.from_policy(policy)
    if not all(isinstance(leaf, jax.Array) for leaf in payload.arrays):
        raise ExecutionPlanningError(
            "Finite profiles require retained canonical JAX policy leaves."
        )
    abstract_payload = jax.tree.map(
        partial(_abstract_policy_leaf, devices=runtime.subject_devices), payload
    )
    bank = _prepare_program(
        runtime=runtime,
        program=programs.policy_prepare[period],
        arguments=policy_prepare_arguments(
            payload=abstract_payload, states=states, params=params, age=age
        ),
        period=period,
        n_subjects=n_subjects,
        widths=widths,
        family="policy_prepare",
        regime_name=name,
        profiles=profiles,
        wave=wave,
        feeds_forward=True,
    )
    return _prepare_program(
        runtime=runtime,
        program=programs.policy_rank[period],
        arguments=policy_rank_arguments(
            payload=abstract_payload,
            bank=bank,
            canonical_states=canonical_states,
            params=params,
            age=age,
            next_values=next_values,
            references=references,
        ),
        period=period,
        n_subjects=n_subjects,
        widths=widths,
        family="decision",
        regime_name=name,
        profiles=profiles,
        wave=wave,
        feeds_forward=True,
    )


# keyword-only-exempt: library-callback=jax.tree.map
def _abstract_policy_leaf(
    leaf: jax.Array, *, devices: tuple[jax.Device, ...]
) -> jax.ShapeDtypeStruct:
    """Bind only required device metadata while abstracting a retained policy leaf."""
    return _placed_abstract(
        leaf=leaf,
        sharding=simulation_value_sharding(
            stored_sharding=leaf.sharding, devices=devices
        ),
    )


def _prepare_program(
    *,
    runtime: SimulationRuntime,
    program: CoreProgram,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    period: int,
    n_subjects: int,
    widths: Mapping[str, int],
    family: str,
    regime_name: RegimeName,
    profiles: dict[str, ForwardProgramProfile],
    wave: CompilationWave | None,
    feeds_forward: bool = False,
) -> ShapeDtypePytree:
    """Compile a forward program here, or lower it into `wave`.

    A compiled program's profile is stored in `profiles` under `family`. A lowered
    one is named in the wave's logs and error notes by regime, family, period and
    widths. When a later program of the unit reads this one's outputs unplaced,
    the caller sets `feeds_forward`: the wave then waits for the compile, so the
    descriptors carry the compiler's shardings, as the ordered walk sees them.

    Returns:
        The program's output descriptors.

    """
    if wave is None:
        profile = _profile_program(
            runtime=runtime,
            program=program,
            arguments=arguments,
            period=period,
            n_subjects=n_subjects,
            widths=widths,
        )
        profiles[family] = profile
        return profile.executable.out_info
    concrete_widths = _concrete_widths(
        program=program, n_subjects=n_subjects, widths=widths
    )
    return runtime.lower_abstract(
        program=program,
        arguments=arguments,
        period=period,
        n_subjects=n_subjects,
        widths=concrete_widths,
        wave=wave,
        label=(f"{regime_name} {family} (period {period}, widths={concrete_widths!r})"),
        wait=feeds_forward,
    )


def _profile_program(
    *,
    runtime: SimulationRuntime,
    program: CoreProgram,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    period: int,
    n_subjects: int,
    widths: Mapping[str, int],
) -> ForwardProgramProfile:
    """Resolve the same declared axes as dispatch at this concrete chunk extent."""
    compiled = runtime.prepare_abstract(
        program=program,
        arguments=arguments,
        period=period,
        n_subjects=n_subjects,
        widths=_concrete_widths(program=program, n_subjects=n_subjects, widths=widths),
    )
    if compiled.memory is None:
        raise ExecutionPlanningError(
            "Chunk profiling requires a compiled executable with cached "
            "compiler memory accounting."
        )
    return ForwardProgramProfile(
        executable=cast("jax.stages.Compiled", compiled.executable),
        # Preparation refused any argument leaf that is not a shape descriptor.
        arguments=cast("Mapping[ReferenceName, ShapeDtypePytree]", arguments),
        memory=compiled.memory,
    )


def _concrete_widths(
    *, program: CoreProgram, n_subjects: int, widths: Mapping[str, int]
) -> dict[str, int]:
    """Clamp the chunk's widths to each declared axis at this subject extent."""
    return {
        axis.name: min(
            widths.get(
                axis.name, n_subjects if axis.name == "subject" else axis.extent
            ),
            n_subjects if axis.name == "subject" else axis.extent,
        )
        for axis in program.requirements.axes
        if axis.name != "subject" or n_subjects > 1
    }


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


# keyword-only-exempt: library-callback=jax.tree.map
def _shared_leaf(
    leaf: jax.Array | jax.ShapeDtypeStruct, *, devices: tuple[jax.Device, ...]
) -> jax.ShapeDtypeStruct:
    """Project one retained leaf to the same shared destination layout.

    Module-level so the beartype claw decorates it once at import instead of on
    every `_shared_tree` call; see `_lcm/utils/functools.py`.
    """
    if not isinstance(leaf, jax.Array | jax.ShapeDtypeStruct):
        raise ExecutionPlanningError(
            "Forward retained inputs must be canonical JAX array metadata."
        )
    return _placed_abstract(
        leaf=leaf,
        sharding=simulation_value_sharding(
            stored_sharding=leaf.sharding, devices=devices
        ),
    )


def _shared_tree(
    *,
    tree: Mapping[str, PytreeValue | ShapeDtypePytree],
    devices: tuple[jax.Device, ...],
) -> Mapping[str, ShapeDtypePytree]:
    """Project actual retained metadata to the same shared destination layout."""
    return jax.tree.map(partial(_shared_leaf, devices=devices), tree)


def _placed_abstract(
    *, leaf: jax.Array | jax.ShapeDtypeStruct, sharding: jax.sharding.Sharding
) -> jax.ShapeDtypeStruct:
    """Keep shape, dtype and weak type while declaring the required placement."""
    return jax.ShapeDtypeStruct(
        leaf.shape, leaf.dtype, sharding=sharding, weak_type=leaf.weak_type
    )
