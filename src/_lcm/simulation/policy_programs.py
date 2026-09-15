"""Declare the finite replay route using its complete published candidate bank."""

import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.engine import NNBEGMPolicyRead, Regime
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ValueRead,
)
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.simulation.program_types import subject_axis
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.solution.continuation_reads import rekeyed_value_reads
from _lcm.typing import FlatRegimeParams, RegimeName
from lcm.solver_api import SIMULATION_POLICY
from lcm.typing import FloatND, IntND, ScalarFloat, ScalarInt

POLICY_PREPARE = "simulate_policy_prepare"
POLICY_RANK = "simulate_policy_rank"


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReplayPayload:
    """Dynamic payload leaves with immutable public-pytree reconstruction metadata.

    An ordinary dataclass traversal would turn numeric auxiliary metadata into
    device operands. Keeping that metadata in the TreeDef preserves the producer's
    registered pytree while exposing concrete array locators to transfer planning.
    """

    arrays: tuple[jax.Array, ...]
    structure: jax.tree_util.PyTreeDef

    @classmethod
    def from_policy(cls, policy: NNBEGMSimPolicy) -> ReplayPayload:
        leaves, structure = jax.tree_util.tree_flatten(policy)
        return cls(arrays=tuple(leaves), structure=structure)

    def restore(self) -> NNBEGMSimPolicy:
        return cast(
            "NNBEGMSimPolicy", jax.tree_util.tree_unflatten(self.structure, self.arrays)
        )


def _flatten_payload(payload: ReplayPayload) -> tuple[tuple, object]:
    return payload.arrays, payload.structure


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
def _unflatten_payload(structure: object, arrays: Sequence[object]) -> ReplayPayload:
    result = object.__new__(ReplayPayload)
    object.__setattr__(result, "arrays", tuple(arrays))
    object.__setattr__(result, "structure", structure)
    return result


jax.tree_util.register_pytree_node(ReplayPayload, _flatten_payload, _unflatten_payload)


def declare_finite_replay_programs(regime: Regime) -> Regime:
    """Select finite replay as the decision only where model authority admits it."""
    if regime.simulation.replay_route.consumer_route != "nnbegm_finite":
        return regime
    read = cast("NNBEGMPolicyRead", regime.simulation.egm_policy_read)
    previous = regime.simulation.programs
    preparation = {}
    ranking = {}
    for period, decision in previous.decision.items():
        payload_reads = _policy_reads(
            regime=regime.name,
            period=period,
            n_arrays=5 if read.discrete_action_names else 4,
            core=POLICY_PREPARE,
        )
        preparation[period] = _program(
            name=POLICY_PREPARE,
            body=_Prepare(regime=regime, period=period),
            subject_names=("states",),
            state_names=regime.simulation.state_names,
            reads=payload_reads,
            roles=("inner_candidate", "outer_candidate", "live", "represented"),
        )
        value_reads = rekeyed_value_reads(
            reads=decision.requirements.value_reads, core_key=POLICY_RANK
        )
        reference_names = tuple(
            dict.fromkeys(
                read.source.argument
                for read in value_reads
                if read.source.argument not in (None, "next_regime_to_V_arr")
            )
        )
        ranking[period] = _program(
            name=POLICY_RANK,
            body=_Rank(
                regime=regime,
                period=period,
                reference_names=reference_names,
            ),
            subject_names=("bank", "canonical_states"),
            state_names=regime.simulation.state_names,
            reads=(
                *value_reads,
                *_policy_reads(
                    regime=regime.name,
                    period=period,
                    n_arrays=5 if read.discrete_action_names else 4,
                    core=POLICY_RANK,
                ),
            ),
            roles=(
                MappingProxyType(
                    dict.fromkeys(
                        (
                            read.inner_action_name,
                            read.outer_action_name,
                            *read.discrete_action_names,
                        ),
                        "action",
                    )
                ),
                "decision_value",
                "nested_policy_fallback",
            ),
        )
    programs = dataclasses.replace(
        previous,
        decision=MappingProxyType(ranking),
        policy_prepare=MappingProxyType(preparation),
        policy_rank=MappingProxyType(ranking),
    )
    return dataclasses.replace(
        regime, simulation=dataclasses.replace(regime.simulation, programs=programs)
    )


def _program(
    *,
    name: str,
    body: Callable[..., object],
    subject_names: tuple[str, ...],
    state_names: tuple[str, ...],
    reads: tuple[ValueRead, ...],
    roles: object,
) -> CoreProgram:
    return CoreProgram(
        name=name,
        function=_SubjectTiled(func=body, subject_arg_names=subject_names),
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name=name, subject_arg_names=subject_names
        ),
        requirements=CoreExecutionRequirements(
            tiled_axes=(subject_axis(state_names=state_names),), value_reads=reads
        ),
        output_roles=roles,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _policy_reads(
    *, regime: str, period: int, n_arrays: int, core: str
) -> tuple[ValueRead, ...]:
    """Match NNBEGMSimPolicy's registered four mandatory and optional fifth leaf."""
    return tuple(
        ValueRead(
            target=ValueArtifactAddress(
                kind=ValueArtifactKind.REPLAY_ARTIFACT_LEAF,
                period=period,
                regime=regime,
                artifact_key=SIMULATION_POLICY,
                leaf_path=(f"FlattenedIndexKey:{jax.tree_util.FlattenedIndexKey(i)}",),
            ),
            source=ValueConsumerAddress(
                source_period=period,
                source_regime=regime,
                core_key=core,
                channel=ValueInputChannel.CURRENT_REPLAY_ARTIFACT,
                argument="payload",
                path=("arrays", i),
            ),
        )
        for i in range(n_arrays)
    )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _Prepare:
    regime: Regime
    period: int

    def __call__(
        self,
        *,
        payload: ReplayPayload,
        states: Mapping[str, FloatND | IntND],
        params: FlatRegimeParams,
        age: ScalarFloat | ScalarInt,
    ) -> object:
        # Imported at execution to keep declaration construction independent of
        # the forward coordinator's import order.
        from _lcm.simulation.simulate import (  # noqa: PLC0415
            _prepare_nnbegm_candidate_bank,
        )

        bank = _prepare_nnbegm_candidate_bank(
            regime=self.regime,
            sim_policy=payload.restore(),
            states=jax.tree.map(lambda value: value[None], states),
            flat_params=params,
            period=self.period,
            age=age,
        )
        return tuple(value[:, 0] for value in bank)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _Rank:
    regime: Regime
    period: int
    reference_names: tuple[str, ...]

    @property
    def __signature__(self) -> inspect.Signature:
        return inspect.Signature(
            tuple(
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                for name in (
                    "payload",
                    "bank",
                    "canonical_states",
                    "params",
                    "age",
                    "next_regime_to_V_arr",
                    *self.reference_names,
                )
            )
        )

    def __call__(self, **arguments: object) -> object:
        from _lcm.simulation.simulate import (  # noqa: PLC0415
            _rank_nnbegm_candidate_bank,
        )

        payload = cast("ReplayPayload", arguments.pop("payload"))
        bank = cast("tuple[jax.Array, ...]", arguments.pop("bank"))
        states = arguments.pop("canonical_states")
        actions, values = _rank_nnbegm_candidate_bank(
            candidate_inner=bank[0][:, None],
            candidate_outer=bank[1][:, None],
            represented=bank[3][:, None],
            optimal_actions=MappingProxyType({}),
            regime=self.regime,
            sim_policy=payload.restore(),
            flat_params=cast("FlatRegimeParams", arguments.pop("params")),
            period=self.period,
            age=cast("ScalarFloat | ScalarInt", arguments.pop("age")),
            canonical_states=jax.tree.map(lambda value: value[None], states),
            action_names=self.regime.simulation.action_names,
            next_regime_to_V_arr=cast(
                "MappingProxyType[RegimeName, FloatND]",
                arguments.pop("next_regime_to_V_arr"),
            ),
            referenced_value_kwargs=arguments,
        )
        return (
            jax.tree.map(lambda value: value[0], actions),
            values[0],
            jnp.zeros((), dtype=bool),
        )
