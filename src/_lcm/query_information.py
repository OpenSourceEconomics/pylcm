"""Finite physical information support for canonical typed policy producers.

Each producer prices its own selected policy over its hidden rows. Ancestor
information axes remain transient native outputs until their actual caller
selects a policy; only the occupant root integrates into a durable value bank.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.current_iid import (
    CURRENT_IID_ARGUMENT,
    CURRENT_IID_NODES_ARGUMENT,
    CURRENT_IID_ROWS_ARGUMENT,
    AdmittedGroupParams,
    CurrentIIDDrawFrame,
    CurrentIIDPlan,
    NodeOperands,
)
from _lcm.policy_queries import PolicyAnswer, QueryNode
from _lcm.probability import is_represented_zero
from _lcm.shock_registry import ShockBinding
from _lcm.typing import PytreeValue, QAndFKwargs
from _lcm.utils.counts import sum_counts
from _lcm.utils.dispatchers import vary_carry_like_body
from _lcm.zero_safe import factored_weighted_term
from lcm.shocks import ShockGroupName
from lcm.typing import BoolND, FloatND, IntND

type SourceStates = Mapping[str, FloatND | IntND]
type Frames = Mapping[str, CurrentIIDDrawFrame]
type QueryArguments = Mapping[str, PytreeValue]
type Dependencies = Mapping[str, PolicyAnswer]


def support_size(*, node: QueryNode, period: int) -> int:
    """Return the declared finite row count without materializing physical draws."""
    plan = node.context.current_iid_plans.get(period)
    size = 1 if plan is None else plan.indices.shape[0]
    return size * math.prod(node.observation_counts)


def _has_local_information(*, node: QueryNode, period: int) -> bool:
    return (
        bool(node.observation_sources)
        or node.context.current_iid_plans.get(period) is not None
    )


def support_shape(
    *, node: QueryNode, definitions: Mapping[str, QueryNode], period: int
) -> tuple[int, ...]:
    """Name only ancestor information axes that have a numerical node operator."""
    if period != node.read.period:
        raise ValueError("Information shape needs the query's actual period.")
    return tuple(
        support_size(node=definitions[name], period=definitions[name].read.period)
        for name in node.ancestors
        if _has_local_information(
            node=definitions[name], period=definitions[name].read.period
        )
    )


def has_information(
    *, node: QueryNode, definitions: Mapping[str, QueryNode], period: int
) -> bool:
    """Whether this producer or its producing context has current node support."""
    if period != node.read.period:
        raise ValueError("Information demand needs the query's actual period.")
    return any(
        _has_local_information(node=candidate, period=candidate.read.period)
        for candidate in (node, *(definitions[name] for name in node.ancestors))
    )


def _row(*, frame: CurrentIIDDrawFrame, row: jax.Array) -> CurrentIIDDrawFrame:
    """Select physical/address leaves without indexing scalar law metadata."""
    return frame._replace(
        physical={name: value[row] for name, value in frame.physical.items()},
        addressed_bases={
            name: value[row] for name, value in frame.addressed_bases.items()
        },
        held_bases={name: value[row] for name, value in frame.held_bases.items()},
        recovery_error={
            name: value[row] for name, value in frame.recovery_error.items()
        },
        source_params=jax.tree.map(lambda value: value[row], frame.source_params),
        source_inputs=jax.tree.map(lambda value: value[row], frame.source_inputs),
        valid=frame.valid[row],
        information_available={
            name: value[row] for name, value in frame.information_available.items()
        },
    )


def _empty_frame() -> CurrentIIDDrawFrame:
    return CurrentIIDDrawFrame({}, {}, {}, {}, {}, {}, jnp.asarray(a=True))


def _batch_metadata(*, frame: CurrentIIDDrawFrame, size: int) -> CurrentIIDDrawFrame:
    def broadcast(value: FloatND | IntND | BoolND) -> FloatND | IntND | BoolND:
        value = jnp.asarray(value)
        return jnp.broadcast_to(value, (size, *value.shape))

    return frame._replace(
        source_params=jax.tree.map(broadcast, frame.source_params),
        source_inputs=jax.tree.map(broadcast, frame.source_inputs),
        valid=jnp.broadcast_to(frame.valid, (size,)),
        information_available=jax.tree.map(broadcast, frame.information_available),
    )


def _bindings(
    *, node: QueryNode, definitions: Mapping[str, QueryNode], period: int
) -> Mapping[ShockGroupName, ShockBinding]:
    plan = node.context.current_iid_plans.get(period)
    bindings = (
        {} if plan is None else {name: binding for name, binding, _ in plan.bindings}
    )
    for group, producer in node.observation_sources.items():
        target = definitions[producer]
        target_plan = target.context.current_iid_plans[target.read.period]
        if target_plan is None:
            raise ValueError("An observed group requires its producing law plan.")
        bindings[group] = next(
            binding for name, binding, _ in target_plan.bindings if name == group
        )
    return bindings


@dataclass(frozen=True, kw_only=True)
class InformationQueryCell:
    """One scalar source cell of an existing native policy-query producer."""

    name: str
    node: QueryNode
    definitions: Mapping[str, QueryNode]
    period: int
    fixed: QueryArguments
    projection_operands: QueryArguments
    information_arguments: Mapping[str, QueryArguments]
    realised_root: str | None = None
    root_frame: CurrentIIDDrawFrame | None = None

    def _size(self, *, name: str, node: QueryNode) -> int:
        return (
            1
            if name == self.realised_root
            else support_size(node=node, period=node.read.period)
        )

    def __call__(
        self, *, live: BoolND, source: SourceStates, dependencies: Dependencies
    ) -> PolicyAnswer:
        """Retain caller axes, reduce only this producer's hidden node support."""
        return self._ancestors(
            position=0,
            frames={},
            indices=(),
            live=live,
            source=source,
            dependencies=dependencies,
        )

    def _available(
        self,
        *,
        node: QueryNode,
        fixed: QueryArguments,
        source: SourceStates,
        frames: Frames,
    ) -> BoolND:
        from _lcm.policy_queries import (  # noqa: PLC0415
            _PHYSICAL_CONTEXTS,
            _PROJECTION,
            _query_availability,
        )

        if node.available is None:
            return jnp.asarray(a=True)
        available = {
            **(
                self.projection_operands
                if node is self.node
                else cast("QueryArguments", fixed[_PROJECTION])
            ),
            **source,
            "period": fixed["period"],
            "age": fixed["age"],
            _PHYSICAL_CONTEXTS: {
                name: frame.physical for name, frame in frames.items()
            },
        }
        return _query_availability(node=node, available=available)

    def _realise(
        self,
        *,
        node: QueryNode,
        fixed: QueryArguments,
        source: SourceStates,
        frames: Frames,
    ) -> tuple[CurrentIIDDrawFrame, FloatND]:
        from _lcm.policy_queries import (  # noqa: PLC0415
            _PHYSICAL_CONTEXTS,
            _PROJECTION,
            _mapped_query_states,
        )

        key = next(
            name for name, candidate in self.definitions.items() if candidate is node
        )
        if key == self.realised_root:
            if self.root_frame is None:
                raise ValueError(
                    "A realised policy query needs its original physical frame."
                )
            observed = node.read.policy_information
            missing = observed - self.root_frame.physical.keys()
            if missing:
                raise ValueError(
                    f"Unrevealed required policy information: {sorted(missing)!r}."
                )
            frame = self.root_frame._replace(
                physical={
                    name: value[None]
                    for name, value in self.root_frame.physical.items()
                    if name in observed
                },
                addressed_bases={
                    name: value[None]
                    for name, value in self.root_frame.addressed_bases.items()
                    if name in observed
                },
                held_bases={
                    name: value[None]
                    for name, value in self.root_frame.held_bases.items()
                    if name in observed
                },
                recovery_error={
                    name: value[None]
                    for name, value in self.root_frame.recovery_error.items()
                    if name in observed
                },
            )
            return _batch_metadata(frame=frame, size=1), jnp.ones((1, 1))
        available = {
            **(
                self.projection_operands
                if node is self.node
                else cast("QueryArguments", fixed[_PROJECTION])
            ),
            **source,
            "period": fixed["period"],
            "age": fixed["age"],
            _PHYSICAL_CONTEXTS: {
                name: frame.physical for name, frame in frames.items()
            },
        }
        mapped = _mapped_query_states(node=node, available=available)
        plan = node.context.current_iid_plans.get(node.read.period)
        parent = node.ancestors[-1] if node.ancestors else None
        previous = None if parent is None else frames[parent]
        if plan is None:
            frame, factors = _empty_frame(), jnp.ones((1, 1))
        else:
            frame, factors = plan.realise_information_nodes(
                arguments=cast("QAndFKwargs", {**fixed, **mapped}),
                group_params=cast("AdmittedGroupParams", fixed[CURRENT_IID_ARGUMENT]),
                node_operands=cast("NodeOperands", fixed[CURRENT_IID_NODES_ARGUMENT]),
                indices=cast("IntND", fixed[CURRENT_IID_ROWS_ARGUMENT]),
                observation=previous,
                observed=frozenset(
                    node.read.price_information
                    & (frozenset() if previous is None else previous.physical.keys())
                ),
                source_bindings={}
                if parent is None
                else _bindings(
                    node=self.definitions[parent],
                    definitions=self.definitions,
                    period=self.definitions[parent].read.period,
                ),
            )
        if not node.observation_sources:
            return (
                frame
                if plan is None
                else _batch_metadata(frame=frame, size=factors.shape[1]),
                factors,
            )
        return self._observations(
            node=node,
            primary=_batch_metadata(frame=frame, size=factors.shape[1]),
            factors=factors,
            source=source,
            frames=frames,
        )

    def _observations(
        self,
        *,
        node: QueryNode,
        primary: CurrentIIDDrawFrame,
        factors: FloatND,
        source: SourceStates,
        frames: Frames,
    ) -> tuple[CurrentIIDDrawFrame, FloatND]:
        key = next(
            name for name, candidate in self.definitions.items() if candidate is node
        )
        extents = (factors.shape[1], *node.observation_counts)

        def point(flat: IntND) -> tuple[CurrentIIDDrawFrame, FloatND]:
            rows = jnp.unravel_index(flat, extents)
            frame = _row(frame=primary, row=rows[0])
            terms = [factors[:, rows[0]]]
            for axis, (group, producer) in enumerate(
                node.observation_sources.items(), start=1
            ):
                target = self.definitions[producer]
                fixed = self.information_arguments[producer]
                chosen, weights = self._observer_row(
                    observer=node,
                    target=target,
                    group=group,
                    fixed=fixed,
                    row=rows[axis],
                    observed=node.read.policy_information,
                    source=source,
                    frames={**frames, key: frame},
                )
                observed = node.read.policy_information & chosen.physical.keys()
                frame = CurrentIIDDrawFrame(
                    physical={
                        **frame.physical,
                        **{name: chosen.physical[name] for name in observed},
                    },
                    addressed_bases={
                        **frame.addressed_bases,
                        **{name: chosen.addressed_bases[name] for name in observed},
                    },
                    held_bases={
                        **frame.held_bases,
                        **{name: chosen.held_bases[name] for name in observed},
                    },
                    source_params={**frame.source_params, **chosen.source_params},
                    source_inputs={**frame.source_inputs, **chosen.source_inputs},
                    recovery_error={
                        **frame.recovery_error,
                        **{name: chosen.recovery_error[name] for name in observed},
                    },
                    valid=frame.valid & chosen.valid,
                    information_available={
                        **frame.information_available,
                        **{
                            name: chosen.information_available[name]
                            for name in observed
                        },
                    },
                )
                terms.append(weights)
            return frame, jnp.concatenate(terms)

        frame, weights = jax.lax.map(
            point,
            jnp.arange(math.prod(extents), dtype=jnp.int32),
        )
        return frame, weights.T

    def _observer_row(
        self,
        *,
        observer: QueryNode,
        target: QueryNode,
        group: ShockGroupName,
        fixed: QueryArguments,
        row: IntND,
        observed: frozenset[str],
        source: SourceStates,
        frames: Frames,
    ) -> tuple[CurrentIIDDrawFrame, FloatND]:
        """An unavailable observation executes neither settlement nor physical law."""
        from _lcm.current_iid import (  # noqa: PLC0415
            law_informativeness,
            law_parameters,
        )
        from _lcm.policy_queries import (  # noqa: PLC0415
            _PHYSICAL_CONTEXTS,
            _observation_available,
            _observation_inputs,
        )

        plan = target.context.current_iid_plans[target.read.period]
        if plan is None:
            raise ValueError("An observation needs an actual producing law.")
        binding, nodes = next(
            (binding, nodes) for name, binding, nodes in plan.bindings if name == group
        )
        available = {
            **source,
            _PHYSICAL_CONTEXTS: {
                name: frame.physical for name, frame in frames.items()
            },
        }

        def produce() -> tuple[CurrentIIDDrawFrame, FloatND]:
            mapped = _observation_inputs(
                node=observer,
                group=group,
                definitions=self.definitions,
                arguments=self.information_arguments,
                period=target.read.period,
                available=available,
            )
            site = CurrentIIDPlan(
                bindings=((group, binding, nodes),),
                indices=jnp.arange(nodes.weights.size, dtype=jnp.int32)[:, None],
            )
            produced, weights = site.realise_information_nodes(
                arguments=cast("QAndFKwargs", {**fixed, **mapped}),
                group_params={
                    group: cast("AdmittedGroupParams", fixed[CURRENT_IID_ARGUMENT])[
                        group
                    ]
                },
                node_operands={
                    group: cast("NodeOperands", fixed[CURRENT_IID_NODES_ARGUMENT])[
                        group
                    ]
                },
                indices=site.indices,
            )
            params = law_parameters(params=produced.source_params[group])
            informative = law_informativeness(
                binding=binding,
                params=params,
                inputs=produced.source_inputs[group],
            )
            mask = jnp.stack(
                tuple(
                    informative[name] & (name in observed)
                    for name in binding.group.shocks
                )
            )
            identity = binding.group._has_identity_correlation(  # noqa: SLF001 - shared exact represented-matrix proof
                params=params
            )
            chosen = _row(
                frame=_batch_metadata(frame=produced, size=nodes.weights.size),
                row=row,
            )
            return chosen._replace(
                valid=chosen.valid & (identity | jnp.all(mask) | ~jnp.any(mask)),
                information_available={
                    name: jnp.asarray(a=True) for name in chosen.physical
                },
            ), weights[:, row]

        def prototype_fields() -> tuple[tuple[PytreeValue, ...], FloatND]:
            frame, weights = produce()
            return tuple(frame), weights

        prototype = jax.eval_shape(prototype_fields)
        neutral_fields, neutral_weights = jax.tree.map(
            lambda leaf: jnp.zeros(leaf.shape, leaf.dtype),
            prototype,
        )
        neutral_frame = CurrentIIDDrawFrame(*neutral_fields)._replace(
            valid=jnp.asarray(a=True)
        )
        # Inactive support is a single numerical no-information representative;
        # it does not consult an unsafe target law to obtain a probability mass.
        neutral_weights = (
            jnp.ones_like(neutral_weights)
            .at[0]
            .set((row == 0).astype(neutral_weights.dtype))
        )
        neutral = vary_carry_like_body(
            body=lambda _carry: produce(), carry=(neutral_frame, neutral_weights)
        )
        return jax.lax.cond(
            _observation_available(
                node=observer,
                group=group,
                definitions=self.definitions,
                arguments=self.information_arguments,
                available=available,
            ),
            produce,
            lambda: neutral,
        )

    def _ancestors(
        self,
        *,
        position: int,
        frames: Frames,
        indices: tuple[IntND, ...],
        live: BoolND,
        source: SourceStates,
        dependencies: Dependencies,
    ) -> PolicyAnswer:
        if position == len(self.node.ancestors):
            return self._own(
                frames=frames,
                indices=indices,
                live=live,
                source=source,
                dependencies=dependencies,
            )
        name = self.node.ancestors[position]
        node = self.definitions[name]
        fixed = self.information_arguments[name]

        def admitted() -> PolicyAnswer:
            frame, factors = self._realise(
                node=node, fixed=fixed, source=source, frames=frames
            )
            if not _has_local_information(node=node, period=node.read.period):
                return self._ancestors(
                    position=position + 1,
                    frames={**frames, name: frame},
                    indices=indices,
                    live=live,
                    source=source,
                    dependencies=dependencies,
                )

            def point(row: IntND) -> PolicyAnswer:
                answer = self._ancestors(
                    position=position + 1,
                    frames={**frames, name: _row(frame=frame, row=row)},
                    indices=(*indices, row),
                    live=live,
                    source=source,
                    dependencies=dependencies,
                )
                present = jnp.all(~is_represented_zero(factors[:, row]))
                return (*answer[:4], jnp.where(present, answer[4], 0))

            return jax.lax.map(
                point,
                jnp.arange(self._size(name=name, node=node), dtype=jnp.int32),
            )

        absent = vary_carry_like_body(
            body=lambda _carry: admitted(), carry=self._absent(position=position)
        )
        return jax.lax.cond(
            live
            & self._available(node=node, fixed=fixed, source=source, frames=frames),
            admitted,
            lambda: absent,
        )

    def _absent(self, *, position: int) -> PolicyAnswer:
        dimensions = tuple(
            self._size(name=name, node=self.definitions[name])
            for name in self.node.ancestors[position:]
            if _has_local_information(
                node=self.definitions[name], period=self.definitions[name].read.period
            )
        )
        from _lcm.policy_queries import _query_landing_checks  # noqa: PLC0415

        members = self.node.value_shape
        return (
            jnp.full((*dimensions, *members), -jnp.inf),
            jnp.full(
                (*dimensions, self._size(name=self.name, node=self.node)),
                -1,
                jnp.int32,
            ),
            jnp.zeros(dimensions, jnp.bool_),
            jnp.ones(dimensions, jnp.bool_),
            jnp.zeros(
                (*dimensions, len(_query_landing_checks(node=self.node))), jnp.int32
            ),
        )

    def _own(
        self,
        *,
        frames: Frames,
        indices: tuple[IntND, ...],
        live: BoolND,
        source: SourceStates,
        dependencies: Dependencies,
    ) -> PolicyAnswer:
        from _lcm.policy_queries import _PHYSICAL_CONTEXTS, _query_cell  # noqa: PLC0415

        def admitted() -> PolicyAnswer:
            frame, factors = self._realise(
                node=self.node,
                fixed=self.fixed,
                source=source,
                frames=frames,
            )

            def point(row: IntND) -> PolicyAnswer:
                local = _has_local_information(
                    node=self.node, period=self.node.read.period
                )
                physical = {} if not local else _row(frame=frame, row=row).physical
                dependency_index = (*indices, row) if local else indices
                projected = {
                    name: (
                        answer[0][dependency_index],
                        jnp.asarray(-1, jnp.int32),
                        answer[2][dependency_index],
                        answer[3],
                        answer[4],
                    )
                    for name, answer in dependencies.items()
                }
                return _query_cell(
                    live,
                    source,
                    projected,
                    node=self.node,
                    period=self.node.read.period,
                    fixed={
                        **self.fixed,
                        **physical,
                        _PHYSICAL_CONTEXTS: {
                            **{name: frame.physical for name, frame in frames.items()},
                            self.name: physical,
                        },
                    },
                    projection_operands=self.projection_operands,
                )

            answers = jax.lax.map(
                point,
                jnp.arange(self._size(name=self.name, node=self.node), dtype=jnp.int32),
            )
            values, actions, support, valid, landings = answers
            expanded = jnp.broadcast_to(
                factors.reshape((*factors.shape, *((1,) * (values.ndim - 1)))),
                (factors.shape[0], *values.shape),
            )
            present = jnp.all(~is_represented_zero(factors), axis=0)
            return (
                jnp.sum(
                    factored_weighted_term(factors=expanded, values=values), axis=0
                ),
                actions,
                jnp.all(~present | support),
                jnp.all(frame.valid) & jnp.all(valid),
                sum_counts(counts=jnp.where(present[:, None], landings, 0), axis=0),
            )

        absent = vary_carry_like_body(
            body=lambda _carry: admitted(),
            carry=self._absent(position=len(self.node.ancestors)),
        )
        return jax.lax.cond(
            live
            & self._available(
                node=self.node,
                fixed=self.fixed,
                source=source,
                frames=frames,
            ),
            admitted,
            lambda: absent,
        )
