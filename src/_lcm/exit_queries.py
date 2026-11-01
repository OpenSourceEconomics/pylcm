"""Bind public exits to the canonical scalar policy-query decision."""

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from itertools import product
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions, with_signature
from dags.signature import rename_arguments
from dags.tree import qname_from_tree_path
from jax.core import Tracer

from _lcm.current_iid import CurrentIIDPlan
from _lcm.engine import Regime
from _lcm.exits import exit_functions, exits_for_phase
from _lcm.grids import DiscreteGrid
from _lcm.policy_queries import (
    _PHYSICAL_CONTEXTS,
    _TASTE,
    PolicyAnswer,
    QueryKind,
    QueryNode,
    QueryRead,
    _PolicyQueryKernel,
    _taste_stay_roster,
    build_policy_query_kernel,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.typing import PytreeValue, QualifiedName
from _lcm.utils.functools import (
    ComposedProjection,
    NamedOperand,
)
from _lcm.utils.functools import (
    call_with_declared_arguments as _call,
)
from lcm.exceptions import RegimeInitializationError, UnsupportedOperationError
from lcm.exits import Consent, Exit, ExitRole, RuleResult
from lcm.shocks import Discrete, ShockGroupName, ShockName, _fingerprint
from lcm.typing import (
    BoolND,
    FloatND,
    Phase,
    ReferenceName,
    RegimeName,
    StateName,
    UserFunction,
)


def _predicate(
    *,
    function: Callable[..., PytreeValue],
    operands: Mapping[ReferenceName, PytreeValue],
) -> BoolND:
    """Normalize scalar Python booleans without reducing undeclared axes."""
    value = jnp.asarray(_call(function=function, operands=operands))
    if value.shape != () or value.dtype != jnp.bool_:
        raise ValueError("Exit availability must be a scalar boolean.")
    return value


@dataclass(frozen=True, kw_only=True)
class _ExitReducer:
    """One scalar rule shared by solve and actual subject replay.

    Target values are already evaluated at one common Participating action.
    This reducer never maximizes the destination again to manufacture consent.
    """

    kind: QueryKind
    """The source policy problem being answered."""
    members: tuple[ExitRole, ...]
    """Canonical source tuple order, or self for a singleton."""
    exits: Mapping[str, Exit]
    """Exactly the current phase's public declarations."""
    targets: Mapping[ReferenceName, Mapping[ExitRole, tuple[str, int | None]]]
    """Exit/member to producer operand and destination stakeholder index."""
    adjustments: Mapping[str, Mapping[str, Callable[..., PytreeValue]]]
    """Source-context utility adjustments, evaluated once in each payoff."""
    availability: Mapping[str, Callable[..., PytreeValue]]
    """Source-context predicates; absent predicates mean available."""
    rules: Mapping[str, Callable[..., PytreeValue]]
    """Canonically bound custom rule callables."""
    consent_reads: Mapping[str, tuple[str, str, str, int | None]]
    """Own Raw, counterparty Raw, couple producer and counterparty tuple index."""
    weights: Callable[..., Mapping[str, FloatND]] | None
    """Canonical source Pareto evaluator; singleton objective has weight one."""
    n_actions: int
    """Flattened canonical stay product, preceding stable exit IDs."""
    n_discrete_action_axes: int = 0
    """A source EV1 component belongs to a discrete combination, not a continuous ID."""

    def __post_init__(self) -> None:
        """Keep declaration containers immutable and IDs inside their native width."""
        if not 0 < self.n_actions <= jnp.iinfo(jnp.int32).max - len(self.exits):
            raise RegimeInitializationError(
                "Exit alternatives must fit stable int32 IDs."
            )
        for name in ("targets", "adjustments"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        key: MappingProxyType(dict(value))
                        for key, value in getattr(self, name).items()
                    }
                ),
            )
        for name in ("exits", "availability", "rules", "consent_reads"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def _payoffs(
        self, *, operands: Mapping[ReferenceName, PytreeValue]
    ) -> tuple[dict[str, dict[str, FloatND]], dict[str, BoolND], BoolND]:
        payoffs = {}
        availability = {}
        valid = jnp.asarray(a=True)
        for name, declaration in self.exits.items():
            active = (
                jnp.asarray(a=True)
                if name not in self.availability
                else jnp.asarray(
                    _call(function=self.availability[name], operands=operands)
                )
            )
            if active.shape != () or active.dtype != jnp.bool_:
                raise ValueError("Exit availability must be a scalar boolean.")
            availability[name] = active
            member_values = {}
            for member in declaration.targets:
                if member not in self.targets[name]:
                    member_values[member] = jnp.asarray(-jnp.inf)
                    continue
                key, index = self.targets[name][member]
                answer = cast("PolicyAnswer", operands[key])
                value = answer[0] if index is None else answer[0][index]
                if member in self.adjustments.get(name, {}):
                    adjustment = jax.lax.cond(
                        active,
                        lambda helper=self.adjustments[name][member]: jnp.asarray(
                            _call(function=helper, operands=operands)
                        ),
                        lambda: jnp.asarray(0.0),
                    )
                    if adjustment.shape != ():
                        raise ValueError("Exit utility adjustment must be scalar.")
                    value = value + adjustment
                member_values[member] = value
                valid &= ~active | (answer[3] & ~jnp.isnan(value))
            payoffs[name] = member_values
        return payoffs, availability, valid

    def keep(self, **operands: PytreeValue) -> BoolND:
        """Apply weak dissolution thresholds before the common stay argmax."""
        q = cast("FloatND", operands["q"])
        payoffs, availability, _ = self._payoffs(operands=operands)
        keep = jnp.asarray(a=True)
        for name, declaration in self.exits.items():
            if declaration.rule not in ("unilateral", "mutual"):
                continue
            threshold = jnp.stack(
                tuple(payoffs[name][member] for member in self.members)
            )
            member_q = q[None] if self.members == ("self",) else q
            comparisons = member_q >= threshold
            willing = (
                jnp.all(comparisons)
                if declaration.rule == "unilateral"
                else jnp.any(comparisons)
            )
            keep &= ~availability[name] | willing
        return keep

    def __call__(self, **operands: PytreeValue) -> PolicyAnswer:
        """Route weak dissolution or strict consent, preserving common tuple/ID."""
        stay = cast("PolicyAnswer", operands["answer"])
        payoffs, availability, valid = self._payoffs(operands=operands)
        result = (*stay[:3], stay[3] & valid, stay[4])
        custom = tuple(self.rules.values())
        if custom:
            return self._custom(
                operands=operands,
                payoffs=payoffs,
                availability=availability,
                valid=result[3],
            )
        if _TASTE in operands and self.kind is QueryKind.FULL:
            return self._full_taste(
                operands=operands,
                payoffs=payoffs,
                availability=availability,
                valid=result[3],
            )
        for offset, (name, declaration) in enumerate(self.exits.items()):
            eligible, integrity = self._eligible(
                name=name,
                operands=operands,
                payoffs=payoffs,
                availability=availability,
            )
            valid &= integrity
            move = eligible & (self.kind is QueryKind.FULL)
            if not isinstance(declaration.rule, Consent):
                move &= ~stay[2]
            value = jnp.stack(tuple(payoffs[name][member] for member in self.members))
            if self.members == ("self",):
                value = value[0]
            result = (
                jnp.where(move, value, result[0]),
                jnp.where(move, self.n_actions + offset, result[1]).astype(jnp.int32),
                result[2] | move,
                result[3] & valid,
                stay[4],
            )
        return result

    def _eligible(
        self,
        *,
        name: str,
        operands: Mapping[ReferenceName, PytreeValue],
        payoffs: Mapping[str, Mapping[str, FloatND]],
        availability: Mapping[str, BoolND],
    ) -> tuple[BoolND, BoolND]:
        """Share exit support and strict common-policy consent across taste routes."""
        if isinstance(self.exits[name].rule, Consent):
            if name not in self.consent_reads:
                return jnp.asarray(a=False), jnp.asarray(a=True)
            own_key, other_key, couple_key, other_index = self.consent_reads[name]
            own = cast("PolicyAnswer", operands[own_key])
            other = cast("PolicyAnswer", operands[other_key])
            couple = cast("PolicyAnswer", operands[couple_key])
            other_value = couple[0] if other_index is None else couple[0][other_index]
            return (
                availability[name]
                & couple[2]
                & (payoffs[name]["self"] > own[0])
                & (other_value > other[0]),
                ~availability[name] | (own[3] & other[3] & couple[3]),
            )
        support = jnp.all(
            jnp.stack(
                tuple(
                    cast("PolicyAnswer", operands[operand])[2]
                    for operand, _ in self.targets[name].values()
                )
                or (jnp.asarray(a=False),)
            )
        )
        return availability[name] & support, jnp.asarray(a=True)

    def _full_taste(
        self,
        *,
        operands: Mapping[ReferenceName, PytreeValue],
        payoffs: Mapping[str, Mapping[str, FloatND]],
        availability: Mapping[str, BoolND],
        valid: BoolND,
    ) -> PolicyAnswer:
        """Append exit components to the source's canonical discrete taste roster."""
        from _lcm.logsum import logsum_and_softmax  # noqa: PLC0415
        from _lcm.regime_building.max_Q_over_a import (  # noqa: PLC0415
            draw_taste_shock_noise,
        )

        stay_values, stay_ids = _taste_stay_roster(
            q=cast("FloatND", operands["q"]),
            feasible=cast("BoolND", operands["feasible"]),
            n_discrete=self.n_discrete_action_axes,
        )
        eligible = []
        kept = jnp.any(cast("BoolND", operands["feasible"]))
        for name in self.exits:
            allowed, integrity = self._eligible(
                name=name,
                operands=operands,
                payoffs=payoffs,
                availability=availability,
            )
            if not isinstance(self.exits[name].rule, Consent):
                allowed &= ~kept
            eligible.append(allowed)
            valid &= integrity
        exit_values = jnp.stack(tuple(payoffs[name]["self"] for name in self.exits))
        values = jnp.concatenate(
            (stay_values, jnp.where(jnp.stack(eligible), exit_values, -jnp.inf))
        )
        identifiers = jnp.concatenate(
            (stay_ids, self.n_actions + jnp.arange(len(self.exits), dtype=jnp.int32))
        )
        taste = cast("Mapping[str, FloatND]", operands[_TASTE])
        scale = taste["scale"]
        if "key" in taste:
            chosen = jnp.argmax(
                values
                + draw_taste_shock_noise(
                    key=taste["key"], shape=values.shape, scale=scale
                )
            )
            value = values[chosen]
        else:
            chosen = jnp.argmax(values)
            value, _ = logsum_and_softmax(values=values, scale=scale, axes=(0,))
        supported = kept | jnp.any(jnp.stack(eligible))
        return (
            value,
            jnp.where(supported, identifiers[chosen], -1).astype(jnp.int32),
            supported,
            valid & jnp.isfinite(scale) & (scale > 0) & ~jnp.isnan(value),
            cast("PolicyAnswer", operands["answer"])[4],
        )

    def _custom(
        self,
        *,
        operands: Mapping[ReferenceName, PytreeValue],
        payoffs: Mapping[str, Mapping[str, FloatND]],
        availability: Mapping[str, BoolND],
        valid: BoolND,
    ) -> PolicyAnswer:
        """Validate the authored global ID before any narrowing integer cast."""
        q = cast("FloatND", operands["q"])
        feasible = cast("BoolND", operands["feasible"])
        weights = (
            {"self": jnp.asarray(1.0)}
            if self.weights is None
            else _call(function=self.weights, operands=operands)
        )
        member_q = (
            {"self": q}
            if self.members == ("self",)
            else {member: q[..., index] for index, member in enumerate(self.members)}
        )
        all_ids = jnp.arange(self.n_actions + len(self.exits), dtype=jnp.int32)
        arguments = {
            **operands,
            **{
                f"Q_{member}": jnp.where(feasible, value, -jnp.inf)
                for member, value in member_q.items()
            },
            "exit_payoffs": payoffs,
            "availability": availability,
            "alternative_ids": all_ids,
            "pareto_weights": weights,
        }
        returned = _call(function=next(iter(self.rules.values())), operands=arguments)
        if not isinstance(returned, RuleResult):
            raise TypeError("A custom exit rule must return RuleResult.")
        supplied_id = returned.action
        if isinstance(supplied_id, jax.Array | Tracer):
            chosen = supplied_id
        else:
            # Host int64 payloads can narrow in jnp.asarray when x64 is off.
            # Admission precedes that boundary, including zero-dimensional arrays.
            host_id = np.asarray(supplied_id)
            if host_id.shape != () or host_id.dtype.kind not in "iu":
                raise ValueError(
                    "RuleResult.action must be a scalar integer global ID."
                )
            if not 0 <= int(host_id) < self.n_actions + len(self.exits):
                raise ValueError(
                    "RuleResult.action is not a known global alternative ID."
                )
            chosen = jnp.asarray(int(host_id), dtype=jnp.int32)
        if chosen.shape != () or chosen.dtype.kind not in "iu":
            raise ValueError("RuleResult.action must be a scalar integer global ID.")
        # Check in the authored integer width: 2**32 must not become valid ID 0.
        known = (chosen >= 0) & (chosen < all_ids.size)
        safe_id = jnp.where(known, chosen, 0).astype(jnp.int32)
        flat_q = q.reshape(
            (self.n_actions,)
            + (() if self.members == ("self",) else (len(self.members),))
        )
        eligible = []
        for name in self.exits:
            allowed, integrity = self._eligible(
                name=name,
                operands=operands,
                payoffs=payoffs,
                availability=availability,
            )
            eligible.append(allowed)
            valid &= integrity
        allowed = jnp.concatenate((feasible.reshape(-1), jnp.stack(eligible)))
        exit_values = jnp.stack(
            tuple(
                jnp.stack(tuple(payoffs[name][member] for member in self.members))
                for name in self.exits
            )
        )
        if self.members == ("self",):
            exit_values = exit_values[:, 0]
        values = jnp.concatenate((flat_q, exit_values), axis=0)
        admitted = known & allowed[safe_id]
        # A Participating read has no exit alternative: selecting an exit means
        # its kept stay policy is empty, rather than recursive Full routing.
        support = admitted & (
            (safe_id < self.n_actions) | (self.kind is QueryKind.FULL)
        )
        return (
            jnp.where(support, values[safe_id], -jnp.inf),
            jnp.where(support, safe_id, -1),
            support,
            valid & admitted,
            cast("PolicyAnswer", operands["answer"])[4],
        )


def bind_exit_reducer(
    *,
    reducer: _ExitReducer,
    operand_names: tuple[str, ...],
) -> tuple[Callable[..., BoolND], Callable[..., PolicyAnswer]]:
    """Expose explicit native signatures instead of free parameter inference."""
    return (
        with_signature(
            reducer.keep,
            args=["q", *(name for name in operand_names if name != _TASTE)],
            enforce=False,
        ),
        with_signature(
            reducer, args=["q", "feasible", "answer", *operand_names], enforce=False
        ),
    )


def _compose(
    *,
    projection: Mapping[StateName, Callable[..., PytreeValue]],
    function: Callable[..., PytreeValue],
    source: ReferenceName,
    physical_names: frozenset[str] = frozenset(),
) -> Callable[..., PytreeValue]:
    projection = {
        name: helper
        for name, helper in projection.items()
        if name in inspect.signature(function).parameters
    }
    physical_reads = physical_names & inspect.signature(function).parameters.keys()
    names = tuple(
        dict.fromkeys(
            (
                *(
                    name
                    for helper in projection.values()
                    for name in inspect.signature(helper).parameters
                ),
                *((_PHYSICAL_CONTEXTS,) if physical_reads else ()),
                *(
                    name
                    for name in inspect.signature(function).parameters
                    if name not in projection and name not in physical_names
                ),
            )
        )
    )
    return with_signature(
        ComposedProjection(
            projection=projection,
            function=function,
            source=source,
            context_argument=_PHYSICAL_CONTEXTS,
        ),
        args=list(names),
        enforce=False,
    )


@dataclass(frozen=True, kw_only=True)
class _Available:
    """Conjoin ancestor availability before materializing the benchmark."""

    predicates: tuple[Callable[..., PytreeValue], ...]

    def __call__(self, **operands: PytreeValue) -> BoolND:
        active = jnp.asarray(a=True)
        for predicate in self.predicates:
            active = jax.lax.cond(
                active,
                lambda helper=predicate: _predicate(function=helper, operands=operands),
                lambda: jnp.asarray(a=False),
            )
        return active


def _guard(
    predicates: tuple[Callable[..., PytreeValue], ...],
) -> Callable[..., BoolND] | None:
    if not predicates:
        return None
    names = tuple(
        dict.fromkeys(
            name
            for helper in predicates
            for name in inspect.signature(helper).parameters
        )
    )
    return with_signature(
        _Available(predicates=predicates), args=list(names), enforce=False
    )


@dataclass(frozen=True, kw_only=True)
class _QueryGraphBuilder:
    """Expand one call-local public policy graph into canonical typed nodes."""

    contexts: Mapping[RegimeName, SolverBuildContext]
    """All actual phase/period action problems."""
    canonical_regimes: Mapping[RegimeName, Regime]
    """Parameter namespaces and fixed operands with their real owners."""
    period: int
    """Exact current computational period."""
    phase: Phase
    """Actual law/projection phase."""
    aliases: Mapping[tuple[RegimeName, QualifiedName], ReferenceName]
    """Collision-safe native names for source-owned parameter paths."""
    operand_paths: Mapping[ReferenceName, tuple[RegimeName, QualifiedName]]
    """Inverse address map consumed by canonical argument builders."""
    nodes: dict[str, QueryNode] = field(default_factory=dict)
    """Transient graph under construction; frozen by the caller on completion."""
    keys: list[str] = field(default_factory=list)
    """Construction-local identities reserved before recursive descent."""

    def bound(
        self,
        *,
        owner: RegimeName,
        path: tuple[str, ...],
        function: Callable[..., PytreeValue],
    ) -> Callable[..., PytreeValue]:
        """Reuse the phase helper DAG and its exact public parameter namespace."""
        from _lcm.regime_building.processing import (  # noqa: PLC0415 - post-all-context cycle
            _process_one_function,
        )

        context = self.contexts[owner]
        renamed = _process_one_function(
            func=cast("UserFunction", function),
            regime_params_template=self.canonical_regimes[owner].regime_params_template,
            param_key=qname_from_tree_path(path),
        )
        # State-only projections need no generated DAG or its captured callbacks.
        combined = renamed
        if not inspect.isfunction(renamed) or (
            inspect.signature(renamed).parameters.keys()
            & (context.functions.keys() | {"_lcm_exit_result"})
        ):
            combined = concatenate_functions(
                functions={**context.functions, "_lcm_exit_result": renamed},
                targets="_lcm_exit_result",
                enforce_signature=False,
                set_annotations=True,
            )
        mapper = {
            name: self.aliases[owner, name]
            for name in inspect.signature(combined).parameters
            if (owner, name) in self.aliases
        }
        return rename_arguments(combined, mapper=mapper) if mapper else combined

    def make(
        self,
        *,
        kind: QueryKind,
        owner: RegimeName,
        projection: Mapping[StateName, Callable[..., PytreeValue]],
        guards: tuple[Callable[..., PytreeValue], ...],
        ancestry: tuple[tuple[QueryKind, str], ...],
        role: str = "",
        ancestors: tuple[str, ...] = (),
        price_information: frozenset[str] = frozenset(),
        source_projection: Mapping[StateName, Callable[..., PytreeValue]] | None = None,
    ) -> str:
        """Bind a demanded typed read and reject genuine recursive policy cycles."""
        from _lcm.model_graph import _exit_is_statically_unavailable  # noqa: PLC0415

        if kind not in {QueryKind.RAW, QueryKind.PARTICIPATING, QueryKind.FULL}:
            raise ValueError(
                "The policy graph builder requires Raw, Participating or Full."
            )
        if (kind, owner) in ancestry:
            raise ValueError(f"Policy query cycle: {(*ancestry, (kind, owner))!r}")
        context = self.contexts[owner]
        if self.period not in context.Q_and_F_functions:
            raise ValueError(
                f"Exit target {owner!r} has no active policy at period {self.period}."
            )
        key = f"query_{len(self.keys)}"
        self.keys.append(key)
        declaration = context.user_regimes_at(period=self.period)[owner]
        exits = exits_for_phase(exits=declaration.exits, phase=self.phase)
        if (
            kind is QueryKind.FULL
            and context.has_taste_shocks
            and any(callable(item.rule) for item in exits.values())
        ):
            raise UnsupportedOperationError(
                "Source Full EV1 does not integrate an arbitrary custom exit selector."
            )
        _validate_custom_rules(exits=exits)
        if kind is QueryKind.RAW:
            exits = {}
        elif kind is QueryKind.PARTICIPATING:
            exits = {
                name: declaration
                for name, declaration in exits.items()
                if not isinstance(declaration.rule, Consent)
                and (
                    not callable(declaration.rule) or declaration._reads_outside_values  # noqa: SLF001 - frozen declaration metadata
                )
            }
        inactive = {
            name
            for name, declaration in exits.items()
            if _exit_is_statically_unavailable(
                predicate=declaration.available,
                period=self.period,
                ages=context.ages,
            )
        }
        functions = {
            path: self.bound(owner=owner, path=path, function=function)
            for path, function in exit_functions(exits=exits, phase=self.phase).items()
            if path[1] not in inactive or path[-1] in {"available", "rule"}
        }
        physical_names = {
            shock
            for candidate in self.contexts.values()
            for plan in candidate.current_iid_plans.values()
            if plan is not None
            for shock in plan.physical_names
        }
        physical_inputs = frozenset(
            physical_names
            & {
                name
                for function in (
                    context.Q_and_F_functions[self.period],
                    *functions.values(),
                )
                for name in inspect.signature(function).parameters
            }
        )
        policy_information = physical_inputs | frozenset(
            shock
            for name, declaration in exits.items()
            if name not in inactive
            for shock in declaration.observes
        )
        current_plan = context.current_iid_plans.get(self.period)
        if current_plan is not None:
            demanded = tuple(
                item
                for item in current_plan.bindings
                if policy_information & item[1].group.shocks.keys()
            )
            plan = (
                None
                if not demanded
                else CurrentIIDPlan(
                    bindings=demanded,
                    indices=jnp.asarray(
                        tuple(
                            product(
                                *(range(nodes.weights.size) for _, _, nodes in demanded)
                            )
                        ),
                        dtype=jnp.int32,
                    ),
                )
            )
            context = replace(
                context, current_iid_plans=MappingProxyType({self.period: plan})
            )
        inputs, reducer = self._expand_exits(
            kind=kind,
            owner=owner,
            projection=projection,
            guards=guards,
            ancestry=(*ancestry, (kind, owner)),
            exits=exits,
            functions=functions,
            source_key=key,
            ancestors=(*ancestors, key),
            policy_information=policy_information,
        )
        needed_aliases = {
            name
            for function in (*projection.values(), *guards, *functions.values())
            for name in inspect.signature(function).parameters
            if name in self.operand_paths
        }
        operand_paths = {
            name: path
            for name, path in self.operand_paths.items()
            if name in needed_aliases
        }
        names = tuple(
            dict.fromkeys(
                (
                    *inputs,
                    *operand_paths,
                    *context.state_action_space.state_names,
                    *context.state_action_space.action_names,
                    *physical_inputs,
                    _PHYSICAL_CONTEXTS,
                    *((_TASTE,) if context.has_taste_shocks else ()),
                    "period",
                    "age",
                )
            )
        )
        keep, decide = bind_exit_reducer(reducer=reducer, operand_names=names)
        self.nodes[key] = QueryNode(
            read=QueryRead(
                kind=kind,
                target=owner,
                phase=self.phase,
                period=self.period,
                role=role,
                projection=cast("Mapping[StateName, UserFunction]", projection),
                state_inputs=frozenset(context.state_action_space.state_names),
                physical_inputs=physical_inputs,
                operand_paths=operand_paths,
                policy_information=policy_information,
                price_information=price_information,
            ),
            context=context,
            inputs=inputs,
            own_raw=(
                inputs[next(iter(reducer.consent_reads.values()))[0]]
                if len(reducer.consent_reads) == 1
                else None
            ),
            participation=keep if exits else None,
            decision=decide if exits else None,
            available=_guard(guards),
            ancestors=ancestors,
            source_projection=cast(
                "Mapping[StateName, UserFunction]",
                projection if source_projection is None else source_projection,
            ),
            exit_targets={
                reducer.n_actions + offset: {
                    member: inputs[operand]
                    for member, (operand, _) in reducer.targets[name].items()
                }
                for offset, name in enumerate(exits)
            },
            exit_adjustments={
                reducer.n_actions + offset: reducer.adjustments[name]
                for offset, name in enumerate(exits)
            },
        )
        return key

    def _expand_exits(  # noqa: C901, PLR0915 - bind one exit's target/consent topology together
        self,
        *,
        kind: QueryKind,
        owner: RegimeName,
        projection: Mapping[StateName, Callable[..., PytreeValue]],
        guards: tuple[Callable[..., PytreeValue], ...],
        ancestry: tuple[tuple[QueryKind, str], ...],
        exits: Mapping[str, Exit],
        functions: Mapping[tuple[str, ...], Callable[..., PytreeValue]],
        source_key: str,
        ancestors: tuple[str, ...],
        policy_information: frozenset[str],
    ) -> tuple[dict[str, str], _ExitReducer]:
        """Expand only declared benchmarks and consent Raw reads."""
        from _lcm.model_graph import _exit_is_statically_unavailable  # noqa: PLC0415

        context = self.contexts[owner]
        current_plan = context.current_iid_plans.get(self.period)
        physical_names = (
            frozenset() if current_plan is None else current_plan.physical_names
        )
        inputs = {}
        targets = {}
        adjustments = {}
        availability = {}
        rules = {}
        consent_reads = {}
        child_ancestry = ancestry
        for exit_index, (name, declaration) in enumerate(exits.items()):
            prefix = ("exits", name)
            predicate = functions.get((*prefix, "available"))
            if predicate is not None:
                availability[name] = predicate
            child_guards = guards + (
                ()
                if predicate is None
                else (
                    _compose(
                        projection=projection,
                        function=predicate,
                        source=source_key,
                        physical_names=physical_names,
                    ),
                )
            )
            targets[name] = {}
            adjustments[name] = {}
            if _exit_is_statically_unavailable(
                predicate=declaration.available,
                period=self.period,
                ages=context.ages,
            ):
                if callable(declaration.rule):
                    rules[name] = functions[*prefix, "rule"]
                continue
            for member, target in declaration.targets.items():
                if not isinstance(target.projection, Mapping):
                    raise TypeError("Exit query requires phase-resolved projections.")
                target_projection = {
                    state: _compose(
                        projection=projection,
                        function=functions[
                            *prefix, "targets", member, "projection", state
                        ],
                        source=source_key,
                        physical_names=physical_names,
                    )
                    for state in target.projection
                }
                operand = f"exit_{exit_index}_{member}"
                producer = self.make(
                    kind=QueryKind.PARTICIPATING,
                    owner=target.regime,
                    projection=target_projection,
                    guards=child_guards,
                    ancestry=child_ancestry,
                    role=target.role or "",
                    ancestors=ancestors,
                    price_information=policy_information
                    | frozenset(declaration.observes),
                    source_projection={
                        state: functions[
                            *prefix, "targets", member, "projection", state
                        ]
                        for state in target.projection
                    },
                )
                inputs[operand] = producer
                destination_members = self.contexts[target.regime].stakeholders
                index = (
                    None
                    if destination_members is None
                    else destination_members.index(target.role)
                )
                targets[name][member] = (operand, index)
                adjustment = functions.get(
                    (*prefix, "targets", member, "utility_adjustment")
                )
                if adjustment is not None:
                    adjustments[name][member] = adjustment
            if callable(declaration.rule):
                rules[name] = functions[*prefix, "rule"]
            if isinstance(declaration.rule, Consent):
                _validate_consent_context(
                    contexts=self.contexts, source=owner, declaration=declaration
                )
                target = declaration.targets["self"]
                counterparty = declaration.rule.counterparty
                if not isinstance(counterparty.projection, Mapping):
                    raise TypeError(
                        "Counterparty query requires a phase-resolved projection."
                    )
                own_operand, other_operand = (
                    f"own_raw_{exit_index}",
                    f"counterparty_raw_{exit_index}",
                )
                inputs[own_operand] = self.make(
                    kind=QueryKind.RAW,
                    owner=owner,
                    projection=projection,
                    guards=child_guards,
                    ancestry=child_ancestry,
                    ancestors=ancestors,
                    price_information=policy_information,
                    source_projection={
                        state: with_signature(
                            NamedOperand(name=state),
                            args=[state],
                            enforce=False,
                        )
                        for state in context.state_action_space.state_names
                    },
                )
                inputs[other_operand] = self.make(
                    kind=QueryKind.RAW,
                    owner=counterparty.regime,
                    projection={
                        state: _compose(
                            projection=projection,
                            function=functions[
                                *prefix, "counterparty", "projection", state
                            ],
                            source=source_key,
                            physical_names=physical_names,
                        )
                        for state in counterparty.projection
                    },
                    guards=child_guards,
                    ancestry=child_ancestry,
                    ancestors=ancestors,
                    price_information=policy_information
                    | frozenset(declaration.observes),
                    source_projection={
                        state: functions[*prefix, "counterparty", "projection", state]
                        for state in counterparty.projection
                    },
                )
                couple_operand = targets[name]["self"][0]
                participating = self.nodes[inputs[couple_operand]]
                hidden = participating.read.policy_information - (
                    policy_information | frozenset(declaration.observes)
                )
                if hidden:
                    raise RegimeInitializationError(
                        f"Consent exit {owner}.{name} cannot form a target policy "
                        f"requiring unobserved decision information {sorted(hidden)!r}."
                    )
                destination_members = self.contexts[target.regime].stakeholders
                consent_reads[name] = (
                    own_operand,
                    other_operand,
                    couple_operand,
                    None
                    if destination_members is None
                    else destination_members.index(counterparty.role),
                )
        members = context.stakeholders or ("self",)
        reducer = _ExitReducer(
            kind=kind,
            members=members,
            exits=MappingProxyType(dict(exits)),
            targets=targets,
            adjustments=adjustments,
            availability=availability,
            rules=rules,
            consent_reads=consent_reads,
            weights=None
            if context.pareto_weights is None
            else context.pareto_weights.compute,
            n_actions=math.prod(
                value.size for value in context.state_action_space.actions.values()
            ),
            n_discrete_action_axes=sum(
                isinstance(context.grids[name], DiscreteGrid)
                for name in context.state_action_space.action_names
            ),
        )
        return inputs, reducer


def bind_exit_query_graph(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    canonical_regimes: Mapping[RegimeName, Regime],
    root: RegimeName,
    period: int,
    phase: Phase = "solve",
    kind: QueryKind = QueryKind.FULL,
) -> tuple[Mapping[str, QueryNode], str]:
    """Bind demanded public reads after every actual canonical context exists."""
    aliases = {
        (owner, name): f"_lcm_exit_operand_{index}"
        for index, (owner, name) in enumerate(
            (owner, name)
            for owner, context in contexts.items()
            for name in sorted(context.flat_param_names)
        )
    }
    builder = _QueryGraphBuilder(
        contexts=contexts,
        canonical_regimes=canonical_regimes,
        period=period,
        phase=phase,
        aliases=aliases,
        operand_paths={alias: path for path, alias in aliases.items()},
    )
    projection = {
        name: with_signature(NamedOperand(name=name), args=[name], enforce=False)
        for name in contexts[root].state_action_space.state_names
    }
    root_key = builder.make(
        kind=kind, owner=root, projection=projection, guards=(), ancestry=()
    )
    _bind_observation_sources(nodes=builder.nodes, period=period)
    _validate_query_projection_bindings(nodes=builder.nodes, period=period)
    return MappingProxyType(builder.nodes), root_key


def _bind_observation_sources(*, nodes: dict[str, QueryNode], period: int) -> None:
    """Locate declared target-only observations at their producing query context."""
    for key, node in tuple(nodes.items()):
        plan = node.context.current_iid_plans.get(period)
        supplied = frozenset() if plan is None else plan.physical_names
        missing = node.read.policy_information - supplied
        requested = missing
        sources = {}
        aliases = {}
        counts = {}
        for producer in node.inputs.values():
            target = nodes[producer]
            target_plan = target.context.current_iid_plans.get(period)
            if target_plan is None:
                continue
            for group, binding, support in target_plan.bindings:
                if requested & binding.group.shocks.keys():
                    if group in sources and (
                        counts[group] != support.weights.size
                        or not _same_observation_context(
                            source=nodes[sources[group]],
                            target=target,
                            group=group,
                            shocks=frozenset(requested & binding.group.shocks.keys()),
                            period=period,
                        )
                    ):
                        raise RegimeInitializationError(
                            f"Policy query {key!r} observes group {group!r} in "
                            "different producing law contexts."
                        )
                    sources.setdefault(group, producer)
                    aliases.setdefault(group, [])
                    if producer not in aliases[group]:
                        aliases[group].append(producer)
                    counts.setdefault(group, support.weights.size)
                    missing -= binding.group.shocks.keys()
        if missing:
            raise ValueError(
                f"Policy query {key!r} has no producing context for observed "
                f"physical inputs {sorted(missing)!r}."
            )
        nodes[key] = replace(
            node,
            observation_sources=MappingProxyType(sources),
            observation_aliases=MappingProxyType(
                {group: tuple(producers) for group, producers in aliases.items()}
            ),
            observation_counts=tuple(counts.values()),
        )


def _same_observation_context(
    *,
    source: QueryNode,
    target: QueryNode,
    group: ShockGroupName,
    shocks: frozenset[ShockName],
    period: int,
) -> bool:
    """Prove equal producing operands, including captured helpers and parameters."""
    from _lcm.shock_projection import _law_record  # noqa: PLC0415

    if source is target:
        return True
    source_plan = source.context.current_iid_plans.get(period)
    target_plan = target.context.current_iid_plans.get(period)
    if (
        source_plan is None
        or target_plan is None
        or (
            (source.read.phase, source.read.period)
            != (target.read.phase, target.read.period)
        )
    ):
        return False
    bindings = [
        {name: binding for name, binding, _ in plan.bindings}
        for plan in (source_plan, target_plan)
    ]
    if any(group not in bound for bound in bindings):
        return False
    for shock in shocks:
        records = [
            _law_record(
                shock=shock, group=group, binding=bound[group], parameter_paths={}
            )
            for bound in bindings
        ]
        if records[0] != records[1]:
            return False
        if any(
            state not in source.read.projection
            or state not in target.read.projection
            or _fingerprint((bindings[0][group].state_domains.get(state),))
            != _fingerprint((bindings[1][group].state_domains.get(state),))
            or _fingerprint((source.read.projection[state],))
            != _fingerprint((target.read.projection[state],))
            for state in records[0][1]
        ):
            return False
    return True


def _validate_query_projection_bindings(
    *, nodes: Mapping[str, QueryNode], period: int
) -> None:
    """Apply S5's exact proof only to physical reads in their own caller context."""
    from _lcm.shock_projection import (  # noqa: PLC0415
        validate_shared_discrete_projection,
    )

    for node in nodes.values():
        if not node.ancestors:
            continue
        parent = nodes[node.ancestors[-1]]
        source = parent.context.current_iid_plans.get(period)
        target = node.context.current_iid_plans.get(period)
        validate_shared_discrete_projection(
            query_kind=node.read.kind.value,
            target=node.read.target,
            physical_inputs=node.read.physical_inputs,
            source_physical_inputs=parent.read.policy_information,
            projection=node.source_projection,
            source_bindings={}
            if source is None
            else {name: binding for name, binding, _ in source.bindings},
            target_bindings={}
            if target is None
            else {name: binding for name, binding, _ in target.bindings},
            source_domains=parent.context.user_regimes_at(period=period)[
                parent.read.target
            ].states,
            target_domains=node.context.user_regimes_at(period=period)[
                node.read.target
            ].states,
        )
        for group, producer in parent.observation_sources.items():
            origin = nodes[producer]
            origin_plan = origin.context.current_iid_plans[period]
            if origin_plan is None:
                raise RegimeInitializationError(
                    "An observed category requires its producing law plan."
                )
            binding = next(
                binding for name, binding, _ in origin_plan.bindings if name == group
            )
            shared = frozenset(
                shock
                for shock, law in binding.group.shocks.items()
                if isinstance(law, Discrete)
                and shock in parent.read.policy_information & node.read.physical_inputs
            )
            if not shared or origin is node:
                continue
            if not _same_observation_context(
                source=origin, target=node, group=group, shocks=shared, period=period
            ):
                raise RegimeInitializationError(
                    f"Shared discrete observation {sorted(shared)!r} for "
                    f"{node.read.kind.value}[{node.read.target}] has unproved "
                    "producing law-input identity."
                )
            validate_shared_discrete_projection(
                query_kind=node.read.kind.value,
                target=node.read.target,
                physical_inputs=shared,
                source_physical_inputs=shared,
                projection={
                    state: with_signature(
                        NamedOperand(name=state),
                        args=[state],
                        enforce=False,
                    )
                    for state in origin.read.projection
                },
                source_bindings={group: binding},
                target_bindings={}
                if target is None
                else {name: item for name, item, _ in target.bindings},
                source_domains=origin.context.user_regimes_at(period=period)[
                    origin.read.target
                ].states,
                target_domains=node.context.user_regimes_at(period=period)[
                    node.read.target
                ].states,
            )


def lower_exit_queries(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    canonical_regimes: Mapping[RegimeName, Regime],
    root: RegimeName,
    period: int,
) -> _PolicyQueryKernel:
    """Lower a bound Full graph into the existing native publication schedule."""
    nodes, root_key = bind_exit_query_graph(
        contexts=contexts,
        canonical_regimes=canonical_regimes,
        root=root,
        period=period,
    )
    root_context = contexts[root]
    kernel = build_policy_query_kernel(
        root_context=root_context,
        period=period,
        definitions=nodes,
        root=root_key,
        chunk_size=root_context.axis_widths.get("query_cell", 64),
    )
    return kernel.with_fixed_params(
        fixed_flat_params=MappingProxyType(
            {
                owner: regime.solution.resolved_fixed_params
                for owner, regime in canonical_regimes.items()
            }
        )
    )


def install_exit_policy_queries(
    *,
    canonical_regimes: Mapping[RegimeName, Regime],
    contexts: Mapping[RegimeName, SolverBuildContext],
    value_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
) -> MappingProxyType[RegimeName, Regime]:
    """Install Full roots for exits and tuple-preserving collective integration."""
    from dataclasses import replace  # noqa: PLC0415

    updated = dict(canonical_regimes)
    for owner, context in contexts.items():
        kernels = dict(canonical_regimes[owner].solution.period_kernels)
        changed = False
        for period in kernels:
            if (
                value_periods_by_regime is not None
                and period not in value_periods_by_regime.get(owner, ())
            ):
                continue
            if (
                context.participation_queries.get(period)
                or context.user_regimes_at(period=period)[owner].exits
                or (
                    context.stakeholders is not None
                    and context.current_iid_plans.get(period) is not None
                )
            ):
                kernels[period] = lower_exit_queries(
                    contexts=contexts,
                    canonical_regimes=canonical_regimes,
                    root=owner,
                    period=period,
                )
                changed = True
        if changed:
            updated[owner] = replace(
                canonical_regimes[owner],
                solution=replace(
                    canonical_regimes[owner].solution,
                    period_kernels=MappingProxyType(kernels),
                    period_signatures=MappingProxyType(
                        {
                            period: (signature, "typed-exit-query", period)
                            if isinstance(kernels[period], _PolicyQueryKernel)
                            else signature
                            for period, signature in canonical_regimes[
                                owner
                            ].solution.period_signatures.items()
                        }
                    ),
                ),
            )
    return MappingProxyType(updated)


def _validate_custom_rules(*, exits: Mapping[str, Exit]) -> None:
    """A global custom choice has one authored rule, rather than rival arbiters."""
    from _lcm.regime_building.value_adapter import ValueAdapter  # noqa: PLC0415
    from lcm.exceptions import RegimeInitializationError  # noqa: PLC0415

    identities = {}
    for name, declaration in exits.items():
        rule = declaration.rule
        if not callable(rule):
            continue
        while isinstance(rule, ValueAdapter):
            rule = rule.func
        identities[name] = id(rule)
    if len(set(identities.values())) > 1:
        raise RegimeInitializationError(
            f"Custom exits {tuple(identities)!r} must share "
            "the same authored rule object."
        )


def _validate_consent_context(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    source: RegimeName,
    declaration: Exit,
) -> None:
    """Do not omit a member or invent an ambiguous counterparty Raw channel."""
    from lcm.exceptions import RegimeInitializationError  # noqa: PLC0415

    counterparty = cast("Consent", declaration.rule).counterparty
    if (
        contexts[source].stakeholders is not None
        or contexts[counterparty.regime].stakeholders is not None
    ):
        raise RegimeInitializationError(
            "Consent requires unambiguous singleton source "
            "and counterparty Raw channels."
        )
    target = declaration.targets["self"]
    if set(contexts[target.regime].stakeholders or ()) != {
        target.role,
        counterparty.role,
    }:
        raise RegimeInitializationError(
            "Consent comparisons must cover every formation-destination stakeholder."
        )
