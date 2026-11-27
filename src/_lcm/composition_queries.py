"""Bind a regime's composition action to the canonical scalar policy-query decision."""

import hashlib
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from itertools import product
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions, with_signature
from dags.signature import rename_arguments
from dags.tree import qname_from_tree_path

from _lcm.composition_actions import (
    compares_at_optimum,
    composition_action_for_phase,
    composition_action_functions,
)
from _lcm.current_iid import CurrentIIDPlan
from _lcm.engine import Regime
from _lcm.off_grid import clip_projection, projected_grid_rules
from _lcm.outside_values import UnalignedOutsideRead, align_outside_read
from _lcm.policy_queries import (
    _PHYSICAL_CONTEXTS,
    OutsideValueRead,
    PolicyAnswer,
    QueryKind,
    QueryNode,
    QueryRead,
    _PolicyQueryKernel,
    _QueryChunks,
    build_policy_query_kernel,
    gated_query_targets,
)
from _lcm.query_information import default_chunk_size
from _lcm.regime_building.collective import landing_member_index
from _lcm.renegotiation import (
    ClippedDefault,
    FixedNode,
    FixedTransfer,
    RenegotiationQueries,
    TransferQueries,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.grid_search import GridSearch
from _lcm.time import landing_step, period_of_step
from _lcm.typing import PytreeValue, QualifiedName
from _lcm.utils.functools import (
    ComposedProjection,
    NamedOperand,
)
from _lcm.utils.functools import (
    call_with_declared_arguments as _call,
)
from lcm.composition_actions import (
    CompositionAction,
    Consent,
    MemberName,
    MutualConsent,
    PCOnActions,
    PCOnWeight,
    Transfer,
)
from lcm.exceptions import RegimeInitializationError, UnsupportedOperationError
from lcm.shocks import Discrete, ShockGroupName, ShockName, _fingerprint
from lcm.typing import (
    BoolND,
    FloatND,
    IntND,
    Phase,
    ReferenceName,
    RegimeName,
    StateName,
    UserFunction,
)

# Whether the solve's composition action graphs mark the outside reads proved aligned.
# Read once per lowering when the model is built; simulation and participation graphs
# never mark them.
_READ_STORED_OUTSIDE_VALUES: bool = True

# Whether the solve's enumerated composition action choices evaluate the nodes where the
# composition action is available and one node for all the others
# (`_bind_available_nodes`), instead of every node. Read once per lowering when the
# model is built; simulation and participation graphs always enumerate every node.
_COLLAPSE_UNAVAILABLE_NODES: bool = True


def _predicate(
    *,
    function: Callable[..., PytreeValue],
    operands: Mapping[ReferenceName, PytreeValue],
) -> BoolND:
    """Return a scalar Boolean, without reducing undeclared axes."""
    value = jnp.asarray(_call(function=function, operands=operands))
    if value.shape != ():
        raise ValueError("Composition action availability must be a scalar Boolean.")
    return value


@dataclass(frozen=True, kw_only=True)
class _CompositionReducer:
    """One scalar rule shared by solve and actual agent replay.

    Target values are already evaluated at one common action: of the target's Raw
    problem, or of the formed couple's Participating policy under consent. This
    reducer never maximizes the destination again to manufacture consent.
    """

    kind: QueryKind
    """The source policy problem being answered."""
    members: tuple[MemberName, ...]
    """Canonical source tuple order, or self for a singleton."""
    declaration: CompositionAction | None
    """The regime's composition action in the current phase, or None."""
    targets: Mapping[MemberName, tuple[str, int]]
    """Member to producer operand and destination member index."""
    adjustments: Mapping[MemberName, Callable[..., PytreeValue]]
    """Source-context utility adjustments, evaluated once in each payoff."""
    available: Callable[..., PytreeValue] | None
    """Source-context predicate; None means available."""
    consent_read: tuple[str, str, str, int, int, int] | None
    """Own Raw, counterparty Raw, couple producer and counterparty tuple index.

    Then the member index of the own and of the counterparty Raw value.
    """
    n_actions: int
    """Flattened canonical stay product; the composition action takes the next ID."""
    renegotiation: RenegotiationQueries | None = None
    """Raw policies across a declared discrete state, before applying participation."""
    transfer: TransferQueries | None = None
    """Outside reads at each candidate transfer under mutual consent divorce."""

    def __post_init__(self) -> None:
        """Keep declaration containers immutable and IDs inside their native width."""
        if not 0 < self.n_actions < jnp.iinfo(jnp.int32).max:
            raise RegimeInitializationError(
                "Composition action alternatives must fit stable int32 IDs."
            )
        for name in ("targets", "adjustments"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def composition_actions(
        self,
    ) -> dict[
        int,
        tuple[
            Mapping[MemberName, tuple[str, int]],
            Mapping[MemberName, Callable[..., PytreeValue]],
        ],
    ]:
        """Each composition action ID with its members' reads and adjustments.

        The composition action at the default takes the ID after the stay actions; a
        separation at candidate transfer `k` takes the `k`-th ID after that.
        """
        composition_actions = {self.n_actions: (self.targets, self.adjustments)}
        if self.transfer is not None:
            composition_actions.update(
                {
                    self.n_actions + 1 + index: pair
                    for index, pair in enumerate(
                        zip(
                            self.transfer.targets,
                            self.transfer.adjustments,
                            strict=True,
                        )
                    )
                }
            )
        return composition_actions

    def _payoffs(
        self,
        *,
        operands: Mapping[ReferenceName, PytreeValue],
        targets: Mapping[MemberName, tuple[str, int]] | None = None,
        adjustments: Mapping[MemberName, Callable[..., PytreeValue]] | None = None,
    ) -> tuple[dict[str, FloatND], BoolND, dict[str, BoolND]]:
        """Each member's outside payoff, by default at the composition action's reads.

        Also returns whether each member's payoff is resolved: true where the
        composition action is not available or the member has no read, else the read's
        status and a payoff that is not NaN.
        """
        targets = self.targets if targets is None else targets
        adjustments = self.adjustments if adjustments is None else adjustments
        declaration = cast("CompositionAction", self.declaration)
        active = (
            jnp.asarray(a=True)
            if self.available is None
            else _predicate(function=self.available, operands=operands)
        )
        resolved = {}
        payoffs = {}
        for member in declaration.destinations:
            if member not in targets:
                payoffs[member] = jnp.asarray(-jnp.inf)
                resolved[member] = jnp.asarray(a=True)
                continue
            key, index = targets[member]
            answer = cast("PolicyAnswer", operands[key])
            value = answer[0][index]
            if member in adjustments:
                adjustment = jnp.where(
                    active,
                    cast(
                        "FloatND",
                        _call(function=adjustments[member], operands=operands),
                    ),
                    0.0,
                )
                if adjustment.shape != ():
                    raise ValueError(
                        "Composition action utility adjustment must be scalar."
                    )
                value = value + adjustment
            payoffs[member] = value
            resolved[member] = ~active | (answer[3] & ~jnp.isnan(value))
        return payoffs, active, resolved

    def keep(self, **operands: PytreeValue) -> BoolND:
        """Apply built-in dissolution thresholds before the common stay argmax.

        A member is willing to stay when its stay value is above its composition action
        payoff, or equal to it when the rule's `tie` is `"stay"`, and always where its
        destination has no feasible action (`_undefined`).
        """
        rule = None if self.declaration is None else self.declaration.rule
        if not isinstance(rule, PCOnActions):
            return jnp.asarray(a=True)
        q = cast("FloatND", operands["q"])
        payoffs, active, _ = self._payoffs(operands=operands)
        threshold = jnp.stack(tuple(payoffs[member] for member in self.members))
        comparisons = q >= threshold if rule.tie == "stay" else q > threshold
        return ~active | jnp.all(comparisons | self._undefined(operands=operands))

    def _undefined(self, *, operands: Mapping[ReferenceName, PytreeValue]) -> BoolND:
        """Mark each member whose destination has no feasible action.

        Such a member's value of the outside option is `-inf`. It accepts every
        feasible stay action, whatever the rule's `tie`.
        """
        return jnp.stack(
            tuple(
                member in self.targets
                and ~cast("PolicyAnswer", operands[self.targets[member][0]])[2]
                for member in self.members
            )
        )

    def __call__(self, **operands: PytreeValue) -> PolicyAnswer:
        """Route weak dissolution or strict consent, preserving common tuple/ID."""
        stay = cast("PolicyAnswer", operands["answer"])
        payoffs, active, resolved = self._payoffs(operands=operands)
        valid = jnp.all(jnp.stack((*resolved.values(), stay[3])))
        if self.renegotiation is not None:
            return self._renegotiated(
                operands=operands, payoffs=payoffs, active=active, valid=valid
            )
        if compares_at_optimum(cast("CompositionAction", self.declaration).rule):
            return self._at_optimum(
                operands=operands, payoffs=payoffs, active=active, valid=valid
            )
        eligible, integrity = self._eligible(
            operands=operands, payoffs=payoffs, active=active
        )
        move = eligible & (self.kind is QueryKind.FULL)
        if not isinstance(cast("CompositionAction", self.declaration).rule, Consent):
            move &= ~stay[2]
        value = jnp.stack(tuple(payoffs[member] for member in self.members))
        return (
            jnp.where(move, value, stay[0]),
            jnp.where(move, self.n_actions, stay[1]).astype(jnp.int32),
            stay[2] | move,
            valid & integrity,
            stay[4],
        )

    def _renegotiated(
        self,
        *,
        operands: Mapping[ReferenceName, PytreeValue],
        payoffs: Mapping[str, FloatND],
        active: BoolND,
        valid: BoolND,
    ) -> PolicyAnswer:
        adjustment = cast("RenegotiationQueries", self.renegotiation)
        outside = jnp.stack(tuple(payoffs[member] for member in self.members))
        index, kept = adjustment.select(
            arguments=operands,
            outside=outside,
            undefined=self._undefined(operands=operands),
            active=active,
        )
        raw = tuple(
            cast("PolicyAnswer", operands[name]) for name in adjustment.operands
        )
        value = jnp.stack(tuple(answer[0] for answer in raw))[index]
        action = jnp.stack(tuple(answer[1] for answer in raw))[index]
        eligible, integrity = self._eligible(
            operands=operands, payoffs=payoffs, active=active
        )
        move = eligible & ~kept & (self.kind is QueryKind.FULL)
        return (
            jnp.where(move, outside, jnp.where(kept, value, -jnp.inf)),
            jnp.where(move, self.n_actions, jnp.where(kept, action, -1)).astype(
                jnp.int32
            ),
            kept | move,
            valid & integrity & jnp.all(jnp.stack(tuple(answer[3] for answer in raw))),
            cast("PolicyAnswer", operands["answer"])[4],
        )

    def _at_optimum(
        self,
        *,
        operands: Mapping[ReferenceName, PytreeValue],
        payoffs: Mapping[str, FloatND],
        active: BoolND,
        valid: BoolND,
    ) -> PolicyAnswer:
        """Compare each member's value at the optimum with its outside option.

        The stay answer is the optimum of the Pareto-weighted objective, since `keep`
        masks no action. A member is willing to stay when that value is above the
        value of its outside option, or equal to it when the rule's `tie` is `"stay"`.
        No feasible action leaves every member unwilling. Under a transfer, the values
        of the outside options are those at the default transfer. A pair in which
        exactly one member refuses separates at the candidate the transfer selects,
        under that candidate's composition action ID. The value of the outside option
        at a candidate must be resolved only where the selection reads it.
        """
        stay = cast("PolicyAnswer", operands["answer"])
        rule = cast(
            "PCOnActions | PCOnWeight | MutualConsent",
            cast("CompositionAction", self.declaration).rule,
        )
        inside = stay[0]
        outside = jnp.stack(tuple(payoffs[member] for member in self.members))
        willing = (
            (inside >= outside if rule.tie == "stay" else inside > outside)
            | self._undefined(operands=operands)
        ) & stay[2]
        stays = (
            jnp.any(willing) if isinstance(rule, MutualConsent) else jnp.all(willing)
        )
        identifier = jnp.asarray(self.n_actions)
        settles = jnp.asarray(a=False)
        eligible, integrity = self._eligible(
            operands=operands, payoffs=payoffs, active=active
        )
        if self.transfer is not None:
            candidates = tuple(
                self._payoffs(operands=operands, targets=targets, adjustments=paid)
                for targets, paid in zip(
                    self.transfer.targets, self.transfer.adjustments, strict=True
                )
            )
            # Row 0 is the default and row `1 + k` candidate `k`, so a row's composition
            # action ID is `n_actions` plus its index.
            settled = jnp.stack(
                (
                    outside,
                    *(
                        jnp.stack(tuple(payoff[member] for member in self.members))
                        for payoff, _, _ in candidates
                    ),
                )
            )
            nearest, settles, decided = self.transfer.select(
                inside=inside,
                willing=willing,
                outside=settled,
                resolved=jnp.stack(
                    tuple(
                        jnp.stack(tuple(checked[member] for member in self.members))
                        for _, _, checked in candidates
                    )
                ),
                supported=jnp.stack(
                    (
                        eligible,
                        *(
                            jnp.all(
                                jnp.stack(
                                    tuple(
                                        cast("PolicyAnswer", operands[key])[2]
                                        for key, _ in targets.values()
                                    )
                                )
                            )
                            for targets in self.transfer.targets
                        ),
                    )
                ),
                default=cast(
                    "FloatND",
                    _call(function=self.transfer.default, operands=operands),
                ),
            )
            settles &= active
            valid &= decided
            outside = jnp.where(settles, settled[nearest], outside)
            identifier = jnp.where(settles, self.n_actions + nearest, identifier)
        kept = jnp.where(active, stays & ~settles, stay[2])
        move = (settles | (eligible & ~kept)) & (self.kind is QueryKind.FULL)
        value = jnp.where(move, outside, jnp.where(kept, inside, -jnp.inf))
        return (
            value,
            jnp.where(move, identifier, jnp.where(kept, stay[1], -1)).astype(jnp.int32),
            kept | move,
            valid & integrity,
            stay[4],
        )

    def updated_state(self, **operands: PytreeValue) -> FloatND | IntND:
        """Carry the selected stay coordinate alongside its action and value."""
        adjustment = cast("RenegotiationQueries", self.renegotiation)
        payoffs, active, _ = self._payoffs(operands=operands)
        index, kept = adjustment.select(
            arguments=operands,
            outside=jnp.stack(tuple(payoffs[member] for member in self.members)),
            undefined=self._undefined(operands=operands),
            active=active,
        )
        return jnp.where(kept, index, cast("IntND", operands[adjustment.state]))

    def _eligible(
        self,
        *,
        operands: Mapping[ReferenceName, PytreeValue],
        payoffs: Mapping[str, FloatND],
        active: BoolND,
    ) -> tuple[BoolND, BoolND]:
        """Share composition action support and strict common-policy consent."""
        if isinstance(cast("CompositionAction", self.declaration).rule, Consent):
            if self.consent_read is None:
                return jnp.asarray(a=False), jnp.asarray(a=True)
            own_key, other_key, couple_key, other_index, own_raw, other_raw = (
                self.consent_read
            )
            own = cast("PolicyAnswer", operands[own_key])
            other = cast("PolicyAnswer", operands[other_key])
            couple = cast("PolicyAnswer", operands[couple_key])
            other_value = couple[0][other_index]
            own_value = own[0][own_raw]
            other_single = other[0][other_raw]
            return (
                active
                & couple[2]
                & (payoffs["self"] > own_value)
                & (other_value > other_single),
                ~active | (own[3] & other[3] & couple[3]),
            )
        support = jnp.all(
            jnp.stack(
                tuple(
                    cast("PolicyAnswer", operands[operand])[2]
                    for operand, _ in self.targets.values()
                )
                or (jnp.asarray(a=False),)
            )
        )
        return active & support, jnp.asarray(a=True)


def bind_composition_reducer(
    *,
    reducer: _CompositionReducer,
    operand_names: tuple[str, ...],
) -> tuple[Callable[..., BoolND], Callable[..., PolicyAnswer]]:
    """Expose explicit native signatures instead of free parameter inference."""
    return (
        with_signature(
            reducer.keep,
            args=["q", *operand_names],
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
            active &= _predicate(function=predicate, operands=operands)
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
    outside_reads: dict[str, tuple[RegimeName, str]] = field(default_factory=dict)
    """Reads of a destination's problem: the deciding source and the member priced.

    The destinations of every composition action that needs no consent, at any
    depth, and the counterparty of a formation by consent. A consent destination and
    the reads of the source's own problem are not outside reads.
    """

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
            & (context.functions.keys() | {"_lcm_composition_result"})
        ):
            combined = concatenate_functions(
                functions={**context.functions, "_lcm_composition_result": renamed},
                targets="_lcm_composition_result",
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
        composition_action_source: RegimeName | None = None,
    ) -> str:
        """Bind a demanded typed read and reject genuine recursive policy cycles.

        With `composition_action_source`, the read is the landing of that regime's
        composition action, and each projected state follows its destination grid's
        `off_grid` rule in this period (`ProjectedGridRule`).
        """
        from _lcm.model_graph import _composition_action_is_statically_unavailable  # noqa: PLC0415

        if kind not in {QueryKind.RAW, QueryKind.PARTICIPATING, QueryKind.FULL}:
            raise ValueError(
                "The policy graph builder requires Raw, Participating or Full."
            )
        if (kind, owner) in ancestry:
            raise ValueError(f"Policy query cycle: {(*ancestry, (kind, owner))!r}")
        context = self.contexts[owner]
        # A regime of a later stage of the period is read at its occurrence.
        occurrence = landing_step(ages=context.ages, step=self.period - 1, target=owner)
        if occurrence == self.period and self.period not in context.Q_and_F_functions:
            raise ValueError(
                f"Destination {owner!r} has no active policy at period "
                f"{period_of_step(ages=context.ages, step=self.period)}."
            )
        key = f"query_{len(self.keys)}"
        self.keys.append(key)
        axes = self.canonical_regimes[owner].solution.period_state_axes
        grid_rules = (
            ()
            if composition_action_source is None
            else projected_grid_rules(
                source=composition_action_source,
                target=owner,
                states=tuple(projection),
                grids=context.grids,
                nodes={
                    **context.state_action_space.states,
                    **({} if axes is None else axes.get(occurrence, {})),
                },
            )
        )
        projection = clip_projection(projection=projection, rules=grid_rules)
        if occurrence != self.period:
            # Only the landing feasibility check uses the destination's draw.
            # The source's policy query must have no information about that draw.
            plan = context.current_iid_plans.get(occurrence)
            context = replace(
                context,
                current_iid_plans=MappingProxyType(
                    {} if plan is None else {occurrence: plan}
                ),
            )
            self.nodes[key] = QueryNode(
                read=QueryRead(
                    kind=kind,
                    target=owner,
                    phase=self.phase,
                    period=self.period,
                    role=role,
                    projection=cast("Mapping[StateName, UserFunction]", projection),
                    state_inputs=frozenset(context.state_action_space.state_names),
                    physical_inputs=frozenset(),
                    operand_paths={
                        name: path
                        for name, path in self.operand_paths.items()
                        if any(
                            name in inspect.signature(function).parameters
                            for function in (*projection.values(), *guards)
                        )
                    },
                    policy_information=frozenset(),
                    price_information=price_information,
                ),
                context=context,
                available=_guard(guards),
                ancestors=ancestors,
                source_projection=cast(
                    "Mapping[StateName, UserFunction]",
                    projection if source_projection is None else source_projection,
                ),
                grid_rules=grid_rules,
                landing=occurrence,
            )
            return key
        declaration = composition_action_for_phase(
            declaration=context.user_regimes_at(period=self.period)[
                owner
            ].composition_action,
            phase=self.phase,
        )
        if kind is QueryKind.RAW or (
            kind is QueryKind.PARTICIPATING
            and declaration is not None
            and isinstance(declaration.rule, Consent)
        ):
            declaration = None
        inactive = (
            declaration is not None
            and _composition_action_is_statically_unavailable(
                predicate=declaration.available, period=self.period, ages=context.ages
            )
        )
        # Only the solved occupant draws; simulation realises its shocks.
        rule = (
            declaration.nodes
            if declaration is not None
            and not inactive
            and kind is QueryKind.FULL
            and not ancestors
            and self.phase == "solve"
            else None
        )
        functions = {
            path: self.bound(owner=owner, path=path, function=function)
            for path, function in composition_action_functions(
                declaration=declaration, phase=self.phase
            ).items()
            if not inactive or path[-1] == "available"
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
            () if declaration is None or inactive else declaration.observes
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
        inputs, reducer = self._expand_composition_action(
            kind=kind,
            owner=owner,
            projection=projection,
            guards=guards,
            ancestry=(*ancestry, (kind, owner)),
            declaration=declaration,
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
                    "period",
                    "age",
                )
            )
        )
        keep, decide = bind_composition_reducer(reducer=reducer, operand_names=names)
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
                None
                if reducer.consent_read is None
                else inputs[reducer.consent_read[0]]
            ),
            participation=None if declaration is None else keep,
            decision=None if declaration is None else decide,
            state_updates={}
            if reducer.renegotiation is None
            else {
                reducer.renegotiation.state: with_signature(
                    reducer.updated_state, args=list(names), enforce=False
                )
            },
            gated_channels={
                target: tuple(
                    index
                    for index, _ in enumerate(self.contexts[target].members)
                    if any(
                        landing_member_index(
                            members=self.contexts[target].members,
                            role=route.target_member,
                        )
                        == index
                        for route in context.laws[owner]
                        .gated_edges[target]
                        .legs.values()
                    )
                )
                for target in gated_query_targets(context=context, period=self.period)
            },
            available=_guard(guards),
            ancestors=ancestors,
            source_projection=cast(
                "Mapping[StateName, UserFunction]",
                projection if source_projection is None else source_projection,
            ),
            composition_action_targets={}
            if declaration is None
            else {
                identifier: {
                    member: inputs[operand] for member, (operand, _) in targets.items()
                }
                for identifier, (targets, _) in reducer.composition_actions().items()
            },
            composition_action_adjustments={}
            if declaration is None
            else {
                identifier: adjustments
                for identifier, (
                    _,
                    adjustments,
                ) in reducer.composition_actions().items()
            },
            sample=None if rule is None else (rule, ()),
            composition_action_availability=reducer.available,
            grid_rules=grid_rules,
        )
        return key

    def _expand_composition_action(
        self,
        *,
        kind: QueryKind,
        owner: RegimeName,
        projection: Mapping[StateName, Callable[..., PytreeValue]],
        guards: tuple[Callable[..., PytreeValue], ...],
        ancestry: tuple[tuple[QueryKind, str], ...],
        declaration: CompositionAction | None,
        functions: Mapping[tuple[str, ...], Callable[..., PytreeValue]],
        source_key: str,
        ancestors: tuple[str, ...],
        policy_information: frozenset[str],
    ) -> tuple[dict[str, str], _CompositionReducer]:
        """Expand only declared benchmarks and consent Raw reads."""
        from _lcm.model_graph import _composition_action_is_statically_unavailable  # noqa: PLC0415

        context = self.contexts[owner]
        current_plan = context.current_iid_plans.get(self.period)
        physical_names = (
            frozenset() if current_plan is None else current_plan.physical_names
        )
        inputs = {}
        targets = {}
        adjustments = {}
        consent_read = None
        predicate = None
        renegotiation = None
        transfer_queries = None
        if declaration is not None:
            predicate = functions.get(("composition_action", "available"))
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
            if not _composition_action_is_statically_unavailable(
                predicate=declaration.available,
                period=self.period,
                ages=context.ages,
            ):
                transfer = (
                    declaration.rule.transfer
                    if isinstance(declaration.rule, MutualConsent)
                    else None
                )
                default = (
                    None
                    if transfer is None
                    else self._transfer_default(
                        transfer=transfer, declaration=declaration, functions=functions
                    )
                )
                inputs, targets, adjustments, consent_read = (
                    self._composition_action_reads(
                        owner=owner,
                        declaration=declaration,
                        projection=projection
                        if transfer is None or default is None
                        else {
                            **projection,
                            transfer.name: _compose(
                                projection=projection,
                                function=default,
                                source=source_key,
                                physical_names=physical_names,
                            ),
                        },
                        guards=child_guards,
                        ancestry=ancestry,
                        functions=functions,
                        source_key=source_key,
                        ancestors=ancestors,
                        policy_information=policy_information,
                        physical_names=physical_names,
                    )
                )
                if transfer is not None and default is not None:
                    settled, adjustments, transfer_queries = self._transfer_reads(
                        transfer=transfer,
                        default=default,
                        targets=targets,
                        adjustments=adjustments,
                        owner=owner,
                        declaration=declaration,
                        projection=projection,
                        guards=child_guards,
                        ancestry=ancestry,
                        functions=functions,
                        source_key=source_key,
                        ancestors=ancestors,
                        policy_information=policy_information,
                        physical_names=physical_names,
                    )
                    inputs.update(settled)
                if (
                    isinstance(declaration.rule, PCOnWeight)
                    and declaration.rule.state is not None
                ):
                    state = declaration.rule.state
                    nodes = context.state_action_space.states[state]
                    names = []
                    for index in range(nodes.size):
                        operand = f"_lcm_renegotiate_{index}"
                        names.append(operand)
                        inputs[operand] = self.make(
                            kind=QueryKind.RAW,
                            owner=owner,
                            projection={
                                **projection,
                                state: FixedNode(value=index),
                            },
                            guards=guards,
                            ancestry=ancestry,
                            ancestors=ancestors,
                            price_information=policy_information,
                        )
                    renegotiation = RenegotiationQueries(
                        state=state,
                        n_nodes=nodes.size,
                        operands=tuple(names),
                        tie=declaration.rule.tie,
                    )
        members = context.members
        reducer = _CompositionReducer(
            kind=kind,
            members=members,
            declaration=declaration,
            targets=targets,
            adjustments=adjustments,
            available=predicate,
            consent_read=consent_read,
            renegotiation=renegotiation,
            transfer=transfer_queries,
            n_actions=math.prod(
                value.size for value in context.state_action_space.actions.values()
            ),
        )
        return inputs, reducer

    def _composition_action_reads(
        self,
        *,
        owner: RegimeName,
        declaration: CompositionAction,
        projection: Mapping[StateName, Callable[..., PytreeValue]],
        guards: tuple[Callable[..., PytreeValue], ...],
        ancestry: tuple[tuple[QueryKind, str], ...],
        functions: Mapping[tuple[str, ...], Callable[..., PytreeValue]],
        source_key: str,
        ancestors: tuple[str, ...],
        policy_information: frozenset[str],
        physical_names: frozenset[str],
    ) -> tuple[
        dict[str, str],
        dict[MemberName, tuple[str, int]],
        dict[MemberName, Callable[..., PytreeValue]],
        tuple[str, str, str, int, int, int] | None,
    ]:
        """Expand an available decision's benchmarks and, for consent, its Raw reads."""
        context = self.contexts[owner]
        prefix = ("composition_action",)
        inputs = {}
        targets = {}
        adjustments = {}
        consent_read = None
        for member, target in declaration.destinations.items():
            if not isinstance(target.projection, Mapping):
                raise TypeError(
                    "Composition action query requires phase-resolved projections."
                )
            target_projection = {
                state: _compose(
                    projection=projection,
                    function=functions[
                        *prefix, "destinations", member, "projection", state
                    ],
                    source=source_key,
                    physical_names=physical_names,
                )
                for state in target.projection
            }
            operand = f"composition_action_{member}"
            # An agent changes its composition at most once per period, so a target
            # is read as its problem, without its own composition action. A formed
            # couple keeps its rule for staying. The values of its outside options
            # are again problems.
            producer = self.make(
                kind=QueryKind.PARTICIPATING
                if isinstance(declaration.rule, Consent)
                else QueryKind.RAW,
                owner=target.regime,
                projection=target_projection,
                guards=guards,
                ancestry=ancestry,
                role=target.role or "",
                ancestors=ancestors,
                price_information=policy_information | frozenset(declaration.observes),
                source_projection={
                    state: functions[
                        *prefix, "destinations", member, "projection", state
                    ]
                    for state in target.projection
                },
                composition_action_source=owner,
            )
            inputs[operand] = producer
            if (
                not isinstance(declaration.rule, Consent)
                and self.nodes[producer].landing is None
            ):
                self.outside_reads[producer] = (owner, f"member {member!r}")
            targets[member] = (
                operand,
                landing_member_index(
                    members=self.contexts[target.regime].members, role=target.role
                ),
            )
            adjustment = functions.get(
                (*prefix, "destinations", member, "utility_adjustment")
            )
            if adjustment is not None:
                adjustments[member] = adjustment
        if isinstance(declaration.rule, Consent):
            _validate_consent_context(
                contexts=self.contexts, source=owner, declaration=declaration
            )
            target = declaration.destinations["self"]
            counterparty = declaration.rule.counterparty
            if not isinstance(counterparty.projection, Mapping):
                raise TypeError(
                    "Counterparty query requires a phase-resolved projection."
                )
            inputs["own_raw"] = self.make(
                kind=QueryKind.RAW,
                owner=owner,
                projection=projection,
                guards=guards,
                ancestry=ancestry,
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
            inputs["counterparty_raw"] = self.make(
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
                guards=guards,
                ancestry=ancestry,
                ancestors=ancestors,
                price_information=policy_information | frozenset(declaration.observes),
                source_projection={
                    state: functions[*prefix, "counterparty", "projection", state]
                    for state in counterparty.projection
                },
                composition_action_source=owner,
            )
            self.outside_reads[inputs["counterparty_raw"]] = (owner, "the counterparty")
            couple_operand = targets["self"][0]
            participating = self.nodes[inputs[couple_operand]]
            hidden = participating.read.policy_information - (
                policy_information | frozenset(declaration.observes)
            )
            if hidden:
                raise RegimeInitializationError(
                    f"The formation by consent of {owner} cannot form a target policy "
                    f"requiring unobserved decision information {sorted(hidden)!r}."
                )
            consent_read = (
                "own_raw",
                "counterparty_raw",
                couple_operand,
                landing_member_index(
                    members=self.contexts[target.regime].members,
                    role=counterparty.role,
                ),
                landing_member_index(members=context.members, role=None),
                landing_member_index(
                    members=self.contexts[counterparty.regime].members, role=None
                ),
            )
        return inputs, targets, adjustments, consent_read

    def _transfer_default(
        self,
        *,
        transfer: Transfer,
        declaration: CompositionAction,
        functions: Mapping[tuple[str, ...], Callable[..., PytreeValue]],
    ) -> Callable[..., PytreeValue]:
        """Bind the source-context default, clipped to the candidates' range."""
        shocks = {
            shock
            for context in self.contexts.values()
            for plan in context.current_iid_plans.values()
            if plan is not None
            for shock in plan.physical_names
        } | set(declaration.observes)
        if transfer.name in shocks:
            raise RegimeInitializationError(
                f"Transfer {transfer.name!r} has the name of a shock."
            )
        points = _transfer_points(transfer)
        function = functions["composition_action", "transfer", "default"]
        return with_signature(
            ClippedDefault(function=function, low=points[0], high=points[-1]),
            args=list(inspect.signature(function).parameters),
            enforce=False,
        )

    def _transfer_reads(
        self,
        *,
        transfer: Transfer,
        default: Callable[..., PytreeValue],
        targets: Mapping[MemberName, tuple[str, int]],
        adjustments: Mapping[MemberName, Callable[..., PytreeValue]],
        owner: RegimeName,
        declaration: CompositionAction,
        projection: Mapping[StateName, Callable[..., PytreeValue]],
        guards: tuple[Callable[..., PytreeValue], ...],
        ancestry: tuple[tuple[QueryKind, str], ...],
        functions: Mapping[tuple[str, ...], Callable[..., PytreeValue]],
        source_key: str,
        ancestors: tuple[str, ...],
        policy_information: frozenset[str],
        physical_names: frozenset[str],
    ) -> tuple[
        dict[str, str], dict[MemberName, Callable[..., PytreeValue]], TransferQueries
    ]:
        """Read each member's value of the outside option at every candidate transfer.

        A member whose projection does not read the transfer keeps its read at
        the default; only its adjustment changes with the candidate. Returns the
        candidate reads, the adjustments at the default and the candidates' rule.
        """
        moving = {
            member: target
            for member, target in declaration.destinations.items()
            if any(
                _reads(
                    function=functions[
                        "composition_action",
                        "destinations",
                        member,
                        "projection",
                        state,
                    ],
                    name=transfer.name,
                )
                for state in cast("Mapping[StateName, UserFunction]", target.projection)
            )
        }
        inputs = {}
        node_targets = []
        node_adjustments = []
        points = _transfer_points(transfer)
        for index, point in enumerate(points):
            fixed = with_signature(FixedTransfer(value=point), args=[], enforce=False)
            node = dict(targets)
            if moving:
                reads, moved, _, _ = self._composition_action_reads(
                    owner=owner,
                    declaration=replace(declaration, destinations=moving),
                    projection={**projection, transfer.name: fixed},
                    guards=guards,
                    ancestry=ancestry,
                    functions=functions,
                    source_key=source_key,
                    ancestors=ancestors,
                    policy_information=policy_information,
                    physical_names=physical_names,
                )
                for member, (operand, slot) in moved.items():
                    key = f"_lcm_transfer_{index}_{member}"
                    inputs[key] = reads[operand]
                    node[member] = (key, slot)
            node_targets.append(MappingProxyType(node))
            node_adjustments.append(
                MappingProxyType(
                    {
                        member: _bind_transfer(
                            function=adjustment,
                            name=transfer.name,
                            helper=fixed,
                            source=source_key,
                        )
                        for member, adjustment in adjustments.items()
                    }
                )
            )
        return (
            inputs,
            {
                member: _bind_transfer(
                    function=adjustment,
                    name=transfer.name,
                    helper=default,
                    source=source_key,
                )
                for member, adjustment in adjustments.items()
            },
            TransferQueries(
                points=points,
                default=default,
                targets=tuple(node_targets),
                adjustments=tuple(node_adjustments),
                tie=cast("MutualConsent", declaration.rule).tie,
            ),
        )


def _transfer_points(transfer: Transfer) -> tuple[float, ...]:
    """The candidate transfers in the working float format, as host scalars."""
    return tuple(float(point) for point in np.asarray(transfer.candidates.to_jax()))


def _reads(*, function: Callable[..., PytreeValue], name: ReferenceName) -> bool:
    return name in inspect.signature(function).parameters


def _bind_transfer(
    *,
    function: Callable[..., PytreeValue],
    name: ReferenceName,
    helper: Callable[..., PytreeValue],
    source: ReferenceName,
) -> Callable[..., PytreeValue]:
    """Evaluate a source-context callback at the transfer `helper` gives."""
    if not _reads(function=function, name=name):
        return function
    return _compose(projection={name: helper}, function=function, source=source)


def bind_composition_query_graph(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    canonical_regimes: Mapping[RegimeName, Regime],
    root: RegimeName,
    period: int,
    phase: Phase = "solve",
    kind: QueryKind = QueryKind.FULL,
    stored_outside_reads: bool = False,
    value_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
    collapse_unavailable_nodes: bool = False,
) -> tuple[Mapping[str, QueryNode], str]:
    """Bind demanded public reads after every actual canonical context exists.

    With `stored_outside_reads`, each outside read proved aligned carries its
    `OutsideValueRead`, and an outside read of a problem solved without grid search is
    refused. `value_periods_by_regime` names the periods each regime is solved in;
    without it, a regime is solved in each of its active periods. With
    `collapse_unavailable_nodes`, the root enumerates only the nodes where its
    composition action is available, and one more (`_bind_available_nodes`).
    """
    aliases = {
        (owner, name): f"_lcm_composition_operand_{index}"
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
    _bind_sample(nodes=builder.nodes, root=root_key)
    if collapse_unavailable_nodes:
        _bind_available_nodes(nodes=builder.nodes, root=root_key)
    if stored_outside_reads:
        _bind_outside_reads(
            nodes=builder.nodes,
            outside_reads=builder.outside_reads,
            canonical_regimes=canonical_regimes,
            value_periods_by_regime={
                name: regime.active_periods
                for name, regime in canonical_regimes.items()
            }
            if value_periods_by_regime is None
            else value_periods_by_regime,
        )
    return MappingProxyType(builder.nodes), root_key


def _bind_outside_reads(
    *,
    nodes: dict[str, QueryNode],
    outside_reads: Mapping[str, tuple[RegimeName, str]],
    canonical_regimes: Mapping[RegimeName, Regime],
    value_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
) -> None:
    """Mark the outside reads proved aligned; refuse reads of non-grid-search problems.

    A problem solved on an endogenous grid publishes no stored array and cannot be
    evaluated at a point, so no outside read of it is admitted.
    """
    for key, (source, member) in outside_reads.items():
        alignment = align_outside_read(
            key=key,
            definitions=nodes,
            canonical_regimes=canonical_regimes,
            value_periods_by_regime=value_periods_by_regime,
        )
        node = nodes[key]
        _fail_if_endogenous_grid_read(
            node=node, alignment=alignment, source=source, member=member
        )
        if isinstance(alignment, OutsideValueRead):
            nodes[key] = replace(node, outside_value=alignment)


def _fail_if_endogenous_grid_read(
    *,
    node: QueryNode,
    alignment: OutsideValueRead | UnalignedOutsideRead,
    source: RegimeName,
    member: str,
) -> None:
    """Refuse an outside read of a problem solved without grid search."""
    target = node.read.target
    solver = node.context.user_regimes_at(period=node.read.period)[target].solver
    if isinstance(solver, GridSearch):
        return
    # A solver bound to its regime subclasses the public configuration it names.
    name = next(
        cls.__name__ for cls in type(solver).__mro__ if not cls.__name__.startswith("_")
    )
    read = (
        f"The value of the outside option of {member} in the composition action of "
        f"{source!r} reads {target!r}"
    )
    if isinstance(alignment, UnalignedOutsideRead):
        raise UnsupportedOperationError(
            f"{read}, whose {name} solver cannot be evaluated at a point, at a "
            f"projected state that is not provably a grid point ({alignment.reason}). "
            f"Put the projected states on the grid of {target!r}."
        )
    raise UnsupportedOperationError(
        "Stored values of the outside option of an endogenous-grid problem are not "
        "produced yet. "
        f"{read}, solved with the {name} solver, at points of its grid."
    )


def acting_kinds(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    readers: tuple[tuple[RegimeName, Mapping[str, QueryNode], str, int], ...],
) -> Mapping[tuple[RegimeName, int], QueryKind]:
    """Return the problem each member who leaves acts in: its decision's price.

    Each reader is a source with a composition action, its simulated query graph, the
    graph's root and its period. A destination can be entered in one period both by
    decisions priced at its Participating policy and by decisions priced at its Raw
    problem. Such a destination is refused, unless the two coincide because a
    Participating read drops the destination's composition action.
    """
    moves: dict[tuple[RegimeName, int], dict[QueryKind, set[RegimeName]]] = {}
    for source, definitions, root, period in readers:
        for targets in definitions[root].composition_action_targets.values():
            for query in targets.values():
                # A member entering a later stage is an agent there, not an actor here.
                if definitions[query].landing is not None:
                    continue
                read = definitions[query].read
                moves.setdefault((read.target, period), {}).setdefault(
                    read.kind, set()
                ).add(source)
    for (owner, period), kinds in moves.items():
        if len(kinds) > 1 and _participating_read_keeps_composition_action(
            context=contexts[owner], owner=owner, period=period
        ):
            readings = "; ".join(
                f"{kind.value} after {', '.join(map(repr, sorted(sources)))}"
                for kind, sources in sorted(kinds.items())
            )
            economic = period_of_step(ages=contexts[owner].ages, step=period)
            raise RegimeInitializationError(
                f"Regime {owner!r} is entered in period {economic} by decisions "
                f"on the composition priced at different problems ({readings}). A "
                "regime with an available composition action that needs no consent "
                "cannot be entered in one period both by consent, which reads it with "
                "its composition action, and by another decision on the composition, "
                "which reads it without."
            )
    return MappingProxyType(
        {
            key: QueryKind.PARTICIPATING
            if QueryKind.PARTICIPATING in kinds
            else QueryKind.RAW
            for key, kinds in moves.items()
        }
    )


def _participating_read_keeps_composition_action(
    *, context: SolverBuildContext, owner: RegimeName, period: int
) -> bool:
    """Whether a simulated Participating read of a regime differs from its Raw one.

    A Participating read applies the regime's composition action, unless that
    composition action needs consent or is unavailable at every state of the period; a
    Raw read never applies it.
    """
    from _lcm.model_graph import _composition_action_is_statically_unavailable  # noqa: PLC0415

    declaration = composition_action_for_phase(
        declaration=context.user_regimes_at(period=period)[owner].composition_action,
        phase="simulate",
    )
    return (
        declaration is not None
        and not isinstance(declaration.rule, Consent)
        and not _composition_action_is_statically_unavailable(
            predicate=declaration.available, period=period, ages=context.ages
        )
    )


def _bind_sample(*, nodes: dict[str, QueryNode], root: str) -> None:
    """Keep the root's stay-read groups as a product; every other axis is drawn.

    A stay problem whose shock reads are not attributable keeps every own group. A root
    with nothing left to draw integrates over its full product. When the root's
    composition action is available only on some nodes of one drawn group, that group is
    drawn among those nodes (`QueryNode.draw_condition`).
    """
    from _lcm.query_information import _frame_shocks, stay_read_shocks  # noqa: PLC0415

    node = nodes[root]
    if node.sample is None:
        return
    plan = node.context.current_iid_plans.get(node.read.period)
    bindings = () if plan is None else plan.bindings
    shocks = stay_read_shocks(name=root, definitions=nodes)
    kept = tuple(
        position
        for position, (_, binding, _) in enumerate(bindings)
        if shocks is None or shocks & binding.group.shocks.keys()
    )
    if len(kept) == len(bindings) and not node.observation_sources:
        nodes[root] = replace(node, sample=None)
        return
    context = node.context
    if plan is not None:
        indices = jnp.asarray(
            tuple(
                product(
                    *(
                        range(support.weights.size if position in kept else 1)
                        for position, (_, _, support) in enumerate(bindings)
                    )
                )
            ),
            dtype=jnp.int32,
        )
        context = replace(
            context,
            current_iid_plans=MappingProxyType(
                {node.read.period: replace(plan, indices=indices)}
            ),
        )
    nodes[root] = replace(
        node,
        context=context,
        sample=(node.sample[0], kept),
        draw_condition=_draw_condition(
            predicate=node.composition_action_availability,
            frame_shocks=_frame_shocks(key=root, definitions=nodes),
            groups=tuple(
                frozenset(binding.group.shocks) if position not in kept else None
                for position, (_, binding, _) in enumerate(bindings)
            ),
        ),
    )


def _draw_condition(
    *,
    predicate: Callable[..., PytreeValue] | None,
    frame_shocks: frozenset[ShockName],
    groups: tuple[frozenset[ShockName] | None, ...],
) -> int | None:
    """The drawn plan position whose shocks alone decide if the decision is available.

    `groups` holds each drawn position's shocks and `None` for a kept one. None when the
    composition action is always available, when its predicate reads no shock, or when
    the shocks it reads are not all in one drawn group.
    """
    reads = (
        frozenset()
        if predicate is None
        else frozenset(inspect.signature(predicate).parameters) & frame_shocks
    )
    owners = [
        position
        for position, shocks in enumerate(groups)
        if shocks is not None and reads <= shocks
    ]
    return owners[0] if reads and len(owners) == 1 else None


def _bind_available_nodes(*, nodes: dict[str, QueryNode], root: str) -> None:
    """Enumerate the root's decision at the nodes where its decision is available.

    It applies when the root enumerates its rows, updates no state (no Pareto weight
    moves), and its composition action's predicate reads only discrete shocks of one
    group that its stay problem ignores (`_draw_condition`). The group's nodes then
    carry fixed codes, so where the composition action is available is known now.
    Where it is not, the row's value is the stay value whatever the node, so the plan
    keeps the available nodes and the first unavailable one, which stands for all of
    them (`QueryNode.available_nodes`). A predicate true at every node changes nothing.
    """
    from _lcm.query_information import (  # noqa: PLC0415
        _composition_action_available,
        _frame_shocks,
        stay_read_shocks,
    )

    node = nodes[root]
    predicate = node.composition_action_availability
    plan = node.context.current_iid_plans.get(node.read.period)
    shocks = stay_read_shocks(name=root, definitions=nodes)
    if (
        predicate is None
        or plan is None
        or shocks is None
        or node.sample is not None
        or node.state_updates
    ):
        return
    column = _draw_condition(
        predicate=predicate,
        frame_shocks=_frame_shocks(key=root, definitions=nodes),
        groups=tuple(
            None
            if shocks & binding.group.shocks.keys()
            else frozenset(binding.group.shocks)
            for _, binding, _ in plan.bindings
        ),
    )
    if column is None:
        return
    _, binding, support = plan.bindings[column]
    names = tuple(inspect.signature(predicate).parameters)
    if not all(isinstance(binding.group.shocks.get(name), Discrete) for name in names):
        return
    available = np.asarray(
        _composition_action_available(
            predicate=predicate,
            arguments={},
            physical={name: support.discrete_codes[name] for name in names},
        ),
        dtype=bool,
    )
    if available.all():
        return
    kept = (*np.flatnonzero(available).tolist(), int(np.flatnonzero(~available)[0]))
    indices = jnp.asarray(
        tuple(
            product(
                *(
                    kept if position == column else range(group.weights.size)
                    for position, (_, _, group) in enumerate(plan.bindings)
                )
            )
        ),
        dtype=jnp.int32,
    )
    nodes[root] = replace(
        node,
        context=replace(
            node.context,
            current_iid_plans=MappingProxyType(
                {node.read.period: replace(plan, indices=indices)}
            ),
        ),
        available_nodes=(column, tuple(available.tolist())),
    )


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
    """Prove one law for every shock a caller sees and its query reads."""
    from _lcm.shock_projection import validate_observed_laws  # noqa: PLC0415

    for node in nodes.values():
        if not node.ancestors:
            continue
        parent = nodes[node.ancestors[-1]]
        source = parent.context.current_iid_plans.get(period)
        target = node.context.current_iid_plans.get(period)
        validate_observed_laws(
            source=parent.read.target,
            target=node.read.target,
            physical_inputs=node.read.policy_information,
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
            validate_observed_laws(
                source=origin.read.target,
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


def lower_composition_queries(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    canonical_regimes: Mapping[RegimeName, Regime],
    root: RegimeName,
    period: int,
    value_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
) -> _PolicyQueryKernel:
    """Lower a bound Full graph into the existing native publication schedule.

    Each destination a stored outside read takes is bound once more, as the root
    of its own Raw graph, which the kernel tabulates.
    """
    nodes, root_key = bind_composition_query_graph(
        contexts=contexts,
        canonical_regimes=canonical_regimes,
        root=root,
        period=period,
        stored_outside_reads=_READ_STORED_OUTSIDE_VALUES,
        value_periods_by_regime=value_periods_by_regime,
        collapse_unavailable_nodes=_COLLAPSE_UNAVAILABLE_NODES,
    )
    tables = {}
    for node in nodes.values():
        read = node.outside_value
        if read is not None and read.label not in tables:
            tables[read.label] = bind_composition_query_graph(
                contexts=contexts,
                canonical_regimes=canonical_regimes,
                root=node.read.target,
                period=period,
                kind=QueryKind.RAW,
            )
    root_context = contexts[root]
    kernel = build_policy_query_kernel(
        root_context=root_context,
        period=period,
        definitions=nodes,
        root=root_key,
        chunk_size=root_context.axis_widths.get(
            "query_cell", default_chunk_size(definitions=nodes)
        ),
        outside_tables=MappingProxyType(tables),
    )
    return kernel.with_fixed_params(
        fixed_flat_params=MappingProxyType(
            {
                owner: regime.solution.resolved_fixed_params
                for owner, regime in canonical_regimes.items()
            }
        )
    )


def install_composition_policy_queries(
    *,
    canonical_regimes: Mapping[RegimeName, Regime],
    contexts: Mapping[RegimeName, SolverBuildContext],
    value_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
) -> MappingProxyType[RegimeName, Regime]:
    """Install Full roots for the periods of regimes that declare a composition action.

    Periods of one regime whose query graphs trace the same programs run the
    functions of the first such period and publish its period in their
    signature, so one lowering serves all of them.
    """
    updated = dict(canonical_regimes)
    for owner, context in contexts.items():
        solution = canonical_regimes[owner].solution
        kernels = dict(solution.period_kernels)
        # ponytail: linear scan over a regime's distinct structures; index them by
        # a hashable digest if a regime ever carries more than a handful.
        groups: list[
            tuple[
                _Structure, int, dict[tuple[str, int, int], Callable[..., PytreeValue]]
            ]
        ] = []
        traced_at: dict[int, int] = {}
        for period in kernels:
            if (
                value_periods_by_regime is not None
                and period not in value_periods_by_regime.get(owner, ())
            ):
                continue
            if (
                context.user_regimes_at(period=period)[owner].composition_action
                is not None
            ):
                kernel = lower_composition_queries(
                    contexts=contexts,
                    canonical_regimes=canonical_regimes,
                    root=owner,
                    period=period,
                    value_periods_by_regime=value_periods_by_regime,
                )
                structure = (
                    solution.period_signatures[period],
                    _traced_structure(kernel=kernel),
                )
                first, functions = next(
                    (
                        (first, functions)
                        for known, first, functions in groups
                        if _is_same_structure(left=known, right=structure)
                    ),
                    (period, {}),
                )
                if first == period:
                    groups.append((structure, period, functions))
                kernels[period] = kernel.sharing(functions=functions)
                traced_at[period] = first
        if traced_at:
            solver_keys = solution.solver_period_group_keys
            updated[owner] = replace(
                canonical_regimes[owner],
                solution=replace(
                    solution,
                    period_kernels=MappingProxyType(kernels),
                    # Grid search, the one solver a query admits, keys each period
                    # with a law plan apart. A query kernel takes the plan's nodes
                    # as operands and its structure holds the rest of the plan, so
                    # the group shares one key.
                    solver_period_group_keys=MappingProxyType(
                        {
                            period: solver_keys[traced_at.get(period, period)]
                            for period in solver_keys
                        }
                    ),
                    period_signatures=MappingProxyType(
                        {
                            period: (
                                signature,
                                # Frozen identity: the tag names compiled
                                # programs and keeps its spelling.
                                "typed-exit-query",
                                traced_at.get(period, period),
                            )
                            if isinstance(kernels[period], _PolicyQueryKernel)
                            else signature
                            for period, signature in (
                                solution.period_signatures.items()
                            )
                        }
                    ),
                ),
            )
    return MappingProxyType(updated)


type _Structure = tuple[object, ...]  # noqa: PAN001 - a query node's fields share only object


def _traced_structure(*, kernel: _PolicyQueryKernel) -> _Structure:
    """Describe what the kernel's programs trace, apart from their operands.

    Two periods with the same structure trace the same programs. The period,
    its age, its draws, parameters and stored values enter as operands and stay
    out of the structure.
    """
    chunks = cast("_QueryChunks", kernel.programs["_chunks"].function)
    return (
        kernel.root,
        kernel.chunk_size,
        chunks.chunks,
        dict(chunks.raw_sources),
        tuple(
            (name, _node_structure(node=node))
            for name, node in kernel.definitions.items()
        ),
    )


def _is_same_structure(*, left: _Structure, right: _Structure) -> bool:
    """Compare two structures; one that cannot be compared matches nothing."""
    try:
        return bool(left == right)
    except TypeError, ValueError:
        return False


def _node_structure(*, node: QueryNode) -> _Structure:
    """One node's fields, with each per-period object replaced by its sources.

    - The read drops its period and keeps the names its projection writes.
    - The context becomes `_context_structure`.
    - A closure bound per period keeps only its presence: it is built from the
      declaration, the closed composition action and the child queries, which the
      context and the remaining fields hold.
    - A utility adjustment keeps its composition action and member: it is bound per
      period from the declaration and the regime's functions, which the context holds.
    - The availability predicate of the composition action keeps only its presence, for
      the same reason as a utility adjustment.
    """
    parts: _Structure = ()
    for item in fields(node):
        value = getattr(node, item.name)
        if item.name == "read":
            value = tuple(
                tuple(getattr(value, part.name))
                if part.name == "projection"
                else getattr(value, part.name)
                for part in fields(value)
                if part.name != "period"
            )
        elif item.name == "context":
            value = _context_structure(node=node)
        elif item.name in {
            "participation",
            "decision",
            "available",
            "composition_action_availability",
        }:
            value = value is not None
        elif item.name == "source_projection":
            value = tuple(value)
        elif item.name == "landing" and value is not None:
            value -= node.read.period
        elif item.name == "composition_action_adjustments":
            value = tuple(
                (composition_action_id, tuple(members))
                for composition_action_id, members in value.items()
            )
        parts = (*parts, value)
    return parts


def _context_structure(*, node: QueryNode) -> _Structure:
    """The node's context, with each period-keyed field read at the node's period.

    A field shared by every period stands by its identity. The law plan stands by
    its bindings and the content of its node arrays, because every period holds
    a plan of its own.
    """
    from _lcm.model_graph import _composition_action_is_statically_unavailable  # noqa: PLC0415

    context, period = node.context, node.read.period
    shared = tuple(
        id(getattr(context, item.name))
        for item in fields(context)
        if item.name != "current_iid_plans"
    )
    if node.landing is not None:
        # A landing read traces its target's grid at the occurrence.
        grids = context.period_to_regime_v_interp or {}
        return (
            shared,
            id(grids.get(node.landing, {}).get(node.read.target)),
            id(context.Q_and_F_functions.get(node.landing)),
            _plan_structure(context.current_iid_plans.get(node.landing)),
        )
    declaration = context.user_regimes_at(period=period)[node.read.target]
    plan = context.current_iid_plans.get(period)
    return (
        shared,
        id(context.Q_and_F_functions[period]),
        id(declaration),
        declaration.composition_action is not None
        and _composition_action_is_statically_unavailable(
            predicate=declaration.composition_action.available,
            period=period,
            ages=context.ages,
        ),
        _plan_structure(plan),
    )


def _plan_structure(plan: CurrentIIDPlan | None) -> _Structure | None:
    """Compare each occurrence's law and node rule independently of array identity."""
    if plan is None:
        return None
    return (
        tuple(
            (name, id(binding), _content_digest(tree=nodes))
            for name, binding, nodes in plan.bindings
        ),
        _content_digest(tree=plan.indices),
    )


def _content_digest(*, tree: PytreeValue) -> str:
    """Digest the shape, dtype and bytes of every array leaf."""
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        digest.update(repr((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_consent_context(
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    source: RegimeName,
    declaration: CompositionAction,
) -> None:
    """Do not omit a member or invent an ambiguous counterparty Raw channel."""
    from lcm.exceptions import RegimeInitializationError  # noqa: PLC0415

    counterparty = cast("Consent", declaration.rule).counterparty
    if contexts[source].members != ("self",) or contexts[
        counterparty.regime
    ].members != ("self",):
        raise RegimeInitializationError(
            "Consent requires unambiguous singleton source "
            "and counterparty Raw channels."
        )
    target = declaration.destinations["self"]
    if set(contexts[target.regime].members) != {
        target.role,
        counterparty.role,
    }:
        raise RegimeInitializationError(
            "Consent comparisons must cover every formation-destination member."
        )
