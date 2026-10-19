"""Lower declared composite states to one generated discrete state each.

A composite state replaces the axes of its member states by one axis whose codes
index the member tuples reachable from the declared start codes. The lowering acts
on the user regimes before anything else reads them:

- a dense build of the declared model supplies the engine's own laws, constraint
  pools and fixed parameters to `enumerate_reachable_states`;
- every regime carrying the members holds the generated grid in their place, one
  decoder function per member, so user functions keep reading member names, and
  one law per carrying target that encodes the members' next values;
- a generated constraint makes each code infeasible at the periods it cannot be
  reached in, so its value there is `-inf` and no reachable cell reads it.

Codes off the generated set read as `MISSING_CAT_CODE`, whose value read is NaN.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, make_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dags import concatenate_functions
from jaxtyping import Array, Int

from _lcm.dtypes import safe_to_int_dtype
from _lcm.engine import Regime as EngineRegime
from _lcm.grids import DiscreteGrid
from _lcm.grids.categorical import categorical
from _lcm.grids.discrete import MISSING_CAT_CODE
from _lcm.identity_transition import _IdentityTransition
from _lcm.params.processing import find_param_candidates
from _lcm.regime_building.reachable_states import (
    ReachableStates,
    enumerate_reachable_states,
)
from _lcm.typing import (
    DataclassInstance,
    FlatRegimeParams,
    FunctionName,
    ReferenceName,
    RegimeName,
    RegimeNamesToIds,
    StateName,
)
from _lcm.utils.functools import publish_signature
from _lcm.utils.namespace import flatten_regime_namespace
from lcm.composite_state import CompositeState
from lcm.exceptions import InvalidInitialConditionsError, ModelInitializationError
from lcm.phased import Phased
from lcm.regime import Regime, StateTransitionEntry
from lcm.transition import fixed_transition
from lcm.typing import (
    BoolND,
    DiscreteState,
    FloatND,
    IntND,
    ScalarInt,
    UserInitialConditions,
    UserParams,
)

if TYPE_CHECKING:
    from lcm.model import Model
else:
    # The model imports this module, so the claw sees a wide fallback.
    type Model = object


@dataclass(frozen=True, kw_only=True)
class CompositeLowering:
    """The generated code set of one composite state and how it was built."""

    name: StateName
    """Name of the generated state."""

    members: tuple[StateName, ...]
    """Member states, in tuple order."""

    reachable: ReachableStates
    """Reachable tuples per node and their union."""

    table: np.ndarray
    """Member codes of every composite code, shape `(n_codes, n_members)`."""

    encoding: np.ndarray
    """Composite code of every member tuple, `MISSING_CAT_CODE` off the set."""

    carriers: frozenset[RegimeName]
    """Regimes holding the composite state."""

    member_labels: tuple[tuple[str, ...], ...]
    """Category labels of every member, in code order."""

    @property
    def labels(self) -> tuple[str, ...]:
        """Category labels of the composite codes, in code order."""
        return tuple(f"h{code}" for code in range(len(self.table)))

    def encode(self, members: tuple[int, ...]) -> int:
        """Return the composite code of one member tuple."""
        return int(self.encoding[members])

    def encode_rows(self, codes: np.ndarray) -> np.ndarray:
        """Return the composite codes of member-code rows, missing off the set."""
        inside = np.all((codes >= 0) & (codes < np.asarray(self.encoding.shape)), 1)
        out = np.full(len(codes), MISSING_CAT_CODE, dtype=np.int32)
        out[inside] = self.encoding[tuple(codes[inside].T)]
        return out


def lower_composite_states(
    *,
    composite_states: Mapping[StateName, CompositeState],
    regimes: Mapping[RegimeName, Regime],
    fixed_params: UserParams,
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
    build_dense_model: Callable[[], Model],
) -> tuple[
    MappingProxyType[RegimeName, Regime],
    MappingProxyType[StateName, CompositeLowering],
    frozenset[str],
]:
    """Replace each composite state's members by one generated state.

    Args:
        composite_states: Mapping of generated state names to their declarations.
        regimes: Mapping of regime names to the declared user regimes.
        fixed_params: Parameters fixed at model initialization.
        derived_categoricals: Mapping of model-level derived categoricals.
        build_dense_model: Build the declared model without composite states.

    Returns:
        The rewritten regimes, each composite state's lowering, and the flat
        `fixed_params` keys consumed by member laws.

    Raises:
        ModelInitializationError: If more than one composite state is declared, a
            generated name collides with a declared one, or the enumeration
            refuses a member.

    """
    if not composite_states:
        return MappingProxyType(dict(regimes)), MappingProxyType({}), frozenset()
    if len(composite_states) > 1:
        msg = f"Only one composite state is supported; got {sorted(composite_states)}."
        raise ModelInitializationError(msg)
    ((name, declaration),) = composite_states.items()
    _fail_if_members_are_unsupported(
        name=name, members=declaration.members, regimes=regimes
    )
    dense = build_dense_model()
    engine = dense._regimes  # noqa: SLF001
    reachable = enumerate_reachable_states(
        regimes=engine,
        initial_nodes=dense.graph.initial_nodes,
        ages=dense._time,  # noqa: SLF001
        state_names=declaration.members,
        start_codes=declaration.start_codes,
    )
    codes = reachable.union
    members = declaration.members
    value_inputs = _value_grid_inputs(
        name=name,
        members=members,
        engine=engine,
        regimes=regimes,
        derived_categoricals=derived_categoricals,
    )
    carriers = reachable.carriers
    declared = regimes[min(carriers)].states
    member_grids = [
        grid
        for member in members
        if isinstance(grid := declared.get(member), DiscreteGrid)
    ]
    if len(member_grids) != len(members):
        msg = f"Every member of {name!r} must be held on a discrete grid."
        raise ModelInitializationError(msg)
    shape = tuple(len(grid.codes) for grid in member_grids)
    table = np.asarray(codes, dtype=np.int32).reshape(len(codes), len(members))
    encoding = np.full(shape, MISSING_CAT_CODE, dtype=np.int32)
    encoding[tuple(table.T)] = np.arange(len(codes), dtype=np.int32)
    lowering = CompositeLowering(
        name=name,
        members=members,
        reachable=reachable,
        table=table,
        encoding=encoding,
        carriers=carriers,
        member_labels=tuple(tuple(grid.categories) for grid in member_grids),
    )
    grid = DiscreteGrid(_category_class(name=name, n_codes=len(codes)))
    period_masks = _period_masks(
        lowering=lowering,
        n_periods=len(dense._time.exact_values),  # noqa: SLF001
    )
    rewritten = {
        regime_name: _rewrite_regime(
            regime=regime,
            regime_name=regime_name,
            lowering=lowering,
            grid=grid,
            engine_transitions=_member_laws(
                regime=engine[regime_name], members=members, carriers=carriers
            ),
            value_inputs=value_inputs,
            fixed=MappingProxyType(
                {
                    **engine[regime_name].resolved_fixed_params,
                    **engine[regime_name].solution.resolved_fixed_params,
                }
            ),
            period_mask=period_masks[regime_name],
        )
        if regime_name in carriers
        else regime
        for regime_name, regime in regimes.items()
    }
    consumed = _consumed_fixed_params(
        engine=engine, carriers=carriers, fixed_params=fixed_params, members=members
    )
    return MappingProxyType(rewritten), MappingProxyType({name: lowering}), consumed


def _fail_if_members_are_unsupported(
    *,
    name: StateName,
    members: tuple[StateName, ...],
    regimes: Mapping[RegimeName, Regime],
) -> None:
    """Refuse phase-specific members and members held on differing grids."""
    grids: dict[StateName, set[tuple[str, ...]]] = {}
    for regime_name, regime in regimes.items():
        for member in members:
            if isinstance(regime.state_transitions.get(member), Phased) or isinstance(
                regime.states.get(member), Phased
            ):
                msg = (
                    f"Member {member!r} of {name!r} is phase-specific in regime "
                    f"{regime_name!r}; composite members need one law and one grid "
                    "for both phases."
                )
                raise ModelInitializationError(msg)
            grid = regime.states.get(member)
            if isinstance(grid, DiscreteGrid):
                grids.setdefault(member, set()).add(tuple(grid.categories))
    for member, categories in grids.items():
        if len(categories) > 1:
            msg = (
                f"Member {member!r} of {name!r} is held on different grids "
                f"{sorted(categories)}; every carrying regime needs the same one."
            )
            raise ModelInitializationError(msg)


def _value_grid_inputs(
    *,
    name: StateName,
    members: tuple[StateName, ...],
    engine: Mapping[RegimeName, EngineRegime],
    regimes: Mapping[RegimeName, Regime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> frozenset[ReferenceName]:
    """Refuse members declared by their values; return the other value-grid reads.

    A value grid hands its values to user functions and its codes to the engine.
    Members are read through generated decoders as codes, so a member declared by
    its values is refused; the returned names (`x` and `next_x` of every other value
    grid) are what a member law may not read.
    """
    grids = (
        [
            (regime_name, variable, grid)
            for regime_name, regime in engine.items()
            for variable, grid in regime.solution.grids.items()
        ]
        + [
            (regime_name, variable, grid)
            for regime_name, regime in regimes.items()
            for variable, grid in regime.derived_categoricals.items()
        ]
        + [(None, variable, grid) for variable, grid in derived_categoricals.items()]
    )
    valued = {
        (regime_name, variable)
        for regime_name, variable, grid in grids
        if isinstance(grid, DiscreteGrid) and grid.values is not None  # noqa: PD011
    }
    for regime_name, variable in sorted(valued, key=str):
        if variable in members:
            msg = (
                f"Member {variable!r} of {name!r} is declared by its values in regime "
                f"{regime_name!r}; composite members must be held on category grids."
            )
            raise ModelInitializationError(msg)
    names = {variable for _, variable in valued}
    return frozenset({*names, *(f"next_{variable}" for variable in names)})


def lower_composite_initial_conditions(
    *,
    initial_conditions: UserInitialConditions | pd.DataFrame,
    lowerings: Mapping[StateName, CompositeLowering],
    regime_names_to_ids: RegimeNamesToIds,
) -> UserInitialConditions | pd.DataFrame:
    """Replace member columns of initial conditions by composite codes.

    Only subjects whose initial regime carries the members are encoded; every other
    subject keeps the missing code (`None` in a labelled frame), as the dense model
    does for a state its regime lacks. Member codes are checked for integrality and
    the int32 range before any conversion.

    Raises:
        InvalidInitialConditionsError: If a carrying subject's member tuple has no
            composite code, or a carrying subject's label is no category of its
            member.
        ValueError: If a member code is fractional or outside the int32 range.

    """
    if not lowerings:
        return initial_conditions
    if isinstance(initial_conditions, pd.DataFrame):
        return _lower_labelled_frame(frame=initial_conditions, lowerings=lowerings)
    out = dict(initial_conditions)
    for name, lowering in lowerings.items():
        present = [member for member in lowering.members if member in out]
        if not present:
            continue
        columns = {
            member: np.asarray(safe_to_int_dtype(value=out.pop(member), name=member))
            for member in present
        }
        n_subjects = (
            len(np.asarray(out["regime_id"]))
            if "regime_id" in out
            else len(columns[present[0]])
        )
        wrong_length = sorted(
            member for member, codes in columns.items() if len(codes) != n_subjects
        )
        if wrong_length:
            msg = (
                f"Initial member columns {wrong_length} must give one code per "
                f"subject ({n_subjects})."
            )
            raise InvalidInitialConditionsError(msg)
        carrying = (
            np.isin(
                np.asarray(out["regime_id"]),
                [int(regime_names_to_ids[regime]) for regime in lowering.carriers],
            )
            if "regime_id" in out
            else np.ones(n_subjects, dtype=bool)
        )
        composite = np.full(n_subjects, MISSING_CAT_CODE, dtype=np.int32)
        if carrying.any():
            missing = sorted(set(lowering.members) - set(present))
            if missing:
                msg = (
                    f"Initial conditions give {present} but not {missing}, which "
                    "subjects starting in a regime that carries them need."
                )
                raise InvalidInitialConditionsError(msg)
            rows = np.stack(
                [columns[member] for member in lowering.members], axis=1
            ).astype(np.int64)
            composite[carrying] = _encode_or_fail(
                lowering=lowering, rows=rows[carrying]
            )
        out[name] = jnp.asarray(composite)
    return MappingProxyType(out)


def _lower_labelled_frame(
    *, frame: pd.DataFrame, lowerings: Mapping[StateName, CompositeLowering]
) -> pd.DataFrame:
    """Replace member label columns by composite labels on carrying subjects."""
    out = frame.copy()
    for name, lowering in lowerings.items():
        if not any(member in out for member in lowering.members):
            continue
        members = out.reindex(columns=list(lowering.members))
        carrying = out["regime_name"].isin(lowering.carriers).to_numpy()
        given = carrying & members.notna().all(axis=1).to_numpy()
        for member, labels in zip(
            lowering.members, lowering.member_labels, strict=True
        ):
            unknown = sorted(set(members[member][given].astype(str)) - set(labels))
            if unknown:
                msg = (
                    f"Initial labels {unknown} of {member!r} are not among its "
                    f"categories {list(labels)}."
                )
                raise InvalidInitialConditionsError(msg)
        rows = np.stack(
            [
                pd.Categorical(members[member], categories=labels).codes
                for member, labels in zip(
                    lowering.members, lowering.member_labels, strict=True
                )
            ],
            axis=1,
        ).astype(np.int64)
        composite = np.full(len(out), None, dtype=object)
        codes = _encode_or_fail(lowering=lowering, rows=rows[given])
        composite[given] = np.asarray(lowering.labels, dtype=object)[codes]
        out = out.drop(columns=[m for m in lowering.members if m in out])
        out[name] = composite
    return out


def _encode_or_fail(*, lowering: CompositeLowering, rows: np.ndarray) -> np.ndarray:
    """Return the composite codes of member-code rows, refusing unreachable ones."""
    composite = lowering.encode_rows(rows)
    unknown = composite == MISSING_CAT_CODE
    if unknown.any():
        first = dict(
            zip(lowering.members, rows[np.argmax(unknown)].tolist(), strict=True)
        )
        msg = (
            f"{int(unknown.sum())} initial member tuples of "
            f"{list(lowering.members)} are not reachable, e.g. {first}."
        )
        raise InvalidInitialConditionsError(msg)
    return composite


# Integer arrays of any width: with x64 enabled, a helper may be int64.
type _SignedInts = Int[Array, "..."]


@dataclass(frozen=True, kw_only=True, eq=False)
class _MemberDecoder:
    """Read one member's code off a composite code."""

    column: IntND
    """The member's code for every composite code."""

    def __call__(self, **kwargs: IntND) -> IntND:
        (code,) = kwargs.values()
        return jnp.take(self.column, code, mode="fill", fill_value=MISSING_CAT_CODE)


@dataclass(frozen=True, kw_only=True, eq=False)
class _CompositeEncoder:
    """Encode the members' next codes toward one target as a composite code."""

    laws: Callable[..., tuple[IntND, ...]]
    """The member laws, returning next member codes in member order.

    They read whatever the member laws read: codes, but also floating or boolean
    helpers, signed integer helpers of any width and the age, so the keyword inputs
    are not restricted to int32 codes.
    """

    law_args: tuple[ReferenceName, ...]
    """Arguments of `laws`."""

    bound: FlatRegimeParams
    """Fixed parameters `laws` reads, by flat name."""

    encoding: IntND
    """Composite code of every member tuple in C order, flattened."""

    shape: tuple[int, ...]
    """Number of codes of every member."""

    def __call__(self, **kwargs: FloatND | _SignedInts | BoolND) -> IntND:
        values = kwargs | dict(self.bound)
        nexts = self.laws(**{arg: values[arg] for arg in self.law_args})
        return _read_flat(table=self.encoding, shape=self.shape, index=nexts)


@dataclass(frozen=True, kw_only=True, eq=False)
class _ReachableAtPeriod:
    """Whether a composite code is reachable at the period."""

    mask: IntND
    """Flattened table of shape `(n_periods, n_codes)`, one where reachable."""

    shape: tuple[int, int]
    """Number of periods and of composite codes."""

    name: StateName
    """Name of the composite state, the argument holding the code."""

    def __call__(self, **kwargs: IntND) -> BoolND:
        reached = _read_flat(
            table=self.mask,
            shape=self.shape,
            index=(kwargs["period"], kwargs[self.name]),
        )
        return reached == 1


def _read_flat(
    *, table: IntND, shape: tuple[int, ...], index: tuple[IntND, ...]
) -> IntND:
    """Read a C-order flattened table, `MISSING_CAT_CODE` off its bounds."""
    inside = jnp.ones((), dtype=bool)
    flat = jnp.zeros((), dtype=jnp.int32)
    for raw, size in zip(index, shape, strict=True):
        position = jnp.asarray(raw, dtype=jnp.int32)
        inside = inside & (position >= 0) & (position < size)
        flat = flat * size + jnp.clip(position, 0, size - 1)
    return jnp.where(inside, jnp.take(table, flat), MISSING_CAT_CODE)


def _publish(
    *,
    target: _MemberDecoder | _CompositeEncoder | _ReachableAtPeriod,
    args: Mapping[ReferenceName, object],  # noqa: PAN001 - annotation objects
    name: str,
    returns: object = DiscreteState,  # noqa: PAN001 - an annotation object
) -> None:
    """Give a callable instance the signature and annotations the DAG reads."""
    publish_signature(
        target=target,
        signature=inspect.Signature(
            [
                inspect.Parameter(arg, inspect.Parameter.KEYWORD_ONLY, annotation=kind)
                for arg, kind in args.items()
            ],
            return_annotation=returns,
        ),
    )
    object.__setattr__(target, "__annotations__", {**args, "return": returns})
    object.__setattr__(target, "__name__", name)


def _category_class(*, name: StateName, n_codes: int) -> type[DataclassInstance]:
    labels = cast(
        "type[DataclassInstance]",
        make_dataclass(
            name.title().replace("_", ""),
            [(f"h{code}", ScalarInt) for code in range(n_codes)],
        ),
    )
    return categorical(ordered=False)(labels)


def _member_laws(
    *,
    regime: EngineRegime,
    members: tuple[StateName, ...],
    carriers: frozenset[RegimeName],
) -> dict[RegimeName, dict[str, Callable[..., IntND]]]:
    """Return the engine's member laws toward every carrying target."""
    laws: dict[RegimeName, dict[str, Callable[..., IntND]]] = {}
    for phase in (regime.solution, regime.simulation):
        for target, bundle in phase.transitions.items():
            if target in carriers and target not in laws:
                laws[target] = {
                    f"next_{member}": bundle[f"next_{member}"] for member in members
                }
    return laws


def _rewrite_regime(
    *,
    regime: Regime,
    regime_name: RegimeName,
    lowering: CompositeLowering,
    grid: DiscreteGrid,
    engine_transitions: Mapping[RegimeName, Mapping[str, Callable[..., IntND]]],
    value_inputs: frozenset[ReferenceName],
    fixed: FlatRegimeParams,
    period_mask: np.ndarray,
) -> Regime:
    """Hold the composite state in place of the members in one carrying regime."""
    name, members = lowering.name, lowering.members
    reachable_name = f"{name}_is_reachable"
    taken = (set(regime.states) - set(members)) | set(regime.functions)
    if (
        name in taken
        or set(members) & set(regime.functions)
        or reachable_name in regime.constraints
    ):
        msg = (
            f"Regime {regime_name!r} already declares {name!r}, the constraint "
            f"{reachable_name!r}, or a function named after a member of "
            f"{list(members)}."
        )
        raise ModelInitializationError(msg)
    functions = dict(regime.functions)
    for column, member in enumerate(members):
        decoder = _MemberDecoder(column=jnp.asarray(lowering.table[:, column]))
        _publish(target=decoder, args={name: DiscreteState}, name=member)
        functions[member] = decoder
    transitions = {
        state: law
        for state, law in regime.state_transitions.items()
        if state not in members
    }
    if regime.state_transitions:
        transitions[name] = _composite_law(
            name=name,
            regime=regime,
            members=members,
            engine_transitions=engine_transitions,
            value_inputs=value_inputs,
            fixed=fixed,
            encoding=jnp.asarray(lowering.encoding),
        )
    check = _ReachableAtPeriod(
        mask=jnp.asarray(period_mask.ravel(), dtype=jnp.int32),
        shape=(period_mask.shape[0], period_mask.shape[1]),
        name=name,
    )
    _publish(
        target=check,
        args={name: DiscreteState, "period": ScalarInt},
        name=reachable_name,
        returns=BoolND,
    )
    constraints = dict(regime.constraints) | {reachable_name: check}
    return dataclasses.replace(
        regime,
        states={name: grid}
        | {state: g for state, g in regime.states.items() if state not in members},
        functions=functions,
        state_transitions=transitions,
        constraints=constraints,
    )


def _composite_law(
    *,
    name: StateName,
    regime: Regime,
    members: tuple[StateName, ...],
    engine_transitions: Mapping[RegimeName, Mapping[str, Callable[..., IntND]]],
    value_inputs: frozenset[ReferenceName],
    fixed: FlatRegimeParams,
    encoding: IntND,
) -> StateTransitionEntry:
    """Return the identity law, one encoder, or one encoder per target."""
    if not engine_transitions or all(
        isinstance(regime.state_transitions[member], _IdentityTransition)
        for member in members
    ):
        return fixed_transition(name)
    encoders = {}
    for target, laws in engine_transitions.items():
        combined = concatenate_functions(
            functions=dict(laws),
            targets=[f"next_{member}" for member in members],
            return_type="tuple",
            enforce_signature=False,
            set_annotations=True,
        )
        parameters = inspect.signature(combined).parameters
        if read := sorted((set(parameters) - set(fixed)) & value_inputs):
            msg = (
                f"The member laws {sorted(laws)} of {name!r} toward {target!r} read "
                f"{read}, which are declared by their values; a composite state's "
                "member laws can only read category grids."
            )
            raise ModelInitializationError(msg)
        encoder = _CompositeEncoder(
            laws=combined,
            law_args=tuple(parameters),
            bound=MappingProxyType({a: fixed[a] for a in parameters if a in fixed}),
            encoding=encoding.ravel(),
            shape=tuple(encoding.shape),
        )
        _publish(
            target=encoder,
            args={
                arg: (
                    DiscreteState
                    if parameter.annotation is inspect.Parameter.empty
                    else parameter.annotation
                )
                for arg, parameter in parameters.items()
                if arg not in fixed
            },
            name=f"next_{name}",
        )
        encoders[target] = encoder
    return next(iter(encoders.values())) if len(encoders) == 1 else encoders


def _period_masks(
    *, lowering: CompositeLowering, n_periods: int
) -> dict[RegimeName, np.ndarray]:
    """Per carrying regime, which codes are reachable at which period."""
    masks = {
        regime_name: np.zeros((n_periods, len(lowering.table)), dtype=bool)
        for regime_name in lowering.carriers
    }
    for (period, regime_name), tuples in lowering.reachable.by_node.items():
        # An empty successor set is a (0, n_members) table.
        rows = np.asarray(sorted(tuples), dtype=np.int64).reshape(
            len(tuples), len(lowering.members)
        )
        masks[regime_name][period, lowering.encode_rows(rows)] = True
    return masks


def _consumed_fixed_params(
    *,
    engine: Mapping[RegimeName, EngineRegime],
    carriers: frozenset[RegimeName],
    fixed_params: UserParams,
    members: tuple[StateName, ...],
) -> frozenset[str]:
    """Return the flat `fixed_params` keys that fill a slot of a member law.

    The engine keys a law's fixed parameter as `<target>__next_<member>__<param>`
    within its source regime, so the template slot is that key under the source.
    """
    params_flat = flatten_regime_namespace(fixed_params)
    laws = {f"next_{member}" for member in members}
    return frozenset(
        candidate
        for source in carriers
        for key in engine[source].resolved_fixed_params
        if laws & set(key.split("__"))
        for candidate in find_param_candidates(
            qname=f"{source}__{key}", params_flat=params_flat
        )
    )
