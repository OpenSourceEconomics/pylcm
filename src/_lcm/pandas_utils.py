"""Utilities for converting between pandas and LCM data structures."""

import functools
import inspect
import warnings
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dags.tree import qname_from_tree_path, tree_path_from_qname
from jax import Array

from _lcm.dtypes import CanonicalArrayWriter, canonical_float_dtype
from _lcm.grids import DiscreteGrid, Grid, IrregSpacedGrid
from _lcm.params.edges import EDGES
from _lcm.params.mapping_leaf import LeafEntry
from _lcm.params.regime_template import (
    EdgeVocabulary,
    iter_edge_callables,
    iter_transition_callables,
)
from _lcm.params.temporal import (
    align_time_varying,
    temporal_parameter_names,
    time_gather_indices,
    validate_temporal_variants,
)
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.reachability import ModelReachability
from _lcm.regime_building.collective import NO_ROLE, build_role_vocabulary
from _lcm.regime_law import RegimeLaws
from _lcm.simulation.initial_conditions import MISSING_CAT_CODE, PSEUDO_STATE_NAMES
from _lcm.time import TimeAxis, coordinate_kind
from _lcm.typing import (
    FlatParams,
    FunctionName,
    InitialConditions,
    RegimeName,
    RegimeNamesToIds,
    StateName,
)
from _lcm.utils.ast_inspection import _get_func_indexing_params, time_index_names
from _lcm.utils.functools import is_user_function
from _lcm.utils.namespace import ParamsQnameDepth
from lcm.exceptions import InvalidParamsError
from lcm.params import (
    TimeVarying,
    UnlabelledTimeParameterWarning,
    UserMappingLeaf,
    UserSequenceLeaf,
)
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import (
    AgeCaseLaw,
    AgeSpecializedGrid,
    ByAge,
    JointTransition,
    Transition,
    _select_periods,
)
from lcm.typing import (
    Float1D,
    FloatND,
    Int1D,
    ParameterName,
    Phase,
    ReferenceName,
    UserFunction,
    UserParamsNode,
    ValueND,
)

_JOINT_TRANSITION_ROLE_PARAM_QNAME_DEPTH = 4

# A params node between broadcast and canonicalization: a user-form leaf or
# mapping, or an entry a leaf holds, with every Series and `TimeVarying`
# replaced by its JAX array.
type _ConvertedParamsNode = UserParamsNode | LeafEntry | ValueND

# A regime slot whose callable may read a parameter: a function, a `ByAge`
# schedule of laws, or a `Phased` pair of those.
type _ParamConsumer = (
    UserFunction | ByAge | Phased[UserFunction | ByAge, UserFunction | ByAge]
)


def has_series(params: Mapping[str, UserParamsNode]) -> bool:
    """Check if any leaf value in a params mapping is a pd.Series."""
    for value in params.values():
        if isinstance(value, pd.Series):
            return True
        if isinstance(value, Mapping) and has_series(value):
            return True
        if isinstance(value, (UserMappingLeaf, UserSequenceLeaf)):
            items = (
                value.data.values()
                if isinstance(value, UserMappingLeaf)
                else value.data
            )
            if any(isinstance(v, pd.Series) for v in items):
                return True
    return False


def initial_conditions_from_dataframe(  # noqa: C901
    *,
    df: pd.DataFrame,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    array_writer: CanonicalArrayWriter | None = None,
    ages: TimeAxis | None = None,
) -> InitialConditions:
    """Convert a DataFrame of initial conditions to LCM initial conditions format.

    Args:
        df: DataFrame with columns for states and a "regime_name" column
            carrying the regime label strings.
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        array_writer: Optional owner admitting each numeric device upload after
            label validation and host conversion.

    Returns:
        Immutable mapping of state names (plus `"regime_id"`, and
        `"own_stakeholder"` where the frame carries it) to JAX arrays. The
        `"regime_id"` entry contains integer codes derived from the
        `"regime_name"` column via `regime_names_to_ids`; `"own_stakeholder"`
        carries role codes derived from that column's labels.

    Raises:
        ValueError: If the DataFrame is empty, the "regime_name" column is
            missing, contains invalid regime names, has unknown columns, is
            missing required states, or categorical columns contain invalid
            labels.

    """
    if "regime_name" not in df.columns:
        msg = "DataFrame must contain a 'regime_name' column."
        raise ValueError(msg)

    if len(df) == 0:
        msg = "DataFrame must not be empty."
        raise ValueError(msg)

    # Validate regime names
    valid_regimes = set(regime_names_to_ids.keys())
    invalid_regimes = set(df["regime_name"]) - valid_regimes
    if invalid_regimes:
        msg = (
            f"Invalid regime names in 'regime_name' column: "
            f"{sorted(invalid_regimes)}. "
            f"Valid regimes: {sorted(valid_regimes)}."
        )
        raise ValueError(msg)

    state_columns = {
        col for col in df.columns if col not in {"regime_name", "own_stakeholder"}
    }
    _validate_state_columns(
        state_columns=state_columns,
        user_regimes=user_regimes,
        initial_regimes=df["regime_name"].tolist(),
    )

    n_subjects = len(df)
    state_cols = [
        col for col in df.columns if col not in {"regime_name", "own_stakeholder"}
    ]

    # Pre-allocate result arrays (NaN default surfaces bugs for missing states)
    result_arrays: dict[str, np.ndarray] = {
        col: (
            np.zeros(n_subjects, dtype=np.int32)
            if col == "age" and ages is not None and coordinate_kind(ages) == "period"
            else np.full(n_subjects, np.nan)
        )
        for col in state_cols
    }
    discrete_state_names: set[StateName] = set()

    # Process per regime group (vectorised .map() within each group)
    for regime_name, group in df.groupby("regime_name"):
        user_regime = user_regimes[str(regime_name)]
        idx = group.index
        discrete_grids = {
            name: grid
            for name, grid in _state_grids_with_carried_domains(
                user_regime.states
            ).items()
            if isinstance(grid, DiscreteGrid)
        }
        discrete_state_names |= discrete_grids.keys()

        regime_state_names = set(user_regime.states.keys()) | PSEUDO_STATE_NAMES

        for col in state_cols:
            if col not in regime_state_names:
                continue

            values = group[col]
            if hasattr(values, "cat"):
                values = values.astype(str)

            if col in discrete_grids:
                _map_discrete_labels(
                    values=values,
                    grid=discrete_grids[col],
                    result_array=result_arrays[col],
                    idx=idx,
                    col=col,
                    regime_name=str(regime_name),
                )
            else:
                result_arrays[col][idx] = values.to_numpy(
                    dtype=result_arrays[col].dtype
                )

    # Replace remaining NaN in discrete columns with an explicit int sentinel
    # before casting to int32. This avoids platform-undefined NaN→int behavior
    # and the associated RuntimeWarning.
    for col in discrete_state_names:
        if col in result_arrays:
            nan_mask = np.isnan(result_arrays[col])
            result_arrays[col][nan_mask] = MISSING_CAT_CODE

    initial_conditions: dict[
        StateName | Literal["regime_id", "own_stakeholder"], Float1D | Int1D
    ] = {
        col: _write_pandas_array(
            value=arr,
            dtype=np.dtype(
                jnp.int32
                if col in discrete_state_names
                or (
                    col == "age"
                    and ages is not None
                    and coordinate_kind(ages) == "period"
                )
                else canonical_float_dtype()
            ),
            name=f"initial_conditions.{col}",
            array_writer=array_writer,
        )
        for col, arr in result_arrays.items()
    }
    initial_conditions["regime_id"] = _write_pandas_array(
        value=df["regime_name"].map(dict(regime_names_to_ids)).to_numpy(),
        dtype=np.dtype(jnp.int32),
        name="initial_conditions.regime_id",
        array_writer=array_writer,
    )
    if "own_stakeholder" in df.columns:
        initial_conditions["own_stakeholder"] = _role_codes_from_labels(
            labels=df["own_stakeholder"],
            user_regimes=user_regimes,
            array_writer=array_writer,
        )

    return MappingProxyType(initial_conditions)


def _role_codes_from_labels(
    *,
    labels: pd.Series,
    user_regimes: Mapping[RegimeName, UserRegime],
    array_writer: CanonicalArrayWriter | None = None,
) -> Int1D:
    """Convert an `own_stakeholder` label column back to role codes.

    A published frame names roles the way the model declares them and leaves
    the entry missing for a row occupying none, so the inverse reads a missing
    entry as the no-role sentinel rather than as a failed lookup.

    Args:
        labels: The frame's `own_stakeholder` column, as labels.
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances, the source of the role vocabulary.
        array_writer: Optional owner admitting the role-code device upload.

    Returns:
        One role code per row.

    Raises:
        ValueError: A label names no stakeholder any regime declares.
    """
    role_ids = build_role_vocabulary(
        {name: regime.stakeholders for name, regime in user_regimes.items()}
    )
    text = labels.astype("string")
    named = text.notna()
    unknown = sorted(set(text[named]) - set(role_ids))
    if unknown:
        msg = (
            f"Invalid stakeholder names in the 'own_stakeholder' column: "
            f"{unknown}. Valid stakeholders: {sorted(role_ids)}."
        )
        raise ValueError(msg)
    codes = np.full(len(text), NO_ROLE, dtype=np.int32)
    codes[named.to_numpy()] = text[named].map(dict(role_ids)).to_numpy(dtype=np.int32)
    return _write_pandas_array(
        value=codes,
        dtype=np.dtype(np.int32),
        name="initial_conditions.own_stakeholder",
        array_writer=array_writer,
    )


def _write_pandas_array(
    *,
    value: np.ndarray,
    dtype: np.dtype,
    name: str,
    array_writer: CanonicalArrayWriter | None,
) -> Array:
    """Materialize one host-converted leaf through its optional admission owner."""
    if array_writer is not None:
        return array_writer(value=value, dtype=dtype, name=name)
    return jnp.array(value, dtype=dtype)


def _map_discrete_labels(
    *,
    values: pd.Series,
    grid: DiscreteGrid,
    result_array: np.ndarray,
    idx: pd.Index,
    col: str,
    regime_name: RegimeName,
) -> None:
    """Map string labels to integer codes for a discrete state column in-place."""
    label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
    mapped = values.map(label_to_code)
    unmapped = mapped.isna() & values.notna()
    if unmapped.any():
        bad = set(values[unmapped])
        msg = (
            f"Invalid labels for state '{col}' in regime "
            f"'{regime_name}': {sorted(bad)}. "
            f"Valid: {list(grid.categories)}."
        )
        raise ValueError(msg)
    result_array[idx] = mapped.to_numpy()


def convert_series_in_params(
    *,
    flat_params: Mapping[RegimeName, Mapping[str, UserParamsNode]],
    ages: TimeAxis,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    regime_names_to_ids: RegimeNamesToIds,
    declared_transitions: Mapping[RegimeName, tuple[Transition, ...]],
    declared_vocabulary: Mapping[RegimeName, EdgeVocabulary],
    array_writer: CanonicalArrayWriter | None = None,
    required_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
    reachability: ModelReachability | None = None,
    phase_transitions: Mapping[Phase, Mapping[RegimeName, Transition]] | None = None,
) -> FlatParams:
    """Convert pd.Series leaves in already-broadcast internal params to JAX arrays.

    Iterate over the template-shaped `flat_params` (produced by
    `process_params`) and convert any `pd.Series` leaf values via
    `array_from_series`. `UserMappingLeaf` and `UserSequenceLeaf` values
    (and their canonical `MappingLeaf` / `SequenceLeaf` subclasses) are
    traversed and any Series inside are converted. Other values (scalars,
    existing arrays) pass through unchanged.

    Each regime's `derived_categoricals` field is used to resolve index
    levels that correspond to DAG function outputs (not states/actions). A
    level named by a discrete state or action the regime declares resolves
    even where demand pruned it, so a parameter a dormant callable reads keeps
    its labels at every horizon.

    Args:
        flat_params: Already-broadcast params in template shape
            (`{regime: {func__param: value}}`).
        ages: Age grid for the model.
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        laws: Each regime's law, whose transition functions read params too.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        declared_transitions: Per source regime, its `Transition`s as
            `Model(edges=...)` declares them; an `edges` slot's Series is indexed
            by the declared callable reading it.
        declared_vocabulary: Per regime, what it declares before demand prunes
            any; its discrete grids label the levels of a pruned variable.
        array_writer: Optional owner admitting each Series upload and retaining
            completed leaves while the parameter mapping is assembled.
        phase_transitions: Declarations retaining their owning phase, including
            inferred targets, for phase-specific temporal coverage.

    Returns:
        Immutable mapping with the same structure, Series replaced by JAX
        arrays.

    """
    # User leaves (scalars, arrays, Series, mapping and sequence leaves).
    result: dict[RegimeName, Mapping[str, _ConvertedParamsNode]] = {}
    for regime_name, regime_params in flat_params.items():
        if regime_name == EDGES:
            result[EDGES] = MappingProxyType(
                {
                    source: MappingProxyType(
                        _convert_edge_params(
                            source=source,
                            leaves=cast("Mapping[str, UserParamsNode]", leaves),
                            declared_transitions=declared_transitions.get(source, ()),
                            ages=ages,
                            user_regimes=user_regimes,
                            declared_vocabulary=declared_vocabulary,
                            regime_names_to_ids=regime_names_to_ids,
                            array_writer=array_writer,
                            required_periods_by_regime=required_periods_by_regime,
                            reachability=reachability,
                            declarations_by_phase=(
                                None
                                if phase_transitions is None
                                else {
                                    phase: declarations[source]
                                    for phase, declarations in phase_transitions.items()
                                    if source in declarations
                                }
                            ),
                        )
                    )
                    for source, leaves in regime_params.items()
                }
            )
            continue
        user_regime = user_regimes[regime_name]
        solve_funcs = user_regime.get_all_functions(
            phase="solve", law=laws[regime_name]
        )
        simulate_funcs = user_regime.get_all_functions(
            phase="simulate", law=laws[regime_name]
        )
        all_funcs: dict[str, _ParamConsumer] = {**solve_funcs, **simulate_funcs}
        for name in solve_funcs.keys() & simulate_funcs.keys():
            if solve_funcs[name] is not simulate_funcs[name]:
                all_funcs[name] = Phased(
                    solve=solve_funcs[name], simulate=simulate_funcs[name]
                )
        # The Koopmans aggregator is not a regime function; its params live
        # under a pseudo-function key of the same name. Under `Phased` the two
        # variants declare different parameters and the template carries their
        # union, so the variant that declares each parameter is the one whose
        # source describes how it is indexed.
        aggregator_variants = tuple(
            aggregator
            for aggregator in (
                user_regime.get_koopmans_aggregator(phase="solve"),
                user_regime.get_koopmans_aggregator(phase="simulate"),
            )
            if aggregator is not None
        )
        if aggregator_variants:
            all_funcs["koopmans_aggregator"] = aggregator_variants[0]
        converted_regime: dict[str, _ConvertedParamsNode] = {}
        for func_param, value in regime_params.items():
            # Function lookup exists only to infer a Series leaf's indexing axes.
            # Scalars and already-materialized arrays need no source inspection;
            # resolving every value would make an unrelated Series elsewhere in
            # the params tree turn a valid nested joint scalar into a KeyError.
            if not _needs_param_preflight(value):
                converted_regime[func_param] = value
                continue

            parts = tree_path_from_qname(func_param)
            param_name = parts[-1]
            func, resolved_func_name = _resolve_param_consumer(
                parts=parts,
                param_name=param_name,
                user_regime=user_regime,
                all_funcs=all_funcs,
                aggregator_variants=aggregator_variants,
            )

            # Runtime grid/process params are scalar — no AST inspection.
            # `certainty_equivalent` and `taste_shocks` are pseudo-function keys
            # whose parameters come from a declared name set rather than a
            # signature, so there is no source to inspect for them either.
            if resolved_func_name in _PSEUDO_KEYS_WITHOUT_A_SIGNATURE or (
                _is_runtime_grid_param(
                    func_name=resolved_func_name, user_regime=user_regime
                )
            ):
                func = None

            converted_regime[func_param] = _convert_param_value(
                value=value,
                func=func,
                param_name=param_name,
                func_name=resolved_func_name,
                ages=ages,
                user_regimes=user_regimes,
                regime_names_to_ids=regime_names_to_ids,
                regime_name=regime_name,
                declared_categoricals=declared_vocabulary[regime_name].categoricals,
                array_writer=array_writer,
                required_periods=_regime_param_periods(
                    parts=parts,
                    func_name=resolved_func_name,
                    regime_name=regime_name,
                    ages=ages,
                    required_periods_by_regime=required_periods_by_regime,
                    reachability=reachability,
                    user_regime=user_regime,
                    phase_functions={"solve": solve_funcs, "simulate": simulate_funcs},
                ),
            )
        result[regime_name] = converted_regime
    return cast(
        "FlatParams",
        MappingProxyType({k: MappingProxyType(v) for k, v in result.items()}),
    )


def _convert_edge_params(
    *,
    source: RegimeName,
    # User leaves, keyed by slot path; the values are heterogeneous.
    leaves: Mapping[str, UserParamsNode],
    declared_transitions: tuple[Transition, ...],
    ages: TimeAxis,
    user_regimes: Mapping[RegimeName, UserRegime],
    declared_vocabulary: Mapping[RegimeName, EdgeVocabulary],
    regime_names_to_ids: RegimeNamesToIds,
    array_writer: CanonicalArrayWriter | None,
    required_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
    reachability: ModelReachability | None = None,
    declarations_by_phase: Mapping[Phase, Transition] | None = None,
) -> MappingProxyType[str, _ConvertedParamsNode]:
    """Convert the Series leaves of one source's `edges` slots.

    A slot's key is the declaration path of the callable reading it, so that
    callable decides the Series' indexing axes. A law over all targets returns a
    probability vector over regimes, which a Series names by a `next_regime`
    level; a gate or a projection runs on its target's grid, so the target's
    categoricals resolve its levels.

    Args:
        source: The source regime.
        leaves: The source's slots, keyed by declaration path.
        declared_transitions: The source's `Transition` declarations.
        ages: Age grid for the model.
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        declared_vocabulary: Per regime, what it declares before demand prunes
            any; its discrete grids label the levels of a pruned variable.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        array_writer: Optional owner admitting each Series upload.

    Returns:
        The slots with every Series replaced by its array.

    """
    readers: dict[tuple[str, ...], list[tuple[UserFunction, bool]]] = {}
    for transition in declared_transitions:
        for path, func, gate in iter_transition_callables(transition):
            readers.setdefault(path, []).append((func, gate is not None))
    # User leaves (scalars, arrays, Series, mapping and sequence leaves).
    converted: dict[str, _ConvertedParamsNode] = {}
    for key, value in leaves.items():
        if not _needs_param_preflight(value):
            converted[key] = value
            continue
        path = tree_path_from_qname(key)
        slot, param_name = path[:-1], path[-1]
        slot_readers = readers.get(slot, [])
        on_target_grid = any(runs_on_target for _, runs_on_target in slot_readers)
        regime_name = slot[0] if on_target_grid else source
        converted[key] = _convert_param_value(
            value=value,
            func=(
                _variant_declaring(
                    variants=tuple(func for func, _ in slot_readers),
                    param_name=param_name,
                )
                if slot_readers
                else None
            ),
            param_name=param_name,
            func_name=(
                "next_regime" if not slot else qname_from_tree_path((EDGES, *slot))
            ),
            ages=ages,
            user_regimes=user_regimes,
            regime_names_to_ids=regime_names_to_ids,
            regime_name=regime_name,
            declared_categoricals=declared_vocabulary[regime_name].categoricals,
            array_writer=array_writer,
            required_periods=_edge_param_periods(
                declarations=declared_transitions,
                declarations_by_phase=declarations_by_phase,
                reachability=reachability,
                source=source,
                slot=slot,
                param_name=param_name,
                ages=ages,
                source_periods=(
                    tuple(range(ages.n_periods - 1))
                    if required_periods_by_regime is None
                    else required_periods_by_regime.get(source, ())
                ),
            ),
        )
    return MappingProxyType(converted)


def _regime_param_periods(
    *,
    parts: tuple[str, ...],
    func_name: FunctionName,
    regime_name: RegimeName,
    ages: TimeAxis,
    required_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None,
    reachability: ModelReachability | None,
    user_regime: UserRegime,
    phase_functions: Mapping[Phase, Mapping[str, _ParamConsumer]],
) -> tuple[int, ...]:
    """Require only slots where a declaring phase can read this parameter."""
    periods = (
        tuple(range(ages.n_periods))
        if required_periods_by_regime is None
        else required_periods_by_regime.get(regime_name, ())
    )
    is_transition = (
        func_name.startswith("next_")
        or len(parts) >= ParamsQnameDepth.TARGETREGIME__FUNC__PARAM
    )
    if is_transition:
        periods = tuple(period for period in periods if period < ages.n_periods - 1)
    if reachability is None:
        return periods
    required: set[int] = set()
    for phase, functions in phase_functions.items():
        aggregator = user_regime.get_koopmans_aggregator(phase=phase)
        consumer, _ = _resolve_param_consumer(
            parts=parts,
            param_name=parts[-1],
            user_regime=user_regime,
            all_funcs=functions,
            aggregator_variants=() if aggregator is None else (aggregator,),
            phase=phase,
        )
        if consumer is None or parts[-1] not in inspect.signature(consumer).parameters:
            continue
        graph = reachability.solution if phase == "solve" else reachability.simulation
        required.update(
            period
            for period in periods
            if regime_name in graph.active_regimes_by_period[period]
            and (
                not is_transition
                or len(parts) < ParamsQnameDepth.TARGETREGIME__FUNC__PARAM
                or graph.has_edge(period=period, source=regime_name, target=parts[0])
            )
        )
    return tuple(sorted(required))


def _edge_param_periods(
    *,
    declarations: tuple[Transition, ...],
    slot: tuple[str, ...],
    param_name: ParameterName,
    ages: TimeAxis,
    source_periods: tuple[int, ...],
    declarations_by_phase: Mapping[Phase, Transition] | None,
    reachability: ModelReachability | None,
    source: RegimeName,
) -> tuple[int, ...]:
    """Read scheduled laws at their source, and gate functions at their target."""
    required: set[int] = set()
    period_by_age: dict[object, int] = {
        age: period for period, age in enumerate(ages.exact_values)
    }
    variants: tuple[tuple[Phase | None, Transition], ...] = (
        tuple((None, declaration) for declaration in declarations)
        if declarations_by_phase is None
        else tuple(declarations_by_phase.items())
    )
    for phase, declaration in variants:
        periods = source_periods
        if phase is not None and reachability is not None:
            graph = (
                reachability.solution if phase == "solve" else reachability.simulation
            )
            periods = tuple(
                period
                for period in periods
                if source in graph.active_regimes_by_period[period]
            )
        selected = (
            declaration.law.resolve(ages=ages).law_by_period
            if isinstance(declaration.law, ByAge)
            else dict.fromkeys(periods, declaration.law)
        )
        targets = (
            None
            if declaration.targets is None
            else {
                target: _select_periods(
                    selector=selector, ages=ages, period_by_age=period_by_age
                )
                for target, selector in declaration.targets.items()
            }
        )
        is_gate = len(slot) > 1 and slot[0] in declaration.gates
        for period in periods:
            if period >= ages.n_periods - 1:
                continue
            if targets is not None and not any(
                period in selected for selected in targets.values()
            ):
                continue
            case = selected.get(period)
            if is_gate:
                active = (
                    period in targets.get(slot[0], ())
                    if targets is not None
                    else _case_names_target(case=case, target=slot[0])
                )
                if active and any(
                    path == slot and param_name in inspect.signature(func).parameters
                    for path, func, gate in iter_transition_callables(
                        declaration, phase=phase
                    )
                    if gate is not None
                ):
                    required.add(period + 1)
            elif (
                targets is None or not slot or period in targets.get(slot[0], ())
            ) and any(
                path == slot and param_name in inspect.signature(func).parameters
                for path, func, _ in iter_edge_callables(law=case, path=(), phase=phase)
            ):
                required.add(period)
    return tuple(sorted(required))


def _case_names_target(*, case: AgeCaseLaw | None, target: RegimeName) -> bool:
    """Read named support without evaluating parameter-dependent probabilities."""
    if isinstance(case, Phased):
        return any(
            _case_names_target(case=variant, target=target)
            for variant in (case.solve, case.simulate)
        )
    return target in case if isinstance(case, Mapping) else case == target


def _needs_param_preflight(value: UserParamsNode) -> bool:
    """Inspect arrays and containers, including scalar-only or empty containers."""
    return isinstance(
        value,
        (
            pd.Series,
            TimeVarying,
            Array,
            np.ndarray,
            UserMappingLeaf,
            UserSequenceLeaf,
            Mapping,
        ),
    )


def _joint_variants(
    *, raw: JointTransition | Phased, phase: Phase | None
) -> tuple[JointTransition, ...]:
    """Return the requested phase variants of one joint declaration."""
    if isinstance(raw, Phased):
        variants = cast(
            "tuple[JointTransition, JointTransition]", (raw.solve, raw.simulate)
        )
        return variants if phase is None else (variants[0 if phase == "solve" else 1],)
    return (raw,)


def _resolve_param_consumer(
    *,
    parts: tuple[str, ...],
    param_name: ParameterName,
    user_regime: UserRegime,
    all_funcs: Mapping[str, _ParamConsumer],
    aggregator_variants: tuple[UserFunction, ...],
    phase: Phase | None = None,
) -> tuple[UserFunction | None, FunctionName]:
    """Resolve a flattened param path to the callable declaring its Series leaf.

    Joint support/probability roles add one qname level, while joint outputs keep
    the ordinary ``target__next_<state>__param`` path.  Resolve both from the
    public declaration so Series indexing follows the exact role and phase
    variant that owns the parameter.
    """
    if len(parts) == _JOINT_TRANSITION_ROLE_PARAM_QNAME_DEPTH and parts[2] in {
        "support",
        "probabilities",
    }:
        target, kernel_name, role, _ = parts
        raw = user_regime.joint_transitions[target][kernel_name]
        variants = _joint_variants(raw=raw, phase=phase)
        role_funcs: tuple[UserFunction, ...]
        if role == "support":
            role_funcs = tuple(
                variant.support
                for variant in variants
                if is_user_function(variant.support)
            )
        else:
            role_funcs = tuple(variant.probabilities for variant in variants)
        if not role_funcs and phase is None:
            msg = (
                f"Parameter path {'__'.join(parts)!r} names a callable joint "
                f"transition {role} role, but the declaration has no callable."
            )
            raise KeyError(msg)
        return (
            _variant_declaring(variants=role_funcs, param_name=param_name)
            if role_funcs
            else None,
            f"{kernel_name}.{role}",
        )

    if len(parts) == ParamsQnameDepth.TARGETREGIME__FUNC__PARAM:
        target, public_func_name, _ = parts
        if public_func_name.startswith("next_"):
            state_name = public_func_name.removeprefix("next_")
            output_variants = tuple(
                joint.outputs[state_name]
                for raw in user_regime.joint_transitions.get(target, {}).values()
                for joint in _joint_variants(raw=raw, phase=phase)
                if state_name in joint.outputs
            )
            if output_variants:
                return (
                    _variant_declaring(variants=output_variants, param_name=param_name),
                    qname_from_tree_path((public_func_name, target)),
                )
        # Ordinary per-target transition param: the engine keys the callable
        # ``func__target`` even though the public params path is target-first.
        resolved = qname_from_tree_path((public_func_name, target))
        return _scheduled_consumer(
            func=all_funcs.get(resolved), param_name=param_name, phase=phase
        ), resolved

    resolved = parts[0]
    if resolved == "koopmans_aggregator":
        return (
            _variant_declaring(variants=aggregator_variants, param_name=param_name)
            if aggregator_variants
            else None,
            cast("FunctionName", resolved),
        )
    if resolved in _PSEUDO_KEYS_WITHOUT_A_SIGNATURE or _is_runtime_grid_param(
        func_name=cast("FunctionName", resolved), user_regime=user_regime
    ):
        return None, cast("FunctionName", resolved)
    return (
        _scheduled_consumer(
            func=all_funcs.get(resolved), param_name=param_name, phase=phase
        ),
        cast("FunctionName", resolved),
    )


def _convert_param_value(
    *,
    value: UserParamsNode | LeafEntry,
    func: UserFunction | None,
    param_name: ParameterName,
    func_name: FunctionName,
    ages: TimeAxis,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    regime_name: RegimeName | None,
    declared_categoricals: Mapping[ReferenceName, DiscreteGrid],
    array_writer: CanonicalArrayWriter | None = None,
    required_periods: tuple[int, ...] | None = None,
) -> _ConvertedParamsNode:
    """Convert a single param value, dispatching on type.

    Args:
        value: The parameter value (Series, `UserMappingLeaf` /
            `UserSequenceLeaf`, or passthrough).
        func: The function that uses this parameter (`None` for runtime
            grid params — triggers scalar passthrough).
        param_name: Parameter name in the function.
        func_name: Function name (for `next_*` outcome axis detection).
        ages: Age grid for the model.
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        regime_name: Regime name for action grid lookup.
        declared_categoricals: The discrete grids `regime_name` declares, demand
            pruned or not.
        array_writer: Optional owner admitting each nested Series upload.

    Returns:
        Converted value: JAX array for Series, a `UserMappingLeaf` /
        `UserSequenceLeaf` with converted Series entries, or the original
        value unchanged.

    """

    recurse = functools.partial(
        _convert_param_value,
        func=func,
        param_name=param_name,
        func_name=func_name,
        ages=ages,
        user_regimes=user_regimes,
        regime_names_to_ids=regime_names_to_ids,
        regime_name=regime_name,
        declared_categoricals=declared_categoricals,
        array_writer=array_writer,
        required_periods=required_periods,
    )

    managed = param_name in temporal_parameter_names(func)
    name = f"params.{regime_name}.{func_name}.{param_name}"
    if managed and isinstance(value, (UserMappingLeaf, UserSequenceLeaf, Mapping)):
        raise InvalidParamsError(
            f"{name}: a managed temporal parameter cannot be a container; "
            "supply a scalar, TimeVarying, or labelled Series."
        )
    support = (
        tuple(range(ages.n_periods)) if required_periods is None else required_periods
    )
    if isinstance(value, TimeVarying):
        if not managed:
            raise InvalidParamsError(
                f"{name}: TimeVarying requires @time_varying_params on its consumer."
            )
        return align_time_varying(
            value=value,
            ages=ages,
            required_periods=support,
            name=name,
            array_writer=array_writer,
        )
    if isinstance(value, pd.Series):
        return array_from_series(
            sr=value,
            func=func,
            param_name=param_name,
            func_name=func_name,
            ages=ages,
            user_regimes=user_regimes,
            regime_names_to_ids=regime_names_to_ids,
            regime_name=regime_name,
            declared_categoricals=declared_categoricals,
            array_writer=array_writer,
            required_periods=support,
        )
    # `convert_series_in_params` runs between broadcast and canonicalization,
    # so leaves are still in user form. Preserve that user form on output:
    # canonicalization happens downstream in `cast_params_to_canonical_dtypes`.
    if isinstance(value, UserMappingLeaf):
        return UserMappingLeaf({k: recurse(value=v) for k, v in value.data.items()})
    if isinstance(value, UserSequenceLeaf):
        return UserSequenceLeaf(tuple(recurse(value=v) for v in value.data))
    if isinstance(value, (Array, np.ndarray)) and value.ndim:
        _check_raw_time_array(
            managed=managed, func=func, param_name=param_name, ages=ages, name=name
        )
    return value


def _check_raw_time_array(
    *,
    managed: bool,
    func: UserFunction | None,
    param_name: ParameterName,
    ages: TimeAxis,
    name: str,
) -> None:
    """Reject ambiguous managed arrays and diagnose visible manual time indexing."""
    if managed:
        raise InvalidParamsError(
            f"{name}: declared temporal parameters require labelled "
            "TimeVarying or Series input."
        )
    indices = (
        frozenset()
        if func is None
        else time_index_names(func=func, array_param_name=param_name)
    )
    if not {"age", "period"}.intersection(indices):
        return
    if coordinate_kind(ages) == "period":
        raise InvalidParamsError(
            f"{name}: unlabelled temporal array; use a labelled Series "
            "or managed TimeVarying."
        )
    warnings.warn(
        f"{name}: unlabelled time array; its age-to-period mapping cannot "
        "be checked. Use labelled data.",
        UnlabelledTimeParameterWarning,
        stacklevel=4,
    )


def array_from_series(
    *,
    sr: pd.Series,
    func: UserFunction | None,
    param_name: ParameterName,
    func_name: FunctionName,
    ages: TimeAxis,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    regime_name: RegimeName | None = None,
    declared_categoricals: Mapping[ReferenceName, DiscreteGrid] = MappingProxyType({}),
    array_writer: CanonicalArrayWriter | None = None,
    required_periods: tuple[int, ...] | None = None,
) -> FloatND:
    """Convert a pandas Series to a JAX array.

    Inspect `func` to determine indexing dimensions (states, actions,
    period) and scatter the labeled Series into an N-dimensional array.

    The Series time level must match the model: `"age"` with actual age labels,
    or `"period"` with integer positions. Extra time coordinates are silently
    dropped; missing required temporal or categorical cells raise an error.

    Derived categoricals are read from
    `user_regimes[regime_name].derived_categoricals` when `regime_name` is
    not None.

    Args:
        sr: Labeled pandas Series.
        func: The function that uses this array parameter. `None` for
            runtime grid/process params (triggers scalar passthrough).
        param_name: The array parameter name in `func`.
        func_name: Function name (for `next_*` outcome axis detection).
        ages: Age grid for the model.
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        regime_name: Regime for grid and derived categorical lookup.
        declared_categoricals: Discrete grids `regime_name` declares, which label
            a level whose variable demand pruned from the regime.
        array_writer: Optional owner admitting the completed numeric device array.

    Returns:
        JAX array with axes corresponding to the indexing parameters in
        declaration order.

    Raises:
        ValueError: If level names don't match or labels are invalid.

    """
    indexing_params = (
        []
        if func is None
        else _get_func_indexing_params(func=func, array_param_name=param_name)
    )
    name = f"params.{regime_name}.{func_name}.{param_name}"
    managed = param_name in temporal_parameter_names(func)
    if "age" in indexing_params:
        raise InvalidParamsError(
            f"{name}: table[age] cannot index period-normalized data. "
            "Use managed selection or table[period]."
        )
    if managed:
        if "period" in indexing_params:
            raise InvalidParamsError(
                f"{name}: managed temporal consumers already receive a time slice; "
                "remove manual period indexing."
            )
        indexing_params = ["period", *indexing_params]

    if not indexing_params:
        return _write_pandas_array(
            value=sr.to_numpy(),
            dtype=np.dtype(canonical_float_dtype()),
            name=name,
            array_writer=array_writer,
        )

    # A declared grid labels a level whose variable demand pruned from the regime.
    grids = {
        **declared_categoricals,
        **_resolve_categoricals(user_regimes=user_regimes, regime_name=regime_name),
    }

    # Replace internal "period" with user-facing "age"
    display_params = [
        coordinate_kind(ages) if p == "period" else p for p in indexing_params
    ]

    level_mappings = _build_level_mappings_for_param(
        indexing_params=display_params, grids=grids, ages=ages
    )

    # Append outcome axis for transition probability arrays (next_* functions
    # where the Series has a next_* level in its MultiIndex)
    if func_name.startswith("next_") and isinstance(sr.index, pd.MultiIndex):
        next_levels = [
            n for n in sr.index.names if isinstance(n, str) and n.startswith("next_")
        ]
        if next_levels:
            outcome_mapping = _build_outcome_mapping(
                func_name=func_name,
                grids=grids,
                user_regimes=user_regimes,
                regime_names_to_ids=regime_names_to_ids,
            )
            level_mappings = (*level_mappings, outcome_mapping)

    if "period" in indexing_params:
        expected_levels = [mapping.name for mapping in level_mappings]
        sr = _validate_and_reorder_levels(series=sr, expected_levels=expected_levels)
        kind = coordinate_kind(ages)
        labels = tuple(sr.index.get_level_values(kind))
        # Validate schema on every row, but duplicate full keys only after
        # filtering surplus time coordinates. Repeated times across categories
        # are ordinary observations.
        time_gather_indices(
            labels=labels,
            kind=kind,
            ages=ages,
            required_periods=(),
            name=name,
            check_duplicates=False,
        )
        wanted = set(ages.exact_values) | (
            {float(v) for v in ages.exact_values} if kind == "age" else set()
        )
        sr = sr.loc[sr.index.get_level_values(kind).isin(wanted)]
        if sr.index.has_duplicates:
            raise InvalidParamsError(
                f"{name}: duplicate selected {kind} keys "
                f"{sr.index[sr.index.duplicated()].tolist()[:8]}."
            )
        support = (
            tuple(range(ages.n_periods))
            if required_periods is None
            else required_periods
        )
        _require_temporal_cells(
            series=sr,
            mappings=level_mappings,
            required_periods=support,
            name=name,
            kind=kind,
        )

    return _scatter_series(
        series=sr,
        level_mappings=level_mappings,
        name=name,
        array_writer=array_writer,
        fill_value=0.0 if "period" in indexing_params else np.nan,
    )


def _resolve_categoricals(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_name: RegimeName | None,
) -> MappingProxyType[str, DiscreteGrid]:
    """Build combined categorical lookup from model grids and regime overrides.

    Collect discrete state and action grids, then merge in the regime's
    `derived_categoricals` (grids for DAG function outputs).

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        regime_name: Regime for grid discovery. When `None`, grids from
            all regimes are merged.

    Returns:
        Dict mapping variable names to `DiscreteGrid` instances.

    Raises:
        ValueError: If a derived categorical conflicts with a model grid.

    """
    grids: dict[str, DiscreteGrid] = {}
    if regime_name is not None:
        user_regime = user_regimes[regime_name]
        for grids_mapping in (
            _state_grids_with_carried_domains(user_regime.states),
            user_regime.actions,
        ):
            grids.update(
                {n: g for n, g in grids_mapping.items() if isinstance(g, DiscreteGrid)}
            )
        for name, grid in user_regime.derived_categoricals.items():
            if name in grids and grids[name].categories != grid.categories:
                msg = (
                    f"Derived categorical '{name}' conflicts with "
                    f"model grid: {grid.categories} vs "
                    f"{grids[name].categories}."
                )
                raise ValueError(msg)
            grids[name] = grid
    else:
        grids.update(_build_discrete_grid_lookup(user_regimes))
        for user_regime in user_regimes.values():
            for name, grid in user_regime.derived_categoricals.items():
                if name in grids and grids[name].categories != grid.categories:
                    msg = (
                        f"Derived categorical '{name}' conflicts with "
                        f"model grid: {grid.categories} vs "
                        f"{grids[name].categories}."
                    )
                    raise ValueError(msg)
                grids[name] = grid
    return MappingProxyType(grids)


def _is_runtime_grid_param(*, func_name: FunctionName, user_regime: UserRegime) -> bool:
    """Check if a template function key refers to a runtime grid param."""
    if func_name in user_regime.states:
        grid = user_regime.states[func_name]
        return (isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime) or (
            isinstance(grid, _ContinuousStochasticProcess)
            and bool(grid.params_to_pass_at_runtime)
        )
    if func_name in user_regime.actions:
        grid = user_regime.actions[func_name]
        return isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime
    return False


def _fail_if_period_level(sr: pd.Series) -> None:
    """Raise if the Series has a 'period' level instead of 'age'."""
    if "period" in sr.index.names:
        msg = (
            "Use 'age' (with actual age values) as the MultiIndex level name "
            "instead of 'period'."
        )
        raise ValueError(msg)


def _filter_to_grid_ages(
    *,
    series: pd.Series,
    ages: TimeAxis,
) -> pd.Series:
    """Keep only rows whose `"age"` level value is on the `AgeGrid`.

    Args:
        series: Series with an `"age"` MultiIndex level.
        ages: The model's `AgeGrid`.

    Returns:
        Filtered Series containing only rows with valid grid ages.

    """
    grid_ages = {float(v) for v in ages.exact_values}
    age_values = series.index.get_level_values("age").astype(float)
    return series.loc[age_values.isin(grid_ages)]


@dataclass(frozen=True)
class _LevelMapping:
    """Specification for mapping one MultiIndex level to array indices."""

    name: str
    """Level name in the MultiIndex (e.g., `"age"`, `"health"`, `"next_health"`)."""

    size: int
    """Number of positions along this axis."""

    get_code_from_label: Callable[[str], int]
    """Return the integer code for a label."""

    valid_labels: tuple[str, ...] = ()
    """Valid label names, for error messages. Empty for age levels."""


@dataclass(frozen=True, kw_only=True)
class _RegimeIdCode:
    """Map a regime name to its integer code."""

    regime_names_to_ids: RegimeNamesToIds
    """Immutable mapping from regime names to integer indices."""

    def __call__(self, label: str) -> int:
        return int(self.regime_names_to_ids[label])


def _age_level_mapping(ages: TimeAxis) -> _LevelMapping:
    """Create a `_LevelMapping` for the age dimension."""
    # Keyed by every label the index may hold, so an unknown one fails the lookup.
    labels: dict[Hashable, int] = {v: i for i, v in enumerate(ages.exact_values)}
    if coordinate_kind(ages) == "age":
        labels.update({float(v): i for i, v in enumerate(ages.exact_values)})
    return _LevelMapping(
        name=coordinate_kind(ages),
        size=ages.n_periods,
        get_code_from_label=labels.__getitem__,
    )


def _grid_level_mapping(*, name: str, grid: DiscreteGrid) -> _LevelMapping:
    """Create a `_LevelMapping` for a categorical dimension.

    Args:
        name: Level name in the MultiIndex.
        grid: The `DiscreteGrid` defining valid categories.

    Returns:
        `_LevelMapping` mapping category labels to integer codes.

    """
    label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
    return _LevelMapping(
        name=name,
        size=len(grid.categories),
        get_code_from_label=label_to_code.__getitem__,
        valid_labels=grid.categories,
    )


def _build_level_mappings_for_param(
    *,
    indexing_params: list[str],
    grids: Mapping[str, DiscreteGrid],
    ages: TimeAxis,
) -> tuple[_LevelMapping, ...]:
    """Build level mappings for `array_from_series` from indexing params.

    Args:
        indexing_params: Parameter names in output axis order, with
            `"period"` already replaced by `"age"`.
        grids: Categorical grid lookup.
        ages: The model's `AgeGrid`.

    Returns:
        Tuple of `_LevelMapping` instances.

    """
    mappings: list[_LevelMapping] = []
    for param in indexing_params:
        if param == coordinate_kind(ages):
            mappings.append(_age_level_mapping(ages))
        elif param in grids:
            mappings.append(_grid_level_mapping(name=param, grid=grids[param]))
        else:
            msg = (
                f"Unrecognised indexing parameter '{param}'. Expected 'age' "
                f"or a discrete grid name ({sorted(grids)}). If "
                f"'{param}' is a DAG function output, add "
                f'derived_categoricals={{"{param}": DiscreteGrid(...)}} '
                f"to the Regime or Model constructor."
            )
            raise ValueError(msg)
    return tuple(mappings)


def _build_outcome_mapping(
    *,
    func_name: FunctionName,
    grids: Mapping[str, DiscreteGrid],
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_names_to_ids: RegimeNamesToIds,
) -> _LevelMapping:
    """Build a `_LevelMapping` for the outcome axis of a `next_*` function.

    For state transitions (e.g. `"next_partner"`), look up the state grid.
    For per-target transitions (e.g. `"next_health__post65"`), use the target
    regime's grid for the outcome axis.
    For regime transitions (`"next_regime"`), use `regime_names_to_ids`.

    Args:
        func_name: Function name starting with `"next_"`.
        grids: Categorical grid lookup.
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.

    Returns:
        `_LevelMapping` for the outcome (last) axis.

    """
    if func_name == "next_regime":
        return _LevelMapping(
            name="next_regime",
            size=len(regime_names_to_ids),
            get_code_from_label=_RegimeIdCode(regime_names_to_ids=regime_names_to_ids),
            valid_labels=tuple(regime_names_to_ids),
        )

    path = tree_path_from_qname(func_name)
    state_name = path[0].removeprefix("next_")

    # Per-target transitions (e.g. "next_health__post65") must use the TARGET
    # regime's grid for the outcome axis, not the source regime's grid.
    if len(path) > 1:
        target_regime_name = path[1]
        target_user_regime = user_regimes.get(target_regime_name)
        if target_user_regime is not None and state_name in target_user_regime.states:
            target_grid = target_user_regime.states[state_name]
            if isinstance(target_grid, DiscreteGrid):
                return _grid_level_mapping(name=f"next_{state_name}", grid=target_grid)

    return _grid_level_mapping(name=f"next_{state_name}", grid=grids[state_name])


def _require_temporal_cells(
    *,
    series: pd.Series,
    mappings: tuple[_LevelMapping, ...],
    required_periods: tuple[int, ...],
    name: str,
    kind: str,
) -> None:
    """Require complete labelled categorical combinations at every read time."""
    codes = {
        mapping.name: _map_level(
            mapping=mapping, level_values=series.index.get_level_values(mapping.name)
        )
        for mapping in mappings
    }
    cells_per_time = int(
        np.prod([mapping.size for mapping in mappings if mapping.name != kind])
    )
    missing = [
        period
        for period in required_periods
        if np.count_nonzero(codes[kind] == period) != cells_per_time
    ]
    if missing:
        raise InvalidParamsError(
            f"{name}: missing required {kind}/categorical keys at "
            f"period positions {missing[:8]}."
        )


def _scatter_series(
    *,
    series: pd.Series,
    level_mappings: tuple[_LevelMapping, ...],
    fill_value: float = np.nan,
    name: str = "series",
    array_writer: CanonicalArrayWriter | None = None,
) -> FloatND:
    """Scatter a MultiIndex Series into an N-dimensional JAX array.

    Each `_LevelMapping` defines one axis: its size, and how to map labels from
    the corresponding MultiIndex level to integer indices. Positions not covered
    by the Series are filled with `fill_value`.

    Args:
        series: Series with a named MultiIndex.
        level_mappings: One mapping per axis, in output axis order.
        fill_value: Value for positions not present in the Series.
        name: Qualified parameter name attached to upload admission.
        array_writer: Optional owner admitting the scattered device array.

    Returns:
        JAX array with shape `[m.size for m in level_mappings]`.

    """
    expected_levels = [m.name for m in level_mappings]
    series = _validate_and_reorder_levels(
        series=series, expected_levels=expected_levels
    )

    shape = [m.size for m in level_mappings]

    if len(series) == 0:
        if array_writer is not None:
            return _write_pandas_array(
                value=np.full(shape, fill_value),
                dtype=np.dtype(canonical_float_dtype()),
                name=name,
                array_writer=array_writer,
            )
        return jnp.full(shape, fill_value, dtype=canonical_float_dtype())

    index_arrays = [
        _map_level(
            mapping=mapping, level_values=series.index.get_level_values(mapping.name)
        )
        for mapping in level_mappings
    ]

    result = np.full(shape, fill_value)
    result[tuple(index_arrays)] = series.to_numpy()
    return _write_pandas_array(
        value=result,
        dtype=np.dtype(canonical_float_dtype()),
        name=name,
        array_writer=array_writer,
    )


def _map_level(*, mapping: _LevelMapping, level_values: pd.Index) -> np.ndarray:
    """Map label values from one MultiIndex level to integer indices.

    Args:
        mapping: The `_LevelMapping` for this level.
        level_values: Index values from the Series MultiIndex level.

    Returns:
        NumPy array of integer indices.

    Raises:
        ValueError: If any label is not valid for the mapping.

    """
    # Categorical levels must use string labels matching grid category names.
    # Reject integer labels early with a clear message instead of a cryptic KeyError.
    if mapping.valid_labels and any(not isinstance(v, str) for v in level_values):
        non_str_types = sorted(
            {type(v).__name__ for v in level_values if not isinstance(v, str)}
        )
        msg = (
            f"Series index level '{mapping.name}' uses non-string labels "
            f"(types: {non_str_types}) but the DiscreteGrid expects string "
            f"category names. Use string labels matching: "
            f"{sorted(mapping.valid_labels)}."
        )
        raise ValueError(msg)

    try:
        return np.array([mapping.get_code_from_label(v) for v in level_values])
    except ValueError:
        # Age levels: age_to_period raises ValueError with a good message
        raise
    except KeyError:
        # Categorical levels: collect all invalid labels
        invalid = sorted(set(level_values) - set(mapping.valid_labels))
        msg = (
            f"Invalid labels for level '{mapping.name}': {invalid}. "
            f"Valid labels: {sorted(mapping.valid_labels)}."
        )
        raise ValueError(msg) from None


def _validate_and_reorder_levels(
    *,
    series: pd.Series,
    expected_levels: list[str],
) -> pd.Series:
    """Validate MultiIndex level names and reorder to match expected order."""
    actual_names = list(series.index.names)

    if "period" in actual_names and "period" not in expected_levels:
        msg = (
            "Use 'age' (with actual age values) as the MultiIndex level name "
            "instead of 'period'."
        )
        raise ValueError(msg)

    if len(actual_names) != len(set(actual_names)):
        msg = (
            f"Series MultiIndex has duplicate level names: {actual_names}. "
            f"All level names must be unique."
        )
        raise ValueError(msg)

    if set(actual_names) != set(expected_levels):
        msg = (
            f"Series MultiIndex level names must be {expected_levels}, "
            f"but got {actual_names}."
        )
        raise ValueError(msg)

    if actual_names != expected_levels:
        series = series.reorder_levels(expected_levels)  # ty: ignore[invalid-argument-type]

    return series


def _validate_state_columns(
    *,
    state_columns: set[str],
    user_regimes: Mapping[RegimeName, UserRegime],
    initial_regimes: list[RegimeName],
) -> None:
    """Validate that DataFrame columns match model states."""
    expected = _collect_state_names(
        user_regimes=user_regimes, initial_regimes=initial_regimes
    )

    unknown = state_columns - expected
    if unknown:
        msg = (
            f"Unknown columns not matching any state of an initial regime: "
            f"{sorted(unknown)}. "
            f"Expected states: {sorted(expected)}."
        )
        raise ValueError(msg)

    missing = expected - state_columns
    if missing:
        required_by: dict[str, list[str]] = {name: [] for name in missing}
        for regime_name in set(initial_regimes):
            for name in user_regimes[regime_name].states:
                if name in required_by:
                    required_by[name].append(regime_name)
        details = ", ".join(
            _format_missing_state_detail(name=name, required_by=required_by[name])
            for name in sorted(missing)
        )
        msg = f"Missing required state columns: {details}."
        raise ValueError(msg)


def _format_missing_state_detail(*, name: str, required_by: list[str]) -> str:
    if name in PSEUDO_STATE_NAMES:
        return f"'{name}' (required for every subject)"
    if required_by:
        return f"'{name}' (required by {sorted(required_by)})"
    return f"'{name}' (required by an initial regime)"


def _collect_state_names(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    initial_regimes: list[RegimeName],
) -> frozenset[str]:
    """Collect all state names from initial regimes.

    Continuous stochastic processes count as states and are included.

    Returns:
        Set of all state names from the initial regimes, plus the pseudo-state
        names from `PSEUDO_STATE_NAMES` (always required).

    """
    names: set[str] = set(PSEUDO_STATE_NAMES)
    for regime_name in set(initial_regimes):
        names.update(user_regimes[regime_name].states.keys())
    return frozenset(names)


def _state_grids_with_carried_domains(
    states: Mapping[StateName, Grid | Phased | AgeSpecializedGrid | None],
) -> MappingProxyType[StateName, Grid | AgeSpecializedGrid]:
    """Replace each carried-state declaration by its simulate-phase grid.

    A carried value (declared via `Phased(solve=..., simulate=Grid)`) is a
    genuine state in simulation input and output, so label/code discovery
    must see the inner grid like any other state grid. `None` masks are
    resolved before the consumers here run; the filter narrows the type. An
    `AgeSpecializedGrid` is passed through unchanged — it is a continuous state,
    so every consumer (all of which filter for `DiscreteGrid`) skips it.
    """
    return MappingProxyType(
        {
            name: cast("Grid", spec.simulate) if isinstance(spec, Phased) else spec
            for name, spec in states.items()
            if spec is not None
        }
    )


def _build_discrete_grid_lookup(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> MappingProxyType[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances from states and actions across regimes.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime` instances.

    Returns:
        Dict mapping variable name to DiscreteGrid.

    Raises:
        ValueError: If two regimes define the same variable with different categories.

    """
    lookup: dict[str, DiscreteGrid] = {}
    for regime_name, user_regime in user_regimes.items():
        for grids_mapping in (
            _state_grids_with_carried_domains(user_regime.states),
            user_regime.actions,
        ):
            for var_name, grid in grids_mapping.items():
                if isinstance(grid, DiscreteGrid):
                    if var_name in lookup:
                        if lookup[var_name].categories != grid.categories:
                            msg = (
                                f"Inconsistent DiscreteGrid for '{var_name}': "
                                f"regime '{regime_name}' has categories "
                                f"{grid.categories}, but a previous regime has "
                                f"{lookup[var_name].categories}."
                            )
                            raise ValueError(msg)
                    else:
                        lookup[var_name] = grid
    return MappingProxyType(lookup)


# Pseudo-function keys in the params template whose parameters are declared as a
# name set rather than by a callable's signature, so there is no source for
# `array_from_series` to inspect.
_PSEUDO_KEYS_WITHOUT_A_SIGNATURE = frozenset({"certainty_equivalent", "taste_shocks"})


def _scheduled_consumer(
    *,
    func: _ParamConsumer | None,
    param_name: ParameterName,
    phase: Phase | None = None,
) -> UserFunction | None:
    """Return the callable law of a `ByAge` schedule that declares `param_name`.

    A schedule is a declaration, not a callable; the law it selects is what
    reads the parameter. Any other consumer is returned unchanged.
    """
    if isinstance(func, Phased):
        if phase is not None:
            return _scheduled_consumer(
                func=func.solve if phase == "solve" else func.simulate,
                param_name=param_name,
                phase=phase,
            )
        return _variant_declaring(
            variants=tuple(
                consumer
                for variant in (func.solve, func.simulate)
                if (
                    consumer := _scheduled_consumer(func=variant, param_name=param_name)
                )
                is not None
            ),
            param_name=param_name,
        )
    if not isinstance(func, ByAge):
        return func
    laws = tuple(
        variant
        for law in func.laws
        for variant in (
            (
                (law.solve, law.simulate)
                if phase is None
                else ((law.solve if phase == "solve" else law.simulate),)
            )
            if isinstance(law, Phased)
            else (law,)
        )
        if is_user_function(variant)
    )
    if not laws:
        msg = (
            f"Parameter {param_name!r} is given for a `ByAge` schedule with no "
            "callable law: each of its laws is a regime name or a per-target "
            "mapping, so none can read a parameter."
        )
        raise InvalidParamsError(msg)

    return _variant_declaring(variants=laws, param_name=param_name)


def _variant_declaring(
    *, variants: tuple[UserFunction, ...], param_name: ParameterName
) -> UserFunction:
    """Return the first variant declaring `param_name`, else the first variant.

    A `Phased` slot contributes both of its variants to the params template, so
    a parameter may be declared by only one of them.
    """
    validate_temporal_variants(functions=variants, name=param_name)
    for variant in variants:
        if param_name in inspect.signature(variant).parameters:
            return variant
    return variants[0]
