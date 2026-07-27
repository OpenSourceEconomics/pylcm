"""Model-level normalization of age specialization.

`normalize_age_specialization` is to `AgeSpecializedFunction` / `AgeSpecializedGrid`
what `normalize_regime_phases` is to `Phased`: the single, early boundary that
resolves the public markers into concrete, build-time objects. It runs immediately
after phase normalization, because — unlike phase, which is known from one regime —
age specialization needs the model `AgeGrid` and each regime's active periods.

After this step:

- every age-specialized **function** factory has been called once per active period,
  and each marker is replaced by a `PeriodizedUserFunction` holding the concrete
  per-period callables (never the factory);
- every age-specialized **grid** factory has been called once per active period, its
  shape-invariance contract validated, and the concrete grids/nodes recorded in an
  `AgeGridSchedule`; the phase specs and the representative regime carry the concrete
  representative-age grid;
- no downstream object retains a `build(age)` factory, so backward induction,
  simulation, AOT compilation, diagnostics, and output only *select* prebuilt objects.

Grid sharing across periods is keyed on the explicit, user-declared
`AgeSpecializedGrid.signature(age)` contract (recorded in the schedule), not on a
numeric fingerprint of the resolved nodes.
"""

import dataclasses
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from _lcm.grids.continuous import ContinuousGrid
from _lcm.regime_building.age_specialization import (
    INVARIANT,
    _describe_trait_mismatch,
    _grid_traits,
    _GridTraits,
    _GridTraitsError,
)
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.phases import PhasedRegimeSpec, RegimePhaseSpec
from _lcm.typing import EconFunction, RegimeName, StateName
from lcm.ages import AgeGrid
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.transition import AgeSpecializedFunction, AgeSpecializedGrid
from lcm.typing import Float1D, UserFunction

# --------------------------------------------------------------------------- #
# Public, build-time-resolved containers                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class PeriodizedUserFunction:
    """Concrete age-specific user functions built during model creation.

    Replaces a public `AgeSpecializedFunction` in the normalized phase specs. It
    holds only concrete callables (one per active period) and the per-period dedup
    signatures; it must never retain the user factory. `representative` is the
    first-active-period concrete callable, used by the age-invariant machinery.
    """

    representative: UserFunction
    concrete_by_period: MappingProxyType[int, UserFunction]
    signature_by_period: MappingProxyType[int, Hashable]

    def resolve(self, period: int) -> UserFunction:
        """Return the concrete callable for `period`."""
        return self.concrete_by_period[period]

    def signature(self, period: int) -> Hashable:
        """Return the dedup signature for `period`."""
        return self.signature_by_period[period]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
        msg = (
            "PeriodizedUserFunction is an internal model-build object and must be "
            "resolved by period before DAG construction."
        )
        raise TypeError(msg)


@dataclass(frozen=True, kw_only=True)
class PeriodizedEconFunction:
    """Processed concrete functions, grouped by explicit user signature.

    The processed counterpart of `PeriodizedUserFunction` (params already renamed
    to qnames), produced by `processing._process_one_function`. Distinct periods
    that share a `signature` share one processed `EconFunction`, so the per-period
    Q/F build resolves by period without ever calling a user factory.
    """

    representative: EconFunction
    function_by_signature: MappingProxyType[Hashable, EconFunction]
    signature_by_period: MappingProxyType[int, Hashable]

    def resolve(self, period: int) -> EconFunction:
        """Return the processed function for `period`."""
        return self.function_by_signature[self.signature_by_period[period]]

    def signature(self, period: int) -> Hashable:
        """Return the dedup signature for `period`."""
        return self.signature_by_period[period]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
        msg = (
            "PeriodizedEconFunction is an internal model-build object and must be "
            "resolved by period before DAG tracing."
        )
        raise TypeError(msg)


@dataclass(frozen=True, kw_only=True)
class ResolvedAgeGrid:
    """One age-specialized grid resolved concretely for one period."""

    grid: ContinuousGrid
    nodes: Float1D
    signature: Hashable


@dataclass(frozen=True, kw_only=True)
class AgeGridSchedule:
    """All concrete age-dependent grid data, built during model creation.

    `by_period` holds only the periods where a regime is active (there are no
    inactive-period placeholder grids). `specialized_states_by_regime` records,
    per regime, which continuous states are age-specialized.
    """

    by_period: MappingProxyType[
        int,
        MappingProxyType[RegimeName, MappingProxyType[StateName, ResolvedAgeGrid]],
    ]
    specialized_states_by_regime: MappingProxyType[RegimeName, frozenset[StateName]]

    def current_nodes(
        self,
        *,
        period: int,
        regime_name: RegimeName,
    ) -> MappingProxyType[StateName, Float1D]:
        """Concrete node arrays for a regime's age-specialized states at `period`."""
        entries = self.by_period.get(period, {}).get(regime_name, {})
        return MappingProxyType(
            {state_name: entry.nodes for state_name, entry in entries.items()}
        )

    def grid_signature(
        self,
        *,
        period: int,
        regime_name: RegimeName,
    ) -> Hashable:
        """Explicit per-state signatures for a regime's grids at `period`.

        The user-declared `AgeSpecializedGrid.signature(age)` values — the sole
        contract that decides whether two periods may share a compiled program.
        """
        entries = self.by_period.get(period, {}).get(regime_name, {})
        return tuple(
            (state_name, entries[state_name].signature)
            for state_name in sorted(entries)
        )


@dataclass(frozen=True, kw_only=True)
class AgeNormalizationResult:
    """Output of `normalize_age_specialization`."""

    representative_user_regimes: MappingProxyType[RegimeName, FinalizedUserRegime]
    phased_specs: MappingProxyType[RegimeName, PhasedRegimeSpec]
    grid_schedule: AgeGridSchedule | None


# --------------------------------------------------------------------------- #
# Period-based resolution of processed periodized functions                    #
# --------------------------------------------------------------------------- #


def resolve_periodized_node(node: object, period: int) -> object:
    """Return a `PeriodizedEconFunction`'s processed function for `period`; else it."""
    if isinstance(node, PeriodizedEconFunction):
        return node.resolve(period)
    return node


def periodized_node_signature(node: object, period: int) -> Hashable:
    """`node`'s period signature: its explicit signature, or `INVARIANT`."""
    if isinstance(node, PeriodizedEconFunction):
        return node.signature(period)
    return INVARIANT


def resolve_periodized_nodes(
    mapping: Mapping[str, object], period: int
) -> Mapping[str, object]:
    """Resolve every `PeriodizedEconFunction` in a flat mapping at `period`.

    Returns the input unchanged when it holds no periodized node, so an
    age-invariant model builds byte-identically to one with no per-age wiring.
    """
    if not any(isinstance(node, PeriodizedEconFunction) for node in mapping.values()):
        return mapping
    return MappingProxyType(
        {name: resolve_periodized_node(node, period) for name, node in mapping.items()}
    )


def periodized_tree_signature(tree: Mapping[str, object], period: int) -> Hashable:
    """Fingerprint a (possibly nested) mapping of nodes at `period`.

    Recurse into `Mapping` values and emit sorted `(path, signature)` pairs, so a
    periodized node nested under one key cannot collide with one under another.
    Mirrors the structure of pylcm's nested transition trees.
    """
    pairs: list[tuple[str, Hashable]] = []
    for key in sorted(tree):
        value = tree[key]
        if isinstance(value, Mapping):
            nested = cast("Mapping[str, object]", value)
            pairs.append((key, periodized_tree_signature(nested, period)))
        else:
            pairs.append((key, periodized_node_signature(value, period)))
    return tuple(pairs)


def continuation_grid_signature_from_schedule(
    *,
    grid_schedule: AgeGridSchedule | None,
    target_period: int,
    target_regimes: tuple[RegimeName, ...],
) -> Hashable:
    """Explicit grid-signature key for a period's continuation targets at `t+1`.

    Contains only the user-declared `AgeSpecializedGrid.signature(age)` values of the
    actual target regimes' grids at `target_period` — not bounds, nodes, bytes, or
    concrete objects. Periods whose continuation grids share these signatures may
    share a compiled `Q_and_F`.
    """
    if grid_schedule is None:
        return ()
    return tuple(
        (
            target_regime,
            grid_schedule.grid_signature(
                period=target_period, regime_name=target_regime
            ),
        )
        for target_regime in target_regimes
        if target_regime in grid_schedule.by_period.get(target_period, {})
    )


# --------------------------------------------------------------------------- #
# Internal per-marker resolution caches                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class _ResolvedFunctionMarker:
    representative: UserFunction
    concrete_by_period: MappingProxyType[int, UserFunction]
    signature_by_period: MappingProxyType[int, Hashable]


@dataclass(frozen=True, kw_only=True)
class _ResolvedGridMarker:
    representative: ContinuousGrid
    concrete_by_period: MappingProxyType[int, ResolvedAgeGrid]


def _resolve_function_marker(
    *,
    marker: AgeSpecializedFunction,
    active_periods: tuple[int, ...],
    ages: AgeGrid,
) -> _ResolvedFunctionMarker:
    """Build one function marker's concrete callables over its active periods."""
    concrete_by_period: dict[int, UserFunction] = {}
    signature_by_period: dict[int, Hashable] = {}
    for period in active_periods:
        age = ages.period_to_age(period)
        concrete_by_period[period] = marker.build(age)
        signature_by_period[period] = marker.signature(age)
    representative_period = active_periods[0]
    return _ResolvedFunctionMarker(
        representative=concrete_by_period[representative_period],
        concrete_by_period=MappingProxyType(concrete_by_period),
        signature_by_period=MappingProxyType(signature_by_period),
    )


def _reject_runtime_points_grid(
    *,
    regime_name: RegimeName,
    state_name: StateName,
    grid: ContinuousGrid,
) -> None:
    """Reject an `AgeSpecializedGrid` that resolves to a runtime-points grid.

    Age specialization is resolved completely at model creation; a grid whose
    points arrive through model parameters at solve/simulation time belongs to the
    separate runtime-points `IrregSpacedGrid` path, not inside an age marker.
    """
    if bool(getattr(grid, "pass_points_at_runtime", False)):
        msg = (
            f"AgeSpecializedGrid state '{state_name}' in regime '{regime_name}' "
            f"resolves to a {type(grid).__name__} whose points are supplied at "
            f"runtime. AgeSpecializedGrid is resolved completely at model creation. "
            f"Use a plain runtime-points IrregSpacedGrid, or supply concrete points "
            f"from build(age)."
        )
        raise RegimeInitializationError(msg)


def _resolve_grid_marker(
    *,
    regime_name: RegimeName,
    state_name: StateName,
    marker: AgeSpecializedGrid,
    active_periods: tuple[int, ...],
    ages: AgeGrid,
) -> _ResolvedGridMarker:
    """Build one grid marker's concrete grids over its active periods, validated.

    Every active period's `build(age)` must return a `ContinuousGrid` of the same
    class, batch_size, points mode, resolved node shape, dtype, and weak_type; only
    node values may vary. Runtime-points grids are rejected outright.
    """
    concrete_by_period: dict[int, ResolvedAgeGrid] = {}
    first_traits: _GridTraits | None = None
    first_period: int | None = None
    for period in active_periods:
        age = ages.period_to_age(period)
        grid = marker.build(age)
        if not isinstance(grid, ContinuousGrid):
            msg = (
                f"AgeSpecializedGrid '{state_name}' in regime '{regime_name}' "
                f"build(age) must return a ContinuousGrid; got "
                f"{type(grid).__name__} at period {period}."
            )
            raise RegimeInitializationError(msg)
        _reject_runtime_points_grid(
            regime_name=regime_name, state_name=state_name, grid=grid
        )
        try:
            traits = _grid_traits(grid)
        except _GridTraitsError as error:
            msg = (
                f"AgeSpecializedGrid '{state_name}' in regime '{regime_name}' at "
                f"period {period}: {error}"
            )
            raise RegimeInitializationError(msg) from error
        if first_traits is None:
            first_traits, first_period = traits, period
        elif first_traits != traits:
            msg = (
                f"AgeSpecializedGrid '{state_name}' in regime '{regime_name}' is not "
                f"shape-invariant: {_describe_trait_mismatch(first_traits, traits)} "
                f"The first active age is period {first_period}, the offending one is "
                f"period {period}. Age-varying grids must keep the same class, "
                f"batch_size, points mode and resolved node shape/dtype at every "
                f"active age; only their bounds or node values may vary."
            )
            raise RegimeInitializationError(msg)
        concrete_by_period[period] = ResolvedAgeGrid(
            grid=grid,
            nodes=grid.to_jax(),
            signature=marker.signature(age),
        )
    representative_period = active_periods[0]
    return _ResolvedGridMarker(
        representative=concrete_by_period[representative_period].grid,
        concrete_by_period=MappingProxyType(concrete_by_period),
    )


# --------------------------------------------------------------------------- #
# Representative-regime and phase-spec rewriting                               #
# --------------------------------------------------------------------------- #


def _representative_function(
    value: object,
    function_cache: dict[int, _ResolvedFunctionMarker],
) -> object:
    """Replace an age-function marker (bare or inside `Phased`) by its concrete.

    Returns the first-active-period concrete callable for a marker, a `Phased`
    with its variants likewise replaced, or the value unchanged.
    """
    if isinstance(value, AgeSpecializedFunction):
        return function_cache[id(value)].representative
    if isinstance(value, Phased):
        return Phased(
            solve=cast(
                "UserFunction", _representative_function(value.solve, function_cache)
            ),
            simulate=cast(
                "UserFunction",
                _representative_function(value.simulate, function_cache),
            ),
        )
    return value


def _representative_regime(
    *,
    user_regime: FinalizedUserRegime,
    function_cache: dict[int, _ResolvedFunctionMarker],
    grid_cache: dict[int, _ResolvedGridMarker],
) -> FinalizedUserRegime:
    """Rebuild one regime with every age marker replaced by its representative.

    The returned regime has no public age markers: functions/constraints markers
    become first-active concrete callables, and `AgeSpecializedGrid` states become
    the concrete representative-age grid. It is the input to parameter-template
    creation, variable/grid discovery, the base state-action space, published
    function sets, and age-invariant validation.
    """
    functions = {
        name: _representative_function(value, function_cache)
        for name, value in user_regime.functions.items()
    }
    constraints = {
        name: _representative_function(value, function_cache)
        for name, value in user_regime.constraints.items()
    }
    states = {
        name: (
            grid_cache[id(spec)].representative
            if isinstance(spec, AgeSpecializedGrid)
            else spec
        )
        for name, spec in user_regime.states.items()
    }
    return user_regime.replace(
        functions=functions,
        constraints=constraints,
        states=states,
    )


def _periodize_functions(
    mapping: Mapping[str, object],
    function_cache: dict[int, _ResolvedFunctionMarker],
) -> MappingProxyType[str, object]:
    """Replace every age-function marker in a phase-slice mapping.

    Phase slices come from `normalize_regime_phases`, so `Phased` is already
    resolved: a value is either a plain callable or a bare `AgeSpecializedFunction`.
    Each marker becomes a `PeriodizedUserFunction` built from the cache.
    """
    return MappingProxyType(
        {
            name: (
                _periodized_from_marker(function_cache[id(value)])
                if isinstance(value, AgeSpecializedFunction)
                else value
            )
            for name, value in mapping.items()
        }
    )


def _periodized_from_marker(
    resolved: _ResolvedFunctionMarker,
) -> PeriodizedUserFunction:
    return PeriodizedUserFunction(
        representative=resolved.representative,
        concrete_by_period=resolved.concrete_by_period,
        signature_by_period=resolved.signature_by_period,
    )


def _rewrite_phase_slice(
    *,
    phase_slice: RegimePhaseSpec,
    function_cache: dict[int, _ResolvedFunctionMarker],
    grid_cache: dict[int, _ResolvedGridMarker],
) -> RegimePhaseSpec:
    """Rewrite one phase slice: markers → periodized functions / representative grid."""
    grid_states = {
        name: (
            grid_cache[id(spec)].representative
            if isinstance(spec, AgeSpecializedGrid)
            else spec
        )
        for name, spec in phase_slice.grid_states.items()
    }
    return dataclasses.replace(
        phase_slice,
        functions=cast(
            "MappingProxyType",
            _periodize_functions(phase_slice.functions, function_cache),
        ),
        constraints=cast(
            "MappingProxyType",
            _periodize_functions(phase_slice.constraints, function_cache),
        ),
        grid_states=MappingProxyType(grid_states),
    )


def _regime_has_markers(user_regime: FinalizedUserRegime) -> bool:
    """Whether a regime declares any age-specialized function or grid marker."""
    for value in (*user_regime.functions.values(), *user_regime.constraints.values()):
        if isinstance(value, AgeSpecializedFunction):
            return True
        if isinstance(value, Phased) and (
            isinstance(value.solve, AgeSpecializedFunction)
            or isinstance(value.simulate, AgeSpecializedFunction)
        ):
            return True
    return any(
        isinstance(spec, AgeSpecializedGrid) for spec in user_regime.states.values()
    )


def _collect_function_markers(
    user_regime: FinalizedUserRegime,
) -> dict[int, AgeSpecializedFunction]:
    """Every distinct age-function marker on a regime, keyed by object identity."""
    markers: dict[int, AgeSpecializedFunction] = {}

    def _visit(value: object) -> None:
        if isinstance(value, AgeSpecializedFunction):
            markers.setdefault(id(value), value)
        elif isinstance(value, Phased):
            _visit(value.solve)
            _visit(value.simulate)

    for value in (*user_regime.functions.values(), *user_regime.constraints.values()):
        _visit(value)
    return markers


def normalize_age_specialization(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    phased_specs: Mapping[RegimeName, PhasedRegimeSpec],
    ages: AgeGrid,
) -> AgeNormalizationResult:
    """Resolve every age-specialized marker into concrete model-creation objects.

    For each regime, build every age-specialized function and grid factory once per
    active period, validate the grids' shape-invariance contract, and:

    - replace public markers in the representative regime by first-active concrete
      objects (functions/constraints) and representative-age grids (states);
    - replace public markers in the phase specs by `PeriodizedUserFunction` (functions
      /constraints) and representative-age grids (grid states);
    - record all concrete period grids in an `AgeGridSchedule`.

    Regimes with no markers pass through unchanged (byte-identical), so an
    age-invariant model normalizes to exactly its input. `grid_schedule` is `None`
    when no regime declares an `AgeSpecializedGrid`.

    Raises:
        RegimeInitializationError: If a regime declares an age-specialized marker
            but is active at no model age, if a grid marker violates the
            shape-invariance contract, or if a grid marker resolves to a
            runtime-points grid.
    """
    representative: dict[RegimeName, FinalizedUserRegime] = {}
    rewritten_specs: dict[RegimeName, PhasedRegimeSpec] = {}
    schedule_by_period: dict[
        int, dict[RegimeName, MappingProxyType[StateName, ResolvedAgeGrid]]
    ] = {}
    specialized_states_by_regime: dict[RegimeName, frozenset[StateName]] = {}
    any_grid = False

    for regime_name, user_regime in user_regimes.items():
        spec = phased_specs[regime_name]
        if not _regime_has_markers(user_regime):
            representative[regime_name] = user_regime
            rewritten_specs[regime_name] = spec
            continue

        active_periods = ages.get_periods_where(user_regime.active)
        if not active_periods:
            msg = (
                f"Regime '{regime_name}' declares age-specialized objects but is "
                f"active at no model age. Remove the marker or make the regime "
                f"active at least once."
            )
            raise RegimeInitializationError(msg)

        function_cache: dict[int, _ResolvedFunctionMarker] = {
            marker_id: _resolve_function_marker(
                marker=marker, active_periods=active_periods, ages=ages
            )
            for marker_id, marker in _collect_function_markers(user_regime).items()
        }
        grid_cache: dict[int, _ResolvedGridMarker] = {
            id(spec_grid): _resolve_grid_marker(
                regime_name=regime_name,
                state_name=state_name,
                marker=spec_grid,
                active_periods=active_periods,
                ages=ages,
            )
            for state_name, spec_grid in user_regime.states.items()
            if isinstance(spec_grid, AgeSpecializedGrid)
        }

        representative[regime_name] = _representative_regime(
            user_regime=user_regime,
            function_cache=function_cache,
            grid_cache=grid_cache,
        )
        rewritten_specs[regime_name] = PhasedRegimeSpec(
            solution=_rewrite_phase_slice(
                phase_slice=spec.solution,
                function_cache=function_cache,
                grid_cache=grid_cache,
            ),
            simulation=_rewrite_phase_slice(
                phase_slice=spec.simulation,
                function_cache=function_cache,
                grid_cache=grid_cache,
            ),
        )

        # Record the concrete grids per active period into the schedule.
        specialized_states = frozenset(
            state_name
            for state_name, spec_grid in user_regime.states.items()
            if isinstance(spec_grid, AgeSpecializedGrid)
        )
        if specialized_states:
            any_grid = True
            specialized_states_by_regime[regime_name] = specialized_states
            state_to_marker = {
                state_name: user_regime.states[state_name]
                for state_name in specialized_states
            }
            for period in active_periods:
                schedule_by_period.setdefault(period, {})[regime_name] = (
                    MappingProxyType(
                        {
                            state_name: grid_cache[id(marker)].concrete_by_period[
                                period
                            ]
                            for state_name, marker in state_to_marker.items()
                        }
                    )
                )

    grid_schedule = (
        AgeGridSchedule(
            by_period=MappingProxyType(
                {
                    period: MappingProxyType(regimes)
                    for period, regimes in schedule_by_period.items()
                }
            ),
            specialized_states_by_regime=MappingProxyType(specialized_states_by_regime),
        )
        if any_grid
        else None
    )

    return AgeNormalizationResult(
        representative_user_regimes=MappingProxyType(representative),
        phased_specs=MappingProxyType(rewritten_specs),
        grid_schedule=grid_schedule,
    )
