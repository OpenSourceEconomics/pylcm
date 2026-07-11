"""Gated edge objects (E3'): mutual-consent marriage / divorce routing.

The design-doc §2 E3' construct that unlocks MIXED singleton/collective regime
topologies. A source regime declares, per target regime, a `GatedEdge`. At the
END of each period ``t``'s solve — after every period-``t`` regime is solved, so
their value arrays and divorce flags are still live — the engine folds, for each
declared edge and each source stakeholder ``s``, a gated continuation object on
the TARGET regime's period-``t`` grid::

    Wbar^s(x) = jnp.where( gate(x), V_target^{leg_s}(x), V_fallback^s(pi_s(x)) )

with ``gate`` a boolean user function on the target grid (EKL consent eq. 27 /
no-divorce eqs. 9/12) and ``V_fallback^s`` a same-period reference regime's value
at a projection (EKL: the source stakeholder's own single regime). The source's
period ``t-1`` continuation then reads ``Wbar`` in place of the raw target V,
threaded through the ordinary transition machinery (`next_regime_to_V_arr`).

**Numerics (non-negotiable).** The mixture is the strict
``jnp.where(gate, V_target, V_fallback)``, never a linear
``gate*V_target + (1-gate)*V_fallback``: the target value carries the slice-3
``-inf`` sentinel in divorce cells, and ``0 * -inf = NaN``. Every read that
lands the target value — including the gate's own ``V_target_<s>`` reads — is an
on-grid identity-projection interpolation, so it is exact.

The whole fold reuses the slice-3 same-period reference-reader machinery
(`_build_same_period_ref_reader`): the target's own value components and divorce
flag are read as identity-projection references of the target regime, and the
gate refs / leg fallbacks as ordinary projected references. The per-cell fold is
product-mapped over the target regime's state grid.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
from dags import concatenate_functions, with_signature

from _lcm.regime_building.Q_and_F import (
    SAME_PERIOD_V_ARG,
    ResolvedSamePeriodRef,
    _build_same_period_ref_reader,
)
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    ConstraintFunction,
    EconFunctionsMapping,
    RegimeName,
    TransitionFunction,
    TransitionFunctionName,
    _ParamsLeaf,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from lcm.typing import BoolND, ContinuousState, DiscreteState, FloatND

# Suffix under which a target regime's divorce flag `D` (cast to float) is passed
# in the same-period value mapping the fold consumes. Never a real regime name.
D_KEY_SUFFIX = "__gated_edge_D__"

# The float divorce flag is 0.0 / 1.0 at grid points; threshold back to boolean.
_D_THRESHOLD = 0.5

# COLLECTIVE-REGIMES (E4). `V_arr_name` under which a fold's grid-level `gate`
# array (cast to float) is bound when interpolated at a simulate-side
# candidate target-state draw (`get_V_interpolator`); the resulting float is
# thresholded the same way as `D_target` above.
GATE_ARR_NAME = "__gate_arr__"
GATE_THRESHOLD = 0.5


@dataclass(frozen=True, kw_only=True)
class ResolvedEdgeLeg:
    """Engine-side form of one source-stakeholder leg of a gated edge (E3')."""

    source_stakeholder: str | None
    """Source stakeholder name, or `None` for a singleton source's single leg."""

    target_component_index: int | None
    """Index of the OPEN-branch value on the target V's trailing stakeholder
    axis, or `None` for a singleton target."""

    fallback: ResolvedSamePeriodRef
    """The CLOSED-branch reference value (regime, projection, stakeholder index)."""


@dataclass(frozen=True, kw_only=True)
class ResolvedGatedEdge:
    """Engine-side form of a user `GatedEdge`, resolved at model processing."""

    target: RegimeName
    """Name of the target regime whose grid the fold lands on."""

    gate: ConstraintFunction
    """Boolean gate predicate (params already renamed to qnames)."""

    gate_refs: Mapping[str, ResolvedSamePeriodRef]
    """Extra same-period references the gate reads (projected from the target grid)."""

    legs: tuple[ResolvedEdgeLeg, ...]
    """One leg per source component, in SOURCE stakeholder order."""

    reference_regimes: tuple[RegimeName, ...]
    """Deduplicated real regimes whose same-period V the fold reads (fallbacks +
    gate refs), excluding the target itself."""


def _pad_reader_to_state_names(
    reader: Callable[..., FloatND],
    *,
    state_names: tuple[str, ...],
) -> Callable[..., FloatND]:
    """Widen a reader's exposed signature to every one of ``state_names``.

    ``reader``'s own args are kept (states it genuinely reads, plus any extra
    runtime params, e.g. grid points for an irregular-grid projection); any
    ``state_names`` entry missing from that set is added as an ignored
    keyword-only argument, so `_grid_reader`'s downstream `productmap` (which
    always maps over the FULL `state_names`) sees every axis in the wrapped
    function's own signature and does not drop it.
    """
    own_args = tuple(get_union_of_args([reader]))
    padded_args = tuple(dict.fromkeys((*own_args, *state_names)))

    @with_signature(args=padded_args, return_annotation="FloatND")
    def padded(**kwargs: _ParamsLeaf) -> FloatND:
        return reader(**{name: kwargs[name] for name in own_args})

    return padded


def get_edge_fold(
    *,
    edge: ResolvedGatedEdge,
    target_v_info: VInterpolationInfo,
    target_functions: EconFunctionsMapping,
    target_deterministic_transitions: Mapping[
        TransitionFunctionName, TransitionFunction
    ],
    reference_v_info: Mapping[RegimeName, VInterpolationInfo],
    target_stakeholders: tuple[str, ...] | None,
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Build one edge's fold: a jittable ``(Wbar, gate)`` producer on the target grid.

    Returns a callable whose keyword arguments are the target regime's state
    grids, the same-period value mapping (under `SAME_PERIOD_V_ARG` — the target
    V, its float divorce flag, and every reference regime's V) and the gate's
    flat params. It returns a pair: ``Wbar`` of shape ``(*target_state_axes,
    n_source_components)`` for a collective source, or ``(*target_state_axes,)``
    for a singleton source (a single leg with no trailing axis); and the raw
    boolean ``gate`` array of shape ``(*target_state_axes,)`` (grid-level,
    shared across every leg — the same predicate value that decided each
    ``jnp.where`` branch of ``Wbar``).

    COLLECTIVE-REGIMES (E4). The solve-side roll (`_roll_gated_edges` in
    `backward_induction.py`) only ever consumed ``Wbar``; the simulate-side
    value router additionally needs the raw ``gate`` to decide REGIME
    ROUTING for a realized subject (`_lcm/simulation/gated_routing.py`) — the
    grid-level array here is interpolated at the subject's candidate
    target-state draw, exactly like any other solved array simulate reads.
    Returning both from one fold avoids evaluating `gate_evaluator` twice.

    **Numerics.** The target regime's OWN value components and divorce flag are
    read by DIRECT array indexing off the same-period mapping — never by
    interpolation. Linear interpolation of the target's ``-inf``-bearing V would
    compute ``0 * -inf = NaN`` at the grid points ADJACENT to a divorce cell
    (the zero-weight neighbour), poisoning the OPEN branch before the
    ``jnp.where`` could guard it. Only the gate references and leg fallbacks —
    which read OTHER (finite) regimes at projected coordinates — are
    interpolated, product-mapped over the target grid.

    Args:
        edge: The resolved edge declaration.
        target_v_info: The target regime's V-interpolation info (its grid).
        target_functions: The target regime's processed functions, so the
            projections and the gate resolve target states / helper functions.
        target_deterministic_transitions: The target regime's merged
            deterministic ``next_<state>`` laws (available to projections).
        reference_v_info: V-interpolation info per reference regime.
        target_stakeholders: The target regime's stakeholders, or `None`.

    Returns:
        The fold callable producing ``(Wbar, gate)`` on the target grid.
    """
    state_names = target_v_info.state_names

    def _grid_reader(reader: Callable[..., FloatND]) -> Callable[..., FloatND]:
        """Product-map an off-grid reference reader over the target grid.

        COLLECTIVE-REGIMES (E3', slice 5): `productmap` derives its OWN
        outward-facing signature from the wrapped function's own parameters
        (`_lcm.utils.dispatchers.productmap` -> `allow_only_kwargs`), and
        silently DROPS any caller-supplied kwarg not in that signature. A
        same-period-ref projection frequently reads only a STRICT SUBSET of
        the target's `state_names` (e.g. a gate ref projected from a single
        newly-drawn state, ignoring a carried-along one) — but `_grid_reader`
        always maps over the FULL `state_names` (every target-grid axis), and
        `batched_vmap`'s internal closure unconditionally needs every one of
        them present in its call kwargs. Left alone, the unused axes get
        dropped by the signature filter before `batched_vmap` ever sees them,
        raising a `KeyError` on the first unused axis. Padding the reader's
        exposed signature to the full `state_names` (ignoring the padding
        args internally) fixes the mismatch without touching `productmap`
        itself, which is shared far beyond gated edges.
        """
        return productmap(
            func=_pad_reader_to_state_names(reader, state_names=state_names),
            variables=state_names,
            batch_sizes=dict.fromkeys(state_names, 0),
        )

    gate_ref_readers = {
        name: _grid_reader(
            _build_same_period_ref_reader(
                ref=ref,
                v_interpolation_info=reference_v_info[ref.regime],
                functions=target_functions,
                deterministic_transitions=target_deterministic_transitions,
            )
        )
        for name, ref in edge.gate_refs.items()
    }
    fallback_readers = [
        _grid_reader(
            _build_same_period_ref_reader(
                ref=leg.fallback,
                v_interpolation_info=reference_v_info[leg.fallback.regime],
                functions=target_functions,
                deterministic_transitions=target_deterministic_transitions,
            )
        )
        for leg in edge.legs
    ]
    # `get_union_of_args` reflects each reader's EXPOSED signature, which —
    # thanks to `_pad_reader_to_state_names` inside `_grid_reader` — already
    # spans the full `state_names` plus any genuine extra params (e.g.
    # runtime grid points for an irregular-grid projection); no separate
    # union with `state_names` is needed here.
    gate_ref_args = {
        name: tuple(get_union_of_args([reader]))
        for name, reader in gate_ref_readers.items()
    }
    fallback_args = [tuple(get_union_of_args([reader])) for reader in fallback_readers]

    # The gate evaluator: the predicate concatenated with the target DAG, so it
    # may read target states / helper functions; its injected leaves (the
    # `V_target_<s>` components, `D_target`, and the gate-ref names) are bound
    # from the elementwise grid arrays below.
    dag_pool = {
        **dict(target_deterministic_transitions),
        **{k: v for k, v in target_functions.items() if k != "H"},
    }
    gate_evaluator = concatenate_functions(
        functions={**dag_pool, "__gate__": edge.gate},
        targets="__gate__",
        enforce_signature=False,
        set_annotations=True,
    )
    gate_arg_names = tuple(get_union_of_args([gate_evaluator]))

    target_component_names = (
        [f"V_target_{s}" for s in target_stakeholders]
        if target_stakeholders is not None
        else ["V_target"]
    )
    injected_names = frozenset({*target_component_names, "D_target", *gate_ref_readers})

    # Outer signature: the target state grids, the same-period value mapping, and
    # any non-injected params/extras the gate or the reference readers need.
    outer_arg_names = sorted(
        {SAME_PERIOD_V_ARG}
        | set(state_names)
        | {arg for args in gate_ref_args.values() for arg in args}
        | {arg for args in fallback_args for arg in args}
        | (set(gate_arg_names) - injected_names)
    )

    singleton_source = all(leg.source_stakeholder is None for leg in edge.legs)

    @with_signature(args=outer_arg_names, return_annotation="tuple[FloatND, BoolND]")
    def fold(**kwargs: _ParamsLeaf) -> tuple[FloatND, BoolND]:
        same_period_V = cast("Mapping[RegimeName, FloatND]", kwargs[SAME_PERIOD_V_ARG])
        # Direct (un-interpolated) reads of the target's own value and flag, so a
        # `-inf` divorce cell never poisons a neighbour through interpolation.
        target_V = same_period_V[edge.target]
        target_components: dict[str, FloatND] = {}
        for index, name in enumerate(target_component_names):
            target_components[name] = (
                target_V[..., index] if target_stakeholders is not None else target_V
            )
        d_value = same_period_V.get(f"{edge.target}{D_KEY_SUFFIX}")

        gate_ref_values = {
            name: reader(
                **{arg: kwargs[arg] for arg in gate_ref_args[name]},
            )
            for name, reader in gate_ref_readers.items()
        }
        # Broadcast the target state grids to the full grid for any gate that
        # reads a state directly (EKL's do not; supported for generality).
        state_mesh = dict(
            zip(
                state_names,
                jnp.meshgrid(
                    *[jnp.asarray(kwargs[s]) for s in state_names], indexing="ij"
                ),
                strict=True,
            )
        )
        gate_kwargs = _assemble_gate_kwargs(
            gate_arg_names=gate_arg_names,
            target_components=target_components,
            d_value=d_value,
            gate_ref_values=gate_ref_values,
            state_mesh=state_mesh,
            cell_kwargs=kwargs,
        )
        gate = jnp.asarray(gate_evaluator(**gate_kwargs))

        component_values: list[FloatND] = []
        for leg, fb_reader, fb_arg_names in zip(
            edge.legs, fallback_readers, fallback_args, strict=True
        ):
            open_name = (
                "V_target"
                if leg.target_component_index is None
                else target_component_names[leg.target_component_index]
            )
            open_branch = target_components[open_name]
            fallback = fb_reader(**{arg: kwargs[arg] for arg in fb_arg_names})
            # STRICT where — never `gate*V + (1-gate)*fallback` (`0*-inf = NaN`).
            component_values.append(jnp.where(gate, open_branch, fallback))

        wbar = (
            component_values[0]
            if singleton_source
            else jnp.stack(component_values, axis=-1)
        )
        return wbar, gate

    return fold


_FALLBACK_PROJECTION_TARGET_PREFIX = "__fallback_state__"


def build_fallback_state_projector(
    *,
    ref: ResolvedSamePeriodRef,
    fallback_v_info: VInterpolationInfo,
    target_functions: EconFunctionsMapping,
    target_deterministic_transitions: Mapping[
        TransitionFunctionName, TransitionFunction
    ],
) -> Callable[..., Mapping[str, FloatND]]:
    """Project a target-grid point onto one edge leg's FALLBACK state coordinates.

    COLLECTIVE-REGIMES (E4). Companion to `_build_same_period_ref_reader`
    (which reads the fallback regime's V at these same projected
    coordinates, for the solve-side fold): the simulate-side value router
    does not need the fallback's VALUE (Wbar already folds that in) but does
    need the fallback's own STATE coordinates, to write the divorced/rejected
    stakeholder's next-period row into `states[fallback.regime]`. Reuses the
    identical projection-function construction (same `dag_pool`, same
    `concatenate_functions` targets) so the coordinates are guaranteed
    consistent with whatever the fold read.

    Args:
        ref: The leg's resolved fallback reference (`ResolvedEdgeLeg.fallback`).
        fallback_v_info: V-interpolation info of the FALLBACK regime (`ref.regime`),
            fixing which state names to project.
        target_functions: The target regime's processed functions (projections
            are expressed in terms of the target's own states/helpers).
        target_deterministic_transitions: The target regime's merged
            deterministic `next_<state>` laws.

    Returns:
        A callable, keyed by (a subset of) the target's state names plus any
        extra params the projections need, returning a dict of the fallback
        regime's own state-coordinate arrays.
    """
    dag_pool = {
        **dict(target_deterministic_transitions),
        **{k: v for k, v in target_functions.items() if k != "H"},
    }
    projection_funcs: dict[str, Callable[..., FloatND]] = {}
    projection_args: dict[str, tuple[str, ...]] = {}
    for state_name in fallback_v_info.state_names:
        target = f"{_FALLBACK_PROJECTION_TARGET_PREFIX}{state_name}"
        projection_funcs[state_name] = concatenate_functions(
            functions={**dag_pool, target: ref.projection[state_name]},
            targets=target,
            enforce_signature=False,
            set_annotations=True,
        )
        projection_args[state_name] = tuple(
            get_union_of_args([projection_funcs[state_name]])
        )
    arg_names = sorted({arg for args in projection_args.values() for arg in args})

    @with_signature(args=arg_names, return_annotation="Mapping[str, FloatND]")
    def project(**kwargs: _ParamsLeaf) -> Mapping[str, FloatND]:
        return {
            state_name: projection_funcs[state_name](
                **{arg: kwargs[arg] for arg in projection_args[state_name]}
            )
            for state_name in fallback_v_info.state_names
        }

    return project


def _assemble_gate_kwargs(
    *,
    gate_arg_names: tuple[str, ...],
    target_components: Mapping[str, FloatND],
    d_value: FloatND | None,
    gate_ref_values: Mapping[str, FloatND],
    state_mesh: Mapping[str, ContinuousState | DiscreteState],
    cell_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Bind each gate argument to its grid array (E3').

    Resolves the gate's declared arguments against the target's own value
    components (``V_target_<s>``), its boolean divorce flag (``D_target``), the
    gate references, the broadcast target-state grids, and remaining cell kwargs.

    COLLECTIVE-REGIMES (E3', slice 5): ``state_mesh`` carries the target
    regime's own state grids broadcast to the full mesh (`jnp.meshgrid`),
    which may include DISCRETE (int-typed) axes — e.g. EKL's encoded
    spouse-type categorical — not just continuous ones; narrowing this to
    `FloatND` was a stale type hint (see the identical fix on
    `_evaluate_edge_fold`'s ``target_states`` in `backward_induction.py`).
    """
    gate_kwargs: dict[str, object] = {}
    for arg in gate_arg_names:
        if arg in target_components:
            gate_kwargs[arg] = target_components[arg]
        elif arg == "D_target":
            if d_value is None:
                # COLLECTIVE-REGIMES (E4). Solve always publishes a divorce
                # flag for every active collective regime, so this branch is
                # unreachable there; it fires only when a SIMULATE caller
                # invoked the internal `simulate()` without threading
                # `period_to_regime_to_divorce_flags` (e.g. the public
                # `Model.simulate()`, which does not yet surface it — see
                # `pylcm-extension-collective-regimes.md` v2.1, slice 6). Fail
                # clearly instead of `None > 0.5`.
                msg = (
                    "This gate reads 'D_target', but no divorce-flag array "
                    "was supplied for the target regime at this period. "
                    "Forward simulation needs `period_to_regime_to_divorce_flags` "
                    "(the third element `backward_induction.solve` returns) "
                    "threaded through to `simulate()`; the public "
                    "`Model.simulate()` does not yet surface it."
                )
                raise NotImplementedError(msg)
            gate_kwargs[arg] = cast("FloatND", d_value) > _D_THRESHOLD
        elif arg in gate_ref_values:
            gate_kwargs[arg] = gate_ref_values[arg]
        elif arg in state_mesh:
            gate_kwargs[arg] = state_mesh[arg]
        else:
            gate_kwargs[arg] = cell_kwargs[arg]
    return gate_kwargs


def build_same_period_mapping_for_fold(
    *,
    edge: ResolvedGatedEdge,
    period_solution: Mapping[RegimeName, FloatND],
    period_divorce_flags: Mapping[RegimeName, BoolND],
) -> MappingProxyType[RegimeName, FloatND]:
    """Assemble the same-period value mapping the fold reads for one edge.

    Carries the target regime's V, its divorce flag cast to float (under the
    reserved `D_KEY_SUFFIX` key), and every reference regime's V — all
    period-``t`` arrays, still live at the fold site.
    """
    mapping: dict[RegimeName, FloatND] = {edge.target: period_solution[edge.target]}
    d_flag = period_divorce_flags.get(edge.target)
    if d_flag is not None:
        mapping[f"{edge.target}{D_KEY_SUFFIX}"] = jnp.asarray(d_flag, dtype=float)
    for regime_name in edge.reference_regimes:
        mapping[regime_name] = period_solution[regime_name]
    return MappingProxyType(mapping)
