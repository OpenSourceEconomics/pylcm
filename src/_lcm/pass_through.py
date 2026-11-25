"""Pass-through regimes: where a subject is in a stage its regime sits out.

An edge of a staged model lands at its target's next occurrence, the first step
after the current one that belongs to the target's stage, and an exit may enter
a regime of a later stage of its period. In every stage the target sits out on
the way, the subject is in the target's pass-through: the target's states and
members, no action, no shocks, zero utility, the identity transition and an
edge to the next step. Its value is the next step's value passed through
unchanged; a pass-through of the last stage discounts into the next period as
every regime of that stage does. The model's declarations are rewritten onto
the pass-throughs, so every link of the engine's object leads to the next step.
The user never names or sees a pass-through: params and their template, regime
codes, starts, values, simulated results and messages name the declared regime.
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.grids.categorical import (
    categorical,
    get_category_codes,
    validate_category_class,
)
from _lcm.params.edges import EDGES
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.schedules import resolve_initial_nodes
from _lcm.time import ModelTime
from _lcm.typing import (
    DataclassInstance,
    EconFunctionArg,
    EdgeParamsTemplate,
    ParamsTemplate,
)
from lcm.collective import CollectiveUtility
from lcm.exceptions import ModelInitializationError, PyLCMError
from lcm.exits import Consent, Exit
from lcm.initial_nodes import UserInitialNodes
from lcm.phased import Phased
from lcm.regime import Regime, StateTransitionEntry
from lcm.shocks import ShockGroup, ShockGroupName
from lcm.transition import (
    AgeCaseLaw,
    AgeRange,
    AgeSelector,
    ByAge,
    DeclaredModelEdges,
    DeterministicTransition,
    PeriodRange,
    PeriodSelector,
    StochasticTransition,
    Transition,
    TransitionLaw,
    fixed_transition,
)
from lcm.typing import (
    FloatND,
    IntND,
    RegimeName,
    ScalarInt,
    UserAge,
    UserFunction,
    UserInitialConditions,
)

type _Renames = Mapping[RegimeName, RegimeName]
# A source mapped to a `Phased` pair of transitions passes unchanged; the graph
# refuses it naming the supported form.
type _EdgeDeclaration = (
    Transition | Mapping[RegimeName, AgeSelector | PeriodSelector] | Phased
)


def pass_through_utility() -> FloatND:
    """A pass-through pays nothing."""
    return jnp.asarray(0.0)


def with_pass_throughs(  # noqa: C901 - one pass over edges, exits and starts
    *,
    time: ModelTime,
    regimes: Mapping[RegimeName, Regime],
    edges: DeclaredModelEdges,
    initial_nodes: UserInitialNodes,
    regime_id_class: type[DataclassInstance],
    shocks: Mapping[ShockGroupName, ShockGroup],
    declares_model_slots: bool,
) -> tuple[
    ModelTime, Mapping[RegimeName, Regime], DeclaredModelEdges, type[DataclassInstance]
]:
    """Rewrite a staged model onto pass-throughs wherever it skips a stage.

    Returns the model unchanged when nothing skips a stage. Otherwise every edge
    lands in the next stage, every exit stays in its stage and every declared
    start is in the first stage, through the pass-throughs added to the clock,
    the regimes, the edges and the regime codes, after the declared ones.

    Returns:
        The clock, the regimes, the edges and the regime-code class.

    """
    stage_of = time.stage_of
    stage_names = [name for name, _ in time.stages]
    added: dict[RegimeName, tuple[RegimeName, int]] = {}

    def landing(*, target: RegimeName, stage: int) -> RegimeName:
        """The regime a subject on its way to `target` is in at `stage`."""
        if target not in stage_of:
            return target
        position = stage
        while position != stage_of[target]:
            added[f"{target}_in_{stage_names[position]}"] = (target, position)
            position = (position + 1) % time.n_stages
        return target if stage == position else f"{target}_in_{stage_names[stage]}"

    phases = (edges.solve, edges.simulate) if isinstance(edges, Phased) else (edges,)
    renames: dict[RegimeName, dict[RegimeName, RegimeName]] = {}
    for phase in phases:
        # A malformed declaration is refused when the graph is bound.
        for source, declaration in phase.items() if isinstance(phase, Mapping) else ():
            if source in stage_of:
                next_stage = (stage_of[source] + 1) % time.n_stages
                renames.setdefault(source, {}).update(
                    {
                        target: landing(target=target, stage=next_stage)
                        for target in _targets(declaration)
                    }
                )
                gated = sorted(
                    target
                    for target in getattr(declaration, "gates", {})
                    if renames[source].get(target, target) != target
                )
                if gated:
                    raise ModelInitializationError(
                        f"The gated edges of '{source}' in stage "
                        f"'{stage_names[stage_of[source]]}' into {gated} skip stage "
                        f"'{stage_names[next_stage]}'; a gated edge lands in the "
                        "next stage."
                    )
    exits = {
        name: {
            destination: landing(target=destination, stage=stage_of[name])
            for destination in _later_destinations(
                name=name, regimes=regimes, shocks=shocks, time=time
            )
        }
        for name, regime in regimes.items()
        if regime.exit is not None and name in stage_of
    }
    for _, name in resolve_initial_nodes(
        initial_nodes=initial_nodes, regime_names=tuple(regimes), ages=time
    ):
        landing(target=name, stage=0)
    if not added or validate_category_class(regime_id_class):
        return time, regimes, edges, regime_id_class
    if taken := sorted(set(added) & set(regimes)):
        raise ModelInitializationError(
            f"The regimes {taken} are named like the pass-throughs the model "
            "generates; rename them."
        )
    for target, stage in added.values():
        if processes := sorted(
            name
            for name, grid in regimes[target].states.items()
            if isinstance(grid, _ContinuousStochasticProcess)
        ):
            raise ModelInitializationError(
                f"'{target}' sits out stage '{stage_names[stage]}' on the way to "
                f"its next occurrence, where it cannot carry its continuous "
                f"stochastic processes {processes} unchanged. Declare a regime "
                f"of its own in stage '{stage_names[stage]}'."
            )
    if declares_model_slots:
        raise ModelInitializationError(
            "A staged model whose regimes skip stages declares its functions, "
            "constraints, states, state transitions and actions on its regimes, "
            f"not on the model; it passes through {sorted(added)}."
        )
    declared = tuple(get_category_codes(regime_id_class))
    names = (*declared, *added)
    code_of = {name: code for code, name in enumerate(names)}
    every = PeriodRange() if time.kind == "period" else AgeRange()
    successors = {
        name: {landing(target=target, stage=(stage + 1) % time.n_stages): every}
        for name, (target, stage) in added.items()
    }

    def rewritten(
        phase: Mapping[RegimeName, _EdgeDeclaration],
    ) -> Mapping[RegimeName, _EdgeDeclaration]:
        if not isinstance(phase, Mapping):
            return phase
        return {
            **{
                source: _redirected(
                    declaration=declaration,
                    renames=renames.get(source, {}),
                    codes=tuple(
                        code_of[renames.get(source, {}).get(name, name)]
                        for name in declared
                    ),
                    n_regimes=len(names),
                )
                for source, declaration in phase.items()
            },
            **successors,
        }

    return (
        dataclasses.replace(
            time,
            stages=tuple(
                (
                    stage,
                    (
                        *members,
                        *(name for name, (_, at) in added.items() if at == position),
                    ),
                )
                for position, (stage, members) in enumerate(time.stages)
            ),
            pass_throughs=frozenset(added),
        ),
        {
            **{
                name: _with_renamed_targets(
                    regime=regime,
                    renames=renames.get(name, {}),
                    exit_renames=exits.get(name, {}),
                )
                for name, regime in regimes.items()
            },
            **{
                name: _pass_through(regimes[target])
                for name, (target, _) in added.items()
            },
        },
        cast(
            "DeclaredModelEdges",
            Phased(solve=rewritten(edges.solve), simulate=rewritten(edges.simulate))
            if isinstance(edges, Phased)
            else rewritten(edges),
        ),
        cast(
            "type[DataclassInstance]",
            categorical(ordered=False)(
                type(
                    regime_id_class.__name__,
                    (),
                    {"__annotations__": dict.fromkeys(names, ScalarInt)},
                )
            ),
        ),
    )


def first_stage_regime(*, name: RegimeName, time: ModelTime) -> RegimeName:
    """The regime a subject declared to start in `name` is in at the first stage."""
    start = f"{name}_in_{time.stages[0][0]}" if time.stages else name
    return start if start in time.pass_throughs else name


def enter_first_stage(
    *,
    initial_conditions: UserInitialConditions | pd.DataFrame,
    time: ModelTime,
    regime_names_to_ids: Mapping[RegimeName, ScalarInt],
) -> UserInitialConditions | pd.DataFrame:
    """Start a subject of a later stage in its pass-through of the first stage.

    Args:
        initial_conditions: The subjects' starts, by `regime_id` code or by
            `regime_name`.
        time: The model's clock.
        regime_names_to_ids: The code of every regime.

    Returns:
        The starts, each in the first stage of its period.

    """
    starts = {
        name: start
        for name in time.stage_of
        if (start := first_stage_regime(name=name, time=time)) != name
    }
    if not starts:
        return initial_conditions
    if isinstance(initial_conditions, pd.DataFrame):
        if "regime_name" not in initial_conditions:
            return initial_conditions
        return initial_conditions.assign(
            regime_name=initial_conditions["regime_name"].astype(object).replace(starts)
        )
    if "regime_id" not in initial_conditions:
        return initial_conditions
    codes = np.asarray(initial_conditions["regime_id"])
    entered = codes.copy()
    for name, start in starts.items():
        entered[codes == regime_names_to_ids[name]] = regime_names_to_ids[start]
    return {**initial_conditions, "regime_id": entered}


def declared_regime(*, name: RegimeName, time: ModelTime) -> RegimeName:
    """The regime a pass-through belongs to; any other regime is its own."""
    if name not in time.pass_throughs:
        return name
    return name.removesuffix(f"_in_{time.stages[time.stage_of[name]][0]}")


def declared_codes(
    *, codes: Mapping[RegimeName, ScalarInt], time: ModelTime
) -> MappingProxyType[RegimeName, ScalarInt]:
    """The codes of the declared regimes, which precede every pass-through's."""
    return MappingProxyType(
        {name: code for name, code in codes.items() if name not in time.pass_throughs}
    )


def declared_nodes(
    *, pairs: frozenset[tuple[UserAge, RegimeName]], time: ModelTime
) -> frozenset[tuple[UserAge, RegimeName]]:
    """The starts as declared, each pass-through named by its regime."""
    return frozenset(
        (age, declared_regime(name=name, time=time)) for age, name in pairs
    )


def target_slots(
    *, template: ParamsTemplate, time: ModelTime
) -> dict[RegimeName, dict[RegimeName, RegimeName]]:
    """Per source, each pass-through its slots name, with its declared regime."""
    edges = cast("Mapping[RegimeName, EdgeParamsTemplate]", template.get(EDGES, {}))
    return {
        source: {
            name: declared_regime(name=name, time=time)
            for name in (*template.get(source, {}), *edges.get(source, {}))
            if name in time.pass_throughs
        }
        for source in template
        if source != EDGES
    }


# keyword-only-exempt: primary-argument=tree
def renamed_slots[Node](
    tree: Mapping[str, Node],
    *,
    slots: Mapping[RegimeName, Mapping[RegimeName, RegimeName]],
    dropped: frozenset[RegimeName] = frozenset(),
) -> dict[str, Node]:
    """`tree` with the keys of each source's branch and edge branch renamed.

    Params and templates name a target by the regime its edge lands in inside the
    engine, and by its declared regime at the public boundary.
    """

    def branch(*, source: RegimeName, node: Node) -> Node:
        renames = slots.get(source, {})
        if not renames or not isinstance(node, Mapping):
            return node
        return cast(
            "Node", {renames.get(key, key): value for key, value in node.items()}
        )

    return {
        name: cast(
            "Node",
            {
                source: branch(source=source, node=edges)
                for source, edges in node.items()
            },
        )
        if name == EDGES and isinstance(node, Mapping)
        else branch(source=name, node=node)
        for name, node in tree.items()
        if name not in dropped
    }


def name_declared_regimes(*, error: PyLCMError, time: ModelTime | None) -> None:
    """Name each pass-through in `error` by its declared regime, in place.

    A quoted pass-through reads as its regime sitting out its stage; inside a
    parameter path it reads as the regime, the path the user writes.
    """
    if time is None or not time.pass_throughs:
        return
    text = str(error)
    for name in sorted(time.pass_throughs, key=len, reverse=True):
        regime = declared_regime(name=name, time=time)
        stage = time.stages[time.stage_of[name]][0]
        text = text.replace(
            f"'{name}'", f"'{regime}' sitting out stage '{stage}'"
        ).replace(name, regime)
    if text != str(error):
        error.args = (text,)


def _targets(declaration: _EdgeDeclaration) -> tuple[RegimeName, ...]:
    """The targets a source's edge declaration names."""
    if isinstance(declaration, Mapping):
        return tuple(declaration)
    if not isinstance(declaration, Transition):
        return ()
    if declaration.targets is not None:
        return tuple(declaration.targets)
    return tuple(dict.fromkeys(_law_targets(declaration.law)))


def _law_targets(law: TransitionLaw) -> tuple[RegimeName, ...]:
    if isinstance(law, str):
        return (law,)
    if isinstance(law, Mapping):
        return tuple(law)
    if isinstance(law, Phased):
        return (*_law_targets(law.solve), *_law_targets(law.simulate))
    if isinstance(law, ByAge):
        return tuple(target for case in law.laws for target in _law_targets(case))
    return ()


def _later_destinations(
    *,
    name: RegimeName,
    regimes: Mapping[RegimeName, Regime],
    shocks: Mapping[ShockGroupName, ShockGroup],
    time: ModelTime,
) -> tuple[RegimeName, ...]:
    """The exit destinations of `name` that decide in a later stage of its period.

    Raises:
        ModelInitializationError: If the exit observes a shock such a destination
            draws, which is not drawn yet when the exit is decided.

    """
    exit_ = cast("Exit", regimes[name].exit)
    stage_of = time.stage_of
    later = sorted(
        {
            *(target.regime for target in exit_.targets.values()),
            *(
                (exit_.rule.counterparty.regime,)
                if isinstance(exit_.rule, Consent)
                else ()
            ),
        }
        & {other for other, stage in stage_of.items() if stage > stage_of[name]}
    )
    stage_names = [stage for stage, _ in time.stages]
    for destination in later:
        drawn = sorted(
            set(exit_.observes)
            & {
                shock
                for group in regimes[destination].shocks
                if group in shocks
                for shock in shocks[group].shocks
            }
        )
        if drawn:
            raise ModelInitializationError(
                f"The exit of '{name}' in stage '{stage_names[stage_of[name]]}' "
                f"observes {drawn} of '{destination}', drawn in its later stage "
                f"'{stage_names[stage_of[destination]]}'. Draw the shock in stage "
                f"'{stage_names[stage_of[name]]}' and carry it as a state."
            )
    return tuple(later)


def _redirected(
    *,
    declaration: _EdgeDeclaration,
    renames: _Renames,
    codes: tuple[int, ...],
    n_regimes: int,
) -> _EdgeDeclaration:
    """A source's edges, landing each target at its next occurrence."""
    if isinstance(declaration, Mapping):
        return {
            renames.get(target, target): item for target, item in declaration.items()
        }
    if not isinstance(declaration, Transition):
        return declaration
    return dataclasses.replace(
        declaration,
        targets=None
        if declaration.targets is None
        else {
            renames.get(target, target): selector
            for target, selector in declaration.targets.items()
        },
        law=_redirected_law(
            declaration.law, renames=renames, codes=codes, n_regimes=n_regimes
        ),
        roles={
            renames.get(target, target): role
            for target, role in declaration.roles.items()
        },
    )


# keyword-only-exempt: library-callback=ByAge.with_mapped_laws
def _redirected_law(  # noqa: PLR0911 - one return per law form
    law: TransitionLaw, *, renames: _Renames, codes: tuple[int, ...], n_regimes: int
) -> TransitionLaw:
    """A law naming the regimes its targets' next occurrences are in."""
    if isinstance(law, str):
        return renames.get(law, law)
    if isinstance(law, Mapping):
        return {renames.get(target, target): cell for target, cell in law.items()}
    redirected = functools.partial(
        _redirected_law, renames=renames, codes=codes, n_regimes=n_regimes
    )
    if isinstance(law, Phased):
        return cast(
            "AgeCaseLaw",
            Phased(solve=redirected(law.solve), simulate=redirected(law.simulate)),
        )
    if isinstance(law, ByAge):
        return law.with_mapped_laws(
            func=cast("Callable[[AgeCaseLaw], AgeCaseLaw]", redirected)
        )
    if isinstance(law, StochasticTransition):
        return dataclasses.replace(
            law, func=_spread(func=law.func, codes=codes, n_regimes=n_regimes)
        )
    if isinstance(law, DeterministicTransition):
        return DeterministicTransition(func=_recoded(func=law.func, codes=codes))
    return _recoded(func=cast("UserFunction", law), codes=codes)


def _recoded(*, func: UserFunction, codes: tuple[int, ...]) -> UserFunction:
    """A regime selector returning the code of its choice's next occurrence."""
    if codes == tuple(range(len(codes))):
        return func

    @functools.wraps(func)
    def recoded(**kwargs: EconFunctionArg) -> IntND:
        return jnp.asarray(codes, dtype=jnp.int32)[func(**kwargs)]

    return recoded


def _spread(
    *, func: Callable[..., FloatND], codes: tuple[int, ...], n_regimes: int
) -> Callable[..., FloatND]:
    """A regime-probability vector over next occurrences, pass-throughs included."""

    @functools.wraps(func)
    def spread(**kwargs: EconFunctionArg) -> FloatND:
        probabilities = func(**kwargs)
        zeros = jnp.zeros((*probabilities.shape[:-1], n_regimes), probabilities.dtype)
        return zeros.at[..., jnp.asarray(codes)].set(probabilities)  # noqa: PD008

    return spread


def _with_renamed_targets(
    *, regime: Regime, renames: _Renames, exit_renames: _Renames
) -> Regime:
    """A source whose per-target laws and exit name the regimes it lands in."""
    if not any(new != old for old, new in {**renames, **exit_renames}.items()):
        return regime
    exit_ = regime.exit
    if exit_ is not None and exit_renames:
        rule = exit_.rule
        exit_ = dataclasses.replace(
            exit_,
            targets={
                role: dataclasses.replace(
                    target, regime=exit_renames.get(target.regime, target.regime)
                )
                for role, target in exit_.targets.items()
            },
            rule=dataclasses.replace(
                rule,
                counterparty=dataclasses.replace(
                    rule.counterparty,
                    regime=exit_renames.get(
                        rule.counterparty.regime, rule.counterparty.regime
                    ),
                ),
            )
            if isinstance(rule, Consent)
            else rule,
        )
    return regime.replace(
        state_transitions={
            state: _per_target(law=law, renames=renames)
            for state, law in regime.state_transitions.items()
        },
        joint_transitions={
            renames.get(target, target): joint
            for target, joint in regime.joint_transitions.items()
        },
        exit=exit_,
    )


def _per_target(
    *, law: StateTransitionEntry, renames: _Renames
) -> StateTransitionEntry:
    """A state law whose per-target cells name the regimes the source lands in."""
    if isinstance(law, Mapping):
        return {renames.get(target, target): cell for target, cell in law.items()}
    if isinstance(law, Phased) and (
        isinstance(law.solve, Mapping) or isinstance(law.simulate, Mapping)
    ):
        return Phased(
            solve=_per_target(law=law.solve, renames=renames),
            simulate=_per_target(law=law.simulate, renames=renames),
        )
    return law


def _pass_through(regime: Regime) -> Regime:
    """The pass-through of `regime`: its states and members, nothing decided."""
    return Regime(
        states=regime.states,
        state_transitions={name: fixed_transition(name) for name in regime.states},
        functions={
            "utility": CollectiveUtility(
                utilities=dict.fromkeys(regime.members, pass_through_utility)
            )
            if isinstance(regime.functions.get("utility"), CollectiveUtility)
            else pass_through_utility
        },
    )
