"""Certify that `GridSearch` evaluates every candidate the action grids declare.

The profile contract claims zero *search* error for both required profiles: the
default solver is exhaustive grid search, so no action combination
representable on the declared grids is left out of the maximization. That claim
is a property of the code, not of a run, and it is only worth what re-checks it
after the code changes — so it is stated here twice, in two instruments that
fail for different reasons.

**Structural.** The enumeration route is read off the syntax tree of
`_lcm.solution.grid_search` and `_lcm.regime_building.max_Q_over_a`: the whole
action-name tuple reaches `get_max_Q_over_a`, the whole action mapping reaches
the compiled core, the core product-maps over that tuple unbatched, and each
reduction covers every action axis. A line-number match would rot on the first
edit, so every obligation resolves a name or an attribute chain instead.

**Executable.** A tiny model is solved once per shape and then re-solved from
params alone, with each declared candidate made in turn the *only* feasible one
and, separately, the *unique* maximizer. A candidate the search never visits
publishes $-\\infty$ in the first sweep and a runner-up value in the second, so
the sweep is sensitive to a dropped grid point wherever it is dropped — in the
solver, in the product map, or upstream in the state-action space.

The singleton and collective reductions are separate code paths and both are
swept. `test_dropping_one_candidate_is_visible_to_the_sweep` masks one cell
inside the action product map and shows the sweep going red on it, so a green
sweep is evidence rather than an assertion nothing ran.

What this does not establish: the sweep is finite, over two action grids of six
combinations, so it cannot certify arbitrary grid shapes on its own — that is
what the structural half is for, and neither half is sufficient alone. It says
nothing about *discretization* error either. Which candidates a grid represents
is the modeller's choice; all that is bounded here is the search over the ones
it does.
"""

import ast
import functools
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.regime_building import max_Q_over_a as max_Q_over_a_module
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.collective import CollectiveUtility
from lcm.regime import Regime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.candidate_certificate.generate_sources import derive_source_paths
from tests.candidate_certificate.verify import (
    nonempty_feasibility_masks,
    reference_masked_argmax,
)
from tests.conftest import DECIMAL_PRECISION

_SRC_ROOT = Path(__file__).parent.parent / "src"

_SOURCE_INVENTORY_PATH = (
    Path(__file__).parent / "candidate_certificate" / "sources.json"
)


def _load_certified_sources() -> tuple[str, ...]:
    """Load the generated source inventory consumed by every certificate layer."""
    payload = json.loads(_SOURCE_INVENTORY_PATH.read_text(encoding="utf-8"))
    return tuple(item["path"] for item in payload["sources"])


#: Generated from the certificate's literal ``_parse`` obligations.
CERTIFIED_SOURCES: tuple[str, ...] = _load_certified_sources()

_WORK_VALUES: tuple[float, ...] = (0.0, 1.0)
_CONSUMPTION_VALUES: tuple[float, ...] = (1.0, 2.0, 3.0)
_WEALTH_VALUES = jnp.array([1.0, 2.0])
_CANDIDATES: tuple[tuple[float, float], ...] = tuple(
    (work, consumption) for work in _WORK_VALUES for consumption in _CONSUMPTION_VALUES
)
_REFERENCE_Q_VALUES: tuple[float, ...] = tuple(
    3.0 * work + consumption for work, consumption in _CANDIDATES
)
_NONEMPTY_FEASIBILITY_MASKS = nonempty_feasibility_masks(len(_CANDIDATES))
_MASK_PARAMETER_NAMES = tuple(f"feasible_{index}" for index in range(len(_CANDIDATES)))


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    working: ScalarInt


@categorical(ordered=False)
class RegimeId:
    acting: ScalarInt
    done: ScalarInt


def _parse(relative: str) -> ast.Module:
    """Parse one certified source into its syntax tree.

    Args:
        relative: Repo-relative path of the source to parse.

    Returns:
        The parsed module.
    """
    path = _SRC_ROOT.parent / relative
    # Named, not defaulted: house style puts literal UTF-8 in sources, and a
    # platform whose default codec is not UTF-8 cannot decode them.
    return ast.parse(path.read_text(encoding="utf-8"))


def _definition(*, tree: ast.Module, qualname: str) -> ast.AST:
    """Return the definition reached by a dotted `Class.method` or `function` path.

    Args:
        tree: Parsed module to search.
        qualname: Dotted path of the definition.

    Returns:
        The matching function or class definition node.
    """
    node: ast.AST = tree
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.ClassDef) and child.name == part:
                node = child
                break
        else:  # pragma: no cover - a missing definition is the failure itself
            msg = f"{qualname!r} has no definition named {part!r}"
            raise AssertionError(msg)
    return node


def _calls_named(*, node: ast.AST, name: str) -> list[ast.Call]:
    """Return every call to `name`, plain or as an attribute.

    Args:
        node: Subtree to search.
        name: Callee name to match.

    Returns:
        List of matching call nodes, in source order.
    """
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (getattr(child.func, "id", None) or getattr(child.func, "attr", None))
        == name
    ]


def _keyword(*, call: ast.Call, name: str) -> ast.expr | None:
    """Return the value passed as `name=`, or `None` when it is not passed.

    Args:
        call: Call node to read.
        name: Keyword argument name.

    Returns:
        The keyword's value expression, or `None`.
    """
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _chain(node: ast.expr | None) -> str | None:
    """Render a bare name or attribute chain as dotted text, else `None`.

    Args:
        node: Expression to render.

    Returns:
        The dotted text, or `None` for any other expression shape.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _splatted_chains(*, node: ast.AST) -> set[str]:
    """Return the dotted sources of every `**` splat below `node`.

    A splat of `dict(x)` is reported as `x`, so `**dict(space.actions)` and
    `**space.actions` read the same. Every other call is left wrapped and so
    reports nothing: `dict` copies a mapping whole, and anything else is exactly
    the transformation this certificate must not let past.

    Args:
        node: Subtree to search.

    Returns:
        Set of dotted chains that are splatted.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        values: list[ast.expr | None] = []
        if isinstance(child, ast.Call):
            values = [word.value for word in child.keywords if word.arg is None]
        elif isinstance(child, ast.Dict):
            values = [
                value
                for key, value in zip(child.keys, child.values, strict=True)
                if key is None
            ]
        for value in values:
            unwrapped = (
                value.args[0]
                if isinstance(value, ast.Call)
                and getattr(value.func, "id", None) == "dict"
                and len(value.args) == 1
                and not value.keywords
                else value
            )
            chain = _chain(unwrapped)
            if chain is not None:
                found.add(chain)
    return found


def test_build_period_kernels_passes_the_whole_action_name_tuple():
    """The kernel builder hands `get_max_Q_over_a` the space's own action names.

    Anything that filtered, sliced or rebuilt the tuple here would shrink the
    search without shrinking the declared grids, which is exactly the omission
    the zero bound denies.
    """
    tree = _parse("src/_lcm/solution/grid_search.py")
    builder = _definition(tree=tree, qualname="GridSearch.build_period_kernels")
    call = _calls_named(node=builder, name="get_max_Q_over_a")[0]

    assert (
        _chain(_keyword(call=call, name="action_names"))
        == "context.state_action_space.action_names"
    )


@pytest.mark.parametrize("method", ["build_lower_args", "__call__"])
def test_period_kernel_splats_the_whole_action_mapping(method: str):
    """Lowering and the call both splat the state-action space's whole actions.

    The compiled core is lowered against the same mapping it is called with, so
    a subset in either place would silently compile a smaller search than the
    one the other announces.
    """
    tree = _parse("src/_lcm/solution/grid_search.py")
    kernel = _definition(tree=tree, qualname=f"_GridSearchPeriodKernel.{method}")

    assert "state_action_space.actions" in _splatted_chains(node=kernel)


def test_actions_are_product_mapped_over_the_whole_action_tuple():
    """`Q_and_F` is product-mapped over the action-name tuple itself."""
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    builder = _definition(tree=tree, qualname="get_max_Q_over_a")
    call = _calls_named(node=builder, name="productmap")[0]

    assert _chain(_keyword(call=call, name="variables")) == "action_names"


def test_the_action_product_map_is_unbatched():
    """Every action axis carries batch size 0, so no chunk boundary can drop one.

    Batching splits an axis into blocks; a block-shaped reduction that missed a
    remainder would omit candidates the grids declare. Actions are the inner
    optimization axis and are never batched, which removes that failure mode by
    construction rather than by test.
    """
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    builder = _definition(tree=tree, qualname="get_max_Q_over_a")
    call = _calls_named(node=builder, name="productmap")[0]
    batch_sizes = _keyword(call=call, name="batch_sizes")
    from_keys = _calls_named(node=batch_sizes, name="fromkeys") if batch_sizes else []

    assert [
        (_chain(node.args[0]), ast.literal_eval(node.args[1])) for node in from_keys
    ] == [("action_names", 0)]


@pytest.mark.parametrize(
    ("builder", "reducer"),
    [
        ("get_max_Q_over_a", "max"),
        ("get_argmax_and_max_Q_over_a", "argmax_and_max"),
    ],
)
def test_the_singleton_reduction_covers_every_action_axis(builder: str, reducer: str):
    """The singleton value is a full reduction: masked, and over no named axis.

    A reduction with no `axis=` covers the whole action product. An `axis=` here
    would leave some action axis unsearched and change the published rank, and
    `where=F_arr` is what makes the mask remove infeasible candidates only —
    never representable ones. Solve and simulate reduce with different callables
    and each owes the obligation separately.
    """
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    node = _definition(tree=tree, qualname=builder)
    reductions = [
        call
        for call in _calls_named(node=node, name=reducer)
        if _chain(_keyword(call=call, name="where")) == "F_arr"
    ]

    assert [_keyword(call=call, name="axis") for call in reductions] == [None]


@pytest.mark.parametrize("builder", ["get_max_Q_over_a", "get_argmax_and_max_Q_over_a"])
def test_the_collective_reduction_treats_every_feasibility_axis_as_an_action(
    builder: str,
):
    """Both collective reductions scalarize over `tuple(range(F_arr.ndim))`.

    The feasibility array carries one axis per action and no stakeholder axis,
    so its full rank *is* the action product. Reducing over a prefix of it would
    hand the household an argmax taken over part of its choice set.
    """
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    node = _definition(tree=tree, qualname=builder)
    assigned = [
        ast.unparse(child.value)
        for child in ast.walk(node)
        if isinstance(child, ast.Assign)
        and [target.id for target in child.targets if isinstance(target, ast.Name)]
        == ["action_axes"]
    ]

    assert assigned == ["tuple(range(F_arr.ndim))"]


def test_the_taste_shock_reduction_covers_every_action_axis():
    """Continuous axes are maximized out and every remaining axis is smoothed.

    With taste shocks the discrete axes are aggregated by `logsum` rather than
    by `max`, but the candidate set is unchanged: the continuous axes run from
    the first non-discrete axis to the last, and the smoothing runs over all of
    what is left.
    """
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    builder = _definition(tree=tree, qualname="get_max_Q_over_a")
    assigned = {
        target.id: ast.unparse(child.value)
        for child in ast.walk(builder)
        if isinstance(child, ast.Assign)
        for target in child.targets
        if isinstance(target, ast.Name) and target.id == "continuous_axes"
    }
    smoothing = _calls_named(node=builder, name="logsum_and_softmax")[0]
    axes = _keyword(call=smoothing, name="axes")

    assert (
        assigned["continuous_axes"],
        ast.unparse(axes) if axes is not None else None,
    ) == ("tuple(range(n_discrete_action_axes, Q_arr.ndim))", "tuple(range(Qc.ndim))")


def test_no_obligation_rests_on_an_undeclared_source():
    """The AST-derived obligations equal the generated inventory exactly."""
    parsed = set(derive_source_paths(Path(__file__)))

    assert parsed == set(CERTIFIED_SOURCES)


def test_certified_sources_are_read_as_utf_8_whatever_the_platform_default_is():
    """The structural reads name their encoding rather than taking the locale's.

    House style puts literal `—`, `→` and `μ` in source files, and a platform
    whose default codec is not UTF-8 cannot decode them: the read raises before
    anything can be parsed.
    """
    recorded: list[str | None] = []
    original = Path.read_text

    def recording_read_text(self: Path, *args: object, **kwargs: object) -> str:
        recorded.append(kwargs.get("encoding"))  # ty: ignore[invalid-argument-type]
        return original(self, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "read_text", recording_read_text)
        for relative in CERTIFIED_SOURCES:
            _parse(relative)

    assert set(recorded) == {"utf-8"}


def _only_target(
    work: DiscreteAction,
    consumption: ContinuousAction,
    target_work: float,
    target_consumption: float,
) -> BoolND:
    """Admit exactly the one candidate the params name."""
    return jnp.isclose(work, target_work) & jnp.isclose(consumption, target_consumption)


def _candidate_mask(
    work: DiscreteAction,
    consumption: ContinuousAction,
    feasible_0: float,
    feasible_1: float,
    feasible_2: float,
    feasible_3: float,
    feasible_4: float,
    feasible_5: float,
) -> BoolND:
    """Admit the candidates named by a six-bit parameterized mask."""
    flat_index = jnp.asarray(work, dtype=jnp.int32) * len(
        _CONSUMPTION_VALUES
    ) + jnp.asarray(consumption - _CONSUMPTION_VALUES[0], dtype=jnp.int32)
    flags = jnp.asarray(
        [feasible_0, feasible_1, feasible_2, feasible_3, feasible_4, feasible_5]
    )
    return flags[flat_index] > 0.5


def _ordinal_utility(
    wealth: ContinuousState, work: DiscreteAction, consumption: ContinuousAction
) -> FloatND:
    """Publish the strict flattened ordering Q = wealth + [1,2,3,4,5,6]."""
    return wealth + 3.0 * work + consumption


def _labelled_utility(
    wealth: ContinuousState, work: DiscreteAction, consumption: ContinuousAction
) -> FloatND:
    """Give every candidate its own value, so the published one identifies it."""
    return wealth + consumption + 10.0 * work


def _peaked_utility(
    wealth: ContinuousState,
    work: DiscreteAction,
    consumption: ContinuousAction,
    target_work: float,
    target_consumption: float,
) -> FloatND:
    """Peak at the state's own value on the named candidate, below it elsewhere."""
    return wealth - (work - target_work) ** 2 - (consumption - target_consumption) ** 2


def _utility_f(
    wealth: ContinuousState, work: DiscreteAction, consumption: ContinuousAction
) -> FloatND:
    """First stakeholder's own value of a candidate."""
    return wealth + consumption + 10.0 * work


def _utility_m(
    wealth: ContinuousState, work: DiscreteAction, consumption: ContinuousAction
) -> FloatND:
    """Second stakeholder's own value of the same candidate."""
    return wealth + 2.0 * consumption + 5.0 * work


def _zero_f() -> FloatND:
    """First stakeholder's terminal payoff."""
    return jnp.array(0.0)


def _zero_m() -> FloatND:
    """Second stakeholder's terminal payoff."""
    return jnp.array(0.0)


def _next_regime() -> ScalarInt:
    """Leave the acting regime after its single period."""
    return RegimeId.done


def _actions() -> dict[str, DiscreteGrid | LinSpacedGrid]:
    """Return the two action grids every model in this module declares."""
    return {
        "work": DiscreteGrid(Work),
        "consumption": LinSpacedGrid(
            start=_CONSUMPTION_VALUES[0],
            stop=_CONSUMPTION_VALUES[-1],
            n_points=len(_CONSUMPTION_VALUES),
        ),
    }


def _build_model(
    *,
    utility: Callable[..., FloatND] | CollectiveUtility,
    constraints: Mapping[str, Callable[..., BoolND]],
    terminal_utility: Callable[..., FloatND] | CollectiveUtility,
) -> Model:
    """Build the one-decision model the sweeps solve.

    Args:
        utility: The acting regime's utility declaration.
        constraints: The acting regime's constraints.
        terminal_utility: The terminal regime's utility declaration.

    Returns:
        The built model.
    """
    acting = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={
            "wealth": LinSpacedGrid(
                start=float(_WEALTH_VALUES[0]),
                stop=float(_WEALTH_VALUES[-1]),
                n_points=len(_WEALTH_VALUES),
            )
        },
        state_transitions={"wealth": fixed_transition("wealth")},
        actions=_actions(),
        functions={"utility": utility},
        constraints=dict(constraints),
    )
    done = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": terminal_utility},
    )
    return Model(
        regimes={"acting": acting, "done": done},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
    )


def _solve_acting(
    *, model: Model, work: float, consumption: float, function_name: str
) -> FloatND:
    """Solve for one named target candidate and return the acting regime's value.

    Args:
        model: The model to solve.
        work: Target value of the discrete action.
        consumption: Target value of the continuous action.
        function_name: Params key of the function reading the two targets.

    Returns:
        The acting regime's period-0 value array.
    """
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"][function_name]["target_work"] = work
    params["acting"][function_name]["target_consumption"] = consumption
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    return model.solve(params=params, log_level="debug")[0]["acting"]


@pytest.fixture(scope="module")
def unique_feasible_model() -> Model:
    """A singleton model whose params name the one admissible candidate."""
    return _build_model(
        utility=_labelled_utility,
        constraints={"only_target": _only_target},
        terminal_utility=lambda: jnp.array(0.0),
    )


@pytest.fixture(scope="module")
def unique_maximizer_model() -> Model:
    """A singleton model with no constraints whose params name the argmax."""
    return _build_model(
        utility=_peaked_utility,
        constraints={},
        terminal_utility=lambda: jnp.array(0.0),
    )


@pytest.fixture(scope="module")
def collective_model() -> Model:
    """A two-stakeholder model whose params name the one admissible candidate."""
    return _build_model(
        utility=CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m}),
        constraints={"only_target": _only_target},
        terminal_utility=CollectiveUtility(utilities={"f": _zero_f, "m": _zero_m}),
    )


@pytest.fixture(scope="module")
def masked_singleton_model() -> Model:
    """Singleton model whose feasibility neighborhood is supplied by parameters."""
    return _build_model(
        utility=_ordinal_utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=lambda: jnp.array(0.0),
    )


@pytest.fixture(scope="module")
def masked_collective_model() -> Model:
    """Collective model with the same strict ordering for both stakeholders."""
    utility = CollectiveUtility(
        utilities={"f": _ordinal_utility, "m": _ordinal_utility}
    )
    return _build_model(
        utility=utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=CollectiveUtility(utilities={"f": _zero_f, "m": _zero_m}),
    )


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_every_declared_candidate_can_be_the_only_feasible_one(
    unique_feasible_model: Model, work: float, consumption: float
):
    """Constraining the search to one candidate publishes that candidate's value.

    A candidate the search never evaluates leaves the feasibility mask empty,
    and the published value is `-inf` rather than the candidate's own.
    """
    V_arr = _solve_acting(
        model=unique_feasible_model,
        work=work,
        consumption=consumption,
        function_name="only_target",
    )

    aaae(V_arr, _WEALTH_VALUES + consumption + 10.0 * work, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_every_declared_candidate_can_win_the_maximization(
    unique_maximizer_model: Model, work: float, consumption: float
):
    """With everything feasible, the search finds whichever candidate peaks.

    The utility peaks at exactly zero on the named candidate and is strictly
    negative on every other, so a published zero says that candidate entered the
    argmax and a published negative says the search never reached it.
    """
    V_arr = _solve_acting(
        model=unique_maximizer_model,
        work=work,
        consumption=consumption,
        function_name="utility",
    )

    aaae(V_arr, _WEALTH_VALUES, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_every_declared_candidate_can_be_the_household_choice(
    collective_model: Model, work: float, consumption: float
):
    """The collective reduction reads each stakeholder's own value at the candidate.

    The collective path scalarizes and argmaxes on its own code, so exhausting
    the action product there is a separate obligation from the singleton case.
    """
    V_arr = _solve_acting(
        model=collective_model,
        work=work,
        consumption=consumption,
        function_name="only_target",
    )

    aaae(
        V_arr,
        _WEALTH_VALUES[:, None]
        + jnp.array([consumption + 10.0 * work, 2.0 * consumption + 5.0 * work]),
        decimal=DECIMAL_PRECISION,
    )


def _simulate_acting(
    *, model: Model, work: float, consumption: float, function_name: str
) -> dict[str, list[float]]:
    """Simulate one target candidate and return the actions it published at period 0.

    Args:
        model: The model to solve and simulate.
        work: Target value of the discrete action.
        consumption: Target value of the continuous action.
        function_name: Params key of the function reading the two targets.

    Returns:
        Mapping of each action name to its published value, one entry per subject.
    """
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"][function_name]["target_work"] = work
    params["acting"][function_name]["target_consumption"] = consumption
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(len(_WEALTH_VALUES)),
            "wealth": _WEALTH_VALUES,
            "regime_id": jnp.full(len(_WEALTH_VALUES), RegimeId.acting),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    frame = result.to_dataframe(use_labels=False)
    period_0 = frame[frame["period"] == 0]
    return {
        name: period_0[name].to_numpy().tolist() for name in ("work", "consumption")
    }


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_simulation_routes_to_every_declared_candidate(
    unique_feasible_model: Model, work: float, consumption: float
):
    """Simulation publishes the one candidate the constraint admits.

    Simulation maximizes with `get_argmax_and_max_Q_over_a`, a different callable from
    the one backward induction reduces with, so a candidate the solve reaches is not
    thereby a candidate simulation reaches. The bound covers every workload, and
    simulation is one.
    """
    rows = _simulate_acting(
        model=unique_feasible_model,
        work=work,
        consumption=consumption,
        function_name="only_target",
    )

    assert rows == {
        "work": [work] * len(_WEALTH_VALUES),
        "consumption": [consumption] * len(_WEALTH_VALUES),
    }


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_simulation_routes_a_household_to_every_declared_candidate(
    collective_model: Model, work: float, consumption: float
):
    """The collective simulate reduction reaches every candidate too.

    `collective_argmax_and_readout` is a third reduction, distinct from both the
    singleton simulate path and the collective solve path.
    """
    rows = _simulate_acting(
        model=collective_model,
        work=work,
        consumption=consumption,
        function_name="only_target",
    )

    assert rows == {
        "work": [work] * len(_WEALTH_VALUES),
        "consumption": [consumption] * len(_WEALTH_VALUES),
    }


def _params_for_mask(*, model: Model, mask: tuple[bool, ...]) -> dict[str, Any]:
    """Populate the model parameter template with one exact feasibility mask."""
    params = cast("dict[str, Any]", model.get_params_template())
    for name, feasible in zip(_MASK_PARAMETER_NAMES, mask, strict=True):
        params["acting"]["candidate_mask"][name] = float(feasible)
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    return params


def _solve_mask_case(*, model: Model, mask: tuple[bool, ...]) -> FloatND:
    """Solve one mask neighborhood and return the acting regime's value."""
    params = _params_for_mask(model=model, mask=mask)
    return model.solve(params=params, log_level="debug")[0]["acting"]


def _simulate_mask_case(
    *, model: Model, mask: tuple[bool, ...]
) -> dict[str, list[float]]:
    """Simulate one mask neighborhood and return actions plus published values."""
    params = _params_for_mask(model=model, mask=mask)
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(len(_WEALTH_VALUES)),
            "wealth": _WEALTH_VALUES,
            "regime_id": jnp.full(len(_WEALTH_VALUES), RegimeId.acting),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    frame = result.to_dataframe(use_labels=False)
    period_0 = frame[frame["period"] == 0]
    return {name: period_0[name].to_numpy().tolist() for name in period_0.columns}


def _reference_for_mask(mask: tuple[bool, ...]) -> tuple[int, float, float, float]:
    """Return flat index, Q value, work, and consumption from the scalar oracle."""
    index, value = reference_masked_argmax(_REFERENCE_Q_VALUES, mask)
    work, consumption = _CANDIDATES[index]
    return index, value, work, consumption


def test_singleton_solve_matches_reference_over_every_nonempty_feasibility_mask(
    masked_singleton_model: Model,
):
    """Singleton solve agrees with the scalar oracle on all 63 nonempty masks."""
    for mask in _NONEMPTY_FEASIBILITY_MASKS:
        _index, value, _work, _consumption = _reference_for_mask(mask)
        observed = _solve_mask_case(model=masked_singleton_model, mask=mask)
        aaae(observed, _WEALTH_VALUES + value, decimal=DECIMAL_PRECISION)


def test_singleton_simulate_matches_reference_over_every_nonempty_feasibility_mask(
    masked_singleton_model: Model,
):
    """Singleton simulation agrees with the scalar oracle on all mask neighborhoods."""
    for mask in _NONEMPTY_FEASIBILITY_MASKS:
        _index, value, work, consumption = _reference_for_mask(mask)
        observed = _simulate_mask_case(model=masked_singleton_model, mask=mask)
        assert observed["work"] == [work] * len(_WEALTH_VALUES), mask
        assert observed["consumption"] == [consumption] * len(_WEALTH_VALUES), mask
        aaae(observed["value"], _WEALTH_VALUES + value, decimal=DECIMAL_PRECISION)


def test_collective_solve_matches_reference_over_every_nonempty_feasibility_mask(
    masked_collective_model: Model,
):
    """Collective solve selects the same strict winner for both stakeholders."""
    for mask in _NONEMPTY_FEASIBILITY_MASKS:
        _index, value, _work, _consumption = _reference_for_mask(mask)
        observed = _solve_mask_case(model=masked_collective_model, mask=mask)
        expected = jnp.stack([_WEALTH_VALUES + value, _WEALTH_VALUES + value], axis=-1)
        aaae(observed, expected, decimal=DECIMAL_PRECISION)


def test_collective_simulate_matches_reference_over_every_nonempty_feasibility_mask(
    masked_collective_model: Model,
):
    """Collective simulation agrees on action and each stakeholder's value."""
    for mask in _NONEMPTY_FEASIBILITY_MASKS:
        _index, value, work, consumption = _reference_for_mask(mask)
        observed = _simulate_mask_case(model=masked_collective_model, mask=mask)
        assert observed["work"] == [work] * len(_WEALTH_VALUES), mask
        assert observed["consumption"] == [consumption] * len(_WEALTH_VALUES), mask
        expected = _WEALTH_VALUES + value
        aaae(observed["value_f"], expected, decimal=DECIMAL_PRECISION)
        aaae(observed["value_m"], expected, decimal=DECIMAL_PRECISION)


def _argmax_masking_the_last_action_cell() -> Callable[..., Any]:
    """Return an `argmax_and_max` that hides the last cell of the action product.

    Patching the simulate-side reducer alone is what makes the control specific: the
    solve reduction is a different callable and keeps the full candidate set, so a
    green solve sweep beside a red simulate sweep is exactly the divergence a
    solve-only certificate cannot see.

    Returns:
        A drop-in replacement for `argmax_and_max`.
    """
    real = max_Q_over_a_module.argmax_and_max

    def patched(a: Any, *args: Any, where: Any = None, **kwargs: Any) -> Any:
        # `argmax_and_max` also reduces rank-0 masks elsewhere in the engine; those
        # carry no action axis to hide a cell along, so they pass through untouched.
        if where is not None and where.ndim >= 1:
            where = where.at[..., -1].set(False)
        return real(a, *args, where=where, **kwargs)

    return patched


def test_masking_one_simulate_candidate_changes_the_published_action():
    """Hiding one cell from the simulate reducer alone moves the published action.

    Without this the simulate sweep's green would only say that nothing raised. The
    masked candidate is the one the constraint admits, so an exhaustive simulate
    search would publish it and this run publishes a different action.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            max_Q_over_a_module,
            "argmax_and_max",
            _argmax_masking_the_last_action_cell(),
        )
        model = _build_model(
            utility=_labelled_utility,
            constraints={"only_target": _only_target},
            terminal_utility=lambda: jnp.array(0.0),
        )
        rows = _simulate_acting(
            model=model,
            work=_WORK_VALUES[-1],
            consumption=_CONSUMPTION_VALUES[-1],
            function_name="only_target",
        )

    assert rows["consumption"] != [_CONSUMPTION_VALUES[-1]] * len(_WEALTH_VALUES)


def test_masking_one_simulate_candidate_leaves_the_others_alone():
    """The simulate mask removes one cell, not the simulation.

    A patch that broke the simulate reduction outright would move every published
    action and make the control above pass for the wrong reason.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            max_Q_over_a_module,
            "argmax_and_max",
            _argmax_masking_the_last_action_cell(),
        )
        model = _build_model(
            utility=_labelled_utility,
            constraints={"only_target": _only_target},
            terminal_utility=lambda: jnp.array(0.0),
        )
        rows = _simulate_acting(
            model=model,
            work=_WORK_VALUES[0],
            consumption=_CONSUMPTION_VALUES[0],
            function_name="only_target",
        )

    assert rows["consumption"] == [_CONSUMPTION_VALUES[0]] * len(_WEALTH_VALUES)


def _productmap_dropping_the_last_action_cell(
    *, action_names: frozenset[str]
) -> Callable[..., Any]:
    """Return a `productmap` that hides the last cell of the action product.

    Args:
        action_names: Names identifying the action product map, so the state
            product map built by the same function is left alone.

    Returns:
        A drop-in replacement for `productmap`.
    """
    real = max_Q_over_a_module.productmap

    def patched(
        *,
        func: Any,
        variables: tuple[str, ...],
        batch_sizes: dict[str, int],
    ) -> Any:
        mapped = real(func=func, variables=variables, batch_sizes=batch_sizes)
        if frozenset(variables) != action_names:
            return mapped

        @functools.wraps(mapped, assigned=("__name__", "__qualname__", "__doc__"))
        def masked(**call_kwargs: Any) -> Any:
            Q_arr, F_arr = mapped(**call_kwargs)
            return Q_arr, F_arr.at[..., -1].set(False)

        return masked

    return patched


def _solve_with_the_last_action_cell_dropped(
    *, work: float, consumption: float
) -> FloatND:
    """Solve the unique-feasible model with one cell hidden from the search.

    Args:
        work: Target value of the discrete action.
        consumption: Target value of the continuous action.

    Returns:
        The acting regime's period-0 value array under the mask.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            max_Q_over_a_module,
            "productmap",
            _productmap_dropping_the_last_action_cell(
                action_names=frozenset({"work", "consumption"})
            ),
        )
        model = _build_model(
            utility=_labelled_utility,
            constraints={"only_target": _only_target},
            terminal_utility=lambda: jnp.array(0.0),
        )
        return _solve_acting(
            model=model,
            work=work,
            consumption=consumption,
            function_name="only_target",
        )


def test_dropping_one_candidate_is_visible_to_the_sweep():
    """Hiding one cell of the action product turns the sweep red on that cell.

    Without this the sweep's green would only say that nothing raised. The
    masked candidate is the one the params name, so an exhaustive search would
    still publish its value and this run publishes `-inf` instead.
    """
    V_arr = _solve_with_the_last_action_cell_dropped(
        work=_WORK_VALUES[-1], consumption=_CONSUMPTION_VALUES[-1]
    )

    assert bool(jnp.all(jnp.isneginf(V_arr)))


def test_dropping_one_candidate_leaves_every_other_candidate_alone():
    """The mask removes one cell, not the solve.

    A patch that broke the search outright would publish `-inf` everywhere and
    make the control above pass for the wrong reason. Naming a candidate the
    mask does not touch separates the two.
    """
    V_arr = _solve_with_the_last_action_cell_dropped(
        work=_WORK_VALUES[0], consumption=_CONSUMPTION_VALUES[0]
    )

    aaae(
        V_arr,
        _WEALTH_VALUES + _CONSUMPTION_VALUES[0] + 10.0 * _WORK_VALUES[0],
        decimal=DECIMAL_PRECISION,
    )
