"""A parameter is cast by the numeric role of the slot it fills.

A slot is a float when the library owns it as a coefficient (a discount factor, a
taste-shock scale, a process law) or when the callback reading it annotates it as a
float; an integer written there becomes the active float dtype. A slot a callback
annotates as an integer, or leaves unannotated, keeps the kind of its value, so an
integer index or a Boolean mask is never turned into a float.
"""

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from _lcm.typing import FlatParams
from lcm import (
    AgeGrid,
    CESAggregator,
    DiscreteGrid,
    InitialNodes,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Phased,
    Regime,
    categorical,
)
from lcm.certainty_equivalent import PowerMean
from lcm.exceptions import ModelInitializationError
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarFloat,
    ScalarInt,
    UserParams,
)
from tests.params.test_edge_params import _mortal_model

type Rate = float


@categorical(ordered=False)
class _Regimes:
    alive: ScalarInt
    final: ScalarInt


@categorical(ordered=True)
class _Binary:
    a: ScalarInt
    b: ScalarInt


@categorical(ordered=False)
class _Terminal:
    terminal: ScalarInt


def _utility(
    *, wealth: ContinuousState, shock: ContinuousState, d: DiscreteAction
) -> FloatND:
    return wealth + shock + d * 0.0


def _utility_with_bonus(
    *,
    wealth: ContinuousState,
    shock: ContinuousState,
    d: DiscreteAction,
    bonus: FloatND,
) -> FloatND:
    return wealth + shock + d * 0.0 + bonus * 0.5


def _utility_with_rate(
    *, wealth: ContinuousState, shock: ContinuousState, d: DiscreteAction, rate: Rate
) -> FloatND:
    return wealth + shock + d * 0.0 + rate * 0.5


def _utility_with_string_annotation(
    *,
    wealth: ContinuousState,
    shock: ContinuousState,
    d: DiscreteAction,
    bonus: ScalarFloat,
) -> FloatND:
    return wealth + shock + d * 0.0 + bonus * 0.5


def _utility_final(*, wealth: ContinuousState) -> FloatND:
    return wealth * 1.0


def _next_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return wealth


def _model(
    *,
    process: NormalIIDProcess | None = None,
    utility: Callable[..., FloatND] = _utility,
    koopmans_aggregator: CESAggregator | None = None,
    certainty_equivalent: PowerMean | None = None,
    taste_shocks: ExtremeValueTasteShocks | None = None,
) -> Model:
    grid = LinSpacedGrid(start=1.0, stop=2.0, n_points=3)
    alive = Regime(
        states={
            "wealth": grid,
            "shock": process
            or NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=0.1),
        },
        actions={"d": DiscreteGrid(category_class=_Binary)},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": utility},
        koopmans_aggregator=koopmans_aggregator,
        certainty_equivalent=certainty_equivalent,
        taste_shocks=taste_shocks,
    )
    final = Regime(states={"wealth": grid}, functions={"utility": _utility_final})
    return Model(
        regimes={"alive": alive, "final": final},
        regime_id_class=_Regimes,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes=((0, "alive"),),
        edges={"alive": {"final": 0}},
        durable_identity=False,
    )


def _alive_params(
    *, koopmans: Mapping[str, float] | None = None, **alive: Mapping[str, float]
) -> UserParams:
    return {
        "alive": {
            "koopmans_aggregator": {"discount_factor": 0.9, **(koopmans or {})},
            **alive,
        },
        "final": {},
    }


def _values(*, model: Model, params: UserParams) -> list[np.ndarray]:
    result = model.solve(params=params, log_level="off")
    return [
        np.asarray(values)
        for _, regimes in sorted(result.values.items())
        for _, values in sorted(regimes.items())
    ]


def _kind(tree: FlatParams, *path: str) -> str:
    """Return the dtype kind of the canonical leaf at `path`."""
    node: object = tree  # noqa: PAN001 - any node of the canonical tree
    for key in path:
        assert isinstance(node, Mapping)
        node = node[key]
    assert isinstance(node, jax.Array)
    return node.dtype.kind


def _assert_twins(*, model: Model, integer: UserParams, floating: UserParams) -> None:
    for left, right in zip(
        _values(model=model, params=integer),
        _values(model=model, params=floating),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def _ces_model() -> Model:
    return _model(koopmans_aggregator=CESAggregator(), certainty_equivalent=PowerMean())


# Each library-owned float coefficient: the model, then the params with the integer.
_LIBRARY_FLOATS = {
    "discount_factor": (
        _model,
        lambda value: _alive_params(koopmans={"discount_factor": value}),
        1,
    ),
    "intertemporal_elasticity_of_substitution": (
        _ces_model,
        lambda value: _alive_params(
            koopmans={"intertemporal_elasticity_of_substitution": value},
            certainty_equivalent={"risk_aversion": 2.0},
        ),
        2,
    ),
    "risk_aversion": (
        _ces_model,
        lambda value: _alive_params(
            koopmans={"intertemporal_elasticity_of_substitution": 0.5},
            certainty_equivalent={"risk_aversion": value},
        ),
        2,
    ),
    "taste_shock_scale": (
        lambda: _model(taste_shocks=ExtremeValueTasteShocks()),
        lambda value: _alive_params(taste_shocks={"scale": value}),
        1,
    ),
    "runtime_process_law": (
        lambda: _model(
            process=NormalIIDProcess(n_points=3, gauss_hermite=True, mu=None, sigma=1.0)
        ),
        lambda value: _alive_params(shock={"mu": value}),
        0,
    ),
}


@pytest.mark.parametrize("name", sorted(_LIBRARY_FLOATS))
def test_library_float_coefficient_accepts_an_integer(name: str) -> None:
    build, params, value = _LIBRARY_FLOATS[name]
    _assert_twins(model=build(), integer=params(value), floating=params(float(value)))


@pytest.mark.parametrize(
    "utility",
    [_utility_with_bonus, _utility_with_string_annotation],
    ids=["jaxtyping", "string"],
)
def test_float_annotated_callback_parameter_accepts_an_integer(
    utility: Callable[..., FloatND],
) -> None:
    model = _model(utility=utility)
    _assert_twins(
        model=model,
        integer=_alive_params(utility={"bonus": 3}),
        floating=_alive_params(utility={"bonus": 3.0}),
    )


def test_float_alias_annotation_accepts_an_integer() -> None:
    model = _model(utility=_utility_with_rate)
    _assert_twins(
        model=model,
        integer=_alive_params(utility={"rate": 3}),
        floating=_alive_params(utility={"rate": 3.0}),
    )


def test_float_annotated_edge_parameter_accepts_an_integer() -> None:
    model = _mortal_model()
    canonical = model._canonicalize_params(
        {"discount_factor": 0.95, "edges": {"working": {"survival_probability": 1}}}
    )
    assert _kind(canonical, "edges", "working", "working__survival_probability") == "f"
    assert _kind(canonical, "edges", "working", "dead__survival_probability") == "f"
    _assert_twins(
        model=model,
        integer={"discount_factor": 0.95, "survival_probability": 1},
        floating={"discount_factor": 0.95, "survival_probability": 1.0},
    )


def _flow(*, wage: FloatND) -> FloatND:
    return wage


def _terminal() -> FloatND:
    return jnp.asarray(8.0)


@categorical(ordered=False)
class _WorkRegimes:
    work: ScalarInt
    done: ScalarInt


def _temporal_model() -> Model:
    return Model(
        n_periods=3,
        regimes={
            "work": Regime(
                functions={"utility": lcm.time_varying_params("wage")(_flow)}
            ),
            "done": Regime(functions={"utility": _terminal}),
        },
        regime_id_class=_WorkRegimes,
        initial_nodes=InitialNodes(by_period={0: "work"}),
        edges={
            "work": {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(1,))}
        },
        fixed_params={"discount_factor": 0.5},
    )


def test_temporal_float_parameter_accepts_integers() -> None:
    model = _temporal_model()
    labels = {"periods": (1, 0)}
    _assert_twins(
        model=model,
        integer={"wage": lcm.TimeVarying(values=jnp.array([2, 1]), **labels)},
        floating={"wage": lcm.TimeVarying(values=jnp.array([2.0, 1.0]), **labels)},
    )


def float_first(*, a_float: FloatND, z_int: ScalarInt) -> FloatND:
    # Trace-time dtype metadata only, never a traced value.
    assert a_float.dtype.kind == "f"
    assert z_int.dtype.kind in "iu"
    return a_float * jnp.asarray([3.0, 11.0])[z_int]


def int_first(*, a_int: ScalarInt, z_float: FloatND) -> FloatND:
    assert a_int.dtype.kind in "iu"
    assert z_float.dtype.kind == "f"
    return z_float * jnp.asarray([3.0, 11.0])[a_int]


def two_floats(*, a: FloatND, b: FloatND) -> FloatND:
    return a + b


def _terminal_model(
    *,
    utility: Callable[..., FloatND] | Phased,
    fixed_params: UserParams | None = None,
    shocks: Mapping[str, ShockGroup] | None = None,
) -> Model:
    groups = shocks or {}
    return Model(
        n_periods=1,
        regimes={
            "terminal": Regime(functions={"utility": utility}, shocks=tuple(groups))
        },
        regime_id_class=_Terminal,
        initial_nodes=InitialNodes(by_period={0: "terminal"}),
        edges={},
        fixed_params=fixed_params or {},
        shocks=groups,
        # The callables assert on trace-time dtypes; pytest rewrites those asserts
        # into references a durable identity cannot fingerprint.
        durable_identity=False,
    )


@pytest.mark.parametrize(
    ("utility", "float_name", "int_name"),
    [(float_first, "a_float", "z_int"), (int_first, "z_float", "a_int")],
)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("array", [False, True])
def test_one_object_feeds_a_float_and_an_integer_slot(
    *,
    utility: Callable[..., FloatND],
    float_name: str,
    int_name: str,
    fixed: bool,
    array: bool,
) -> None:
    shared = np.asarray(1, dtype=np.int32) if array else 1
    params = {float_name: shared, int_name: shared}
    model = _terminal_model(utility=utility, fixed_params=params if fixed else None)
    result = model.solve(params={} if fixed else params, log_level="off")
    np.testing.assert_array_equal(np.asarray(result.values[0]["terminal"]), 11.0)


def test_one_object_in_two_float_slots_is_one_canonical_leaf() -> None:
    shared = np.asarray([1, 2, 3], dtype=np.int32)
    model = _terminal_model(utility=two_floats)
    canonical = model._canonicalize_params({"a": shared, "b": shared})
    a = canonical["terminal"]["utility__a"]
    b = canonical["terminal"]["utility__b"]
    assert a is b
    assert _kind(canonical, "terminal", "utility__a") == "f"


def _masked(*, weight: FloatND, mask: lcm.typing.BoolND, index: ScalarInt) -> FloatND:
    assert mask.dtype.kind == "b"
    assert index.dtype.kind in "iu"
    return jnp.where(mask, weight, 0.0).sum() + index


def test_integer_and_boolean_slots_keep_their_kind() -> None:
    model = _terminal_model(utility=_masked)
    canonical = model._canonicalize_params(
        {"weight": 2, "mask": np.asarray([True, False]), "index": 1}
    )
    assert _kind(canonical, "terminal", "utility__weight") == "f"
    assert _kind(canonical, "terminal", "utility__mask") == "b"
    assert _kind(canonical, "terminal", "utility__index") == "i"


def _indexed_mean(*, selector: ScalarInt) -> FloatND:
    return jnp.asarray([0.0, 4.0])[selector]


def _shock_utility(*, innovation: FloatND) -> FloatND:
    return innovation


@pytest.mark.parametrize("fixed", [False, True])
def test_integer_shock_callback_input_keeps_its_kind(*, fixed: bool) -> None:
    group = ShockGroup(
        shocks={"innovation": Normal(mu=_indexed_mean, sigma=0.0)},
        nodes=Product(innovation=GaussHermite(1)),
    )
    params = {"shock_group": {"selector": 1}}
    model = _terminal_model(
        utility=_shock_utility,
        shocks={"shock_group": group},
        fixed_params=params if fixed else None,
    )
    result = model.solve(params={} if fixed else params, log_level="off")
    np.testing.assert_array_equal(np.asarray(result.values[0]["terminal"]), 4.0)


def _scaled_mean(*, level: FloatND) -> FloatND:
    assert level.dtype.kind == "f"
    return level * 2.0


def test_float_shock_callback_input_accepts_an_integer() -> None:
    group = ShockGroup(
        shocks={"innovation": Normal(mu=_scaled_mean, sigma=0.0)},
        nodes=Product(innovation=GaussHermite(1)),
    )
    model = _terminal_model(utility=_shock_utility, shocks={"shock_group": group})
    for level in (2, 2.0):
        result = model.solve(params={"shock_group": {"level": level}}, log_level="off")
        np.testing.assert_array_equal(np.asarray(result.values[0]["terminal"]), 4.0)


def _float_reader(*, gain: FloatND) -> FloatND:
    return gain * 1.0


def _int_reader(*, gain: ScalarInt) -> FloatND:
    return gain * 1.0


@pytest.mark.parametrize(
    ("solve", "simulate"),
    [(_float_reader, _int_reader), (_int_reader, _float_reader)],
    ids=["float-then-int", "int-then-float"],
)
def test_conflicting_slot_roles_are_refused_in_either_order(
    *, solve: Callable[..., FloatND], simulate: Callable[..., FloatND]
) -> None:
    with pytest.raises(ModelInitializationError, match="gain"):
        _terminal_model(utility=Phased(solve=solve, simulate=simulate))


@categorical(ordered=False)
class _EdgeRegimes:
    source: ScalarInt
    sink: ScalarInt
    unused: ScalarInt


def _flat_utility() -> FloatND:
    return jnp.asarray(0.0)


def _certain() -> FloatND:
    return jnp.asarray(1.0)


def _scaled_zero(*, hazard: float) -> FloatND:
    return jnp.asarray(0.0) * hazard


def test_fixed_float_law_argument_is_a_float_when_evaluated_at_build() -> None:
    """A fixed law is evaluated at build to prove a zero edge; a float argument
    fixed to an integer past the int32 range is read as a float there too."""
    model = Model(
        regimes={
            name: Regime(functions={"utility": _flat_utility})
            for name in ("source", "sink", "unused")
        },
        ages=AgeGrid(start=25, inclusive_stop=26, step="Y"),
        regime_id_class=_EdgeRegimes,
        edges={
            "source": lcm.Transition(
                law={
                    "sink": lcm.StochasticTransition(func=_certain),
                    "unused": lcm.StochasticTransition(func=_scaled_zero),
                }
            ),
            "unused": {"sink": 25},
        },
        initial_nodes=InitialNodes(by_age={25: "source"}),
        fixed_params={"hazard": 2**31},
    )

    assert model.graph.pruned_edges["simulate"] == {
        (25, "source", "unused"): "fixed_zero_probability"
    }
