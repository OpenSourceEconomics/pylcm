"""What a replay may publish when the winning post-decision is a domain endpoint.

Both outer searches recover the outer action by inverting the declared map at
the post-decision the solve retained, then re-evaluate the map at that action.
The stock the action actually reaches decides whether the candidate may be
published, and the rule is the same for either search:

- a target at a declared endpoint is represented only when the recovered
  action reproduces it exactly — an inward image is a different stock, and the
  declared domain has no room the other side of it to absorb the difference;
- an interior target only has to land inside the declared domain.

Recovering the action is exact whenever the stock and the target share a sign,
because the subtraction then stays inside the operands' own binades. It is the
sign-crossing case that rounds: the action lands in a coarser binade than the
image, so the rounding it picks up is not undone when the map is re-evaluated.
Each endpoint below is therefore approached from the far side of zero, and the
two cases are mirror images of one another.

A nonnegative domain reached through a direct law never crosses, which is why
the accept-side controls carry as much weight as the refusals: a replay that
dropped every candidate would satisfy the refusals on its own.
"""

import logging
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import _lcm.simulation.simulate as simulation_module
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy, OuterPolicyBank
from _lcm.egm.outer_inversion import (
    DeclaredOuterInverse,
    invert_declared_outer_target,
)
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import Regime
from lcm.exceptions import InvalidSimulationInputError
from lcm.solvers import AdaptiveOuterMesh
from lcm.typing import BoolND, FloatND
from tests.test_models import n_nbegm_toy as toy

# The declared lower endpoint, approached from a positive realized stock.
_LOWER_CASE = {
    "domain": (-1.0, 10.0),
    "target": -1.0,
    "nodes": (-1.0, 4.5, 10.0),
    "winner": 0,
    "approach_from": 1.0,
    "unsafe_investment": -20.0,
    "reproduced_endpoint": 10.0,
}

# The declared upper endpoint, approached from a negative realized stock.
_UPPER_CASE = {
    "domain": (-10.0, 1.0),
    "target": 1.0,
    "nodes": (-10.0, -4.5, 1.0),
    "winner": 2,
    "approach_from": -1.0,
    "unsafe_investment": 20.0,
    "reproduced_endpoint": -10.0,
}

_ENDPOINT_CASES = [_LOWER_CASE, _UPPER_CASE]

# The direct-law control domain: recovering the action never crosses zero.
_NONNEGATIVE_DOMAIN = (0.0, 20.0)

# A domain lying wholly on the far side of zero from the subject, close
# enough to it that the action recovered for either endpoint lands in a
# coarser binade than the endpoint itself. Both endpoints are then missed,
# so no projection is publishable and the read has nothing left to emit.
_DOUBLE_MISS_DOMAIN: tuple[float, float] = (-1.5, -1.0)
_DOUBLE_MISS_NODES: tuple[float, ...] = (-1.5, -1.25, -1.0)
_DOUBLE_MISS_APPROACH_FROM = 1.0
_DOUBLE_MISS_INVESTMENT = -20.0

_PARAMS = {"discount_factor": 0.95}
_SEED = 42
_N_PERIODS = 3

_MESH = AdaptiveOuterMesh(
    initial_grid=toy.OUTER_GRID,
    max_nodes=513,
    max_refinement_rounds=10,
    value_atol=1e-4,
    value_rtol=1e-4,
    golden_iterations=40,
)

_ROUTES = {"finite": None, "adaptive": _MESH}

_INITIAL = {
    "wealth": jnp.array([4.3, 11.7, 19.9, 8.1]),
    "illiquid": jnp.array([1.37, 6.6, 13.2, 17.5]),
    "age": jnp.full(4, 20.0),
    "regime_id": jnp.zeros(4, dtype=jnp.int32),
}


class _StubRegime(Regime):
    """Engine regime carrying only the fields the policy read reaches."""

    def __init__(self, *, simulation: object) -> None:
        object.__setattr__(self, "simulation", simulation)


def _working_dtype() -> np.dtype:
    """The float format the suite is running at."""
    return np.dtype(jnp.zeros(1).dtype)


def _walk_away_from_zero(*, base: float, target: float, want_miss: bool) -> float:
    """Return a stock near `base` that does or does not reach `target` exactly.

    Steps through representable neighbours of `base`, away from zero, until the
    round trip `stock + (target - stock)` either misses `target` or lands on
    it. Stepping happens in the format the suite is running at, so each witness
    is genuine at float32 and at float64 alike rather than pinned to one.
    """
    dtype = _working_dtype()
    scalar = dtype.type
    stock = np.asarray(base, dtype=dtype)
    away = np.asarray(np.inf if base > 0 else -np.inf, dtype=dtype)
    for _ in range(128):
        missed = stock + (scalar(target) - stock) != scalar(target)
        if missed == want_miss:
            return float(stock)
        stock = np.nextafter(stock, away)
    wanted = "misses" if want_miss else "reaches"
    raise AssertionError(
        f"no representable stock near {base} {wanted} {target} at {dtype.name}"
    )


def _double_miss_stock() -> float:
    """A stock whose recovered action misses both double-miss endpoints.

    Walks representable neighbours away from zero until one stock fails to
    reproduce either endpoint, so the witness is genuine in whichever format
    the suite is running rather than a constant that only holds at one.
    """
    dtype = _working_dtype()
    scalar = dtype.type
    endpoints = (_DOUBLE_MISS_NODES[0], _DOUBLE_MISS_NODES[-1])
    stock = np.asarray(_DOUBLE_MISS_APPROACH_FROM, dtype=dtype)
    away = np.asarray(np.inf, dtype=dtype)
    for _ in range(4096):
        if all(stock + (scalar(e) - stock) != scalar(e) for e in endpoints):
            return float(stock)
        stock = np.nextafter(stock, away)
    raise AssertionError(
        f"no representable stock near {_DOUBLE_MISS_APPROACH_FROM} misses both "
        f"{endpoints} at {dtype.name}"
    )


def _payload(*, domain: tuple[float, float], nodes: tuple[float, ...], winner: int):
    """A replayable continuous-outer payload whose `winner` node ranks highest.

    Every branch publishes the same liquid row, so the outer ranking is set by
    the branch values alone and the refined optimum settles on the winning node.
    """
    liquid = jnp.array([1.0, 2.0, 3.0])
    values = jnp.asarray(
        [
            [10.0, 10.0, 10.0] if i == winner else [0.0, 0.0, 0.0]
            for i in range(len(nodes))
        ]
    )
    stacked = jnp.stack([liquid] * len(nodes))
    return NestedEGMSimPolicy(
        keeper=EGMSimPolicy(
            endog_grid=liquid,
            policy=jnp.array([0.5, 0.5, 0.5]),
            value=jnp.array([-1.0, -1.0, -1.0]),
            marginal_utility=jnp.array([1.0, 1.0, 1.0]),
        ),
        adjuster=OuterPolicyBank(
            outer_nodes=jnp.asarray(nodes),
            policies=EGMSimPolicy(
                endog_grid=stacked,
                policy=jnp.stack([jnp.array([0.4, 0.4, 0.4])] * len(nodes)),
                value=values,
                marginal_utility=jnp.stack([jnp.array([1.0, 1.0, 1.0])] * len(nodes)),
            ),
        ),
        outer_action_name="investment",
        outer_state_name="illiquid",
        outer_post_decision_name="new_illiquid",
        inner_action_name="consumption",
        liquid_state_name="wealth",
        outer_no_adjustment_name=None,
        resources_target_name="resources",
        savings_lower_bound=0.0,
        golden_iterations=4,
        replay_capability=OuterReplayCapability(
            inverse=DeclaredOuterInverse(
                coefficient=Fraction(1), low=domain[0], high=domain[1]
            ),
            undeclared_functions=(),
            unbindable_functions=(),
            unavailable_keeper_states=(),
            unaddressable_passive_states=(),
            unaddressable_discrete_actions=(),
        ),
    )


def _new_illiquid(illiquid, investment):
    """`s' = Z + Iz`: the outer post-decision the replay inverts."""
    return illiquid + investment


def _resources(wealth):
    """Liquid resources the recovered pair is checked against."""
    return wealth + 20.0


def _falls_back(*, domain, nodes, winner: int, stock: float) -> bool:
    """Replay one subject at `stock` and report whether it fell back."""
    _, fallback, _ = simulation_module._read_nested_policy(
        payload=_payload(domain=domain, nodes=nodes, winner=winner),
        optimal_actions=MappingProxyType(
            {"consumption": jnp.array([1.0]), "investment": jnp.array([0.0])}
        ),
        regime=_StubRegime(
            simulation=SimpleNamespace(
                grids={},
                functions={"new_illiquid": _new_illiquid, "resources": _resources},
                constraints={},
                compute_regime_transition_probs=None,
                age_specialized_function_names=frozenset(),
            )
        ),
        states=MappingProxyType(
            {"wealth": jnp.array([2.0]), "illiquid": jnp.array([stock])}
        ),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
    )
    return bool(fallback[0])


def _admissible(*, domain, target: float, stock: float) -> bool:
    """Report whether finite replay may publish the candidate reaching `target`."""
    at_zero = jnp.asarray([stock])
    inversion = invert_declared_outer_target(
        inverse=DeclaredOuterInverse(
            coefficient=Fraction(1), low=domain[0], high=domain[1]
        ),
        target=jnp.asarray([target]),
        at_zero=at_zero,
        forward=lambda action: at_zero + action,
    )
    return bool(inversion.admissible[0])


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_adaptive_replay_falls_back_when_the_action_misses_a_declared_endpoint(
    case: dict,
) -> None:
    """A recovered action landing beside a declared endpoint is not published."""
    assert _falls_back(
        domain=case["domain"],
        nodes=case["nodes"],
        winner=case["winner"],
        stock=_walk_away_from_zero(
            base=case["approach_from"], target=case["target"], want_miss=True
        ),
    )


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_adaptive_replay_publishes_when_the_action_reaches_a_declared_endpoint(
    case: dict,
) -> None:
    """A recovered action reproducing a declared endpoint exactly is published."""
    assert not _falls_back(
        domain=case["domain"],
        nodes=case["nodes"],
        winner=case["winner"],
        stock=_walk_away_from_zero(
            base=case["approach_from"], target=case["target"], want_miss=False
        ),
    )


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_adaptive_replay_publishes_an_interior_target_inside_the_domain(
    case: dict,
) -> None:
    """An interior target only has to land inside the declared domain."""
    assert not _falls_back(
        domain=case["domain"],
        nodes=case["nodes"],
        winner=1,
        stock=_walk_away_from_zero(
            base=case["approach_from"], target=case["target"], want_miss=True
        ),
    )


@pytest.mark.parametrize("endpoint", [0, 1], ids=["lower", "upper"])
def test_adaptive_replay_publishes_a_nonnegative_direct_law_endpoint(
    endpoint: int,
) -> None:
    """A nonnegative domain reached through a direct law publishes its endpoints."""
    assert not _falls_back(
        domain=_NONNEGATIVE_DOMAIN,
        nodes=(0.0, 10.0, 20.0),
        winner=0 if endpoint == 0 else 2,
        stock=1.0,
    )


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_finite_replay_drops_a_candidate_that_misses_a_declared_endpoint(
    case: dict,
) -> None:
    """Finite replay applies the same endpoint rule the adaptive reader does."""
    assert not _admissible(
        domain=case["domain"],
        target=case["target"],
        stock=_walk_away_from_zero(
            base=case["approach_from"], target=case["target"], want_miss=True
        ),
    )


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_finite_replay_admits_a_candidate_that_reaches_a_declared_endpoint(
    case: dict,
) -> None:
    """A recovered action reproducing the endpoint exactly stays admissible."""
    assert _admissible(
        domain=case["domain"],
        target=case["target"],
        stock=_walk_away_from_zero(
            base=case["approach_from"], target=case["target"], want_miss=False
        ),
    )


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_finite_replay_admits_an_interior_target_inside_the_domain(case: dict) -> None:
    """An interior candidate is admitted on containment alone."""
    assert _admissible(
        domain=case["domain"],
        target=case["nodes"][1],
        stock=_walk_away_from_zero(
            base=case["approach_from"], target=case["target"], want_miss=True
        ),
    )


@pytest.mark.parametrize("endpoint", [0, 1], ids=["lower", "upper"])
def test_finite_replay_admits_a_nonnegative_direct_law_endpoint(endpoint: int) -> None:
    """The nonnegative direct-law control is admitted at either endpoint."""
    assert _admissible(
        domain=_NONNEGATIVE_DOMAIN,
        target=_NONNEGATIVE_DOMAIN[endpoint],
        stock=1.0,
    )


def _build(route: str):
    """The smooth two-asset toy under the requested outer search."""
    return toy.build_model(
        variant="n_nbegm", n_periods=_N_PERIODS, outer_search=_ROUTES[route]
    )


def _simulate(model, *, period_to_regime_to_V_arr=None, policies=None):
    """Simulate the toy from the shared initial conditions."""
    return model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        policies=policies,
        log_level="debug",
        seed=_SEED,
    ).to_dataframe()


@pytest.fixture(scope="module", params=["finite", "adaptive"])
def route(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def solved(route: str):
    """The model plus the `(values, policies)` pair its own `solve()` returned."""
    model = _build(route)
    values, policies = model.solve(
        params=_PARAMS, log_level="debug", return_simulation_policy=True
    )
    return model, values, policies


def test_both_workflows_publish_the_same_endpoint_admissions(solved) -> None:
    """Automatic and split solve-and-simulate admit the same candidates."""
    model, values, policies = solved

    assert_frame_equal(
        _simulate(model, period_to_regime_to_V_arr=values, policies=policies),
        _simulate(model, period_to_regime_to_V_arr=None),
    )


def test_a_simulated_nonnegative_model_admits_every_realized_subject(
    monkeypatch,
) -> None:
    """The endpoint rule admits the subjects a nonnegative toy actually reaches.

    The refusals above are only half the contract: a replay that dropped every
    candidate would satisfy them and publish the action-grid winner throughout.
    Recording what the shared predicate returns during an ordinary simulation
    of the toy — whose durable domain is nonnegative and whose law is direct —
    pins the other half, that admission is the normal outcome.
    """
    admissions = []
    admissible = simulation_module.outer_candidate_is_admissible

    def record(*, image, target, low, high):
        verdict = admissible(image=image, target=target, low=low, high=high)
        admissions.append(verdict)
        return verdict

    monkeypatch.setattr(simulation_module, "outer_candidate_is_admissible", record)
    _simulate(_build("adaptive"), period_to_regime_to_V_arr=None)

    verdicts = [bool(jnp.all(verdict)) for verdict in admissions]
    # An empty recording fails this too: `set()` is not `{True}`.
    assert set(verdicts) == {True}


def _q_and_f(**kwargs: object) -> tuple[FloatND, BoolND]:
    """Score a pair by how little it moves the outer stock, and call it feasible.

    Ranking on `-|investment|` puts the endpoint nearest the subject's stock
    above the far one. The near endpoint is the one a sign-crossing action
    misses, so a fallback that admits an endpoint on containment alone will
    prefer exactly the projection it should have refused.
    """
    investment = jnp.asarray(kwargs["investment"])
    return -jnp.abs(investment), jnp.ones(investment.shape, dtype=bool)


def _regime() -> _StubRegime:
    """Engine stub carrying the simulate-phase pieces a nested read reaches."""
    return _StubRegime(
        simulation=SimpleNamespace(
            grids={},
            functions={"new_illiquid": _new_illiquid, "resources": _resources},
            constraints={},
            compute_regime_transition_probs=None,
            age_specialized_function_names=frozenset(),
            Q_and_F=MappingProxyType({0: _q_and_f}),
        )
    )


def _image_of(*, stock: float, investment: float) -> float:
    """The stock `investment` actually reaches, in the format the suite runs at."""
    scalar = _working_dtype().type
    return float(scalar(scalar(stock) + scalar(investment)))


def _baseline(*, domain, nodes, winner: int, stock: float, investment: float):
    """The fallback baseline for one subject holding `stock`.

    `investment` is the raw simulation-grid winner. Passing one that leaves the
    published mesh is what sends the baseline down the endpoint-projection
    branch, which is the branch under test.
    """
    states = MappingProxyType(
        {"wealth": jnp.array([2.0]), "illiquid": jnp.array([stock])}
    )
    return simulation_module._nested_grid_baseline(
        payload=_payload(domain=domain, nodes=nodes, winner=winner),
        grid_actions=MappingProxyType(
            {"consumption": jnp.array([1.0]), "investment": jnp.array([investment])}
        ),
        regime=_regime(),
        states=states,
        canonical_states=states,
        action_names=("consumption", "investment"),
        next_regime_to_V_arr=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(40.0),
    )


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_projected_fallback_reaches_the_endpoint_it_was_projected_onto(
    case: dict,
) -> None:
    """A fallback projection is published only where it lands on its endpoint.

    The endpoint nearest the subject is missed by a sign-crossing action, and
    ranks above the far endpoint, so publishing the near projection is what an
    admission rule that asks only for domain containment would do.
    """
    stock = _walk_away_from_zero(
        base=case["approach_from"], target=case["target"], want_miss=True
    )
    actions, _, admissible = _baseline(
        domain=case["domain"],
        nodes=case["nodes"],
        winner=case["winner"],
        stock=stock,
        investment=case["unsafe_investment"],
    )

    assert bool(admissible[0])
    assert _image_of(
        stock=stock, investment=float(actions["investment"][0])
    ) == pytest.approx(case["reproduced_endpoint"], abs=0.0)


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_projected_fallback_publishes_the_endpoint_it_reaches_exactly(
    case: dict,
) -> None:
    """A projection whose action reproduces its endpoint stays publishable."""
    stock = _walk_away_from_zero(
        base=case["approach_from"], target=case["target"], want_miss=False
    )
    actions, _, admissible = _baseline(
        domain=case["domain"],
        nodes=case["nodes"],
        winner=case["winner"],
        stock=stock,
        investment=case["unsafe_investment"],
    )

    assert bool(admissible[0])
    assert _image_of(
        stock=stock, investment=float(actions["investment"][0])
    ) == pytest.approx(case["target"], abs=0.0)


@pytest.mark.parametrize("endpoint", [0, 1], ids=["lower", "upper"])
def test_projected_fallback_publishes_a_nonnegative_direct_law_endpoint(
    endpoint: int,
) -> None:
    """A nonnegative domain projects onto either endpoint without refusal."""
    nodes = (0.0, 10.0, 20.0)
    actions, _, admissible = _baseline(
        domain=_NONNEGATIVE_DOMAIN,
        nodes=nodes,
        winner=0 if endpoint == 0 else 2,
        stock=1.0,
        investment=-40.0,
    )

    assert bool(admissible[0])
    reached = _image_of(stock=1.0, investment=float(actions["investment"][0]))
    assert reached in {nodes[0], nodes[-1]}


def test_a_grid_pair_inside_the_published_mesh_is_kept_unprojected() -> None:
    """A raw grid pair the solver can serve is the baseline, unchanged."""
    actions, _, admissible = _baseline(
        domain=_NONNEGATIVE_DOMAIN,
        nodes=(0.0, 10.0, 20.0),
        winner=1,
        stock=1.0,
        investment=4.0,
    )

    assert bool(admissible[0])
    assert float(actions["investment"][0]) == 4.0


def test_a_fallback_missing_every_endpoint_is_refused() -> None:
    """No endpoint projection is publishable when the action misses them all."""
    _, _, admissible = _baseline(
        domain=_DOUBLE_MISS_DOMAIN,
        nodes=_DOUBLE_MISS_NODES,
        winner=0,
        stock=_double_miss_stock(),
        investment=_DOUBLE_MISS_INVESTMENT,
    )

    assert not bool(admissible[0])


def test_a_subject_with_no_publishable_pair_fails_loud() -> None:
    """Simulation raises rather than emit a pair no admission rule allows."""
    stock = _double_miss_stock()
    states = MappingProxyType(
        {"wealth": jnp.array([2.0]), "illiquid": jnp.array([stock])}
    )
    with pytest.raises(InvalidSimulationInputError, match="neither the nested policy"):
        simulation_module._replace_continuous_action_with_policy_read(
            optimal_actions=MappingProxyType(
                {
                    "consumption": jnp.array([1.0]),
                    "investment": jnp.array([_DOUBLE_MISS_INVESTMENT]),
                }
            ),
            regime=_regime(),
            sim_policy=_payload(
                domain=_DOUBLE_MISS_DOMAIN, nodes=_DOUBLE_MISS_NODES, winner=0
            ),
            states=states,
            flat_params=MappingProxyType({}),
            period=0,
            age=jnp.asarray(40.0),
            canonical_states=states,
            action_names=("consumption", "investment"),
            next_regime_to_V_arr=MappingProxyType({}),
            grid_values=jnp.array([-20.0]),
            in_regime=jnp.array([True]),
            logger=logging.getLogger(__name__),
        )


@pytest.mark.parametrize("case", _ENDPOINT_CASES, ids=["lower", "upper"])
def test_containment_alone_would_publish_a_missed_endpoint(
    case: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weakening the endpoint verdict to containment re-admits the missed stock.

    The refusals above only mean something if the rule that replaces them can
    be shown to decide differently. Substituting plain domain containment for
    the shared predicate publishes the near projection again, which is the
    behaviour the endpoint verdict exists to prevent.
    """

    def containment_only(*, image, target, low, high):  # noqa: ARG001
        return (image >= low) & (image <= high)

    monkeypatch.setattr(
        simulation_module, "outer_candidate_is_admissible", containment_only
    )
    stock = _walk_away_from_zero(
        base=case["approach_from"], target=case["target"], want_miss=True
    )
    actions, _, admissible = _baseline(
        domain=case["domain"],
        nodes=case["nodes"],
        winner=case["winner"],
        stock=stock,
        investment=case["unsafe_investment"],
    )

    assert bool(admissible[0])
    reached = _image_of(stock=stock, investment=float(actions["investment"][0]))
    assert reached != case["reproduced_endpoint"]
    assert reached != case["target"]
