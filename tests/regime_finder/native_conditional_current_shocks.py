"""Actual-Model conditional-shock checks, isolated from metadata-only test RSS."""

from copy import deepcopy
import pytest

import jax.numpy as jnp

from lcm.exits import Unilateral
from lcm import DiscreteGrid, Exit, ExitTarget, InitialNodes, Model, Regime, categorical
from lcm.shocks import Discrete, GaussHermite, Normal, Product, ShockGroup
from lcm.typing import DiscreteState, FloatND, ScalarInt

from _lcm_regime_finder import analyze_production, verify_production_result
from _lcm_regime_finder.model_snapshot import read_model
from _lcm_regime_finder.semantic_check import check_semantics


@categorical(ordered=True)
class Binary:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=True)
class Regimes:
    source: ScalarInt
    target: ScalarInt


def offer_probabilities(lagged_school: DiscreteState) -> FloatND:
    probability = (1.0 + lagged_school) / 3.0
    return jnp.stack((1.0 - probability, probability), axis=-1)


def same_history(lagged_school: DiscreteState) -> DiscreteState:
    return lagged_school


def conditional_location(lagged_school: DiscreteState) -> FloatND:
    return 1.0 + lagged_school


def stay() -> float:
    return 0.5


def payoff(offer: DiscreteState, choice: DiscreteState) -> FloatND:
    return offer + choice * (2.0 * offer - 1.0)


def model(*, shock_group=None):
    return Model(
        n_periods=1,
        regime_id_class=Regimes,
        initial_nodes=InitialNodes(by_period={0: ("source", "target")}),
        edges={},
        regimes={
            "source": Regime(
                states={"lagged_school": DiscreteGrid(Binary)},
                functions={"utility": stay},
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="target",
                            projection={"lagged_school": same_history},
                        )
                    },
                    rule=Unilateral(),
                    observes=("offer",),
                ),
            ),
            "target": Regime(
                states={"lagged_school": DiscreteGrid(Binary)},
                actions={"choice": DiscreteGrid(Binary)},
                functions={"utility": payoff},
                shocks=("offers",),
            ),
        },
        shocks={
            "offers": shock_group
            or ShockGroup(
                shocks={"offer": Discrete(categories=Binary, probs=offer_probabilities)}
            ),
            "unregistered": ShockGroup(
                shocks={"other": Discrete(categories=Binary, probs=offer_probabilities)}
            ),
        },
        enable_jit=False,
    )


def test_capture_to_recommendation_retains_probability_only_history() -> None:
    captured = read_model(model())
    report = analyze_production(captured)
    # Literal observed-exit oracle: max(stay=.5, target=0/2) in
    # the two offer outcomes. Only their probabilities use this state.
    probabilities = offer_probabilities(jnp.asarray([0, 1], dtype=jnp.int32))
    values = probabilities[:, 0] * 0.5 + probabilities[:, 1] * 2.0
    assert float(values[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(values[1]) == pytest.approx(1.5, abs=1e-6)
    assert (verify_production_result(captured, report)["status"]) == ("PASS")
    cells = report["analysis"]["declarations"]["cells"]
    assert ({cell["regime"] for cell in cells}) == ({"source", "target"})
    for cell in cells:
        options = report["analysis"]["interfaces"][cell["id"]]["options"]
        assert all("lagged_school" in option["states"] for option in options)


def test_continuous_law_inputs_and_shared_callable_references_remain_live() -> None:
    group = ShockGroup(
        shocks={"offer": Normal(mu=conditional_location, sigma=conditional_location)},
        nodes=Product(offer=GaussHermite(n=3)),
    )
    captured = read_model(model(shock_group=group))
    law = captured["shocks"]["offers"]["fields"]["shocks"]["offer"]["fields"]
    # The compact capture format stores reused callbacks by identity.
    # Exercise that spelling as well as the expanded native mu record.
    law["sigma"] = {
        "kind": "callable-reference",
        **{
            key: law["sigma"][key]
            for key in ("capture_ref", "identity_sha256", "capture_graph_sha256")
        },
    }
    report = analyze_production(captured)
    assert (verify_production_result(captured, report)["status"]) == ("PASS")
    for cell in report["analysis"]["declarations"]["cells"]:
        assert all(
            "lagged_school" in option["states"]
            for option in report["analysis"]["interfaces"][cell["id"]]["options"]
        )


def test_unregistered_conditional_group_does_not_keep_an_unused_state() -> None:
    captured = read_model(model())
    # Change only the captured law to a constant. Reconstructing a Model
    # would reject its now-unused history before the checker can inspect it.
    captured = deepcopy(captured)
    captured["shocks"]["offers"]["fields"]["shocks"]["offer"]["fields"]["probs"] = [
        0.5,
        0.5,
    ]
    report = analyze_production(captured)
    assert (verify_production_result(captured, report)["status"]) == ("PASS")
    targets = [
        c
        for c in report["analysis"]["declarations"]["cells"]
        if c["regime"] == "target"
    ]
    assert targets
    for cell in targets:
        assert all(
            "lagged_school" not in option["states"]
            for option in report["analysis"]["interfaces"][cell["id"]]["options"]
        )


def test_independent_source_check_rejects_missing_or_miswired_law(
    *, subtests: pytest.Subtests
) -> None:
    captured = read_model(model())
    report = analyze_production(captured)
    for mutation in ("omit", "rename", "phase", "erase_inputs", "missing_group"):
        analysis = deepcopy(report["analysis"])
        declaration = analysis["declarations"]
        for cell in declaration["cells"]:
            laws = [c for c in cell["consumers"] if c["kind"] == "shock_law"]
            if not laws:
                continue
            law = laws[0]
            if mutation == "omit":
                cell["consumers"].remove(law)
            elif mutation == "rename":
                law["group"] = "unregistered"
            elif mutation == "phase":
                law["phase"] = "both"
            elif mutation == "missing_group":
                del law["group"]
            else:
                callback = declaration["callbacks"][law["callback"]]
                callback["arguments"] = callback["reads"] = []
        with subtests.test(mutation=mutation):
            checked = check_semantics(
                captured, declaration, analysis, check_coverage=False
            )
            assert (checked["status"]) == ("FAIL"), checked


def test_original_shock_probability_mutation_invalidates_saved_recommendation() -> None:
    captured = read_model(model())
    report = analyze_production(captured)
    source = deepcopy(captured)
    source["shocks"]["offers"] = source["shocks"]["unregistered"]
    assert (verify_production_result(source, report)["status"]) == ("FAIL")
