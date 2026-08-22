"""Executable contracts for claims made by the reorganized documentation."""

import importlib
from pathlib import Path

import jax.numpy as jnp
import yaml

from lcm import GridBreakpoint, JointTransition
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    NestedConsumptionSavingsRegime,
)

_ROOT = Path(__file__).parents[1]
_DOCS = _ROOT / "docs"


def test_getting_started_sources_live_with_their_navigation_section() -> None:
    """Getting Started owns its installation and tiny-example source files."""
    config = yaml.safe_load((_DOCS / "myst.yml").read_text())
    getting_started = config["project"]["toc"][1]
    children = [entry["file"] for entry in getting_started["children"]]

    assert children[:2] == [
        "getting_started/installation.md",
        "getting_started/tiny_example.ipynb",
    ]
    assert not (_DOCS / "user_guide" / "installation.md").exists()
    assert not (_DOCS / "user_guide" / "tiny_example.ipynb").exists()


def test_public_constructor_examples_use_supported_shapes() -> None:
    """Reference examples use the public breakpoint and joint-transition fields."""
    grid_breakpoint = GridBreakpoint(value=1.0, owner="right")
    transition_text = (_DOCS / "reference" / "transitions.md").read_text()
    grid_text = (_DOCS / "reference" / "grids_and_processes.md").read_text()

    joint = JointTransition(
        support_size=2,
        support={"wealth": jnp.array([0.0, 1.0])},
        probabilities=lambda: jnp.array([0.5, 0.5]),
        outputs={"wealth": lambda joint_draw: joint_draw["wealth"]},
    )
    declaration = {"target_regime": {"joint_draw": joint}}

    assert grid_breakpoint.owner == "right"
    assert declaration["target_regime"]["joint_draw"] is joint
    assert "joint_transitions={" in transition_text
    assert '"target_regime": {' in transition_text
    assert "GridBreakpoint(value, owner=...)" in grid_text
    assert "ownership=" not in grid_text


def test_specialized_authoring_examples_build_the_declared_regime_types() -> None:
    """The User Guide's specialized examples are backed by executable builders."""
    examples = importlib.import_module("lcm_examples.specialized_consumption_savings")

    one_margin = examples.build_one_margin_model(enable_jit=False)
    nested = examples.build_nested_model(enable_jit=False)

    assert isinstance(one_margin.user_regimes["working"], ConsumptionSavingsRegime)
    assert isinstance(nested.user_regimes["working"], NestedConsumptionSavingsRegime)


def test_curated_egm_examples_instantiate_the_specialized_paths() -> None:
    """Curated EGM pages run their specialized builders as the principal path."""
    iskhakov = (_DOCS / "examples" / "iskhakov_et_al_2017.md").read_text()
    mahler_yum = (_DOCS / "examples" / "mahler_yum_2024.md").read_text()

    assert "model = get_dcegm_model(n_periods=6)" in iskhakov
    assert 'create_mahler_yum_model(implementation="paper")' in mahler_yum
    assert "model = MAHLER_YUM_MODEL" not in mahler_yum

    iskhakov_module = importlib.import_module("lcm_examples.iskhakov_et_al_2017")
    paper_module = importlib.import_module("lcm_examples.mahler_yum_2024.paper")

    dcegm_model = iskhakov_module.get_dcegm_model(n_periods=3)
    paper_model = paper_module.create_mahler_yum_model(
        implementation="paper", enable_jit=False
    )

    assert isinstance(
        dcegm_model.user_regimes["working_life"], ConsumptionSavingsRegime
    )
    assert isinstance(
        paper_model.user_regimes["working"], NestedConsumptionSavingsRegime
    )
    assert isinstance(
        paper_model.user_regimes["retirement"], NestedConsumptionSavingsRegime
    )


def test_specialized_authoring_examples_solve_and_simulate() -> None:
    """Both specialized User Guide declarations run through their public workflow."""
    examples = importlib.import_module("lcm_examples.specialized_consumption_savings")

    for builder, nested in (
        (examples.build_one_margin_model, False),
        (examples.build_nested_model, True),
    ):
        model = builder(enable_jit=False)
        params = examples.example_params()
        solution = model.solve(params=params, log_level="debug")
        result = model.simulate(
            params=params,
            initial_conditions=examples.example_initial_conditions(nested=nested),
            period_to_regime_to_V_arr=solution,
            log_level="debug",
        )
        frame = result.to_dataframe().sort_values(["subject_id", "period"])

        assert result.n_subjects == 2
        assert frame["regime_name"].tolist() == [
            "working",
            "dead",
            "working",
            "dead",
        ]
