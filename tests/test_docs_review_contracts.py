"""Executable contracts for claims made by the reorganized documentation."""

import importlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from numpy.testing import assert_array_almost_equal as aaae

from lcm import GridBreakpoint, JointTransition
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    NestedConsumptionSavingsRegime,
)
from lcm.solvers import NBEGM
from tests.conftest import DECIMAL_PRECISION

_ROOT = Path(__file__).parents[1]
_DOCS = _ROOT / "docs"


def test_getting_started_sources_live_with_their_navigation_section() -> None:
    """Getting Started owns its installation and tiny-example source files."""
    config = yaml.safe_load((_DOCS / "myst.yml").read_text(encoding="utf-8"))
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
    transition_text = (_DOCS / "reference" / "transitions.md").read_text(
        encoding="utf-8"
    )
    grid_text = (_DOCS / "reference" / "grids_and_processes.md").read_text(
        encoding="utf-8"
    )

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


def test_kinked_tax_example_runs_nbegm_on_both_sides_of_the_bracket() -> None:
    """The small NBEGM example solves and exposes both sides of its tax kink."""
    examples = importlib.import_module("lcm_examples.specialized_consumption_savings")

    model = examples.build_kinked_tax_model(enable_jit=False)
    regime = model.user_regimes["working"]
    params = examples.kinked_tax_params()
    solution = model.solve(params=params, log_level="debug")
    result = model.simulate(
        params=params,
        initial_conditions=examples.kinked_tax_initial_conditions(),
        solution=solution,
        log_level="debug",
    )
    working = result.to_dataframe(additional_targets=["tax"]).query(
        "regime_name == 'working'"
    )

    assert isinstance(regime, ConsumptionSavingsRegime)
    assert isinstance(regime.solver, NBEGM)
    aaae(
        working["tax"].to_numpy(),
        np.array([0.0, 0.6]),
        decimal=DECIMAL_PRECISION,
    )


def test_curated_egm_examples_instantiate_the_specialized_paths() -> None:
    """Curated EGM pages run their specialized builders as the principal path."""
    iskhakov = (_DOCS / "examples" / "iskhakov_et_al_2017.md").read_text(
        encoding="utf-8"
    )
    mahler_yum = (_DOCS / "examples" / "mahler_yum_2024.md").read_text(encoding="utf-8")

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
    """The smooth and nested User Guide declarations run through their workflow."""
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
            solution=solution,
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


def test_nested_example_demonstrates_adjustment_and_no_adjustment() -> None:
    """The small NEGM example contains both an adjuster and a keeper."""
    examples = importlib.import_module("lcm_examples.specialized_consumption_savings")

    model = examples.build_nested_model(enable_jit=False)
    params = examples.example_params()
    solution = model.solve(params=params, log_level="debug")
    result = model.simulate(
        params=params,
        initial_conditions=examples.example_initial_conditions(nested=True),
        solution=solution,
        log_level="debug",
    )
    working = result.to_dataframe().query("regime_name == 'working'")
    investments = working["illiquid_investment"].to_numpy()

    assert np.any(investments == 0.0)
    assert np.any(investments > 0.0)


def test_solver_capability_guidance_stays_aligned() -> None:
    """Guide, Methods, Reference, and public docstrings state one capability map."""
    chooser = (_DOCS / "user_guide" / "choosing_a_solver.md").read_text(
        encoding="utf-8"
    )
    authoring = (_DOCS / "user_guide" / "authoring_specialized_solvers.md").read_text(
        encoding="utf-8"
    )
    reference = (_DOCS / "reference" / "solvers.md").read_text(encoding="utf-8")
    notebook = json.loads(
        (_DOCS / "examples" / "epstein_zin.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    solver_source = (_ROOT / "src" / "_lcm" / "solution" / "nbegm.py").read_text(
        encoding="utf-8"
    )

    assert 'plain_egm{"Plain `EGM` contract satisfied?"}' in chooser
    assert "DCEGM-compatible resources, states, or processes" in chooser
    assert "genuine resources node and optional discrete choice" in reference
    assert "## Nonlinear certainty equivalents" in reference
    assert "`NBEGM` and `NNBEGM` implement" in reference
    assert "supported only with the\n  `GridSearch` solver" not in notebook_text
    assert "not GridSearch-only" in notebook_text
    assert "## A kinked tax schedule" in authoring
    assert "@lcm.piecewise_affine" in authoring
    assert 'additional_targets=["consumption"' not in authoring
    assert "double-double precision" not in solver_source
    assert "The fold stays available" not in solver_source
    assert "fixed-width integer arithmetic" in solver_source


def test_nnbegm_phase_replay_boundary_is_linked_across_docs() -> None:
    """Solver choice and phase grammar point to NNBEGM's replay restriction."""
    chooser = (_DOCS / "user_guide" / "choosing_a_solver.md").read_text(
        encoding="utf-8"
    )
    methods = (_DOCS / "methods" / "nested_egm.md").read_text(encoding="utf-8")
    reference = (_DOCS / "reference" / "solvers.md").read_text(encoding="utf-8")
    notebook = json.loads(
        (_DOCS / "explanations" / "phase_grammar.ipynb").read_text(encoding="utf-8")
    )
    phase_grammar = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "exact same callable object" in chooser
    assert "rejected during `Model(...)` construction" in chooser
    assert "../reference/solvers.md#nnbegm" in chooser
    assert "../reference/solvers.md#nnbegm" in methods
    assert "../reference/solvers.md#nnbegm" in phase_grammar
    assert "two distinct functions" in reference


def test_certified_nbegm_and_batch_widths_have_actionable_user_contracts() -> None:
    """User docs separate native capability from mapped batch widths."""
    installation = " ".join(
        (_DOCS / "getting_started" / "installation.md")
        .read_text(encoding="utf-8")
        .split()
    )
    solver_reference = " ".join(
        (_DOCS / "reference" / "solvers.md").read_text(encoding="utf-8").split()
    )
    tuning = " ".join(
        (_DOCS / "user_guide" / "tuning.md").read_text(encoding="utf-8").split()
    )
    paper_source = (
        _ROOT / "src" / "lcm_examples" / "mahler_yum_2024" / "paper.py"
    ).read_text(encoding="utf-8")

    assert 'NBEGM(envelope_arithmetic="ordinary")' in installation
    assert "fails before returning a certified result" in installation
    assert "first certified envelope evaluation" not in installation
    assert "during an NBEGM solve" not in installation
    assert "ExactAffineKernelUnavailableError" in installation
    assert "installed exact-affine CPU/CUDA payload" in solver_reference
    assert "compiled batch widths" in tuning
    assert "bounds how many entries are evaluated together" in tuning
    assert "They do not cap surrounding arrays" in tuning
    assert "explicit paper-mode scheduling profile" not in paper_source
    assert "Commit one `labor_supply` branch" not in paper_source
    assert "value-invariant memory knobs" not in tuning


def test_dcegm_methods_notebook_ends_with_counterpart_links() -> None:
    """The DCEGM Methods notebook links to every neighboring documentation role."""
    notebook = json.loads(
        (_DOCS / "explanations" / "iskhakov_et_al_2017.ipynb").read_text(
            encoding="utf-8"
        )
    )
    final_source = "".join(notebook["cells"][-1]["source"])

    for label in ("Guide", "Methods", "Example", "Reference"):
        assert f"**{label}:**" in final_source
