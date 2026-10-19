"""Independent economic-loss and structural-value checks on the original panel."""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.solution import _nb_scientific_reference as reference
from tests.solution import test_state_dependent_borrowing_limit as fixture
from tests.solution._nb_scientific_capture import (
    NativeIncomeCapture,
    capture_original_nb_income,
)

_DECISION = "03d2b414e4fcb87e853f3da5374b662439a5a1388d59ace60717918785b938e2"
_REFERENCE = "86085c1e12f5807cada427403eb16e67afa2ba5bb924314dccf4271efc8afde8"
type _Array = NDArray[np.generic]
type _Json = bool | int | float | str | list[_Json] | dict[str, _Json] | None


@dataclass(frozen=True, kw_only=True)
class _Inputs:
    """Actual captured physical coordinates and public actions for one precision."""

    wealth: tuple[Fraction, ...]
    income: tuple[int, ...]
    actions: dict[str, tuple[Fraction, ...]]
    grid: tuple[Fraction, ...]
    value_wealth: tuple[Fraction, ...]
    values: dict[str, _Array]
    shadow: dict[str, reference.Economics]


def _number(value: np.generic) -> Fraction:
    """Embed a captured native scalar exactly, refusing nonfinite physical data."""
    assert isinstance(value, np.generic)
    assert value.dtype.name in ("float32", "float64", "int32", "int64")
    assert np.isfinite(value)
    return Fraction(float(value))


def _load(*, root: Path, filename: str) -> dict[str, _Array]:
    """Read already admitted actual capture bytes without replacing model inputs."""
    with np.load(root / filename, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _processed_scalar(
    *, capture: NativeIncomeCapture, root: Path, solver: str, path: tuple[str, str]
) -> Fraction:
    """Join qualified original method leaves to their verified native NPZ payload."""
    calls = [
        call
        for call in capture.manifest["processed_parameter_calls"]
        if call["solver"] == solver and call["operation"] == "solve"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert call["returned_mapping_id"] == id(capture.processed_params[solver, "solve"])
    leaves = [leaf for leaf in call["leaves"] if tuple(leaf["path"]) == path]
    assert len(leaves) == 1, path
    arrays = _load(root=root, filename=call["file"])
    values = []
    for leaf in leaves:
        array = arrays[leaf["array"]]
        assert array.shape == ()
        assert array.dtype.name == f"float{capture.manifest['precision']}"
        assert array.tobytes(order="C").hex() == leaf["bit_pattern_hex"]
        scalar = array[()]
        assert isinstance(scalar, np.generic)
        values.append(_number(scalar))
    assert len(set(values)) == 1, path
    return values[0]


def _native_shadow(
    *, capture: NativeIncomeCapture, root: Path, solver: str
) -> reference.Economics:
    """Keep processed coefficients separate from source-bound auxiliary primitives."""
    primitives = _load(root=root, filename=f"{solver}-income-primitives.npz")
    np.testing.assert_array_equal(primitives["income_codes"], [0, 1])
    amounts = tuple(_number(value) for value in primitives["income_amounts"])
    assert len(amounts) == 2
    rows = primitives["transition_rows"]
    assert rows.shape == (2, 2)
    shares = [
        -_number(floor) / income
        for floor, income in zip(primitives["declared_floor"], amounts, strict=True)
    ]
    assert len(set(shares)) == 1
    if solver == "nbegm":
        assert shares[0] == _processed_scalar(
            capture=capture,
            root=root,
            solver=solver,
            path=("saving", "borrowing_limit_lower__limit_share"),
        )
    return reference.Economics(
        beta=_processed_scalar(
            capture=capture,
            root=root,
            solver=solver,
            path=("saving", "koopmans_aggregator__discount_factor"),
        ),
        tax_rate=_processed_scalar(
            capture=capture,
            root=root,
            solver=solver,
            path=("saving", "tax__tax_rate"),
        ),
        tax_kink=_processed_scalar(
            capture=capture,
            root=root,
            solver=solver,
            path=("saving", "tax__tax_exemption"),
        ),
        share=shares[0],
        incomes=(amounts[0], amounts[1]),
        probability_rows=(
            (_number(rows[0, 0]), _number(rows[0, 1])),
            (_number(rows[1, 0]), _number(rows[1, 1])),
        ),
        shadow=True,
    )


def _bind_inputs(*, capture: NativeIncomeCapture, root: Path) -> _Inputs:
    """Require complete named populations and genuine public-action lineage."""
    panels = {}
    actions = {}
    values = {}
    shadow = {}
    grids = {}
    coordinates = {}
    for solver in ("nbegm", "grid"):
        panel = _load(root=root, filename=f"{solver}-period0.npz")
        np.testing.assert_array_equal(panel["subject_id"], np.arange(60))
        np.testing.assert_array_equal(panel["period"], np.zeros(60))
        public = capture.simulations[solver].to_dataframe(use_categorical_codes=True)
        actual = public.query("period == 0").sort_values("subject_id")
        np.testing.assert_array_equal(actual["subject_id"], panel["subject_id"])
        np.testing.assert_array_equal(actual["consumption"], panel["consumption"])
        np.testing.assert_array_equal(capture.actions[solver], panel["consumption"])
        np.testing.assert_array_equal(actual["liquid"], panel["liquid"])
        np.testing.assert_array_equal(actual["income"], panel["income"])
        panels[solver] = panel
        actions[solver] = tuple(_number(value) for value in panel["consumption"])
        grids[solver] = _load(root=root, filename=f"{solver}-actions.npz")[
            "consumption"
        ]
        assert grids[solver].shape == (2500,)
        saved = _load(root=root, filename=f"{solver}-public-values.npz")
        schemas = capture.solutions[solver].metadata.value_schemas
        assert schemas[1, "saving"].axis_names == ("income", "liquid")
        values[solver] = saved["p1__saving"]
        assert values[solver].shape == (2, 120)
        np.testing.assert_array_equal(
            values[solver], capture.solutions[solver].values[1]["saving"]
        )
        coordinates[solver] = _load(
            root=root, filename=f"{solver}-value-coordinates.npz"
        )
        np.testing.assert_array_equal(coordinates[solver]["p1__saving__income"], [0, 1])
        shadow[solver] = _native_shadow(capture=capture, root=root, solver=solver)
    for field in ("liquid", "income", "subject_id"):
        np.testing.assert_array_equal(panels["nbegm"][field], panels["grid"][field])
    np.testing.assert_array_equal(grids["nbegm"], grids["grid"])
    np.testing.assert_array_equal(grids["nbegm"], fixture._CONSUMPTION_GRID.to_jax())
    physical = coordinates["nbegm"]["p1__saving__liquid"]
    assert physical.shape == (120,)
    np.testing.assert_array_equal(physical, fixture._WEALTH_GRID.to_jax())
    np.testing.assert_array_equal(physical, coordinates["grid"]["p1__saving__liquid"])
    np.testing.assert_array_equal(panels["nbegm"]["income"], np.repeat([0, 1], 30))
    expected_wealth = np.tile(np.linspace(0.5, 15.0, 30), 2)
    np.testing.assert_array_equal(
        panels["nbegm"]["liquid"],
        np.asarray(expected_wealth, dtype=panels["nbegm"]["liquid"].dtype),
    )
    return _Inputs(
        wealth=tuple(_number(value) for value in panels["nbegm"]["liquid"]),
        income=tuple(int(value) for value in panels["nbegm"]["income"]),
        actions=actions,
        grid=tuple(_number(value) for value in grids["nbegm"]),
        value_wealth=tuple(_number(value) for value in physical),
        values=values,
        shadow=shadow,
    )


def _require_action_lineage_sensitivity(
    *, capture: NativeIncomeCapture, root: Path
) -> None:
    """Reject a changed action even when every coordinate and other capture agrees."""
    changed = capture.actions["nbegm"].copy()
    changed[0] = np.nextafter(changed[0], np.inf, dtype=changed.dtype)
    assert changed[0] != capture.actions["nbegm"][0]
    altered = replace(capture, actions={**capture.actions, "nbegm": changed})
    with pytest.raises(AssertionError):
        _bind_inputs(capture=altered, root=root)


def _interval(interval: reference.Interval) -> list[_Json]:
    """Serialize outward endpoints as exact decimal strings."""
    return [str(interval.lower), str(interval.upper)]


def _subject_at_precision(
    *, inputs: _Inputs, subject: int, precision: int, bisections: int
) -> tuple[dict[str, _Json], reference.Interval]:
    """Score both public actions with one independently bounded nominal optimum."""
    economics = reference.Economics()
    wealth = inputs.wealth[subject]
    income = economics.incomes[inputs.income[subject]]
    optimum = reference.optimum0(
        economics=economics,
        wealth=wealth,
        income=income,
        precision=precision,
        max_bisections=bisections,
    )
    assert inputs.grid[0] <= optimum.consumption_lower
    assert optimum.consumption_upper <= inputs.grid[-1]
    methods: dict[str, _Json] = {}
    losses = {}
    for solver in ("nbegm", "grid"):
        action = inputs.actions[solver][subject]
        scored = reference.objective0(
            economics=economics,
            wealth=wealth,
            income=income,
            consumption=action,
            precision=precision,
        )
        regret = reference.nonnegative_regret(
            optimum=optimum.value, action=scored, precision=precision
        )
        losses[solver] = reference.loss_basis_points(
            regret=regret, beta=economics.beta, precision=precision
        )
        shadow = inputs.shadow[solver]
        shadow_value = reference.objective0(
            economics=shadow,
            wealth=wealth,
            income=shadow.incomes[inputs.income[subject]],
            consumption=action,
            precision=precision,
        )
        methods[solver] = {
            "consumption": str(action),
            "Q_nominal": _interval(scored),
            "regret_nominal": _interval(regret),
            "loss_bp_nominal": _interval(losses[solver]),
            "Q_rounded_parameter_shadow": _interval(shadow_value),
            "shadow_minus_nominal_Q": _interval(
                reference.subtract(left=shadow_value, right=scored, precision=precision)
            ),
        }
    resources = wealth - economics.tax_rate * max(wealth - economics.tax_kink, 0)
    upper = resources + economics.share * income
    feasible_grid = sum(0 < action <= upper for action in inputs.grid)
    assert feasible_grid > 0
    return {
        "subject_id": subject,
        "wealth": str(wealth),
        "income_code": inputs.income[subject],
        "equal_weight": "1/60",
        "precision_digits": precision,
        "optimum_iterations": optimum.iterations,
        "feasible_original_grid_nodes_nominal": feasible_grid,
        "optimum_consumption_bracket": [
            str(optimum.consumption_lower),
            str(optimum.consumption_upper),
        ],
        "Qopt_nominal": _interval(optimum.value),
        "methods": methods,
    }, losses["nbegm"]


def _subject(*, inputs: _Inputs, subject: int) -> tuple[dict[str, _Json], Decimal]:
    """Refine only unresolved oracle classifications within two fixed budgets."""
    for precision, bisections in ((80, 128), (160, 256)):
        try:
            row, loss = _subject_at_precision(
                inputs=inputs,
                subject=subject,
                precision=precision,
                bisections=bisections,
            )
        except reference.OracleUnresolvedError:
            if precision == 160:
                raise
            continue
        if loss.lower <= 1 < loss.upper and precision == 80:
            continue
        return row, loss.upper
    raise reference.OracleUnresolvedError("One-bp classification is unresolved.")


def _value_node(
    *, inputs: _Inputs, code: int, index: int
) -> tuple[dict[str, _Json], bool]:
    """Compare one actual NB physical node to the full structural V1 enclosure."""
    economics = reference.Economics()
    wealth = inputs.value_wealth[index]
    observed = _number(inputs.values["nbegm"][code, index])
    for precision in (80, 160):
        truth = reference.last_period(
            economics=economics,
            wealth=wealth,
            income=economics.incomes[code],
            precision=precision,
        )
        try:
            accepted = reference.value_budget(
                observed=observed, truth=truth.value, precision=precision
            )
        except reference.OracleUnresolvedError:
            if precision == 160:
                raise
            continue
        shadow = inputs.shadow["nbegm"]
        shadow_truth = reference.last_period(
            economics=shadow,
            wealth=wealth,
            income=shadow.incomes[code],
            precision=precision,
        )
        return {
            "income_code": code,
            "wealth_index": index,
            "wealth": str(wealth),
            "observed_nb": str(observed),
            "truth_nominal": _interval(truth.value),
            "structural_budget_met": accepted,
            "precision_digits": precision,
            "truth_rounded_parameter_shadow": _interval(shadow_truth.value),
            "shadow_minus_nominal_V1": _interval(
                reference.subtract(
                    left=shadow_truth.value, right=truth.value, precision=precision
                )
            ),
        }, accepted
    raise reference.OracleUnresolvedError("Structural V1 classification is unresolved.")


@fixture._EXACT
def test_original_nb_income_meets_independent_scientific_contract(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All sixty actions and 240 physical nodes meet independent economic budgets."""
    supplied_decision = os.environ.get("NB_SCIENTIFIC_DECISION")
    if supplied_decision is not None:
        decision = Path(supplied_decision)
        assert hashlib.sha256(decision.read_bytes()).hexdigest() == _DECISION
    supplied_root = os.environ.get("NB_COMPLETE_OBSERVER_SOURCE_ROOT")
    supplied_map = os.environ.get("NB_COMPLETE_OBSERVER_SOURCE_MAP")
    if (supplied_root is None) != (supplied_map is None):
        raise ValueError("Audit provenance requires both source root and source map")
    source_root = None if supplied_root is None else Path(supplied_root).resolve()
    source_map = None if supplied_map is None else Path(supplied_map)
    reference_path = Path(reference.__file__).resolve()
    actual_reference = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if source_map is None:
        assert reference_path.parent == Path(__file__).resolve().parent
    else:
        assert source_root is not None
        identities = json.loads(source_map.read_text())
        relative = reference_path.relative_to(source_root).as_posix()
        assert identities[relative]["sha256"] == actual_reference
    assert fixture._params(limit_param="limit_share", taxed=True) == {
        "discount_factor": 0.95,
        "tax_rate": 0.2,
        "tax_exemption": 10.0,
        "limit_share": 0.5,
    }
    assert fixture._LIMIT_SHARE == 0.5
    assert fixture._INCOME_LOW == 1.0
    assert fixture._INCOME_HIGH == 2.0
    exported_root = os.environ.get("NB_COMPLETE_OBSERVER_OUTPUT")
    root = tmp_path / "capture" if exported_root is None else Path(exported_root)
    capture = capture_original_nb_income(
        root=root,
        source_map=source_map,
        source_root=source_root,
        monkeypatch=monkeypatch,
    )
    simulation_calls = [
        call for call in capture.manifest["calls"] if call["operation"] == "simulate"
    ]
    assert len(simulation_calls) == 2
    assert {call["seed"] for call in simulation_calls} == {528}
    inputs = _bind_inputs(capture=capture, root=root)
    _require_action_lineage_sensitivity(capture=capture, root=root)
    scored = [_subject(inputs=inputs, subject=subject) for subject in range(60)]
    nodes = [
        _value_node(inputs=inputs, code=code, index=index)
        for code in range(2)
        for index in range(120)
    ]
    losses = sorted(upper for _, upper in scored)
    difference = np.abs(capture.actions["nbegm"] - capture.actions["grid"])
    budget = 3 * fixture._CONSUMPTION_STEP
    original = _load(root=root, filename="original-policy-comparison.npz")
    np.testing.assert_array_equal(original["absolute_difference"], difference)
    np.testing.assert_array_equal(
        original["violates_original_budget"], difference > budget
    )
    assert original["original_budget"][()] == budget
    legacy = {
        "role": "diagnostic",
        "primary_gate": False,
        "historical_assertion_status": "LEGACY_FAILED",
        "historical_threshold_exceedances": {"float32": 4, "float64": 5},
        "threshold": repr(budget),
        "threshold_exact_binary": str(Fraction(budget)),
        "all_absolute_differences": [float(value) for value in difference],
        "all_threshold_exceedances": [bool(value > budget) for value in difference],
        "all_action_disagreements": [bool(value != 0) for value in difference],
    }
    report = {
        "decision_sha256": _DECISION,
        "reviewed_raw_reference_sha256": _REFERENCE,
        "actual_reference_sha256": actual_reference,
        "source_map_sha256": capture.manifest["source_map_sha256"],
        "native_precision": capture.manifest["precision"],
        "nominal_economy": {
            "scope": "Intended declarations, not captured native leaves",
            "beta": "19/20",
            "tax_rate": "1/5",
            "tax_kink": "10",
            "floor_share": "1/2",
            "incomes": ["1", "2"],
            "transition_rows": [["4/5", "1/5"], ["1/5", "4/5"]],
        },
        "capture_sha256": hashlib.sha256(
            (root / "CAPTURE.json").read_bytes()
        ).hexdigest(),
        "subjects": [row for row, _ in scored],
        "period_one_nodes": [row for row, _ in nodes],
        "max_loss_bp_upper": str(losses[-1]),
        "q95_nearest_rank_loss_bp_upper": str(losses[56]),
        "legacy_comparison": legacy,
        "candidate_omission": "unknown/report_unknown",
        "shadow_scope": (
            "Exact-real expression from captured processed parameters and "
            "source-bound auxiliary income/transition/floor evaluations; "
            "raw transition mass retained; not native execution replay or "
            "nominal welfare acceptance"
        ),
        "shadow_row_masses": {
            solver: [str(sum(row)) for row in shadow.probability_rows]
            for solver, shadow in inputs.shadow.items()
        },
        "overall_acceptance": "UNVERIFIED; this is one native precision leg",
    }
    (root / "scientific-contract-leg.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    assert len(scored) == 60
    assert len(nodes) == 240
    assert losses[-1] <= Decimal(1)
    assert all(accepted for _, accepted in nodes)
