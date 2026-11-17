"""ACA's labelled parameter Series retain their values and index metadata."""

import pytest
import importlib.util
from pathlib import Path
import tempfile

from _lcm_regime_finder.callable_identity import capture_session
from _lcm_regime_finder.model_snapshot import normalize
from _lcm_regime_finder.pytask_api import export_model
from _lcm_regime_finder.cli import load_problem
from _lcm_regime_finder.structural_store import final_input_check
from tests.regime_finder.test_round6_capture import FakeModel


def _create_series():
    import pandas as pd

    return pd.Series(
        [0.7, 0.9],
        name="survival",
        index=pd.MultiIndex.from_tuples(
            [(51, "bad"), (51, "good")], names=["age", "health"]
        ),
    )


@pytest.mark.skipif(
    not (importlib.util.find_spec("pandas")), reason="actual pandas required"
)
def test_production_series_retains_labelled_domain_and_values() -> None:
    model = FakeModel()
    model.fixed_params = {"survival": _create_series()}
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "model.json"
        receipt = export_model(
            model,
            path,
            production_metadata={"fixture": True},
            seconds=None,
            host_gib=None,
            source_transaction={
                "kind": "exclusive-owner",
                "authority": "test owns the Model",
            },
        )
        assert (receipt["status"]) == ("COMPLETE"), receipt
        captured = load_problem(path)
        fields = captured["fixed_params"]["survival"]["fields"]
        assert (fields["name"]) == ("survival")
        assert (fields["index_names"]) == (["age", "health"])
        assert (fields["index"]) == ([[51, "bad"], [51, "good"]])
        assert (fields["values"]["values"]) == ([0.7, 0.9])
        final_input_check(captured)


@pytest.mark.skipif(
    not (importlib.util.find_spec("pandas")), reason="actual pandas required"
)
def test_value_and_index_name_mutation_prevent_complete_capture(
    *, subtests: pytest.Subtests
) -> None:
    for label_only in (False, True):
        with subtests.test(label_only=label_only):
            series = _create_series()
            with capture_session() as session:
                record = normalize(series)
                if label_only:
                    series.index.names = ["wrong_age", "health"]
                else:
                    series.iloc[0] = 0.1
            assert (session.report()["status"]) != ("COMPLETE")
            assert (record["capture_status"]) == ("UNRESOLVED_MUTATION")


@pytest.mark.skipif(
    not (importlib.util.find_spec("pandas")), reason="actual pandas required"
)
def test_unsupported_labels_remain_incomplete_instead_of_losing_them() -> None:
    import pandas as pd

    model = FakeModel()
    model.fixed_params = {
        "table": pd.Series([1.0], index=pd.date_range("2020", periods=1))
    }
    with tempfile.TemporaryDirectory() as temp:
        receipt = export_model(
            model,
            Path(temp) / "model.json",
            production_metadata={"fixture": True},
            seconds=None,
            host_gib=None,
        )
        assert (receipt["status"]) == ("INCOMPLETE")
        assert ("Unsupported labelled Series index") in (
            receipt["capture_report"]["exhaustion_reason"]
        )


@pytest.mark.skipif(
    not (importlib.util.find_spec("pandas")), reason="actual pandas required"
)
def test_series_with_external_buffer_captures_owned_logical_values() -> None:
    import numpy as np
    import pandas as pd

    model = FakeModel()
    model.fixed_params = {"table": pd.Series(np.frombuffer(bytes([1, 2]), dtype="u1"))}
    with tempfile.TemporaryDirectory() as temp:
        receipt = export_model(
            model,
            Path(temp) / "model.json",
            production_metadata={"fixture": True},
            seconds=None,
            host_gib=None,
        )
        assert (receipt["status"]) == ("COMPLETE"), receipt
