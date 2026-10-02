"""Tests for the benchmark dashboard patches applied after `asv publish`."""

# ruff: noqa: SLF001

import json
import shutil
from pathlib import Path

import asv
import asv.results

from benchmarks import asv_machine, publish

_ASV_WWW = Path(asv.__file__).parent / "www"


def _patched_graphdisplay_js(tmp_path: Path) -> str:
    graphdisplay_js = tmp_path / "graphdisplay.js"
    shutil.copy(_ASV_WWW / "graphdisplay.js", graphdisplay_js)
    publish._default_y_axis_to_log(graphdisplay_js)
    return graphdisplay_js.read_text(encoding="utf-8")


def test_graph_detail_view_defaults_to_log_y_axis(tmp_path: Path) -> None:
    """The installed asv detail view turns log scale on when no URL param is given."""
    assert (
        "} else {\n            $('#log-scale').addClass('active');\n"
        "            log_scale = true;\n"
    ) in _patched_graphdisplay_js(tmp_path)


def test_graph_log_toggle_switched_off_stays_linear(tmp_path: Path) -> None:
    """Switching the log toggle off records `linear` rather than the default."""
    assert "log_scale ? ['log'] : ['linear']" in _patched_graphdisplay_js(tmp_path)


def test_summary_thumbnails_use_log_y_axis(tmp_path: Path) -> None:
    """The installed asv front-page thumbnails get a log transform on the y-axis."""
    summarygrid_js = tmp_path / "summarygrid.js"
    shutil.copy(_ASV_WWW / "summarygrid.js", summarygrid_js)

    publish._log_scale_summary_thumbnails(summarygrid_js)

    assert "return v > 0 ? Math.log(v) : null;" in summarygrid_js.read_text(
        encoding="utf-8"
    )


_COLUMNS = ["result", "params", "version", "started_at", "duration"]
_MACHINE = {
    "arch": "x86_64",
    "cpu": "Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz",
    "machine": "gpu-01",
    "num_cpu": "32",
    "os": "Linux 6.8.0-106-generic",
    "ram": "134807396352",
    "version": 1,
}
_PYTHON = "/runner/.pixi/envs/benchmarks-cuda12/bin/python"


def _write_result(
    *, machine_dir: Path, commit: str, ram: str, results: dict[str, list]
) -> Path:
    params = {key: val for key, val in _MACHINE.items() if key != "version"}
    params.update(ram=ram, python=_PYTHON)
    data = {
        "commit_hash": commit * 5,
        "env_name": "existing-py",
        "date": 1_790_000_000_000,
        "params": params,
        "python": _PYTHON,
        "requirements": {},
        "env_vars": {},
        "result_columns": _COLUMNS,
        "results": results,
        "durations": {},
        "version": 2,
    }
    path = machine_dir / f"{commit}-existing-py.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _results_dir(tmp_path: Path) -> Path:
    machine_dir = tmp_path / "results" / "gpu-01"
    machine_dir.mkdir(parents=True)
    (machine_dir / "machine.json").write_text(json.dumps(_MACHINE), encoding="utf-8")
    _write_result(
        machine_dir=machine_dir,
        commit="aaaaaaaa",
        ram="134807384064",
        results={"bench_m.Mahler.track_execution_time": [[176.0], [], "3", 1, 0.1]},
    )
    _write_result(
        machine_dir=machine_dir,
        commit="bbbbbbbb",
        ram="134807396352",
        results={"bench_m.Mahler.track_execution_time": [[7.3], [], "4", 1, 0.1]},
    )
    return tmp_path / "results"


def test_stable_ram_rounds_bytes_to_whole_gigabytes() -> None:
    """Both RAM figures the runner has reported map to one label, idempotently."""
    labels = {asv_machine.stable_ram(r) for r in ("134807384064", "134807396352")}
    assert labels == {"135GB"}
    assert asv_machine.stable_ram("135GB") == "135GB"


def test_normalised_history_is_one_series_that_asv_keeps(tmp_path: Path) -> None:
    """Stored results share the current machine params and survive a version bump."""
    results_dir = _results_dir(tmp_path)

    publish._normalise_results(results_dir)

    machine_dir = results_dir / "gpu-01"
    benchmarks = {"bench_m.Mahler.track_execution_time": {"version": "5"}}
    loaded = [
        asv.results.Results.load(str(path))
        for path in sorted(machine_dir.glob("*-existing-py.json"))
    ]
    assert [r.params["ram"] for r in loaded] == ["135GB", "135GB"]
    assert {r.params["python"] for r in loaded} == {_PYTHON}
    assert [r.get_result_keys(benchmarks) for r in loaded] == [set(benchmarks)] * 2
    machine = json.loads((machine_dir / "machine.json").read_text(encoding="utf-8"))
    assert machine["ram"] == "135GB"


def test_normalising_results_is_idempotent(tmp_path: Path) -> None:
    """A second pass leaves every file byte-identical."""
    results_dir = _results_dir(tmp_path)
    publish._normalise_results(results_dir)
    first = {p: p.read_bytes() for p in results_dir.rglob("*.json")}

    publish._normalise_results(results_dir)

    assert {p: p.read_bytes() for p in results_dir.rglob("*.json")} == first


def test_padding_extends_the_x_range_without_interior_gaps(tmp_path: Path) -> None:
    """Nulls go only where the series lies outside the folder-wide range."""
    folder = tmp_path / "summary"
    folder.mkdir()
    (folder / "bench_a.json").write_text(
        json.dumps([[1, 1.0], [2, 1.0], [3, 1.0], [4, 1.0], [5, 1.0]])
    )
    (folder / "bench_b.json").write_text(json.dumps([[2, 2.0], [4, 2.0]]))

    publish._pad_graphs_in_folder(folder)

    padded = json.loads((folder / "bench_b.json").read_text(encoding="utf-8"))
    assert padded == [[1, None], [2, 2.0], [4, 2.0], [5, None]]
