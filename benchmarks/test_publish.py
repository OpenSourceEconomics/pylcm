"""Tests for the benchmark dashboard patches applied after `asv publish`."""

# ruff: noqa: SLF001

import shutil
from pathlib import Path

import asv

from benchmarks import publish

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
