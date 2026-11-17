"""The source-bound tiny example: one original declaration fixture."""

from __future__ import annotations

import json
from pathlib import Path

from _lcm_regime_finder.records import JSONValue


def original() -> dict[str, JSONValue]:
    return json.loads(
        Path(__file__)
        .with_name("fixtures_metadata")
        .joinpath("tiny_original.json")
        .read_text()
    )
