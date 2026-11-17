"""The source-bound tiny example: one original declaration fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def original() -> Any:  # noqa: PAN002 - The parsed fixture JSON; tests index and edit it freely.
    return json.loads(
        Path(__file__)
        .with_name("fixtures_metadata")
        .joinpath("tiny_original.json")
        .read_text()
    )
