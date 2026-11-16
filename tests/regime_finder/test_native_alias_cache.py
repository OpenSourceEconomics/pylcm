"""The native FloatND alias check compiles `lcm/typing.py` once per process.

Every callback read replays the check; recompiling the file on each call made it
dominate the analysis time on French–Jones.
"""

import builtins

import pytest

from _lcm_regime_finder.current_api_contract import (
    inspect_native_floatnd_alias,
    matches_native_floatnd_alias,
)
from lcm import typing as lcm_typing


def test_a_repeated_alias_check_compiles_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = inspect_native_floatnd_alias(lcm_typing.FloatND)
    assert expected is not None, "positive control: the native alias authenticates"

    def forbidden(*args: bytes | str | int, **kwargs: int | bool) -> None:
        raise AssertionError(f"recompiled {args[1:2]}")

    monkeypatch.setattr(builtins, "compile", forbidden)
    assert matches_native_floatnd_alias(expected)
