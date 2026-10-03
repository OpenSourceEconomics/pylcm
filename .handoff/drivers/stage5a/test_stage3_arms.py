"""CLI and execution-layout checks without building an ACA model."""

# ruff: noqa: INP001, S101, PLC0415, SLF001

import importlib.util
import sys
from pathlib import Path

import pytest

_DRIVER = Path(__file__).with_name("stage3_arms.py")
_SPEC = importlib.util.spec_from_file_location("stage3_arms", _DRIVER)
assert _SPEC is not None
assert _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def test_cli_exposes_action_and_state_layouts(
    *, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The benchmark CLI exposes action partitions and independent state sharding."""
    monkeypatch.setattr(sys, "argv", [str(_DRIVER), "--help"])
    with pytest.raises(SystemExit) as error:
        driver.main()
    help_text = capsys.readouterr().out
    assert (
        error.value.code,
        "--action-partitions" in help_text,
        "--sharded-states" in help_text,
    ) == (0, True, True)


@pytest.mark.parametrize(
    ("layout", "sharded_states", "block_widths", "action_partitions"),
    [
        ([], ("assets",), {}, {}),
        (["--invariant-blocking"], ("assets",), {"pref_type": 1}, {}),
        (
            [
                "--invariant-blocking",
                "--sharded-states",
                "--action-partitions",
                "work=4",
            ],
            (),
            {"pref_type": 1},
            {"work": 4},
        ),
        (
            [
                "--invariant-blocking",
                "--sharded-states",
                "assets",
                "--action-partitions",
                "work=2",
                "--action-partitions",
                "retired=2",
            ],
            ("assets",),
            {"pref_type": 1},
            {"work": 2, "retired": 2},
        ),
    ],
)
def test_cli_config_expresses_four_layouts(
    *,
    layout: list[str],
    sharded_states: tuple[str, ...],
    block_widths: dict[str, int],
    action_partitions: dict[str, int],
) -> None:
    """The four layouts keep devices and width policy while selecting the axes."""
    from lcm import ExecutionConfig

    driver.parse_args(
        ["--workload", "reduced3", "--arm", "check", "--out", "unused", *layout]
    )
    config = driver._blocked(
        ExecutionConfig(
            devices=(0, 1, 2, 3),
            sharded_states=("assets",),
            axis_widths={"cell": 16},
        )
    )
    assert (
        config.sharded_states,
        dict(config.invariant_block_widths),
        dict(config.action_partitions),
        config.devices,
        dict(config.axis_widths),
    ) == (sharded_states, block_widths, action_partitions, (0, 1, 2, 3), {"cell": 16})


@pytest.mark.parametrize(
    "invalid",
    [
        ["--action-partitions", "work=0"],
        ["--action-partitions", "work=-1"],
        ["--action-partitions", "work=2.5"],
        ["--action-partitions", "=2"],
        ["--action-partitions", "work"],
        ["--action-partitions", "work=2=3"],
        ["--action-partitions", "work=2", "--action-partitions", "work=4"],
        ["--sharded-states", "assets", "assets"],
        ["--sharded-states", ""],
    ],
)
def test_cli_refuses_invalid_layout_flags(
    *,
    invalid: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Malformed or conflicting layout requests fail before touching a GPU."""
    with pytest.raises(SystemExit) as error:
        driver.parse_args(
            ["--workload", "reduced3", "--arm", "check", "--out", "unused", *invalid]
        )
    assert (error.value.code, invalid[0] in capsys.readouterr().err) == (2, True)


def test_cli_omitted_flags_preserve_the_supplied_config() -> None:
    """Absent overrides preserve the builder's complete execution policy."""
    from lcm import ExecutionConfig

    driver.parse_args(["--workload", "reduced3", "--arm", "check", "--out", "unused"])
    config = ExecutionConfig(
        sharded_states=("assets",),
        action_partitions={"work": 2},
        invariant_block_widths={"pref_type": 1},
    )
    assert driver._blocked(config) is config
