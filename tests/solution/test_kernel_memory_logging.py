"""`LCM_LOG_KERNEL_MEMORY` compile-time memory dump is decoupled from `log_level`.

The env var is the sole opt-in for the per-kernel `[mem]` lines. They must surface
at every solve `log_level` — including `"off"`, whose whole purpose is to silence the
per-period NaN/Inf diagnostic (its own full-V transient) so the kernel's true peak is
not masked. So the dump emits at a level that always clears the logger's threshold.
"""

import logging

from tests.solution.test_egm_discrete import (
    _get_skill_model,
    _get_skill_model_params,
)


def test_kernel_memory_dump_emits_at_log_level_off(monkeypatch, caplog):
    """With the env var on, the `[mem]` lines surface even at `log_level="off"`."""
    monkeypatch.setenv("LCM_LOG_KERNEL_MEMORY", "1")
    model = _get_skill_model()
    params = _get_skill_model_params()

    with caplog.at_level(logging.NOTSET, logger="lcm"):
        model.solve(params=params, log_level="off")

    mem_lines = [r.getMessage() for r in caplog.records if "[mem]" in r.getMessage()]
    assert mem_lines, "expected per-kernel [mem] lines with the env var on at log=off"
    assert any("temp=" in line and "peak=" in line for line in mem_lines)


def test_kernel_memory_dump_silent_without_env_var(monkeypatch, caplog):
    """Without the env var the dump is a no-op at any level (zero cost)."""
    monkeypatch.delenv("LCM_LOG_KERNEL_MEMORY", raising=False)
    model = _get_skill_model()
    params = _get_skill_model_params()

    with caplog.at_level(logging.NOTSET, logger="lcm"):
        model.solve(params=params, log_level="debug")

    assert not [r for r in caplog.records if "[mem]" in r.getMessage()]
