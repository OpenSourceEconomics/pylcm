"""The engine package has no import-order cycles.

`_lcm.typing` imports type aliases from `lcm.typing`, and reaching
`lcm.typing` boots the public `lcm` package. When an `_lcm` submodule is
imported before `lcm` (as the benchmark suite does via `lcm_examples`),
that boot must not cycle back into a half-initialized `_lcm.typing`.
"""

import subprocess
import sys


def test_engine_submodule_imports_without_lcm_first():
    """An `_lcm` engine submodule imports cleanly in a fresh interpreter.

    Importing `_lcm.utils.dispatchers` as the first pylcm import — with no
    prior `import lcm` — must not raise a circular-import `ImportError`.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import _lcm.utils.dispatchers"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
