"""Build-time configuration constants resolved from the source tree layout.

`TEST_DATA` locates the analytical-solution and regression fixtures the test
suite reads, so no consumer has to rebuild that path itself.
"""

from pathlib import Path

TEST_DATA = Path(__file__).parent.parent.parent.resolve().joinpath("tests", "data")
