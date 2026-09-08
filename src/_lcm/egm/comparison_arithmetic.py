"""Which arithmetic settles a comparison between two candidate lines.

The alias is declared in its own leaf module because both the numerical
envelope kernels and the solver configuration that selects them name it, and
the kernels must stay importable without the model's own import graph.
"""

from typing import Literal

# It names what varies between the envelope paths — the comparison — rather than
# the arithmetic itself.
type ComparisonArithmetic = Literal["certified", "ordinary"]
