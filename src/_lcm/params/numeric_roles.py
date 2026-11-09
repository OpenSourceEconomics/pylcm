"""The numeric role of a parameter slot, which decides how its value is cast.

A slot's role comes from what consumes it, never from its value:

- `FLOAT`: a library-owned coefficient (a process law, a correlation, a discount
  factor) or a callback argument annotated as a float. An `int` value, scalar or
  array, is converted to the active float dtype.
- `INT`: a callback argument annotated as an integer — a category code, an index.
  Its value keeps its kind.
- `AS_GIVEN`: anything else, including an unannotated argument. Its value keeps its
  kind.

A Boolean value keeps its kind in every role.
"""

from enum import Enum


class NumericRole(Enum):
    """How the boundary cast treats a parameter slot's value."""

    FLOAT = "float"
    INT = "int"
    AS_GIVEN = "as_given"

    @property
    def converts_to_float(self) -> bool:
        """Whether an integer value of this slot becomes the active float dtype."""
        return self is NumericRole.FLOAT
