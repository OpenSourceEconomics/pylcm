"""The `@categorical` class decorator.

Used to declare a category enumeration class whose fields become the
`DiscreteGrid` codes for a discrete state / action. The decorator and its
supporting validators live in `lcm.grids.categorical` (private
implementation detail today; will consolidate under `lcm.api` in a
follow-up).
"""

from lcm.grids.categorical import categorical, validate_category_class

__all__ = [
    "categorical",
    "validate_category_class",
]
