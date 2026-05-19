"""The `@categorical` class decorator.

Used to declare a category enumeration class whose fields become the
`DiscreteGrid` codes for a discrete state / action. The decorator and its
supporting validators live in `lcm._grids.categorical`.
"""

from lcm._grids.categorical import categorical, validate_category_class

__all__ = [
    "categorical",
    "validate_category_class",
]
