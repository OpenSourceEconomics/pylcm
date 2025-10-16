from dataclasses import dataclass, field
from typing import Any

from lcm.typing import UserFunction


@dataclass
class ModelMock:
    """A model mock for testing the params creation functions.

    This dataclass has the same attributes as the Model dataclass, but does not perform
    any checks, which helps us to test the params creation functions in isolation.

    """

    n_periods: int | None = None
    actions: dict[str, Any] | None = None
    states: dict[str, Any] | None = None
    utility: UserFunction | None = None
    constraints: dict[str, UserFunction] = field(default_factory=dict)
    transitions: dict[str, UserFunction] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)
