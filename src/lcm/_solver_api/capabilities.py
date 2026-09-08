"""Describe a solver configuration without building its numerical programs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    _NamesBoundary: TypeAlias = tuple[str, ...]  # noqa: UP040
else:
    # The constructor snapshots sequences before the published tuple contract
    # applies; runtime annotation sampling must not reject a mutable input first.
    _NamesBoundary = object


@dataclass(frozen=True, kw_only=True)
class SolverExecutionCapabilities:
    """Describe potential execution roles and structural solver requirements.

    A model's concrete programs determine which axes it actually accepts. This
    metadata describes the solver configuration's potential roles; it does not
    authorize an axis absent from those programs or waive model validation.
    """

    required_declaration: str
    """User-facing regime declaration required by this solver."""
    problem_shape: str
    """Economic problem structure the algorithm handles."""
    prerequisites: str
    """Structural requirements and supported constraints."""
    main_tradeoff: str
    """Principal accuracy, representation or computational tradeoff."""
    reduced_axes: _NamesBoundary = ()
    """Potential compiled reduction axes, in presentation order."""
    tiled_axes: _NamesBoundary = ()
    """Potential output tiling axes, in presentation order."""
    host_axes: _NamesBoundary = ()
    """Axes traversed by a host driver rather than a compiled reduction."""
    host_driven_programs: _NamesBoundary = ()
    """Graph keys repeated by the host; their cores may still be planned."""
    donation_candidates: _NamesBoundary = ()
    """Graph keys with a declared eligible input donation candidate."""
    supports_ev1_taste_shocks: bool = False
    """Whether supported model configurations may use EV1 taste shocks."""
    supports_nonlinear_certainty_equivalent: bool = False
    """Whether supported routes implement a nonlinear certainty equivalent."""

    def __post_init__(self) -> None:
        for name in (
            "required_declaration",
            "problem_shape",
            "prerequisites",
            "main_tradeoff",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(
                    f"SolverExecutionCapabilities.{name} must be a nonempty string."
                )
        for name in (
            "reduced_axes",
            "tiled_axes",
            "host_axes",
            "host_driven_programs",
            "donation_candidates",
        ):
            raw = getattr(self, name)
            if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
                raise TypeError(f"SolverExecutionCapabilities.{name} needs a sequence.")
            values = tuple(raw)
            if any(type(value) is not str or not value for value in values):
                raise ValueError(
                    f"SolverExecutionCapabilities.{name} needs nonempty names."
                )
            if len(set(values)) != len(values):
                raise ValueError(
                    f"SolverExecutionCapabilities.{name} has duplicate names."
                )
            object.__setattr__(self, name, values)
        axes = self.reduced_axes + self.tiled_axes + self.host_axes
        if len(set(axes)) != len(axes):
            raise ValueError("An execution axis must have exactly one capability role.")
        for name in (
            "supports_ev1_taste_shocks",
            "supports_nonlinear_certainty_equivalent",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"SolverExecutionCapabilities.{name} must be a bool.")

    @property
    def axis_names(self) -> frozenset[str]:
        """Return the union of potential compiled and host axis names."""
        return frozenset(self.reduced_axes + self.tiled_axes + self.host_axes)
