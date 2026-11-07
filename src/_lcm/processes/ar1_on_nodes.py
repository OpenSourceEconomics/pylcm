"""AR(1) transition rules on user-supplied physical nodes.

`lcm.processes.TauchenAR1Process.on_nodes` builds `_TauchenOnNodesProcess`, a
private subclass whose axis does not depend on the law parameters, so unit-root
persistence is admissible.
"""

from dataclasses import dataclass, fields
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype

from _lcm.beartype_conf import GRID_CONF
from _lcm.dtypes import canonical_float_dtype
from _lcm.grids import coordinates as grid_coordinates
from _lcm.processes.ar1 import TransitionRule, _normal_density_row
from _lcm.processes.base import _ContinuousStochasticProcess, sealed
from _lcm.processes.state_conditioned import tauchen_row
from _lcm.typing import ProcessLawScalar
from lcm.exceptions import GridInitializationError
from lcm.processes import TauchenAR1Process
from lcm.typing import Float1D, FloatND, IntND, ScalarFloat, ScalarInt


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class _TauchenOnNodesProcess(TauchenAR1Process):
    """AR(1) transition rules on a fixed physical axis independent of law parameters."""

    nodes: tuple[float, ...]
    transition_rule: TransitionRule = "midpoint"
    innovation_nodes: tuple[float, ...] = ()
    innovation_weights: tuple[float, ...] = ()

    _NON_PARAM_FIELDS: ClassVar[frozenset[str]] = (
        TauchenAR1Process._NON_PARAM_FIELDS  # noqa: SLF001
        | {
            "nodes",
            "n_std",
            "transition_rule",
            "innovation_nodes",
            "innovation_weights",
        }
    )

    def has_same_law_as(self, other: _ContinuousStochasticProcess) -> bool:
        """Compare the physical law, including state-conditioned scale declarations."""
        return isinstance(other, _TauchenOnNodesProcess) and all(
            getattr(self, spec.name) == getattr(other, spec.name)
            for spec in fields(self)
            if spec.name != "simulated_draws"
        )

    def __post_init__(self) -> None:
        points = np.asarray(self.nodes, dtype=canonical_float_dtype())
        if (
            len(points) < 2  # noqa: PLR2004
            or self.n_points != len(points)
            or not np.isfinite(points).all()
            or not np.all(points[1:] > points[:-1])
        ):
            raise GridInitializationError(
                "nodes must contain at least two finite strictly increasing values "
                "at the active precision."
            )
        super().__post_init__()

    @property
    def constructor_name(self) -> str:
        """Name of the public call that builds this process, for error messages."""
        return "TauchenAR1Process.on_nodes"

    def fail_if_nonstationary_rho(
        self, *, rho: ProcessLawScalar | FloatND | IntND
    ) -> None:
        """Require finite persistence; supplied nodes need no stationary law."""
        if not np.isfinite(np.asarray(rho)).all():
            raise GridInitializationError("rho must be finite for supplied nodes.")

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:
        """Return the supplied nodes, independent of runtime law parameters."""
        del kwargs
        return jnp.asarray(self.nodes, dtype=canonical_float_dtype())

    def get_gridpoints(self) -> Float1D:
        """Return the physical axis even when law parameters are unspecified."""
        return self.compute_gridpoints()

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Interpolate a coordinate on the supplied physical axis."""
        return grid_coordinates.get_irreg_coordinate(value=value, points=self.to_jax())

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:
        """Evaluate the selected transition row at each supplied source node."""
        return jax.vmap(partial(self.compute_transition_row, **kwargs))(self.to_jax())

    def transition_row_at(
        self, *, from_value: ScalarFloat, **kwargs: ScalarFloat | ScalarInt
    ) -> Float1D:
        """Return the selected rule's row from any current value."""
        return self.compute_transition_row(
            from_value, rho=kwargs["rho"], sigma=kwargs["sigma"], mu=kwargs["mu"]
        )

    # keyword-only-exempt: library-callback=jax.vmap
    def compute_transition_row(
        self,
        from_value: ScalarFloat,
        *,
        rho: ScalarFloat | ScalarInt,
        sigma: ScalarFloat | ScalarInt,
        mu: ScalarFloat | ScalarInt,
    ) -> Float1D:
        """Return destination probabilities at an actual physical source value."""
        nodes = self.compute_gridpoints()
        if self.transition_rule == "midpoint":
            return tauchen_row(
                nodes=nodes, from_value=from_value, rho=rho, sigma=sigma, mu=mu
            )
        if self.transition_rule == "density":
            return _normal_density_row(
                (
                    nodes,
                    jnp.asarray(mu, dtype=nodes.dtype),
                    jnp.asarray(rho, dtype=nodes.dtype),
                    jnp.asarray(from_value, dtype=nodes.dtype),
                    jnp.asarray(sigma, dtype=nodes.dtype),
                )
            )
        mean = mu + sealed(rho * from_value)
        innovations = jnp.asarray(self.innovation_nodes, dtype=nodes.dtype)
        weights = jnp.asarray(self.innovation_weights, dtype=nodes.dtype)
        targets = jnp.clip(mean + sigma * innovations, nodes[0], nodes[-1])
        lower = jnp.clip(
            jnp.searchsorted(nodes, targets, side="right") - 1, 0, len(nodes) - 2
        )
        left, right = nodes[lower], nodes[lower + 1]
        # Finite endpoints need not have a finite difference. Scale the operands
        # before subtraction, but retain the resolution of small/adjacent cells.
        wide = (
            (left < 0)
            & (right > 0)
            & (
                jnp.maximum(jnp.abs(left), jnp.abs(right))
                > jnp.finfo(nodes.dtype).max / 2
            )
        )
        scale = jnp.where(wide, 0.5, 1.0)
        upper_share = (targets * scale - left * scale) / (right * scale - left * scale)
        row = jnp.zeros_like(nodes).at[lower].add(weights * (1 - upper_share))
        return row.at[lower + 1].add(weights * upper_share)
