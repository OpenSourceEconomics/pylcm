"""The public shocks module separates declarations from model consumers."""

import lcm


def test_public_shock_declarations() -> None:
    """Model authors can declare a law and its independent quadrature rule."""
    shocks = getattr(lcm, "shocks", None)
    assert shocks is not None, "lcm.shocks must expose pure shock declarations"
    group = shocks.ShockGroup(
        shocks={"eps": shocks.Normal(mu=0.0, sigma=1.0)},
        nodes=shocks.Product(eps=shocks.GaussHermite(3)),
    )
    assert group.shocks["eps"].mu == 0.0
