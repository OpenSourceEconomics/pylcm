"""Layouts for simulation's period-local reads of solved values."""

import jax


def simulation_value_sharding(
    *, stored_sharding: jax.sharding.Sharding, devices: tuple[jax.Device, ...]
) -> jax.sharding.Sharding:
    """Replicate a value across the devices evaluating simulated agents.

    Preserve a full simulation mesh's identity so a sharded value reaches its
    replicated layout through the catalogue's ``ALL_GATHER`` operation. Values
    stored on fewer devices use the common agent mesh instead. Comparing the
    ordered devices also keeps the argument device assignment consistent with
    the agent arrays when a stored mesh has a different device ordering.

    This chooses a layout only. The period's transfer owner applies it to the
    required reads and owns their copies; the stored solution stays unchanged.
    """
    if not devices:
        raise ValueError("Simulation value placement requires at least one device.")
    if len(devices) == 1:
        return jax.sharding.make_single_device_sharding(devices[0])
    if (
        isinstance(stored_sharding, jax.NamedSharding)
        and tuple(stored_sharding.mesh.devices.flat) == devices
    ):
        mesh = stored_sharding.mesh
    else:
        mesh = jax.make_mesh(
            (len(devices),),
            ("X",),
            (jax.sharding.AxisType.Auto,),
            devices=devices,
        )
    return jax.NamedSharding(mesh=mesh, spec=jax.P())
