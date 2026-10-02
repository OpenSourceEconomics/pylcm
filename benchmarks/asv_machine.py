"""Register this host with ASV under machine params that do not drift.

Usage: pixi run asv-machine

`asv machine --yes` records the kernel's `MemTotal` to the byte, and the kernel's
reservation moves that figure by a few kilobytes across reboots and updates. ASV keys
a graph series by every machine param, so such a drift splits one machine's history
into two series. Recording the RAM in whole gigabytes keeps the series whole.
"""

import subprocess

from asv.util import get_memsize


def stable_ram(ram: str) -> str:
    """Return a RAM figure in bytes as whole gigabytes, e.g. `135GB`.

    Args:
        ram: The RAM figure, in bytes as `asv machine` records it, or a label this
            function already produced.

    Returns:
        The RAM rounded to whole decimal gigabytes; a non-numeric label unchanged.

    """
    if not ram.isdigit():
        return ram
    return f"{round(int(ram) / 1e9)}GB"


def main() -> None:
    """Record this host's defaults, then overwrite the RAM with its stable label."""
    # `--ram` alone only updates an existing entry, so the first call writes (or
    # refreshes) every default field.
    subprocess.run(["asv", "machine", "--yes"], check=True)
    subprocess.run(
        ["asv", "machine", "--yes", "--ram", stable_ram(str(get_memsize()))],
        check=True,
    )


if __name__ == "__main__":
    main()
