"""Measure a small subprocess tree independently of the test runner's imports."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from _lcm_regime_finder.resources import run_bounded


def main() -> None:
    command = [
        sys.executable,
        "-c",
        'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",'
        '"import time; x=bytearray(16*1024**2); time.sleep(0.12)"]); p.wait()',
    ]
    with TemporaryDirectory() as directory:
        code = run_bounded(command, output=directory, seconds=3, host_gib=0.5)
        record = json.loads((Path(directory) / "resource.json").read_text())
        print(json.dumps({"exit_code": code, "resource": record}))


if __name__ == "__main__":
    main()
