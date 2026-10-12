"""Resolve and hash declared native tools without executing them."""

import hashlib
import os
import shutil
from pathlib import Path


def capture_native_tool_identity(*, command: str) -> tuple[str, str, str, str]:
    """Bind the declared command, PATH-selected origin, real file and its bytes."""
    assert isinstance(command, str), "Native tool command must be a string"
    assert command, "Native tool command must not be empty"
    declared = Path(command)
    if not declared.is_absolute():
        assert declared.name == command, (
            "Relative native tool paths are unsupported; "
            "use a PATH command or absolute path"
        )
    selected = shutil.which(command)
    assert selected is not None, f"Native tool is absent or not executable: {command}"
    origin = Path(selected).absolute()
    real_file = origin.resolve(strict=True)
    assert real_file.is_file(), "Resolved native tool must be a regular file"
    assert os.access(real_file, os.X_OK), "Resolved native tool must be executable"
    with real_file.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return command, str(origin), str(real_file), digest
