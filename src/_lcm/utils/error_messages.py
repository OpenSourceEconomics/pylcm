"""Internal validation plumbing for assembling error messages."""

from collections.abc import Iterable, Sequence

from dags.tree import QNAME_DELIMITER


def format_messages(errors: str | Sequence[str]) -> str:
    """Convert message or sequence of messages into a single string."""
    if isinstance(errors, str):
        formatted = errors
    elif len(errors) == 1:
        formatted = errors[0]
    else:
        enumerated = "\n\n".join([f"{i}. {error}" for i, error in enumerate(errors, 1)])
        formatted = f"The following errors occurred:\n\n{enumerated}"
    return formatted


def path_segment_name_errors(*, kind: str, names: Iterable[str]) -> list[str]:
    """Collect the error for user-chosen names that cannot be a parameter-path segment.

    Parameter paths are joined with `QNAME_DELIMITER` and split on it again, so a
    segment may not contain the delimiter. Nor may it start or end with `_`: an
    underscore at the seam merges into the delimiter, and `"x_"` + `"y"` would join
    to the same key as `"x"` + `"_y"`.

    Args:
        kind: What the names are, as the subject of the message (e.g. `"Regime"`).
        names: The names to check.

    Returns:
        A list holding one message that names every invalid name, or an empty list
        when every name is valid.

    """
    invalid = [
        name
        for name in names
        if QNAME_DELIMITER in name or name.startswith("_") or name.endswith("_")
    ]
    if not invalid:
        return []
    return [
        (
            f"{kind} names cannot contain the reserved separator character "
            f"'{QNAME_DELIMITER}' or start or end with '_'. Invalid names: {invalid}."
        )
    ]
