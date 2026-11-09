"""Literal scalar call oracle for captured-source dependencies."""
import inspect


class WrappedUnary:
    """Same effective/raw-signature distinction as an array-library wrapper."""

    __signature__ = inspect.Signature([
        inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ])

    def __call__(self, *args, **kwargs):
        value = args[0] if args else kwargs["x"]
        return value * value + 1


def literal_transitive(history, *, shift=0, scale=1):
    derived = shift + scale * history
    return WrappedUnary()(derived)


def varies_in_history(function, values=(-3, -1, 0, 2, 5)):
    outputs = [function(value) for value in values]
    return any(value != outputs[0] for value in outputs[1:])
