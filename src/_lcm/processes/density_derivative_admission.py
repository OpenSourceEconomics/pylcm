"""Certify normalized normal-density derivatives using bounded IEEE enclosures."""

import math
from itertools import accumulate

import numpy as np
from jaxtyping import Bool, Float, Int

from lcm.exceptions import DensityDerivativeAccuracyError

type _HostFloat = Float[np.ndarray, "..."]
type _HostInt = Int[np.ndarray, "..."]
type _HostBool = Bool[np.ndarray, "..."]

# Fixed-size words retain physical exponents without physical underflow.
type _Word = tuple[float, int]
type _Interval = tuple[_Word, _Word]
_ZERO: _Word = (0.0, -100_000)
_EXACT_ZERO: _Interval = (_ZERO, _ZERO)
_BINARY64_ALIGNMENT_LIMIT = np.finfo(np.float64).nmant + 3
_NORMAL_MANTISSA_MIN = 0.5
_WORD_EXPONENT_LIMIT = -_ZERO[1]
_SELECTED_EXP_MIN = 0.25
_SELECTED_EXP_MAX = 4.0


def _normal(*, value: float, exponent: int = 0) -> _Word:
    mantissa, shift = math.frexp(value)
    return (mantissa, exponent + shift) if mantissa else _ZERO


def _stored(value: np.float32 | np.float64 | float | Float[np.ndarray, ""]) -> _Word:
    """Read stored subnormals through integer bits, without arithmetic on them."""
    array = np.asarray(value)
    info = np.finfo(array.dtype)
    bits = int(array.view(f"uint{info.bits}"))
    fraction = bits & ((1 << info.nmant) - 1)
    power = (bits >> info.nmant) & ((1 << (info.bits - info.nmant - 1)) - 1)
    significand = fraction | ((1 << info.nmant) if power else 0)
    exponent = (power - info.maxexp + 1 if power else info.minexp) - info.nmant
    return _normal(
        value=(-1.0 if bits >> (info.bits - 1) else 1.0) * float(significand),
        exponent=exponent,
    )


def _negative(value: _Word) -> _Word:
    return -value[0], value[1]


def _absolute(value: _Word) -> _Word:
    return abs(value[0]), value[1]


def _two_sum(*, left: _Word, right: _Word) -> tuple[_Word, _Word]:
    if left[1] < right[1]:
        left, right = right, left
    distance = left[1] - right[1]
    if distance > _BINARY64_ALIGNMENT_LIMIT:
        return left, right
    smaller = math.ldexp(right[0], -distance)
    total = left[0] + smaller
    right_part = total - left[0]
    error = (left[0] - (total - right_part)) + (smaller - right_part)
    return _normal(value=total, exponent=left[1]), _normal(
        value=error, exponent=left[1]
    )


def _product(*, left: _Word, right: _Word) -> tuple[_Word, _Word]:
    splitter = 134217729.0
    a, b = left[0], right[0]
    split_a, split_b = splitter * a, splitter * b
    high_a, high_b = split_a - (split_a - a), split_b - (split_b - b)
    low_a, low_b = a - high_a, b - high_b
    product = a * b
    error = (
        ((high_a * high_b - product) + high_a * low_b) + low_a * high_b
    ) + low_a * low_b
    exponent = left[1] + right[1]
    return _normal(value=product, exponent=exponent), _normal(
        value=error, exponent=exponent
    )


def _up(value: _Word) -> _Word:
    return (
        _normal(value=math.nextafter(value[0], math.inf), exponent=value[1])
        if value[0]
        else value
    )


def _positive_add(*, left: _Word, right: _Word) -> _Word:
    if not left[0]:
        return right
    if not right[0]:
        return left
    high, low = _two_sum(left=left, right=right)
    return _up(high) if low[0] > 0 else high


def _positive_product(*, left: _Word, right: _Word) -> _Word:
    high, low = _product(left=left, right=right)
    return _up(high) if low[0] > 0 else high


def _positive_quotient(*, left: _Word, right: _Word) -> _Word:
    if not left[0]:
        return _ZERO
    return _up(_normal(value=left[0] / right[0], exponent=left[1] - right[1]))


def _sum_upper(values: tuple[_Word, ...]) -> _Word:
    result = _ZERO
    for value in values:
        result = _positive_add(left=result, right=_absolute(value))
    return result


def _exact_sum(values: tuple[_Word, ...]) -> tuple[_Word, ...]:
    expansion: tuple[_Word, ...] = (values[0],)
    for value in values[1:]:
        carry = value
        residuals: tuple[_Word, ...] = ()
        for component in expansion:
            carry, low = _two_sum(left=carry, right=component)
            residuals = (*residuals, low)
        expansion = (*residuals, carry)
    return expansion


def _from_words(words: tuple[_Word, ...]) -> _Interval:
    expansion = _exact_sum(words)
    index = next((i for i in range(len(expansion) - 1, -1, -1) if expansion[i][0]), 0)
    return expansion[index], _sum_upper(expansion[:index])


# keyword-only-exempt: library-callback=itertools.accumulate
def _add(left: _Interval, right: _Interval) -> _Interval:
    """Accumulate actual addition residuals and outward input radii."""
    high, low = _two_sum(left=left[0], right=right[0])
    return high, _sum_upper((left[1], right[1], low))


def _negate(value: _Interval) -> _Interval:
    return _negative(value[0]), value[1]


def _multiply(*, left: _Interval, right: _Interval) -> _Interval:
    high, low = _product(left=left[0], right=right[0])
    radius = _sum_upper(
        (
            low,
            _positive_product(left=_absolute(left[0]), right=right[1]),
            _positive_product(left=_absolute(right[0]), right=left[1]),
            _positive_product(left=left[1], right=right[1]),
        )
    )
    return high, radius


def _less(*, left: _Word, right: _Word) -> bool:
    difference = _from_words((left, _negative(right)))[0]
    return difference[0] < 0


def _positive_lower(value: _Interval) -> _Word:
    high, low = _two_sum(left=value[0], right=_negative(value[1]))
    if high[0] <= 0:
        raise DensityDerivativeAccuracyError(
            "density derivative denominator is not certified positive"
        )
    return (
        _normal(value=math.nextafter(high[0], -math.inf), exponent=high[1])
        if low[0] < 0
        else high
    )


def _divide(*, left: _Interval, right: _Interval) -> _Interval:
    lower = _positive_lower(right)
    quotient = _normal(
        value=left[0][0] / right[0][0], exponent=left[0][1] - right[0][1]
    )
    product = _product(left=quotient, right=right[0])
    residual = _from_words((left[0], _negative(product[0]), _negative(product[1])))
    error = _sum_upper(
        (
            residual[0],
            residual[1],
            left[1],
            _positive_product(left=_absolute(quotient), right=right[1]),
        )
    )
    return quotient, _positive_quotient(left=error, right=lower)


def _exclusive(values: tuple[_Interval, ...]) -> tuple[_Interval, ...]:
    prefix = tuple(accumulate(values, _add, initial=_EXACT_ZERO))
    suffix = tuple(accumulate(reversed(values), _add, initial=_EXACT_ZERO))
    return tuple(
        _add(prefix[i], suffix[len(values) - i - 1]) for i in range(len(values))
    )


def _coefficient(
    *, weights: tuple[_Interval, ...], gaps: tuple[_Interval, ...], divisor: _Interval
) -> tuple[_Interval, ...]:
    other_weights = _exclusive(weights)
    products = tuple(
        _multiply(left=w, right=g) for w, g in zip(weights, gaps, strict=True)
    )
    other_products = _exclusive(products)
    total = tuple(accumulate(weights, _add, initial=_EXACT_ZERO))[-1]
    total_squared = _multiply(left=total, right=total)
    denominator = _multiply(left=total_squared, right=divisor)
    output: list[_Interval] = []
    for weight, gap, other_weight, other_product in zip(
        weights, gaps, other_weights, other_products, strict=True
    ):
        difference = _add(
            _multiply(left=other_weight, right=gap), _negate(other_product)
        )
        output.append(
            _divide(left=_multiply(left=weight, right=difference), right=denominator)
        )
    return tuple(output)


def _lower(value: _Word) -> _Word:
    return (
        _normal(value=math.nextafter(value[0], 0.0), exponent=value[1])
        if value[0]
        else value
    )


def _accuracy_bound(
    *,
    candidate: _Word,
    enclosure: _Interval,
    scale: _Word,
    epsilon: float,
    min_normal_exponent: int,
) -> tuple[_Word, _Word]:
    """Bound final component error in dimensionless parameter units."""
    scaled_candidate = _multiply(left=(candidate, _ZERO), right=(scale, _ZERO))
    scaled_law = _multiply(left=enclosure, right=(scale, _ZERO))
    difference = _from_words((scaled_candidate[0], _negative(scaled_law[0])))
    absolute_candidate = _sum_upper(
        (_absolute(scaled_candidate[0]), scaled_candidate[1])
    )
    # Normal-result power-of-two factors are exact after the first rounded product.
    materialization_factor = (
        2.0 if candidate[0] and candidate[1] <= min_normal_exponent else epsilon
    )
    materialization = _positive_product(
        left=_stored(materialization_factor), right=absolute_candidate
    )
    error = _sum_upper(
        (
            difference[0],
            difference[1],
            scaled_candidate[1],
            scaled_law[1],
            materialization,
        )
    )
    return error, scaled_candidate[0]


def _admitted(
    *,
    candidate: _Word,
    enclosure: _Interval,
    scale: _Word,
    tolerance: float,
    epsilon: float,
    min_normal_exponent: int,
    largest_finite: _Word,
    max_exponent: int,
) -> bool:
    if candidate[0] and candidate[1] > max_exponent:
        try:
            true_lower = _positive_lower((_absolute(enclosure[0]), enclosure[1]))
        except ArithmeticError:
            return False
        if not _less(left=largest_finite, right=true_lower):
            return False
    error, scaled_candidate = _accuracy_bound(
        candidate=candidate,
        enclosure=enclosure,
        scale=scale,
        epsilon=epsilon,
        min_normal_exponent=min_normal_exponent,
    )
    tau_lower = _stored(math.nextafter(tolerance, 0.0))
    tau_upper = _stored(math.nextafter(tolerance, math.inf))
    relative, residual = _product(left=tau_lower, right=_absolute(scaled_candidate))
    if residual[0] < 0:
        relative = _lower(relative)
    numerator, residual = _two_sum(left=tau_lower, right=relative)
    if residual[0] < 0:
        numerator = _lower(numerator)
    denominator = _positive_add(left=_stored(1.0), right=tau_upper)
    allowance = _lower(
        _normal(
            value=numerator[0] / denominator[0],
            exponent=numerator[1] - denominator[1],
        )
    )
    return not _less(left=allowance, right=error)


def _weight_enclosure(
    *,
    contrast: _Interval,
    residual: np.float32 | np.float64,
    power: int,
    selected_exp: np.float32 | np.float64,
    active: bool,
    floor: int,
) -> _Interval:
    logarithm_two: _Interval = (_stored(0.6931471805599453), (0.5, -54))
    if not active:
        threshold = _multiply(left=logarithm_two, right=(_stored(float(floor)), _ZERO))
        upper_difference = _add(contrast, _negate(threshold))
        if _from_words((upper_difference[0], upper_difference[1]))[0][0] > 0:
            raise DensityDerivativeAccuracyError(
                "density derivative omitted tail is not certified"
            )
        return _ZERO, (0.5, floor + 1)
    reduced = _add(
        _add(contrast, _negate((_stored(residual), _ZERO))),
        _negate(_multiply(left=logarithm_two, right=(_stored(float(power)), _ZERO))),
    )
    perturbation = _sum_upper((reduced[0], reduced[1]))
    if _less(left=(0.5, 0), right=perturbation):
        raise DensityDerivativeAccuracyError(
            "density derivative exponential reduction is not certified"
        )
    y = _stored(selected_exp)
    next_y = np.nextafter(selected_exp, np.asarray(math.inf, dtype=selected_exp.dtype))
    spacing = _from_words((_stored(next_y), _negative(y)))
    primitive_error = _positive_product(left=_stored(4.0), right=_sum_upper(spacing))
    exponential_change = _positive_product(
        left=_positive_add(left=_absolute(y), right=primitive_error),
        right=_positive_product(left=_stored(2.0), right=perturbation),
    )
    radius = _positive_add(left=primitive_error, right=exponential_change)
    return (y[0], y[1] + power), (radius[0], radius[1] + power)


def _fail_if_invalid_density_arrays(
    *,
    nodes: _HostFloat,
    location: _HostFloat,
    rho: _HostFloat,
    source: _HostFloat,
    sigma: _HostFloat,
    reference: _HostInt,
    residuals: _HostFloat,
    powers: _HostInt,
    selected_exp: _HostFloat,
    active: _HostBool,
    coefficient_mantissas: _HostFloat,
    coefficient_exponents: _HostInt,
) -> None:
    """Require the callback shapes and IEEE formats used by the bound."""
    if nodes.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise DensityDerivativeAccuracyError(
            "density derivative admission requires binary32 or binary64"
        )
    if any(
        value.dtype != nodes.dtype
        for value in (
            location,
            rho,
            source,
            sigma,
            residuals,
            selected_exp,
            coefficient_mantissas,
        )
    ):
        raise DensityDerivativeAccuracyError(
            "density derivative admission requires matching floating dtypes"
        )
    if any(
        value.dtype.kind != "i" for value in (reference, powers, coefficient_exponents)
    ) or active.dtype != np.dtype("bool"):
        raise DensityDerivativeAccuracyError(
            "density derivative admission requires integer exponents "
            "and Boolean activity"
        )
    if nodes.ndim != 1 or any(
        value.shape != () for value in (location, rho, source, sigma, reference)
    ):
        raise DensityDerivativeAccuracyError(
            "density derivative admission requires a scalar process row"
        )
    if any(
        value.shape != nodes.shape
        for value in (residuals, powers, selected_exp, active)
    ):
        raise DensityDerivativeAccuracyError(
            "density derivative admission exponential metadata has the wrong shape"
        )
    if (
        coefficient_mantissas.shape != (4, len(nodes))
        or coefficient_exponents.shape != coefficient_mantissas.shape
    ):
        raise DensityDerivativeAccuracyError(
            "density derivative admission coefficient words have the wrong shape"
        )
    if not all(
        np.isfinite(value).all()
        for value in (
            nodes,
            location,
            rho,
            source,
            sigma,
            residuals,
            selected_exp,
            coefficient_mantissas,
        )
    ):
        raise DensityDerivativeAccuracyError(
            "density derivative admission requires finite operands"
        )


# keyword-only-exempt: library-callback=jax.experimental.io_callback
def _checked_density_coefficients(
    nodes: _HostFloat,
    location: _HostFloat,
    rho: _HostFloat,
    source: _HostFloat,
    sigma: _HostFloat,
    reference: _HostInt,
    residuals: _HostFloat,
    powers: _HostInt,
    selected_exp: _HostFloat,
    active: _HostBool,
    coefficient_mantissas: _HostFloat,
    coefficient_exponents: _HostInt,
) -> tuple[_HostFloat, _HostInt]:
    """Return only coefficients whose represented-input law is certified.

    The host callback uses fixed-size IEEE words, not arbitrary precision.
    It certifies coefficients, never numerical tangent values. Unresolved
    cancellation raises before opaque words enter the linear tangent map.
    """
    _fail_if_invalid_density_arrays(
        nodes=nodes,
        location=location,
        rho=rho,
        source=source,
        sigma=sigma,
        reference=reference,
        residuals=residuals,
        powers=powers,
        selected_exp=selected_exp,
        active=active,
        coefficient_mantissas=coefficient_mantissas,
        coefficient_exponents=coefficient_exponents,
    )
    magnitudes = np.abs(coefficient_mantissas)
    if np.any(
        (magnitudes != 0) & ((magnitudes < _NORMAL_MANTISSA_MIN) | (magnitudes >= 1))
    ):
        raise DensityDerivativeAccuracyError(
            "density derivative admission requires normalized coefficient words"
        )
    if np.any(
        (coefficient_exponents < -_WORD_EXPONENT_LIMIT)
        | (coefficient_exponents > _WORD_EXPONENT_LIMIT)
    ):
        raise DensityDerivativeAccuracyError(
            "density derivative admission coefficient exponents "
            "exceed their format bound"
        )
    info = np.finfo(nodes.dtype)
    if np.any((powers < -16 * info.maxexp - 2) | (powers > 16 * info.maxexp + 2)):
        raise DensityDerivativeAccuracyError(
            "density derivative admission exponential power exceeds its format bound"
        )
    if np.any(np.abs(residuals) > 1) or np.any(
        active
        & ((selected_exp < _SELECTED_EXP_MIN) | (selected_exp > _SELECTED_EXP_MAX))
    ):
        raise DensityDerivativeAccuracyError(
            "density derivative admission requires a compact normal exponential result"
        )
    node_words = tuple(_stored(value) for value in nodes)
    location_word, rho_word, source_word, sigma_word = (
        _stored(value[()]) for value in (location, rho, source, sigma)
    )
    if sigma_word[0] <= 0:
        raise DensityDerivativeAccuracyError(
            "density derivative admission requires positive sigma"
        )
    center = int(reference)
    if not 0 <= center < len(nodes):
        raise DensityDerivativeAccuracyError(
            "density derivative admission reference is out of bounds"
        )
    center_word = node_words[center]
    product = _product(left=rho_word, right=source_word)
    negative_mean = tuple(_negative(word) for word in (location_word, *product))
    gaps = tuple(_from_words((word, _negative(center_word))) for word in node_words)
    half_gaps = tuple(
        ((gap[0][0], gap[0][1] - 1), (gap[1][0], gap[1][1] - 1)) for gap in gaps
    )
    midpoint_residuals = tuple(
        _from_words(
            (
                (word[0], word[1] - 1),
                (center_word[0], center_word[1] - 1),
                *negative_mean,
            )
        )
        for word in node_words
    )
    sigma_squared = _multiply(left=(sigma_word, _ZERO), right=(sigma_word, _ZERO))
    contrasts = tuple(
        _EXACT_ZERO
        if i == center
        else _divide(
            left=_multiply(
                left=(_stored(-2.0), _ZERO), right=_multiply(left=gap, right=midpoint)
            ),
            right=sigma_squared,
        )
        for i, (gap, midpoint) in enumerate(
            zip(half_gaps, midpoint_residuals, strict=True)
        )
    )
    weights = tuple(
        _weight_enclosure(
            contrast=contrast,
            residual=residual,
            power=int(power),
            selected_exp=value,
            active=bool(enabled),
            floor=-16 * info.maxexp,
        )
        for contrast, residual, power, value, enabled in zip(
            contrasts, residuals, powers, selected_exp, active, strict=True
        )
    )
    mu_column = _coefficient(weights=weights, gaps=gaps, divisor=sigma_squared)
    sigma_column = tuple(
        _multiply(left=(_stored(-2.0), _ZERO), right=value)
        for value in _coefficient(
            weights=weights, gaps=contrasts, divisor=(sigma_word, _ZERO)
        )
    )
    rho_column = tuple(
        _multiply(left=value, right=(source_word, _ZERO)) for value in mu_column
    )
    source_column = tuple(
        _multiply(left=value, right=(rho_word, _ZERO)) for value in mu_column
    )
    tolerance = 1e-12 if nodes.dtype == np.dtype("float64") else 1e-5
    for column, (enclosures, scale) in enumerate(
        zip(
            (mu_column, rho_column, source_column, sigma_column),
            (sigma_word, _stored(1.0), sigma_word, sigma_word),
            strict=True,
        )
    ):
        for index, enclosure in enumerate(enclosures):
            candidate = (
                float(coefficient_mantissas[column, index]),
                int(coefficient_exponents[column, index]),
            )
            if not _admitted(
                candidate=candidate,
                enclosure=enclosure,
                scale=scale,
                tolerance=tolerance,
                epsilon=float(info.eps),
                min_normal_exponent=int(info.minexp),
                largest_finite=_stored(info.max),
                max_exponent=int(info.maxexp),
            ):
                raise DensityDerivativeAccuracyError(
                    "Cannot certify the density transition derivative dimensionless "
                    f"accuracy budget for {('mu', 'rho', 'source', 'sigma')[column]}, "
                    f"destination node {index}."
                )
    return coefficient_mantissas, coefficient_exponents
