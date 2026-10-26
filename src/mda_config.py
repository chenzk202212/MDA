"""Validation and rank selection shared by MDA experiment entry points."""

import math
from numbers import Integral, Real


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def resolve_rank(shape, options, num_tasks):
    """Use k=1/T by default; k is a retention fraction, not an absolute rank."""
    num_tasks = positive_integer(num_tasks, "number of merged tasks")
    max_rank = min(shape)
    if options.k is None:
        return max(1, max_rank // num_tasks)
    fraction = options.k
    if (isinstance(fraction, bool) or not isinstance(fraction, Real)
            or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("mda.k must be a retention fraction in (0, 1], or null for 1/T")
    return max(1, int(max_rank * fraction))


def validate_parameter_options(options):
    dimension = positive_integer(options.dimension, "mda.dimension")
    if dimension < 2:
        raise ValueError("mda.dimension must be at least 2 for ETF construction")
    alpha = options.parameter_alpha
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("mda.parameter_alpha must be in [0, 1]")
    return dimension, float(alpha)
