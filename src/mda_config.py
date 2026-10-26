"""Validation and rank selection shared by MDA experiment entry points."""

import math
from numbers import Integral, Real


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def resolve_rank(shape, options):
    """Select an explicit rank or a fraction of the layer's available SVD rank."""
    max_rank = min(shape)
    if options.k is not None:
        rank = positive_integer(options.k, "mda.k")
    else:
        fraction = options.sv_reduction
        if (isinstance(fraction, bool) or not isinstance(fraction, Real)
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("mda.sv_reduction must be in (0, 1]")
        rank = max(1, int(max_rank * fraction))
    if rank > max_rank:
        raise ValueError(f"mda.k={rank} exceeds the layer's available rank {max_rank}")
    return rank


def validate_parameter_options(options):
    dimension = positive_integer(options.dimension, "mda.dimension")
    if dimension < 2:
        raise ValueError("mda.dimension must be at least 2 for ETF construction")
    alpha = options.parameter_alpha
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("mda.parameter_alpha must be in [0, 1]")
    return dimension, float(alpha)
