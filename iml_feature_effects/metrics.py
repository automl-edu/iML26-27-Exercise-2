"""Stability and fidelity of feature effect curves across repeated refits.

Let :math:`g_1, \\dots, g_R` be the centered effect curves of one feature, computed by one method on the
same grid :math:`z_1, \\dots, z_K` for models refit on :math:`R` independent training sets, let
:math:`\\bar g` be their pointwise mean and :math:`g^*` the curve the same method gives for the true
function. With :math:`\\|h\\|^2 = \\frac{1}{K} \\sum_k h(z_k)^2`:

.. math::
    \\underbrace{\\frac{1}{R}\\sum_r \\|g_r - g^*\\|^2}_{\\text{fidelity error}}
    = \\underbrace{\\|\\bar g - g^*\\|^2}_{\\text{bias}^2}
    + \\underbrace{\\frac{1}{R}\\sum_r \\|g_r - \\bar g\\|^2}_{\\text{instability}}

All three are divided by ``scale**2`` (the variance of the target), so a value of 0.01 means the curves
are off by about 10% of the target's standard deviation (root mean square over the grid).

This is an *empirical* decomposition on a fixed reference set, a fixed grid and :math:`R` refits:

- :math:`g^*` is not the population effect but the same estimator (same reference data, same grid, same
  ALE intervals) applied to the known function.
- Grid points are weighted equally. With a quantile grid that is close to weighting by the feature's
  distribution; where quantiles coincide and are merged (discrete features), it is not.
- :math:`\\|\\bar g - g^*\\|^2` includes the Monte Carlo error of :math:`\\bar g`: its expectation is the
  squared bias plus instability / :math:`R`, so with :math:`R = 30` bias² is overestimated by about 3% of
  the instability.

What they mean for a user: *instability* is how much the explanation would change had a different
sample of the same size been collected; *bias²* is the systematic error that remains even after
averaging many refits; *fidelity error* is the expected error of the single explanation a user gets.
"""

from __future__ import annotations

import math

import numpy as np


def _check_scale(scale: float) -> float:
    if not (isinstance(scale, int | float) and math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    return float(scale)


def instability(curves: np.ndarray, scale: float = 1.0) -> float:
    """Mean squared deviation of the curves from their pointwise mean.

    :param curves: Array of shape ``(R, K)``, one centered curve per repetition.
    :param scale: Normalization; the result is divided by ``scale**2``.
    """
    curves, scale = np.asarray(curves, float), _check_scale(scale)
    return float(((curves - curves.mean(axis=0)) ** 2).mean() / scale**2)


def bias_squared(curves: np.ndarray, truth: np.ndarray, scale: float = 1.0) -> float:
    """Mean squared distance between the mean curve and the true curve, divided by ``scale**2``."""
    curves, scale = np.asarray(curves, float), _check_scale(scale)
    return float(((curves.mean(axis=0) - np.asarray(truth, float)) ** 2).mean() / scale**2)


def fidelity_error(curves: np.ndarray, truth: np.ndarray, scale: float = 1.0) -> float:
    """Mean squared distance of each curve to the true curve, divided by ``scale**2``.

    Equals :func:`bias_squared` + :func:`instability`.
    """
    curves, scale = np.asarray(curves, float), _check_scale(scale)
    return float(((curves - np.asarray(truth, float)) ** 2).mean() / scale**2)


def curve_distance(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> float:
    """Mean squared distance between two curves on the same grid, divided by ``scale**2``.

    Used for the disagreement between the mean PDP and the mean ALE curve of a model.
    """
    scale = _check_scale(scale)
    return float(((np.asarray(a, float) - np.asarray(b, float)) ** 2).mean() / scale**2)
