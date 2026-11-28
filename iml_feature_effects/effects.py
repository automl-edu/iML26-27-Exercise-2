"""Feature effect explanations: partial dependence, accumulated local effects and ICE.

A :class:`FeatureEffect` turns a fitted model into a curve on a grid of one feature. All methods here
evaluate on a *given* grid, so curves of different repetitions, models, methods and of the true
function lie on the same points and can be compared pointwise.

All curves are centered the same way, by subtracting their (unweighted) mean over the grid points.
With a quantile grid of a continuous feature this approximates centering by the expectation over the
feature's marginal distribution; for discrete features, where coinciding quantiles are merged into one
grid point, it does not weight values by their frequency. Using the same centering for every method
matters: PDP and ALE are only defined up to a constant, and different default centerings would
otherwise show up as spurious disagreement.

These are effects of the *function* (fitted model or true regression function), not causal effects of
the feature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd


def feature_grid(x: pd.Series, resolution: int = 30) -> np.ndarray:
    """Grid of feature values: all unique values if few, otherwise (unique) empirical quantiles.

    The grid always contains the minimum and the maximum of ``x``.

    :param x: Observed values of the feature.
    :param resolution: Maximum number of grid points.
    :return: Sorted grid values.
    :rtype: np.ndarray
    """
    unique = np.unique(x)
    if len(unique) <= resolution:
        return unique
    return np.unique(np.quantile(x, np.linspace(0, 1, resolution)))


class FunctionModel:
    """Wrap a function ``f(X) -> predictions`` so it can be explained like a fitted model.

    Used to compute the effects of the true regression function of a synthetic dataset.
    """

    def __init__(self, f: Callable[[pd.DataFrame], np.ndarray]):
        self.f = f

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.f(X), dtype=float)


def _with_feature(X: pd.DataFrame, feature: str, values: np.ndarray) -> pd.DataFrame:
    X_mod = X.copy()
    X_mod[feature] = values
    return X_mod


def _center(effect: np.ndarray) -> np.ndarray:
    return effect - effect.mean()


class FeatureEffect(ABC):
    """Base class for a method that explains the effect of a single feature on a model's prediction."""

    #: Name under which the effect is reported.
    name: str

    @abstractmethod
    def explain(self, model, X: pd.DataFrame, feature: str, grid: np.ndarray | None = None) -> pd.DataFrame:
        """Compute the centered effect curve of ``feature``.

        :param model: Anything with ``predict(X)``, e.g. a fitted regressor or a :class:`FunctionModel`.
        :param X: Reference data on which the effect is computed.
        :param feature: Name of the column in ``X`` to explain.
        :param grid: Sorted grid of feature values. Defaults to :func:`feature_grid` of ``X[feature]``.
        :return: Columns ``value`` (the grid) and ``effect``.
        :rtype: pd.DataFrame
        """


class PartialDependence(FeatureEffect):
    """Partial dependence (Friedman, 2001): the mean prediction when ``feature`` is set to a grid value.

    .. math:: \\mathrm{PD}(z) = \\frac{1}{n} \\sum_{i=1}^{n} \\hat f(z, x_{i,-j})

    Every reference point is combined with every grid value, including combinations that are rare in
    the data when ``feature`` is correlated with others. The model is then evaluated in low-density
    regions, where few training points constrain it.

    :param resolution: Maximum number of grid points if no grid is given.
    """

    name = "pdp"

    def __init__(self, resolution: int = 30):
        self.resolution = resolution

    def explain(self, model, X: pd.DataFrame, feature: str, grid: np.ndarray | None = None) -> pd.DataFrame:
        grid = feature_grid(X[feature], self.resolution) if grid is None else np.asarray(grid, float)
        n = len(X)
        # One stacked prediction call: block k holds all reference points with feature = grid[k].
        X_stacked = _with_feature(pd.concat([X] * len(grid), ignore_index=True), feature, np.repeat(grid, n))
        effect = model.predict(X_stacked).reshape(len(grid), n).mean(axis=1)
        return pd.DataFrame({"value": grid, "effect": _center(effect)})


class AccumulatedLocalEffects(FeatureEffect):
    """First-order accumulated local effects (Apley & Zhu, 2020).

    The grid :math:`z_0 < \\dots < z_K` splits the feature range into intervals. In interval :math:`k`,
    only the reference points whose feature value lies in :math:`(z_{k-1}, z_k]` are used (points at
    :math:`z_0` belong to the first interval); for them the prediction difference when moving the feature
    from :math:`z_{k-1}` to :math:`z_k` is averaged:

    .. math:: \\mathrm{ALE}(z_k) = \\sum_{l=1}^{k} \\frac{1}{n_l} \\sum_{i:\\, x_{ij} \\in (z_{l-1}, z_l]}
              \\left[\\hat f(z_l, x_{i,-j}) - \\hat f(z_{l-1}, x_{i,-j})\\right]

    Because each point is only moved within its own interval, the model is evaluated close to the data
    even when features are correlated. Intervals without reference points contribute no change: this
    is a convention (the curve is flat there), not evidence that the effect is flat. It happens when
    two neighbouring quantiles fall between the same pair of observed values, i.e. when the grid is fine
    compared with the reference data (in this experiment: one of the 29 intervals of ``cement`` on
    Concrete).

    :param resolution: Maximum number of grid points if no grid is given.
    """

    name = "ale"

    def __init__(self, resolution: int = 30):
        self.resolution = resolution

    def explain(self, model, X: pd.DataFrame, feature: str, grid: np.ndarray | None = None) -> pd.DataFrame:
        grid = feature_grid(X[feature], self.resolution) if grid is None else np.asarray(grid, float)
        if len(grid) < 2:  # constant feature: no change is observable
            return pd.DataFrame({"value": grid, "effect": np.zeros(len(grid))})
        x = X[feature].to_numpy(float)
        # interval[i] = k such that grid[k-1] < x_i <= grid[k]; values outside the grid are clipped.
        interval = np.clip(np.searchsorted(grid, x, side="left"), 1, len(grid) - 1)
        diff = model.predict(_with_feature(X, feature, grid[interval])) - model.predict(
            _with_feature(X, feature, grid[interval - 1])
        )
        counts = np.bincount(interval, minlength=len(grid))[1:]
        sums = np.bincount(interval, weights=diff, minlength=len(grid))[1:]
        local = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        effect = np.concatenate([[0.0], np.cumsum(local)])
        return pd.DataFrame({"value": grid, "effect": _center(effect)})


def ice_curves(model, X: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    """Individual conditional expectation curves (Goldstein et al., 2015), each centered.

    Their mean is the centered partial dependence. Used to evaluate instance level effect fidelity.

    :return: Array of shape ``(len(X), len(grid))``.
    :rtype: np.ndarray
    """
    n = len(X)
    X_stacked = _with_feature(pd.concat([X] * len(grid), ignore_index=True), feature, np.repeat(grid, n))
    curves = model.predict(X_stacked).reshape(len(grid), n).T
    return curves - curves.mean(axis=1, keepdims=True)
