"""Feature effect explanations and properties of them.

A :class:`FeatureEffect` turns a fitted model into a curve: for each value on a grid of one
feature, the (average) effect of that feature on the prediction. :class:`PartialDependence` is the
example. You may add others (ICE, ALE, SHAP dependence, ...).

An :class:`EffectProperty` scores such a curve, e.g. to compare explanations across models of
different fit. :class:`Roughness` is a minimal example related to *continuity*. Replace or extend it
with the properties you want to study (consistency, continuity, stability across refits, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def feature_grid(x: pd.Series, resolution: int = 30) -> np.ndarray:
    """Grid of feature values: all unique values if few, otherwise empirical quantiles.

    :param x: Observed values of the feature.
    :param resolution: Maximum number of grid points.
    :return: Sorted grid values.
    :rtype: np.ndarray
    """
    unique = np.unique(x)
    if len(unique) <= resolution:
        return unique
    return np.unique(np.quantile(x, np.linspace(0, 1, resolution)))


class FeatureEffect(ABC):
    """Base class for a method that explains the effect of a single feature on a model's prediction."""

    #: Name under which the effect is reported.
    name: str

    @abstractmethod
    def explain(self, model: BaseEstimator, X: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Compute the effect curve of ``feature``.

        :param model: A fitted model.
        :param X: Data on which the effect is computed (e.g. the training or test set).
        :param feature: Name of the column in ``X`` to explain.
        :return: Columns ``value`` (grid of feature values) and ``effect``.
        :rtype: pd.DataFrame
        """


class PartialDependence(FeatureEffect):
    """Partial dependence (Friedman, 2001): the mean prediction when ``feature`` is set to each grid value.

    :param resolution: Maximum number of grid points.
    :param centered: If true, subtract the mean of the curve so curves of different models are comparable.
    """

    name = "pdp"

    def __init__(self, resolution: int = 30, centered: bool = True):
        self.resolution = resolution
        self.centered = centered

    def explain(self, model: BaseEstimator, X: pd.DataFrame, feature: str) -> pd.DataFrame:
        grid = feature_grid(X[feature], self.resolution)
        effect = np.empty(len(grid))
        X_mod = X.copy()
        for i, value in enumerate(grid):
            X_mod[feature] = value
            effect[i] = model.predict(X_mod).mean()
        if self.centered:
            effect -= effect.mean()
        return pd.DataFrame({"value": grid, "effect": effect})


class EffectProperty(ABC):
    """Base class for a scalar property of a feature effect curve."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, curve: pd.DataFrame) -> float:
        """Score a feature effect curve.

        :param curve: Output of :meth:`FeatureEffect.explain`.
        :return: The score. Document what higher and lower values mean.
        :rtype: float
        """


class Roughness(EffectProperty):
    """Mean absolute change in slope between neighbouring grid segments.

    Slopes are measured in units of (effect range) / (feature range), so the score is scale-free. It is
    0 for a straight line; larger values mean a more jagged curve. This is only an example: argue why
    your properties capture what you want to study, and what they mean for someone using the explanation.
    """

    name = "roughness"

    def score(self, curve: pd.DataFrame) -> float:
        value, effect = curve["value"].to_numpy(float), curve["effect"].to_numpy(float)
        if len(effect) < 3 or np.ptp(effect) == 0:
            return 0.0
        slopes = np.diff(effect / np.ptp(effect)) / np.diff(value / np.ptp(value))
        return float(np.abs(np.diff(slopes)).mean())


def summarize_effects(curves: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "alpha", "feature"]
    variance = curves.groupby([*keys, "value"])["effect"].var(ddof=0)
    mean_variance = variance.groupby(level=keys).mean().rename("mean_pdp_variance")

    ordered = curves.sort_values("value")
    variation = ordered.groupby([*keys, "repeat"])["effect"].agg(
        total_variation=lambda values: values.diff().abs().sum(),
        net_change=lambda values: abs(values.iloc[-1] - values.iloc[0]),
    )
    variation["excess_variation"] = variation["total_variation"] - variation.pop("net_change")
    mean_variation = variation.groupby(level=keys).mean().add_prefix("mean_")
    return mean_variance.to_frame().join(mean_variation).reset_index()
