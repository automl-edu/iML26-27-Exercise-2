"""Fit every model on every dataset, quantify its fit and compute feature effects."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from iml_feature_effects.data import Dataset
from iml_feature_effects.effects import EffectProperty, FeatureEffect, PartialDependence


def evaluate(
    datasets: Iterable[tuple[Dataset, list[str]]],
    models: dict[str, BaseEstimator],
    effect: FeatureEffect,
    properties: Iterable[EffectProperty],
    seed: int = 0,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every model on every dataset, then explain the requested features.

    Train and test R² are reported so you can argue whether a model under- or overfits. Train R² is
    computed on the data the model was actually fit on (``X_fit_``/``y_fit_`` if the model has them, as
    :class:`~iml_feature_effects.models.SubsampledRegressor` does). Effects are computed on the test set.

    :param datasets: Pairs of a dataset and the names of the features to explain on it.
    :param models: Unfitted models keyed by name, e.g. from :func:`iml_feature_effects.models.get_models`.
    :param effect: Feature effect method.
    :param properties: Properties computed for each effect curve.
    :param seed: Random seed for the train/test split.
    :param test_size: Fraction of each dataset held out.
    :return: ``scores`` with one row per (dataset, model, feature) and columns ``train_r2``, ``test_r2``
        and one per property; ``curves`` with one row per grid point and columns ``dataset``, ``model``,
        ``feature``, ``value`` and ``effect``.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    properties = list(properties)
    score_rows, curves = [], []
    for dataset, features in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=test_size, random_state=seed
        )
        for model_name, model in models.items():
            # Clone so that the same unfitted model can be reused across datasets.
            fitted = clone(model).fit(X_train, y_train)
            X_fit, y_fit = getattr(fitted, "X_fit_", X_train), getattr(fitted, "y_fit_", y_train)
            fit = {
                "dataset": dataset.name,
                "model": model_name,
                "train_r2": r2_score(y_fit, fitted.predict(X_fit)),
                "test_r2": r2_score(y_test, fitted.predict(X_test)),
            }
            for feature in features:
                curve = effect.explain(fitted, X_test, feature)
                score_rows.append({**fit, "feature": feature, **{p.name: p.score(curve) for p in properties}})
                curves.append(curve.assign(dataset=dataset.name, model=model_name, feature=feature))
    curves_df = pd.concat(curves, ignore_index=True)[["dataset", "model", "feature", "value", "effect"]]
    return pd.DataFrame(score_rows), curves_df


def evaluate_gam(
    dataset: Dataset,
    models: dict[float, Pipeline],
    repeats: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str | float]]:
    X, y = dataset.X, dataset.y
    groups = dataset.groups.to_numpy() if dataset.groups is not None else np.arange(len(X))
    test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    development, test = next(test_split.split(X, groups=groups))
    validation_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train, validation = next(validation_split.split(X.iloc[development], groups=groups[development]))
    train, validation = development[train], development[validation]

    X_train, y_train = X.iloc[train], y.iloc[train]
    X_validation, y_validation = X.iloc[validation], y.iloc[validation]
    subsamples = GroupShuffleSplit(n_splits=repeats, train_size=0.8, random_state=seed)
    effect = PartialDependence()
    score_rows, curves = [], []
    for repeat, (sample, _) in enumerate(subsamples.split(X_train, groups=groups[train]), start=1):
        X_sample, y_sample = X_train.iloc[sample], y_train.iloc[sample]
        for alpha, model in models.items():
            fitted = clone(model).fit(X_sample, y_sample)
            score_rows.append(
                {
                    "dataset": dataset.name,
                    "alpha": alpha,
                    "repeat": repeat,
                    "train_mse": mean_squared_error(y_sample, fitted.predict(X_sample)),
                    "validation_mse": mean_squared_error(y_validation, fitted.predict(X_validation)),
                }
            )
            for feature in dataset.feature_names:
                curve = effect.explain(fitted, X_validation, feature)
                curves.append(curve.assign(dataset=dataset.name, alpha=alpha, feature=feature, repeat=repeat))

    scores = pd.DataFrame(score_rows)
    selected_alpha = scores.groupby("alpha")["validation_mse"].mean().idxmin()
    selected = clone(models[selected_alpha]).fit(X_train, y_train)
    test_score = {
        "dataset": dataset.name,
        "alpha": selected_alpha,
        "test_mse": mean_squared_error(y.iloc[test], selected.predict(X.iloc[test])),
    }
    return scores, pd.concat(curves, ignore_index=True), test_score
