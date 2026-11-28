"""Repeatedly refit every model, explain it, and summarize the stability and fidelity of the explanations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import r2_score

from iml_feature_effects.data import RepeatedTask
from iml_feature_effects.effects import FeatureEffect, FunctionModel, feature_grid, ice_curves
from iml_feature_effects.metrics import bias_squared, curve_distance, fidelity_error, instability

CURVE_COLUMNS = ["dataset", "model", "rep", "method", "feature", "value", "effect"]


def run_repetitions(
    task: RepeatedTask,
    features: Sequence[str],
    models: dict[str, BaseEstimator],
    methods: Sequence[FeatureEffect],
    n_reps: int = 30,
    resolution: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit every model on ``n_reps`` training sets and explain every feature with every method.

    In repetition ``r`` all models are fit on the *same* training set (common random numbers), so
    differences between models are not confounded with differences between samples. Grids and the
    reference data are fixed across repetitions and models.

    :param task: Provides the per-repetition training sets, the test set and the reference data.
    :param features: Features to explain.
    :param models: Unfitted models keyed by name, in order of complexity.
    :param methods: Feature effect methods.
    :param n_reps: Number of repetitions.
    :param resolution: Maximum number of grid points per feature.
    :return: ``fits`` (one row per dataset, model and repetition: train/test :math:`R^2`, number of leaves
        and depth of the fitted tree), ``curves`` (one row per grid point, columns :data:`CURVE_COLUMNS`)
        and ``truth`` (curves of the true function, same columns with model ``"truth"``; empty for real data).
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    grids = {f: feature_grid(task.X_ref[f], resolution) for f in features}

    def explain_all(model, model_name: str, rep: int) -> list[pd.DataFrame]:
        return [
            method.explain(model, task.X_ref, f, grids[f]).assign(
                dataset=task.name, model=model_name, rep=rep, method=method.name, feature=f
            )
            for method in methods
            for f in features
        ]

    truth = []
    if task.true_function is not None:
        truth = explain_all(FunctionModel(task.true_function), "truth", -1)

    fit_rows, curves = [], []
    for rep in range(n_reps):
        X_train, y_train = task.train_sample(rep)
        for name, model in models.items():
            fitted = _refit(model, X_train, y_train, rep)
            fit_rows.append(
                {
                    "dataset": task.name,
                    "model": name,
                    "rep": rep,
                    "train_r2": r2_score(y_train, fitted.predict(X_train)),
                    "test_r2": r2_score(task.y_test, fitted.predict(task.X_test)),
                    "n_leaves": fitted.get_n_leaves() if hasattr(fitted, "get_n_leaves") else np.nan,
                    "depth": fitted.get_depth() if hasattr(fitted, "get_depth") else np.nan,
                }
            )
            curves += explain_all(fitted, name, rep)

    def to_frame(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=CURVE_COLUMNS)
        return pd.concat(parts, ignore_index=True)[CURVE_COLUMNS]

    return pd.DataFrame(fit_rows), to_frame(curves), to_frame(truth)


def _refit(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, rep: int) -> BaseEstimator:
    """Fit a clone of ``model``; ``random_state`` is set to the repetition so refits are reproducible."""
    fitted = clone(model)
    if "random_state" in fitted.get_params():
        fitted.set_params(random_state=rep)
    return fitted.fit(X, y)


def ice_fidelity(
    task: RepeatedTask,
    features: Sequence[str],
    models: dict[str, BaseEstimator],
    n_reps: int = 30,
    n_instances: int = 200,
    resolution: int = 30,
) -> pd.DataFrame:
    """Stability and fidelity of centered ICE curves, i.e. of *local* (per-instance) feature effects.

    Uses exactly the same refits as :func:`run_repetitions` (same training sets and seeds). Each ICE
    curve is centered on its own mean, so only the *shape* of the effect is compared, not the prediction
    level of the instance.

    For a like-for-like comparison, the PDP is also computed from *the same* ``n_instances`` points, as
    the mean of their (centered) ICE curves, and reported as method ``"pdp_ice_points"``. Note that by
    Jensen's inequality its fidelity error is never larger than the ICE fidelity error of the same refits,
    so the *level* of the two is not informative; what can be compared is how they change with depth.

    :param task: A task with a ``true_function``.
    :param features: Features to explain.
    :param models: Unfitted models keyed by name.
    :param n_reps: Number of repetitions.
    :param n_instances: Number of reference points (the first ones of ``task.X_ref``) whose ICE curves are used.
    :param resolution: Maximum number of grid points per feature (same grids as :func:`run_repetitions`).
    :return: One row per (dataset, model, method, feature), method ``"ice"`` or ``"pdp_ice_points"``, with
        ``instability``, ``bias2`` and ``fidelity_error`` averaged over instances (ICE) and grid points, in
        units of the target variance.
    :rtype: pd.DataFrame
    """
    if task.true_function is None:
        raise ValueError("ICE fidelity needs a task with a known true function")
    X_ice = task.X_ref.iloc[:n_instances]
    grids = {f: feature_grid(task.X_ref[f], resolution) for f in features}
    truth = {f: ice_curves(FunctionModel(task.true_function), X_ice, f, grids[f]) for f in features}
    ice = {(name, f): [] for name in models for f in features}
    pdp = {(name, f): [] for name in models for f in features}
    for rep in range(n_reps):
        X_train, y_train = task.train_sample(rep)
        for name, model in models.items():
            fitted = _refit(model, X_train, y_train, rep)
            for f in features:
                curves = ice_curves(fitted, X_ice, f, grids[f])
                ice[(name, f)].append(curves.ravel())
                pdp[(name, f)].append(curves.mean(axis=0))
    rows = []
    for method, collected, true_of in [
        ("ice", ice, lambda f: truth[f].ravel()),
        ("pdp_ice_points", pdp, lambda f: truth[f].mean(axis=0)),
    ]:
        for (name, f), reps in collected.items():
            matrix, true = np.stack(reps), true_of(f)
            rows.append(
                {
                    "dataset": task.name,
                    "model": name,
                    "method": method,
                    "feature": f,
                    "n_reps": n_reps,
                    "instability": instability(matrix, task.scale),
                    "bias2": bias_squared(matrix, true, task.scale),
                    "fidelity_error": fidelity_error(matrix, true, task.scale),
                }
            )
    return pd.DataFrame(rows)


def _curve_matrix(group: pd.DataFrame) -> np.ndarray:
    """Rows = repetitions, columns = grid points."""
    return group.pivot(index="rep", columns="value", values="effect").to_numpy()


def summarize(curves: pd.DataFrame, truth: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    """Instability, and (where a true curve exists) bias² and fidelity error, per curve family.

    :param curves: Output of :func:`run_repetitions` (possibly concatenated over datasets).
    :param truth: True curves (possibly empty).
    :param scales: Target standard deviation per dataset, see :class:`~iml_feature_effects.data.RepeatedTask`.
    :return: One row per (dataset, model, method, feature) with ``instability``, ``bias2`` and
        ``fidelity_error`` (NaN without ground truth), all in units of the target variance.
    :rtype: pd.DataFrame
    """
    true_curves = {
        key: g.sort_values("value")["effect"].to_numpy()
        for key, g in truth.groupby(["dataset", "method", "feature"], sort=False)
    }
    rows = []
    for (dataset, model, method, feature), group in curves.groupby(
        ["dataset", "model", "method", "feature"], sort=False
    ):
        matrix, scale = _curve_matrix(group), scales[dataset]
        row = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "feature": feature,
            "n_reps": len(matrix),
            "instability": instability(matrix, scale),
            "bias2": np.nan,
            "fidelity_error": np.nan,
        }
        g_true = true_curves.get((dataset, method, feature))
        if g_true is not None:
            row["bias2"] = bias_squared(matrix, g_true, scale)
            row["fidelity_error"] = fidelity_error(matrix, g_true, scale)
        rows.append(row)
    return pd.DataFrame(rows)


def method_disagreement(
    curves: pd.DataFrame, truth: pd.DataFrame, scales: dict[str, float], methods: Iterable[str] = ("pdp", "ale")
) -> pd.DataFrame:
    """Disagreement between two methods, per (dataset, model, feature), in two variants.

    - ``<a>_vs_<b>``: distance between the *mean* curves (each method first averaged over the refits).
      This is the systematic disagreement; differences that change sign between refits cancel.
    - ``<a>_vs_<b>_per_refit``: distance between the two curves of the *same* refit, averaged over the
      refits. This is the disagreement a user of one fitted model sees. It is never smaller than the
      first variant (Jensen's inequality).

    For synthetic data the same distances for the true function (model ``"truth"``, a single curve per
    method, so both variants coincide) are included: the part of the disagreement that is due to the
    methods themselves rather than to the model.

    :rtype: pd.DataFrame
    """
    a, b = methods
    combined = pd.concat([curves, truth], ignore_index=True) if len(truth) else curves
    keys = ["dataset", "model", "feature"]
    wide = combined.pivot_table(index=[*keys, "rep", "value"], columns="method", values="effect", sort=False)
    mean = wide.groupby(level=[*keys, "value"], sort=False).mean()
    rows = []
    for (dataset, model, feature), g in mean.groupby(level=keys, sort=False):
        per_rep = wide.xs((dataset, model, feature), level=keys)
        per_refit = [
            curve_distance(r[a].to_numpy(), r[b].to_numpy(), scales[dataset])
            for _, r in per_rep.groupby(level="rep", sort=False)
        ]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "feature": feature,
                f"{a}_vs_{b}": curve_distance(g[a].to_numpy(), g[b].to_numpy(), scales[dataset]),
                f"{a}_vs_{b}_per_refit": float(np.mean(per_refit)),
            }
        )
    return pd.DataFrame(rows)
