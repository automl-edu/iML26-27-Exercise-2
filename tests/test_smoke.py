"""Smoke tests that run offline on a synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from iml_feature_effects import (
    AccumulatedLocalEffects,
    CorrelatedFriedman1,
    Dataset,
    FunctionModel,
    PartialDependence,
    SubsampleTask,
    SyntheticTask,
    bias_squared,
    feature_grid,
    fidelity_error,
    get_models,
    ice_curves,
    instability,
    method_disagreement,
    run_repetitions,
    summarize,
)


@pytest.fixture
def linear_data() -> Dataset:
    X, y = make_regression(n_samples=300, n_features=4, noise=0.1, random_state=0)
    return Dataset(name="linear", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


@pytest.mark.parametrize("method", [PartialDependence(resolution=10), AccumulatedLocalEffects(resolution=10)])
def test_effect_of_linear_model_is_centered_line_with_coefficient_slope(linear_data, method):
    model = LinearRegression().fit(linear_data.X, linear_data.y)
    curve = method.explain(model, linear_data.X, "x0")
    assert len(curve) == 10
    assert curve["effect"].mean() == pytest.approx(0, abs=1e-8)
    slope, _ = np.polyfit(curve["value"], curve["effect"], 1)
    assert slope == pytest.approx(model.coef_[0])
    residual = curve["effect"] - np.polyval(np.polyfit(curve["value"], curve["effect"], 1), curve["value"])
    assert np.abs(residual).max() == pytest.approx(0, abs=1e-8)


def test_pdp_equals_mean_of_ice(linear_data):
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(linear_data.X, linear_data.y)
    grid = feature_grid(linear_data.X["x1"], 15)
    pdp = PartialDependence().explain(model, linear_data.X, "x1", grid)
    ice = ice_curves(model, linear_data.X, "x1", grid)
    assert ice.shape == (len(linear_data.X), len(grid))
    np.testing.assert_allclose(ice.mean(axis=0), pdp["effect"], atol=1e-10)


def test_ale_differs_from_pdp_with_interaction_and_correlation():
    # f = x1 * x2 with x2 = x1: ALE follows d/dx1 f = x2 = x1 along the data (a parabola x1^2 / 2 ...),
    # PDP averages over x2 independently of x1 (a line with slope E[x2]).
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 1, 2000)
    X = pd.DataFrame({"x1": x1, "x2": x1})
    model = FunctionModel(lambda X: X["x1"].to_numpy() * X["x2"].to_numpy())
    grid = np.linspace(0, 1, 21)
    pdp = PartialDependence().explain(model, X, "x1", grid)["effect"].to_numpy()
    ale = AccumulatedLocalEffects().explain(model, X, "x1", grid)["effect"].to_numpy()
    np.testing.assert_allclose(pdp, (grid - grid.mean()) * x1.mean(), atol=1e-10)
    expected_ale = grid**2 / 2  # integral of x from 0 to z
    np.testing.assert_allclose(ale, expected_ale - expected_ale.mean(), atol=5e-3)


def test_correlated_friedman_marginals_correlation_and_function():
    gen = CorrelatedFriedman1()
    X = gen.sample_X(5000, np.random.default_rng(0))
    assert list(X.columns) == [f"x{j}" for j in range(1, 11)]
    assert X.to_numpy().min() >= 0 and X.to_numpy().max() <= 1
    np.testing.assert_allclose(X.mean(), 0.5, atol=0.02)
    corr = X.corr(method="spearman")
    assert corr.loc["x1", "x2"] == pytest.approx(6 / np.pi * np.arcsin(0.4), abs=0.03)
    assert corr.loc["x4", "x6"] > 0.7
    assert abs(corr.loc["x1", "x3"]) < 0.05
    point = pd.DataFrame([[0.5, 1.0, 0.5, 0.2, 0.4] + [0.0] * 5], columns=gen.feature_names)
    assert gen.f(point)[0] == pytest.approx(10 * np.sin(np.pi * 0.5) + 0 + 2 + 2)


def test_decomposition_identity():
    rng = np.random.default_rng(0)
    curves, truth = rng.normal(size=(30, 12)), rng.normal(size=12)
    total = fidelity_error(curves, truth, scale=2.0)
    assert total == pytest.approx(bias_squared(curves, truth, 2.0) + instability(curves, 2.0))
    assert instability(np.tile(truth, (5, 1))) == pytest.approx(0)


def test_synthetic_task_fixed_test_set_and_fresh_training_sets():
    task = SyntheticTask(CorrelatedFriedman1(), n_train=50, n_test=100, n_ref=80, seed=1)
    X0, _ = task.train_sample(0)
    X0_again, _ = task.train_sample(0)
    X1, _ = task.train_sample(1)
    pd.testing.assert_frame_equal(X0, X0_again)
    assert not np.allclose(X0.to_numpy(), X1.to_numpy())
    assert len(task.X_test) == 100 and len(task.X_ref) == 80


def test_subsample_task_keeps_groups_out_of_training_pool():
    rng = np.random.default_rng(0)
    groups = pd.Series(np.repeat(np.arange(100), 5))
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    data = Dataset(name="grouped", X=X, y=pd.Series(rng.normal(size=500)), groups=groups)
    task = SubsampleTask(data, n_train=100, test_size=0.3, seed=0)
    test_groups = set(groups[task.X_test.index])
    for rep in range(3):
        X_train, _ = task.train_sample(rep)
        assert len(X_train) == 100 and not X_train.index.duplicated().any()
        assert test_groups.isdisjoint(groups[X_train.index])
    assert task.pool_size + len(task.X_test) == 500


def test_pipeline_shapes():
    task = SyntheticTask(CorrelatedFriedman1(), n_train=60, n_test=100, n_ref=50, seed=0)
    models = get_models(depths=(1, 3, None))
    methods = [PartialDependence(), AccumulatedLocalEffects()]
    fits, curves, truth = run_repetitions(task, ["x1", "x6"], models, methods, n_reps=3, resolution=8)
    assert len(fits) == 3 * 3
    np.testing.assert_allclose(fits.loc[fits["model"] == "depth_full", "train_r2"], 1.0)
    assert set(curves["method"]) == {"pdp", "ale"} and set(truth["model"]) == {"truth"}
    metrics = summarize(curves, truth, {task.name: task.scale})
    assert len(metrics) == 3 * 2 * 2
    np.testing.assert_allclose(metrics["fidelity_error"], metrics["bias2"] + metrics["instability"])
    # x6 does not enter f, so its true effect is exactly zero for both methods.
    assert np.allclose(truth.loc[truth["feature"] == "x6", "effect"], 0)
    disagreement = method_disagreement(curves, truth, {task.name: task.scale})
    assert set(disagreement["model"]) == {"depth_01", "depth_03", "depth_full", "truth"}
    # Jensen: the distance of the mean curves never exceeds the mean distance of the per-refit curves.
    assert (disagreement["pdp_vs_ale"] <= disagreement["pdp_vs_ale_per_refit"] + 1e-12).all()
    is_truth = disagreement["model"] == "truth"
    np.testing.assert_allclose(
        disagreement.loc[is_truth, "pdp_vs_ale"], disagreement.loc[is_truth, "pdp_vs_ale_per_refit"]
    )


def test_load_concrete_deduplicates_and_groups_mixtures(monkeypatch):
    from types import SimpleNamespace

    import iml_feature_effects.data as data_module

    ingredients = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer"]
    ingredients += ["Coarse Aggregate", "Fine Aggregate"]
    mix_a, mix_b = [540.0, 0, 0, 162, 2.5, 1040, 676], [332.5, 142.5, 0, 228, 0, 932, 594]
    rows = [mix_a + [28], mix_a + [28], mix_a + [90], mix_b + [28]]
    frame = pd.DataFrame(rows, columns=ingredients + ["Age (days)"])
    bunch = SimpleNamespace(data=frame, target=pd.Series([79.99, 79.99, 61.89, 40.27]))
    monkeypatch.setattr(data_module, "fetch_openml", lambda **kwargs: bunch)
    dataset = data_module.load_concrete()
    assert "age" in dataset.feature_names and "superplasticizer" in dataset.feature_names
    assert len(dataset.X) == 3  # exact duplicate removed
    assert list(dataset.groups) == [0, 0, 1]  # same mixture at two ages shares a group


def test_ice_fidelity_matches_pdp_refits_and_decomposes():
    from iml_feature_effects import ice_fidelity

    task = SyntheticTask(CorrelatedFriedman1(), n_train=60, n_test=100, n_ref=50, seed=0)
    models = get_models(depths=(2, None))
    ice = ice_fidelity(task, ["x1", "x6"], models, n_reps=3, n_instances=20, resolution=8)
    assert len(ice) == 2 * 2 * 2 and set(ice["method"]) == {"ice", "pdp_ice_points"}
    np.testing.assert_allclose(ice["fidelity_error"], ice["bias2"] + ice["instability"])
    # Jensen: the PDP of the same points (mean of their ICE curves) is never further from the truth.
    wide = ice.pivot_table(index=["model", "feature"], columns="method", values="fidelity_error")
    assert (wide["pdp_ice_points"] <= wide["ice"] + 1e-12).all()
    with pytest.raises(ValueError):
        rng = np.random.default_rng(0)
        real = Dataset("real", pd.DataFrame({"a": rng.normal(size=300)}), pd.Series(rng.normal(size=300)))
        ice_fidelity(SubsampleTask(real, n_train=50), ["a"], models, n_reps=1)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_metrics_reject_invalid_scale(scale):
    curves, truth = np.ones((3, 4)), np.zeros(4)
    for metric in (lambda: instability(curves, scale), lambda: bias_squared(curves, truth, scale)):
        with pytest.raises(ValueError):
            metric()
    with pytest.raises(ValueError):
        fidelity_error(curves, truth, scale)


def test_scale_uses_population_variance_like_r2():
    from sklearn.metrics import r2_score

    task = SyntheticTask(CorrelatedFriedman1(), n_train=60, n_test=50, n_ref=20, seed=0)
    model = get_models(depths=(3,))["depth_03"].fit(*task.train_sample(0))
    prediction = model.predict(task.X_test)
    mse = np.mean((task.y_test.to_numpy() - prediction) ** 2)
    assert 1 - r2_score(task.y_test, prediction) == pytest.approx(mse / task.scale**2)


@pytest.mark.parametrize("method", [PartialDependence(), AccumulatedLocalEffects()])
def test_constant_feature_has_single_grid_point_and_zero_effect(method):
    X = pd.DataFrame({"a": np.full(50, 3.0), "b": np.linspace(0, 1, 50)})
    model = FunctionModel(lambda X: X["a"].to_numpy() * 2 + X["b"].to_numpy())
    curve = method.explain(model, X, "a")
    assert len(curve) == 1 and curve["effect"].iloc[0] == 0


def test_feature_grid_merges_repeated_quantiles():
    x = pd.Series(np.concatenate([np.zeros(700), np.linspace(1, 2, 300)]))  # zero-inflated
    grid = feature_grid(x, resolution=30)
    assert len(grid) < 30 and np.all(np.diff(grid) > 0) and grid[0] == 0 and grid[-1] == 2


def test_ale_empty_interval_contributes_no_change():
    # No reference point lies in (1, 2], so the ALE curve must be flat between grid values 1 and 2.
    X = pd.DataFrame({"a": [0.0, 0.5, 1.0, 2.5, 3.0]})
    model = FunctionModel(lambda X: X["a"].to_numpy() ** 2)
    curve = AccumulatedLocalEffects().explain(model, X, "a", grid=np.array([0.0, 1.0, 2.0, 3.0]))
    effect = curve["effect"].to_numpy()
    assert effect[2] - effect[1] == pytest.approx(0)
    assert effect[1] - effect[0] == pytest.approx(1)  # f(1) - f(0)
    assert effect[3] - effect[2] == pytest.approx(5)  # f(3) - f(2)
