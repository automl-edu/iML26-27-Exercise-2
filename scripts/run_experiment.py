"""Fit all models, compute feature effects and store scores, curves and plots.

Usage::

    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from iml_feature_effects.data import load_pga_tour, load_wine_quality
from iml_feature_effects.effects import summarize_effects
from iml_feature_effects.evaluate import evaluate_gam
from iml_feature_effects.models import get_gam_models
from iml_feature_effects.visualization import plot_effects, plot_mse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ALPHAS = (1000.0, 10.0, 1.0, 0.1, 0.0001)
REPEATS = 10
SEED = 0


def main() -> None:
    models = get_gam_models(ALPHAS)
    for load_dataset in (load_pga_tour, load_wine_quality):
        dataset = load_dataset(DATA)
        scores, curves, test_score = evaluate_gam(dataset, models, repeats=REPEATS, seed=SEED)
        output = RESULTS / dataset.name
        output.mkdir(parents=True, exist_ok=True)
        scores.to_csv(output / "scores.csv", index=False)
        curves.to_csv(output / "curves.csv", index=False)
        summarize_effects(curves).to_csv(output / "effect_metrics.csv", index=False)
        pd.DataFrame([test_score]).to_csv(output / "test_score.csv", index=False)
        plot_mse(scores, output / "mse.png")
        plot_effects(curves, output / "effects.png")
        print(scores.groupby(["dataset", "alpha"], sort=False)[["train_mse", "validation_mse"]].mean().to_string())
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
