from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_mse(scores: pd.DataFrame, path: Path) -> None:
    by_alpha = scores.groupby("alpha")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for column, label in (("train_mse", "Training"), ("validation_mse", "Validation")):
        summary = by_alpha[column].agg(["mean", "std"])
        ax.errorbar(summary.index, summary["mean"], yerr=summary["std"], marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(xlabel="Regularization (alpha)", ylabel="Mean squared error")
    ax.legend(title="Mean ± 1 SD")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_effects(curves: pd.DataFrame, path: Path) -> None:
    features = curves.groupby("feature", sort=False)
    columns = min(3, len(features))
    rows = (len(features) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False, layout="constrained")
    for ax, (feature, feature_curves) in zip(axes.flat, features, strict=False):
        for alpha, alpha_curves in feature_curves.groupby("alpha", sort=False):
            summary = alpha_curves.groupby("value")["effect"].agg(["mean", "std"])
            mean, std = summary["mean"], summary["std"]
            (line,) = ax.plot(summary.index, mean, label=f"alpha = {alpha:g}")
            ax.fill_between(summary.index, mean - std, mean + std, color=line.get_color(), alpha=0.12)
        ax.set(xlabel=feature, ylabel="Centered partial dependence")
    for ax in axes.flat[len(features) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Mean ± 1 SD", loc="outside lower center", ncols=len(labels))
    fig.savefig(path, dpi=150)
    plt.close(fig)
