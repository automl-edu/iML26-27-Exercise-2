"""Figures for the experiment. All functions save a PNG and close the figure."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
TEXT, TEXT_MUTED, GRID = "#191919", "#666666", "#e5e5e5"
METHOD_COLORS = {"pdp": BLUE, "ale": ORANGE, "ice": AQUA}
METHOD_LABELS = {"pdp": "PDP", "ale": "ALE", "ice": "ICE"}

plt.rcParams.update(
    {
        "axes.edgecolor": TEXT_MUTED,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 10,
    }
)


def depth_label(model: str) -> str:
    """``"depth_03" -> "3"``, ``"depth_full" -> "full"``."""
    suffix = model.removeprefix("depth_")
    return suffix if suffix == "full" else str(int(suffix))


def _set_depth_axis(ax, models: Sequence[str]) -> None:
    ax.set_xticks(range(len(models)), [depth_label(m) for m in models])
    ax.set_xlabel("max_depth")


def plot_fit(fits: pd.DataFrame, dataset: str, path: Path) -> None:
    """Validation curve (train/test R² over depth, mean ± 1 sd over repetitions) and tree size.

    Tree size is drawn in its own panel rather than on a second y-axis.
    """
    data = fits[fits["dataset"] == dataset]
    models = list(dict.fromkeys(data["model"]))
    stats = data.groupby("model", sort=False)[["train_r2", "test_r2", "n_leaves"]].agg(["mean", "std"])
    stats = stats.loc[models]
    x = np.arange(len(models))
    fig, (ax, ax_leaves) = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for col, color, label in [("train_r2", BLUE, "train"), ("test_r2", ORANGE, "test")]:
        mean, sd = stats[(col, "mean")].to_numpy(), stats[(col, "std")].to_numpy()
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, marker="o", markersize=5, label=label)
    best = int(np.argmax(stats[("test_r2", "mean")]))
    ax.axvline(best, color=TEXT_MUTED, linestyle=":", linewidth=1)
    ax.annotate(
        "best test $R^2$",
        (best, ax.get_ylim()[0]),
        xytext=(4, 4),
        textcoords="offset points",
        color=TEXT_MUTED,
        fontsize=8,
    )
    ax.set_ylabel("$R^2$")
    ax.legend(loc="lower right")
    ax.set_title(f"{dataset}: fit over complexity (mean ± 1 sd, {data['rep'].nunique()} refits)")
    ax_leaves.plot(x, stats[("n_leaves", "mean")], color=TEXT_MUTED, marker="o", markersize=4)
    ax_leaves.set_ylabel("leaves")
    _set_depth_axis(ax_leaves, models)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_effects(
    curves: pd.DataFrame,
    truth: pd.DataFrame,
    dataset: str,
    method: str,
    models: Sequence[str],
    path: Path,
) -> None:
    """Grid of effect curves: one row per feature, one column per model.

    Thin lines are the individual refits, the thick line is their mean, the dashed line (synthetic data
    only) is the same method applied to the true function.
    """
    data = curves[(curves["dataset"] == dataset) & (curves["method"] == method)]
    true = truth[(truth["dataset"] == dataset) & (truth["method"] == method)] if len(truth) else truth
    features = list(dict.fromkeys(data["feature"]))
    color = METHOD_COLORS.get(method, BLUE)
    fig, axes = plt.subplots(
        len(features),
        len(models),
        figsize=(3.1 * len(models), 2.3 * len(features)),
        squeeze=False,
        sharey="row",
    )
    for i, feature in enumerate(features):
        for j, model in enumerate(models):
            ax = axes[i][j]
            sub = data[(data["feature"] == feature) & (data["model"] == model)]
            for _, rep in sub.groupby("rep"):
                ax.plot(rep["value"], rep["effect"], color=color, alpha=0.18, linewidth=0.8)
            mean = sub.groupby("value")["effect"].mean()
            ax.plot(mean.index, mean.to_numpy(), color=color, linewidth=2.2, label="mean of refits")
            if len(true):
                t = true[true["feature"] == feature]
                ax.plot(t["value"], t["effect"], color=TEXT, linestyle="--", linewidth=1.5, label="true function")
            if i == 0:
                ax.set_title(f"max_depth = {depth_label(model)}")
            if j == 0:
                ax.set_ylabel(f"{feature}\ncentered effect")
    axes[0][-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{dataset}: {METHOD_LABELS.get(method, method)} across refits", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, dataset: str, path: Path) -> None:
    """Per feature: instability, bias² and fidelity error over depth, PDP solid and ALE dashed.

    For real data only instability is available.
    """
    data = metrics[metrics["dataset"] == dataset]
    models = list(dict.fromkeys(data["model"]))
    features = list(dict.fromkeys(data["feature"]))
    has_truth = data["bias2"].notna().any()
    components = [("instability", ORANGE, "instability")]
    if has_truth:
        components = [("fidelity_error", TEXT, "fidelity error"), ("bias2", BLUE, "bias²")] + components
    fig, axes = plt.subplots(1, len(features), figsize=(3.4 * len(features), 3.4), squeeze=False, sharey=True)
    x = np.arange(len(models))
    for ax, feature in zip(axes[0], features, strict=True):
        for method, style in [("pdp", "-"), ("ale", "--")]:
            sub = data[(data["feature"] == feature) & (data["method"] == method)].set_index("model").loc[models]
            for col, color, label in components:
                if not has_truth:  # only instability: distinguish the methods by color as well
                    color = METHOD_COLORS[method]
                ax.plot(
                    x,
                    sub[col],
                    color=color,
                    linestyle=style,
                    marker="o",
                    markersize=3.5,
                    label=f"{label} ({METHOD_LABELS[method]})",
                )
        ax.set_title(feature)
        _set_depth_axis(ax, models)
    axes[0][0].set_ylabel("error / Var(y)")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    title = "bias² + instability = fidelity error" if has_truth else "instability across refits"
    fig.suptitle(f"{dataset}: {title}", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_disagreement(disagreement: pd.DataFrame, dataset: str, path: Path) -> None:
    """Distance between the mean PDP and mean ALE curve over depth, per feature.

    For synthetic data a horizontal line marks the same distance for the true function.
    """
    data = disagreement[disagreement["dataset"] == dataset]
    fitted = data[data["model"] != "truth"]
    models = list(dict.fromkeys(fitted["model"]))
    features = list(dict.fromkeys(fitted["feature"]))
    palette = [BLUE, ORANGE, AQUA, TEXT_MUTED]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(models))
    for feature, color in zip(features, palette * 3, strict=False):
        sub = fitted[fitted["feature"] == feature].set_index("model").loc[models]
        ax.plot(x, sub["pdp_vs_ale"], color=color, marker="o", markersize=4, label=feature)
        true = data[(data["model"] == "truth") & (data["feature"] == feature)]
        if len(true):
            ax.axhline(true["pdp_vs_ale"].iloc[0], color=color, linestyle="--", linewidth=1)
    _set_depth_axis(ax, models)
    ax.set_ylabel("‖mean PDP − mean ALE‖² / Var(y)")
    ax.legend(title="feature (dashed: true function)" if len(data) > len(fitted) else "feature", fontsize=8)
    ax.set_title(f"{dataset}: disagreement between PDP and ALE")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_decoupling(fits: pd.DataFrame, metrics: pd.DataFrame, dataset: str, path: Path) -> None:
    """Prediction error next to the fidelity error of global (PDP, ALE) and local (ICE) explanations.

    All errors are in units of the target variance; prediction error is :math:`1 - R^2_{test}`. Each panel
    has its own y-axis (no panel mixes measures of different scale), and the dotted line marks the depth
    with the best mean test :math:`R^2`.
    """
    data = metrics[metrics["dataset"] == dataset]
    fit = fits[fits["dataset"] == dataset]
    models = list(dict.fromkeys(fit["model"]))
    features = list(dict.fromkeys(data["feature"]))
    prediction_error = 1 - fit.groupby("model", sort=False)["test_r2"].mean().loc[models]
    best = int(np.argmin(prediction_error.to_numpy()))
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, len(features) + 1, figsize=(3.2 * (len(features) + 1), 3.4), squeeze=False)
    axes = axes[0]
    axes[0].plot(x, prediction_error, color=TEXT, marker="o", markersize=3.5)
    axes[0].set_title("prediction error")
    axes[0].set_ylabel("error / Var(y)")
    for ax, feature in zip(axes[1:], features, strict=True):
        for method in ["ice", "pdp", "ale"]:
            sub = data[(data["feature"] == feature) & (data["method"] == method)]
            if len(sub):
                sub = sub.set_index("model").loc[models]
                ax.plot(
                    x,
                    sub["fidelity_error"],
                    color=METHOD_COLORS[method],
                    marker="o",
                    markersize=3.5,
                    label=METHOD_LABELS[method],
                )
        ax.set_title(f"explanation error: {feature}")
    for ax in axes:
        ax.axvline(best, color=TEXT_MUTED, linestyle=":", linewidth=1)
        ax.set_ylim(bottom=0)
        _set_depth_axis(ax, models)
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"{dataset}: prediction error vs. fidelity error of local (ICE) and global (PDP, ALE) effects", color=TEXT
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
