"""Run the full experiment: refit trees of every depth, explain them, summarize and plot.

Usage::

    uv run python scripts/run_experiment.py                      # all datasets
    uv run python scripts/run_experiment.py --datasets friedman1 # a subset
    uv run python scripts/run_experiment.py --n-reps 5           # quick check -> results/quick/

Outputs go to ``results/<dataset>/``: ``fits.csv``, ``curves.csv``, ``truth.csv``, ``metrics.csv``,
``disagreement.csv``, for synthetic data ``ice_metrics.csv``, and the figures. Runs with a number of
refits other than the default 30 go to ``results/quick/<dataset>/`` instead (or to ``--out``), so a quick
check never overwrites the full results. Every run is seeded and deterministic. The first run downloads
Concrete from OpenML into ``data/``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from iml_feature_effects import (
    AccumulatedLocalEffects,
    CorrelatedFriedman1,
    PartialDependence,
    RepeatedTask,
    SubsampleTask,
    SyntheticTask,
    get_models,
    ice_fidelity,
    load_concrete,
    method_disagreement,
    model_name,
    run_repetitions,
    summarize,
)
from iml_feature_effects.plotting import plot_decoupling, plot_disagreement, plot_effects, plot_fit, plot_metrics

RESULTS = Path("results")
N_REPS = 30
SEED = 0
N_TRAIN = 200
RESOLUTION = 30
#: Number of reference points whose ICE curves are evaluated (synthetic data only).
ICE_INSTANCES = 200
#: Depths whose individual curves are shown (underfit, around the best test R², fully grown).
SHOWN_DEPTHS = (2, 4, None)

FRIEDMAN_FEATURES = ["x1", "x3", "x4", "x6"]

#: Dataset name -> (task factory, features to explain).
TASKS: dict[str, tuple[Callable[[], RepeatedTask], list[str]]] = {
    # Original noise level of Friedman #1.
    "friedman1": (
        lambda: SyntheticTask(CorrelatedFriedman1(noise=1.0, name="friedman1"), n_train=N_TRAIN, seed=SEED),
        FRIEDMAN_FEATURES,
    ),
    # Same function with more noise, so that deep trees overfit clearly in test R².
    "friedman1_noise3": (
        lambda: SyntheticTask(CorrelatedFriedman1(noise=3.0, name="friedman1_noise3"), n_train=N_TRAIN, seed=SEED),
        FRIEDMAN_FEATURES,
    ),
    "concrete": (
        lambda: SubsampleTask(load_concrete(), n_train=N_TRAIN, test_size=0.3, seed=SEED),
        ["age", "cement", "water", "superplasticizer"],
    ),
}


def run(name: str, n_reps: int, root: Path) -> None:
    factory, features = TASKS[name]
    task = factory()
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    methods = [PartialDependence(RESOLUTION), AccumulatedLocalEffects(RESOLUTION)]
    fits, curves, truth = run_repetitions(task, features, get_models(seed=SEED), methods, n_reps, RESOLUTION)
    scales = {task.name: task.scale}
    metrics = summarize(curves, truth, scales)
    disagreement = method_disagreement(curves, truth, scales)

    for frame, file in [
        (fits, "fits"),
        (curves, "curves"),
        (truth, "truth"),
        (metrics, "metrics"),
        (disagreement, "disagreement"),
    ]:
        frame.to_csv(out / f"{file}.csv", index=False)
    plot_fit(fits, task.name, out / "fit.png")
    shown = [model_name(d) for d in SHOWN_DEPTHS]
    for method in metrics["method"].unique():
        plot_effects(curves, truth, task.name, method, shown, out / f"effects_{method}.png")
    plot_metrics(metrics, task.name, out / "metrics.png")
    plot_disagreement(disagreement, task.name, out / "disagreement.png")
    if task.true_function is not None:
        # Local explanations: same refits, ICE curves of the first ICE_INSTANCES reference points.
        ice = ice_fidelity(task, features, get_models(seed=SEED), n_reps, ICE_INSTANCES, RESOLUTION)
        ice.to_csv(out / "ice_metrics.csv", index=False)
        plot_decoupling(fits, pd.concat([metrics, ice], ignore_index=True), task.name, out / "decoupling.png")

    info = f"n_train={N_TRAIN}, test={len(task.X_test)}, ref={len(task.X_ref)}"
    if isinstance(task, SubsampleTask):
        info += f", pool={task.pool_size} (expected overlap between refits {N_TRAIN / task.pool_size:.0%})"
    print(f"\n== {name} ({info}, {n_reps} refits, {time.perf_counter() - start:.1f}s) -> {out} ==")
    print(fits.groupby("model", sort=False)[["train_r2", "test_r2", "n_leaves", "depth"]].mean().round(3))
    cols = ["instability", "bias2", "fidelity_error"]
    print(metrics.pivot_table(index="model", columns=["feature", "method"], values=cols, sort=False).round(4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--n-reps", type=int, default=N_REPS, help=f"refits per depth (default: {N_REPS})")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default: results/, or results/quick/ if --n-reps differs)",
    )
    args = parser.parse_args()
    root = args.out if args.out is not None else RESULTS if args.n_reps == N_REPS else RESULTS / "quick"
    for name in args.datasets:
        run(name, args.n_reps, root)


if __name__ == "__main__":
    main()
