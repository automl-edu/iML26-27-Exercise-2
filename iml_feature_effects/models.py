"""Regression trees of increasing complexity, from underfit to overfit.

All models are :class:`~sklearn.tree.DecisionTreeRegressor` with ``min_samples_leaf=1``; only
``max_depth`` changes. Depth 1 (a single split) cannot represent any of the structure in the data, while
an unrestricted tree grows until no further split is possible and therefore (nearly) interpolates
the training set: train :math:`R^2 = 1` unless identical inputs have different targets (on Concrete the
fully grown trees reach 0.998). Whether a level is under-, well or overfit is argued from the validation curve
(train vs. test :math:`R^2` over depth) that :mod:`scripts.run_experiment` produces.

Fully grown regression trees are unbalanced, so their depth is typically well above
:math:`\\log_2 n_{train}`; the actual depth of every fitted tree is recorded to check that the deepest
levels still differ.
"""

from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor

#: Complexity levels; ``None`` means unrestricted depth.
DEPTHS: tuple[int | None, ...] = (1, 2, 3, 4, 5, 6, 8, 12, 16, None)


def model_name(depth: int | None) -> str:
    """``3 -> "depth_03"``, ``None -> "depth_full"``; names sort in order of complexity."""
    return "depth_full" if depth is None else f"depth_{depth:02d}"


def get_models(depths: tuple[int | None, ...] = DEPTHS, seed: int = 0) -> dict[str, DecisionTreeRegressor]:
    """Return unfitted trees keyed by :func:`model_name`, in order of increasing complexity.

    ``random_state`` only affects tie-breaking between equally good splits; the experiment overrides it
    per repetition.

    :param depths: Values of ``max_depth``.
    :param seed: Default ``random_state`` of every tree.
    :rtype: dict[str, DecisionTreeRegressor]
    """
    return {model_name(d): DecisionTreeRegressor(max_depth=d, min_samples_leaf=1, random_state=seed) for d in depths}
