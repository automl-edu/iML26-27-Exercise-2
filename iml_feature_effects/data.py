"""Datasets and the repeated train/test sampling used in the experiment.

Two kinds of data are used:

- :class:`CorrelatedFriedman1`, a synthetic data generating process with a known regression function, so
  that feature effects of fitted models can be compared with the effects of the true function (fidelity).
- Real datasets (:func:`load_concrete`), where no ground truth exists and only stability can be measured.

A :class:`RepeatedTask` fixes everything that must not change between repetitions (test set, reference
data on which effects are computed) and draws a new training set for every repetition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GroupShuffleSplit

# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")

#: OpenML id of the Concrete Compressive Strength dataset (Yeh, 1998), version used in the CTR-23 suite.
CONCRETE_OPENML_ID = 44959


@dataclass(frozen=True)
class Dataset:
    """A preprocessed tabular regression dataset.

    :param name: Human-readable dataset name, used in result tables and plots.
    :param X: Numeric feature matrix without missing values.
    :param y: Target vector.
    :param groups: Optional group label per row. Rows of the same group are never split between train
        and test (see :class:`SubsampleTask`).
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def _clean_name(name: str) -> str:
    """``"Age (days)"`` -> ``"age"``: drop units in parentheses, lower case, underscores."""
    return name.split("(")[0].strip().lower().replace(" ", "_")


def load_concrete(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the Concrete Compressive Strength dataset from OpenML.

    The target is the compressive strength (MPa) of a concrete mixture after ``age`` days. Exact duplicate
    rows are removed. The same mixture (identical amounts of the seven ingredients) is often measured at
    several ages, so the mixture is returned as group label: splitting by group prevents a model from
    being tested on a mixture it has already seen at another age.

    :param data_home: Cache directory for the download. Network access is only needed on the first call.
    :return: The deduplicated dataset with mixture ids as ``groups``.
    :rtype: Dataset
    """
    bunch = fetch_openml(data_id=CONCRETE_OPENML_ID, as_frame=True, parser="liac-arff", data_home=data_home)
    X = bunch.data.astype(float)
    X.columns = [_clean_name(c) for c in X.columns]
    y = bunch.target.astype(float).rename("strength")
    keep = ~pd.concat([X, y], axis=1).duplicated()
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    ingredients = [c for c in X.columns if c != "age"]
    groups = pd.Series(pd.factorize(pd.MultiIndex.from_frame(X[ingredients]))[0], name="mixture")
    return Dataset(name="concrete", X=X, y=y, groups=groups)


def load_bike_sharing(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the hourly Bike Sharing Demand dataset from OpenML (data id 42712). Not used in the main experiment.

    :param data_home: Cache directory for the download.
    :return: The preprocessed Bike Sharing dataset.
    :rtype: Dataset
    """
    bunch = fetch_openml(data_id=42712, as_frame=True, parser="liac-arff", data_home=data_home)
    # This OpenML version already excludes ``casual`` and ``registered``, which sum to the target.
    X = pd.get_dummies(bunch.data, drop_first=True, dtype=float)
    y = bunch.target.astype(float).rename("count")
    return Dataset(name="bike_sharing", X=X.astype(float), y=y)


@dataclass(frozen=True)
class CorrelatedFriedman1:
    """Friedman #1 regression problem with correlated features.

    .. math:: y = 10 \\sin(\\pi x_1 x_2) + 20 (x_3 - 0.5)^2 + 10 x_4 + 5 x_5 + \\varepsilon,
              \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)

    Features ``x6`` ... ``x10`` do not enter the function. As in the original problem every feature is
    marginally :math:`U(0, 1)`, but dependence is introduced with a Gaussian copula: :math:`z \\sim
    \\mathcal{N}(0, \\Sigma)`, :math:`x_j = \\Phi(z_j)`. ``correlations`` sets the off-diagonal entries of
    :math:`\\Sigma` (the Spearman correlation of the uniforms is :math:`\\frac{6}{\\pi}\\arcsin(\\rho/2)`,
    about 0.79 for :math:`\\rho = 0.8`).

    The default correlations make two situations in which explanations can mislead: ``x1``/``x2``
    interact *and* are correlated, so partial dependence evaluates the model at combinations of low
    density (e.g. :math:`P(x_1 > 0.8, x_2 < 0.2) \\approx 0.0003` for :math:`\\rho = 0.8`, compared with
    0.04 under independence; the copula still has full support, so such points are rare, not
    impossible); ``x6`` has no effect on ``y`` but is correlated with ``x4``, so a model may attribute
    part of the ``x4`` effect to it.

    :param n_features: Total number of features (at least 5).
    :param noise: Standard deviation :math:`\\sigma` of the additive Gaussian noise.
    :param correlations: Triples ``(feature_a, feature_b, rho)`` of the latent Gaussian correlations.
    """

    n_features: int = 10
    noise: float = 1.0
    correlations: tuple[tuple[str, str, float], ...] = (("x1", "x2", 0.8), ("x4", "x6", 0.8))
    name: str = "friedman1"

    @property
    def feature_names(self) -> list[str]:
        return [f"x{j}" for j in range(1, self.n_features + 1)]

    def covariance(self) -> np.ndarray:
        """Correlation matrix :math:`\\Sigma` of the latent Gaussian."""
        names = self.feature_names
        sigma = np.eye(self.n_features)
        for a, b, rho in self.correlations:
            i, j = names.index(a), names.index(b)
            sigma[i, j] = sigma[j, i] = rho
        return sigma

    def sample_X(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        z = rng.multivariate_normal(np.zeros(self.n_features), self.covariance(), size=n)
        return pd.DataFrame(ndtr(z), columns=self.feature_names)

    def f(self, X: pd.DataFrame) -> np.ndarray:
        """The noise-free regression function :math:`E[y \\mid x]`."""
        x = {c: X[c].to_numpy(float) for c in ["x1", "x2", "x3", "x4", "x5"]}
        return 10 * np.sin(np.pi * x["x1"] * x["x2"]) + 20 * (x["x3"] - 0.5) ** 2 + 10 * x["x4"] + 5 * x["x5"]

    def sample(self, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
        X = self.sample_X(n, rng)
        y = self.f(X) + self.noise * rng.standard_normal(n)
        return X, pd.Series(y, name="y")


class RepeatedTask(ABC):
    """A dataset prepared for repeated training.

    Attributes set by subclasses:

    - ``name``: dataset name used in result tables.
    - ``X_test``, ``y_test``: fixed test set for train/test :math:`R^2`.
    - ``X_ref``: fixed reference data on which all feature effects are computed (and from which the
      grids are built), so curves differ between repetitions only because the model differs.
    - ``scale``: standard deviation of the test target with ``ddof=0``. Effect errors are reported in
      units of ``scale**2``, i.e. relative to the empirical variance of the test target, which is
      comparable across features, also defined for features whose true effect is zero, and the same
      normalization as in :math:`1 - R^2_{test}` (MSE divided by the ``ddof=0`` variance).
    - ``true_function``: noise-free regression function, or ``None`` for real data.
    """

    name: str
    X_test: pd.DataFrame
    y_test: pd.Series
    X_ref: pd.DataFrame
    scale: float
    true_function: Callable[[pd.DataFrame], np.ndarray] | None = None

    @abstractmethod
    def train_sample(self, rep: int) -> tuple[pd.DataFrame, pd.Series]:
        """Training set of repetition ``rep``. Deterministic in ``rep``."""


@dataclass
class SyntheticTask(RepeatedTask):
    """Every repetition draws a fresh, independent training set from the data generating process.

    :param generator: The data generating process.
    :param n_train: Training set size per repetition.
    :param n_test: Size of the fixed test set.
    :param n_ref: Size of the fixed reference set for feature effects.
    :param seed: Seed for test set, reference set and the per-repetition training sets.
    """

    generator: CorrelatedFriedman1
    n_train: int = 200
    n_test: int = 2000
    n_ref: int = 1000
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng([self.seed, 10**6])
        self.name = self.generator.name
        self.X_test, self.y_test = self.generator.sample(self.n_test, rng)
        self.X_ref = self.generator.sample_X(self.n_ref, rng)
        self.scale = float(self.y_test.std(ddof=0))
        self.true_function = self.generator.f

    def train_sample(self, rep: int) -> tuple[pd.DataFrame, pd.Series]:
        return self.generator.sample(self.n_train, np.random.default_rng([self.seed, rep]))


@dataclass
class SubsampleTask(RepeatedTask):
    """Real data: a fixed group-wise test split, and per repetition a subsample of the remaining pool.

    The test set (which is also the reference set for effects) is split off once by group, if the
    dataset has groups. Each repetition then draws ``n_train`` rows *without* replacement from the
    remaining pool. Unlike bootstrap samples, the subsamples contain no duplicated rows, which deep trees
    would memorize. With a pool of size :math:`N`, two subsamples share on average a fraction
    :math:`n_{train}/N` of their rows, so stability is somewhat overestimated when the pool is small.

    :param dataset: The dataset.
    :param n_train: Training set size per repetition.
    :param test_size: Fraction of groups (or rows, without groups) in the fixed test set.
    :param seed: Seed for the split and the per-repetition subsamples.
    """

    dataset: Dataset
    n_train: int = 200
    test_size: float = 0.3
    seed: int = 0

    def __post_init__(self):
        d = self.dataset
        groups = d.groups if d.groups is not None else np.arange(len(d.X))
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
        pool_idx, test_idx = next(splitter.split(d.X, d.y, groups=groups))
        if len(pool_idx) < self.n_train:
            raise ValueError(f"pool of {len(pool_idx)} rows is smaller than n_train={self.n_train}")
        self.name = d.name
        self._X_pool, self._y_pool = d.X.iloc[pool_idx], d.y.iloc[pool_idx]
        self.X_test, self.y_test = d.X.iloc[test_idx], d.y.iloc[test_idx]
        self.X_ref = self.X_test
        self.scale = float(self.y_test.std(ddof=0))
        self.true_function = None

    @property
    def pool_size(self) -> int:
        return len(self._X_pool)

    def train_sample(self, rep: int) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng([self.seed, rep])
        idx = rng.choice(self.pool_size, size=self.n_train, replace=False)
        return self._X_pool.iloc[idx], self._y_pool.iloc[idx]
