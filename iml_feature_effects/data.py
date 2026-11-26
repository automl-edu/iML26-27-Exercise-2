"""Dataset loading.

Every dataset is returned as a :class:`Dataset` with numeric, imputed features so that all
models in :mod:`iml_feature_effects.models` can be fit on it directly. Add further loaders
(e.g. other OpenML or Kaggle datasets) following the pattern of :func:`load_bike_sharing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.datasets import fetch_openml

# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")
PGA_URL = "https://raw.githubusercontent.com/jongmp/PGA-Tour/4700e10c4c240839561a93f0aaaef94f8c10dd19/pgaTourData.csv"
WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
PGA_FEATURES = ["Avg Distance", "Fairway Percentage", "gir", "Average Putts", "Average Scrambling"]


@dataclass(frozen=True)
class Dataset:
    """A preprocessed tabular regression dataset.

    :param name: Human-readable dataset name, used in result tables and plots.
    :param X: Numeric feature matrix without missing values.
    :param y: Target vector.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def load_bike_sharing(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the hourly Bike Sharing Demand dataset from OpenML (data id 42712).

    The target is the number of rented bikes per hour. The download is cached in ``data_home``, so
    network access is only needed on the first call.

    :param data_home: Cache directory for the download, relative paths are resolved against the
        current working directory.
    :return: The preprocessed Bike Sharing dataset.
    :rtype: Dataset
    """
    bunch = fetch_openml(data_id=42712, as_frame=True, parser="liac-arff", data_home=data_home)
    # This OpenML version already excludes ``casual`` and ``registered``, which sum to the target.
    X = pd.get_dummies(bunch.data, drop_first=True, dtype=float)
    y = bunch.target.astype(float).rename("count")
    return Dataset(name="bike_sharing", X=X.astype(float), y=y)


def _load_csv(path: Path, url: str, sep: str = ",") -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    data = pd.read_csv(url, sep=sep)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp_dir:
        temp_path = Path(temp_dir) / path.name
        data.to_csv(temp_path, index=False)
        temp_path.replace(path)
    return data


def load_pga_tour(data_home: Path = DATA_HOME) -> Dataset:
    data = _load_csv(data_home / "pga_tour.csv", PGA_URL)
    target = "Average Score"
    stats = data[[*PGA_FEATURES, target]].apply(pd.to_numeric, errors="coerce")
    stats = stats.assign(player=data["Player Name"]).dropna()
    return Dataset(name="pga_tour", X=stats[PGA_FEATURES], y=stats[target], groups=stats["player"])


def load_wine_quality(data_home: Path = DATA_HOME) -> Dataset:
    data = _load_csv(data_home / "wine_quality_red.csv", WINE_URL, sep=";")
    features = data.drop(columns="quality")
    groups = features.groupby(list(features.columns), sort=False).ngroup()
    return Dataset(name="wine_quality_red", X=features, y=data["quality"], groups=groups)
