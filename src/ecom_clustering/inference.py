"""Inferenza: dato un utente e i suoi eventi storici, predici il cluster atteso."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import MODELS_DIR

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH: Path = MODELS_DIR / "user_pipeline.joblib"


@lru_cache(maxsize=4)
def _load_artifacts(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {path}. Esegui prima `ecom-cluster`."
        )
    return joblib.load(path)


def predict_user_cluster(
    user_features_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.Series:
    """Predice il cluster di appartenenza per ogni utente nel DataFrame.

    `user_features_df` deve avere le colonne in `select_user_modeling_features`.
    Returns: Serie con cluster id, indicizzata per user_id.
    """
    art = _load_artifacts(str(model_path))
    cluster_model = art["cluster_model"]
    scaler = art["scaler"]
    feat_cols: list[str] = art["feature_columns"]

    X = scaler.transform(user_features_df[feat_cols].to_numpy())
    labels = cluster_model.predict(X)
    return pd.Series(labels, index=user_features_df.index, name="cluster")


def predict_future_cluster_supervised(
    user_features_df: pd.DataFrame,
    classifier_path: Path = MODELS_DIR / "future_cluster_classifier.joblib",
) -> pd.Series:
    """Predice il cluster futuro (orizzonte fissato in training) usando il classificatore."""
    art = _load_artifacts(str(classifier_path))
    classifier = art["classifier"]
    scaler = art["scaler"]
    feat_cols: list[str] = art["feature_columns"]
    X = scaler.transform(user_features_df[feat_cols].to_numpy())
    return pd.Series(classifier.predict(X), index=user_features_df.index, name="future_cluster")


__all__ = ["predict_user_cluster", "predict_future_cluster_supervised", "DEFAULT_MODEL_PATH"]
