"""Classificatore supervisionato: predire il cluster futuro di un utente.

Definizione del problema:
- Input: feature dell'utente alla data `as_of` (calcolate su eventi storici).
- Output: cluster di appartenenza dell'utente alla data `as_of + horizon_days`.

È un problema di classificazione **multiclasse** dove il numero di classi è
il `K` scelto per il clustering. Le classi sono squilibrate (alcuni cluster
contengono più utenti di altri).

Modelli candidati:
- **RandomForest**: baseline non lineare, robusto.
- **XGBoost**: gradient boosting, tipicamente migliore performance.

Metriche multiclasse: macro-F1, balanced accuracy, confusion matrix.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from .config import RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class SupervisedResult:
    name: str
    model: object
    train_metrics: dict
    test_metrics: dict
    feature_importances: pd.DataFrame


def fit_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2,
        max_features="sqrt", n_jobs=-1, random_state=random_state,
        class_weight="balanced",  # gestisce squilibrio classi
    )
    model.fit(X_train, y_train)
    return model


def fit_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    n_classes: int,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=n_classes,
        random_state=random_state, n_jobs=-1, tree_method="hist",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model, X: np.ndarray, y: np.ndarray,
) -> dict:
    """Macro-F1, balanced accuracy, classification report, confusion matrix."""
    y_pred = model.predict(X)
    return {
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "accuracy": float((y_pred == y).mean()),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(
            y, y_pred, zero_division=0, output_dict=True,
        ),
    }


def cross_val_macro_f1(
    model, X: np.ndarray, y: np.ndarray, cv: int = 5,
) -> tuple[float, float]:
    """K-fold CV macro-F1 (mean ± std)."""
    scores = cross_val_score(model, X, y, scoring="f1_macro", cv=cv, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def feature_importances(model, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return pd.DataFrame(columns=["feature", "importance"])


__all__ = [
    "SupervisedResult",
    "fit_random_forest", "fit_xgboost",
    "evaluate_classifier", "cross_val_macro_f1", "feature_importances",
]
