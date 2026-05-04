"""Clustering di utenti/prodotti.

KMeans + GaussianMixture come da PW (almeno due approcci confrontati).
La selezione di K avviene via silhouette score sul subset campionato.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .config import KMEANS_K_RANGE, RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    name: str
    model: object
    k: int
    silhouette: float
    inertia: float | None
    labels: np.ndarray


def fit_kmeans(X: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> ClusteringResult:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=500)
    labels = model.fit_predict(X)
    sil = _sample_silhouette(X, labels, random_state=random_state)
    return ClusteringResult(
        name=f"KMeans(k={k})", model=model, k=k,
        silhouette=sil, inertia=float(model.inertia_), labels=labels,
    )


def fit_gmm(X: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> ClusteringResult:
    model = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=random_state, n_init=3, max_iter=200)
    labels = model.fit_predict(X)
    sil = _sample_silhouette(X, labels, random_state=random_state)
    return ClusteringResult(
        name=f"GMM(k={k})", model=model, k=k,
        silhouette=sil, inertia=None, labels=labels,
    )


def select_k(
    X: np.ndarray,
    k_range: tuple[int, ...] = KMEANS_K_RANGE,
    fit_fn=fit_kmeans,
    random_state: int = RANDOM_STATE,
) -> tuple[int, list[ClusteringResult]]:
    results: list[ClusteringResult] = []
    for k in k_range:
        r = fit_fn(X, k=k, random_state=random_state)
        logger.info("  k=%d → silhouette=%.4f, inertia=%s", k, r.silhouette, r.inertia)
        results.append(r)
    best = max(results, key=lambda r: r.silhouette)
    return best.k, results


def compare_kmeans_vs_gmm(
    X: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Confronta KMeans e GMM con stesso K. Output: DataFrame con metriche."""
    rk = fit_kmeans(X, k=k, random_state=random_state)
    rg = fit_gmm(X, k=k, random_state=random_state)
    return pd.DataFrame([
        {"model": rk.name, "silhouette": rk.silhouette, "inertia": rk.inertia},
        {"model": rg.name, "silhouette": rg.silhouette, "inertia": rg.inertia},
    ])


def _sample_silhouette(
    X: np.ndarray, labels: np.ndarray, sample_size: int = 5000,
    random_state: int = RANDOM_STATE,
) -> float:
    n = len(X)
    if n <= sample_size:
        return float(silhouette_score(X, labels))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=sample_size, replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


__all__ = ["ClusteringResult", "fit_kmeans", "fit_gmm", "select_k", "compare_kmeans_vs_gmm"]
