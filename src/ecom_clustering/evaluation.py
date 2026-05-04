"""Plot diagnostici per clustering e classificazione.

Diviso in due gruppi:
- Cluster: distribuzione cluster, profiling, silhouette plot.
- Classificatore: confusion matrix, importance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# --- Cluster diagnostics ---

def cluster_size_plot(
    labels: np.ndarray, title: str = "Distribuzione cluster",
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = pd.Series(labels).value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Cluster"); ax.set_ylabel("# utenti"); ax.set_title(title)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f' {v}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def cluster_profile_table(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Profilo medio di ogni cluster: media delle feature per cluster."""
    df = df_features[feature_names].copy()
    df["cluster"] = labels
    profile = df.groupby("cluster").mean()
    return profile.round(3)


def cluster_profile_heatmap(
    profile: pd.DataFrame,
    title: str = "Profilo cluster (z-score per feature)",
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap del profilo, normalizzato per feature (z-score sulle colonne)."""
    z = (profile - profile.mean()) / profile.std().replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(profile))))
    sns.heatmap(z, annot=False, cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "z-score (per colonna)"})
    ax.set_title(title); ax.set_xlabel("Feature"); ax.set_ylabel("Cluster")
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


# --- Classifier diagnostics ---

def plot_confusion_matrix(
    cm: np.ndarray | list,
    class_names: list[str] | None = None,
    title: str = "Confusion matrix",
    save_path: Path | None = None,
) -> plt.Figure:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predetto"); ax.set_ylabel("Reale"); ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def plot_feature_importance(
    df_importance: pd.DataFrame,
    title: str = "Top feature importance",
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df_importance))))
    ax.barh(df_importance["feature"][::-1], df_importance["importance"][::-1])
    ax.set_xlabel("Importance"); ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


__all__ = [
    "cluster_size_plot",
    "cluster_profile_table", "cluster_profile_heatmap",
    "plot_confusion_matrix", "plot_feature_importance",
]
