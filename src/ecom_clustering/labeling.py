"""Costruzione delle label future per il classificatore supervisionato.

**Definizione critica**: la label per l'utente `u` osservata alla data
`as_of_date` è il **cluster di appartenenza al tempo `as_of_date + horizon`**.

In altre parole:
1. Calcoliamo le feature di `u` con eventi `< as_of_date`.
2. Calcoliamo le feature di `u` con eventi `< as_of_date + horizon`.
3. Applichiamo il modello di clustering (fittato sul training) alle feature (2).
4. La label è il cluster ottenuto al punto (3).

**Anti-leakage**: le feature di training (1) sono basate sui dati storici;
la label (3) è basata su dati futuri. Niente sovrapposizione.

In produzione: alleniamo il classificatore sul training (snapshot a t0,
label a t0+H). Lo testiamo su un nuovo snapshot (t1, label a t1+H).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    EVENT_TIME_COL,
    USER_ID_COL,
)
from .data import EcomData, filter_events_until
from .features import compute_user_features, select_user_modeling_features

logger = logging.getLogger(__name__)


def make_future_user_clusters(
    ecom: EcomData,
    as_of: pd.Timestamp,
    horizon_days: int,
    cluster_model,
    scaler,
    user_universe: pd.Series | None = None,
) -> pd.Series:
    """Calcola, per ogni utente, il cluster di appartenenza alla data `as_of + horizon`.

    Args:
        ecom: dataset completo.
        as_of: snapshot di riferimento per l'osservazione del classificatore.
        horizon_days: orizzonte in giorni.
        cluster_model: modello di clustering già fittato (KMeans/GMM).
        scaler: scaler già fittato.
        user_universe: utenti su cui calcolare la label.

    Returns:
        Serie indicizzata per `user_id` con il cluster (intero) come valore.
    """
    future_t = as_of + pd.Timedelta(days=horizon_days)
    if future_t > ecom.time_max:
        logger.warning(
            "horizon=%dd supera la fine del dataset (%s); useremo tutti gli eventi disponibili.",
            horizon_days, ecom.time_max,
        )
    events_future = filter_events_until(ecom.events, future_t)
    feats_future = compute_user_features(events_future, future_t, user_universe=user_universe)
    cols = select_user_modeling_features(feats_future)
    X = scaler.transform(feats_future[cols].to_numpy())
    labels = cluster_model.predict(X)
    return pd.Series(labels, index=feats_future.index, name=f"future_cluster_{horizon_days}d")


__all__ = ["make_future_user_clusters"]
