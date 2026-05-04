"""RFM (Recency / Frequency / Monetary) base + arricchimenti.

Le tre feature classiche dell'analisi e-commerce:

- **Recency**: giorni dall'ultimo acquisto (alla data di riferimento).
- **Frequency**: numero di acquisti nello storico.
- **Monetary**: spesa totale nello storico.

Vengono calcolate **per utente** rispetto a una `as_of_date` (snapshot
temporale). Tutti gli eventi successivi a `as_of_date` sono ESCLUSI —
è la barriera anti-leakage del progetto.

`features.py` aggiunge gli arricchimenti richiesti dal PW (propensione
alla conversione, varietà esplorazione, sensibilità prezzo, comportamento
recente vs storico).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    ACTION_COL,
    ACTION_PURCHASE,
    EVENT_TIME_COL,
    PRICE_COL,
    PRODUCT_ID_COL,
    USER_ID_COL,
)

logger = logging.getLogger(__name__)


def compute_rfm(
    events_history: pd.DataFrame,
    as_of: pd.Timestamp,
    user_universe: pd.Series | None = None,
) -> pd.DataFrame:
    """Calcola Recency, Frequency, Monetary per ogni utente.

    Args:
        events_history: eventi filtrati con `events.timestamp < as_of`.
        as_of: data di riferimento.
        user_universe: lista degli utenti da considerare (per coprire utenti senza
            acquisti). Se None, usa solo gli utenti presenti negli eventi.

    Returns:
        DataFrame indicizzato per `user_id` con colonne:
            recency_days, frequency, monetary, has_purchased.
    """
    purchases = events_history[events_history[ACTION_COL] == ACTION_PURCHASE]

    if len(purchases) == 0:
        # Nessun acquisto storico — tutti gli utenti sono "cold".
        if user_universe is None:
            return pd.DataFrame(columns=["recency_days", "frequency", "monetary", "has_purchased"])
        df = pd.DataFrame(index=pd.Index(user_universe.unique(), name=USER_ID_COL))
        df["recency_days"] = np.nan
        df["frequency"] = 0
        df["monetary"] = 0.0
        df["has_purchased"] = 0
        return df

    grp = purchases.groupby(USER_ID_COL)
    last_purchase = grp[EVENT_TIME_COL].max()
    rfm = pd.DataFrame({
        "recency_days": (as_of - last_purchase).dt.total_seconds() / 86400.0,
        "frequency": grp.size(),
        "monetary": grp[PRICE_COL].sum(),
    })
    rfm["has_purchased"] = 1

    if user_universe is not None:
        all_users = pd.Index(user_universe.unique(), name=USER_ID_COL)
        rfm = rfm.reindex(all_users)
        rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
        rfm["monetary"] = rfm["monetary"].fillna(0.0)
        rfm["has_purchased"] = rfm["has_purchased"].fillna(0).astype(int)
        # `recency_days` resta NaN per utenti senza acquisti — gestito a downstream.

    rfm.index.name = USER_ID_COL
    logger.info(
        "RFM computati per %d utenti (di cui %d hanno acquistato).",
        len(rfm), int(rfm["has_purchased"].sum()),
    )
    return rfm


def fill_recency_for_cold_users(
    rfm: pd.DataFrame,
    fill_value: float | None = None,
) -> pd.DataFrame:
    """Imputa la recency dei cold users.

    Strategia: per gli utenti senza acquisti, settare `recency_days` al massimo
    osservato + 1 (sentinella esplicita): "non hanno mai acquistato, quindi sono
    ancora più vecchi del più vecchio noto". Questo evita NaN propagati al
    clustering, e il segnale "cold" è codificato anche da `has_purchased=0`.
    """
    out = rfm.copy()
    if fill_value is None:
        max_recency = out["recency_days"].max()
        fill_value = (max_recency + 1) if pd.notna(max_recency) else 365.0
    out["recency_days"] = out["recency_days"].fillna(fill_value)
    return out


__all__ = ["compute_rfm", "fill_recency_for_cold_users"]
