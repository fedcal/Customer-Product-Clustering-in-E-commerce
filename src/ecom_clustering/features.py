"""Feature engineering ricco per il clustering utenti.

Oltre al RFM base, il PW richiede esplicitamente:

- **Propensione alla conversione**: rapporto tra purchase e tutte le interazioni.
- **Varietà di esplorazione**: numero di categorie/prodotti distinti viewati.
- **Sensibilità al prezzo**: relazione tra prezzo medio dei prodotti viewati e
  prezzo medio dei prodotti acquistati.
- **Differenza recente vs storico**: rapporti fra metriche calcolate sulla
  finestra recente (es. ultimi 14 giorni) e l'intero storico.

Tutte le aggregazioni rispettano la `as_of_date` (no leakage).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    ACTION_CART,
    ACTION_COL,
    ACTION_PURCHASE,
    ACTION_VIEW,
    CATEGORY_ID_COL,
    EVENT_TIME_COL,
    PRICE_COL,
    PRODUCT_ID_COL,
    USER_ID_COL,
)
from .rfm import compute_rfm, fill_recency_for_cold_users

logger = logging.getLogger(__name__)


def compute_user_features(
    events_history: pd.DataFrame,
    as_of: pd.Timestamp,
    recent_window_days: int = 14,
    user_universe: pd.Series | None = None,
) -> pd.DataFrame:
    """Tutte le feature utente, calcolate alla `as_of_date`.

    Returns:
        DataFrame indicizzato per user_id con feature numeriche pronte per il clustering.
    """
    rfm = compute_rfm(events_history, as_of, user_universe=user_universe)
    rfm = fill_recency_for_cold_users(rfm)

    # Conteggi per tipo azione (intero storico).
    action_counts = events_history.pivot_table(
        index=USER_ID_COL, columns=ACTION_COL, values=PRODUCT_ID_COL,
        aggfunc="count", fill_value=0,
    )
    for a in (ACTION_VIEW, ACTION_CART, ACTION_PURCHASE):
        if a not in action_counts.columns:
            action_counts[a] = 0
    action_counts = action_counts.rename(columns=lambda c: f"n_{c}")
    rfm = rfm.join(action_counts, how="left").fillna({c: 0 for c in action_counts.columns})

    # Propensione alla conversione: purchase / (view + cart + purchase).
    interactions = rfm["n_view"] + rfm["n_add_to_cart"] + rfm["n_purchase"]
    rfm["conversion_rate"] = (rfm["n_purchase"] / interactions.replace(0, np.nan)).fillna(0.0)
    rfm["cart_to_purchase_rate"] = (
        rfm["n_purchase"] / rfm["n_add_to_cart"].replace(0, np.nan)
    ).fillna(0.0)

    # Varietà di esplorazione: numero categorie/prodotti distinti viewati.
    views = events_history[events_history[ACTION_COL] == ACTION_VIEW]
    if len(views) > 0:
        n_cat_viewed = views.groupby(USER_ID_COL)[CATEGORY_ID_COL].nunique()
        n_prod_viewed = views.groupby(USER_ID_COL)[PRODUCT_ID_COL].nunique()
        rfm = rfm.join(n_cat_viewed.rename("n_categories_viewed"), how="left")
        rfm = rfm.join(n_prod_viewed.rename("n_products_viewed"), how="left")
    else:
        rfm["n_categories_viewed"] = 0
        rfm["n_products_viewed"] = 0
    rfm[["n_categories_viewed", "n_products_viewed"]] = rfm[
        ["n_categories_viewed", "n_products_viewed"]
    ].fillna(0).astype(int)

    # Sensibilità al prezzo: prezzo medio viewato vs acquistato.
    avg_price_viewed = views.groupby(USER_ID_COL)[PRICE_COL].mean()
    purchases = events_history[events_history[ACTION_COL] == ACTION_PURCHASE]
    avg_price_purchased = purchases.groupby(USER_ID_COL)[PRICE_COL].mean()
    rfm["avg_price_viewed"] = rfm.index.map(avg_price_viewed).astype(float)
    rfm["avg_price_purchased"] = rfm.index.map(avg_price_purchased).astype(float)

    # `price_sensitivity_ratio` < 1 se l'utente acquista in media a prezzi più bassi
    # di quelli che guarda (cerca occasioni); > 1 se acquista i più cari (premium).
    rfm["price_sensitivity_ratio"] = (
        rfm["avg_price_purchased"] / rfm["avg_price_viewed"].replace(0, np.nan)
    )
    rfm[["avg_price_viewed", "avg_price_purchased", "price_sensitivity_ratio"]] = rfm[
        ["avg_price_viewed", "avg_price_purchased", "price_sensitivity_ratio"]
    ].fillna(0.0)

    # Comportamento recente vs storico.
    recent_start = as_of - pd.Timedelta(days=recent_window_days)
    recent_events = events_history[events_history[EVENT_TIME_COL] >= recent_start]
    if len(recent_events) > 0:
        recent_interactions = recent_events.groupby(USER_ID_COL).size().rename("n_recent_interactions")
        rfm = rfm.join(recent_interactions, how="left")
        recent_purchases = (
            recent_events[recent_events[ACTION_COL] == ACTION_PURCHASE]
            .groupby(USER_ID_COL).size().rename("n_recent_purchases")
        )
        rfm = rfm.join(recent_purchases, how="left")
    else:
        rfm["n_recent_interactions"] = 0
        rfm["n_recent_purchases"] = 0

    rfm[["n_recent_interactions", "n_recent_purchases"]] = rfm[
        ["n_recent_interactions", "n_recent_purchases"]
    ].fillna(0).astype(int)

    # Rapporto attività recente / storica: > 1 = utente in crescita; < 1 = in calo.
    total_interactions = (rfm["n_view"] + rfm["n_add_to_cart"] + rfm["n_purchase"]).replace(0, np.nan)
    rfm["recent_to_total_ratio"] = (rfm["n_recent_interactions"] / total_interactions).fillna(0.0)
    # Clip a 1.0: non può essere > 1 per definizione, ma errori float possono dare 1.001.
    rfm["recent_to_total_ratio"] = rfm["recent_to_total_ratio"].clip(0.0, 1.0)

    logger.info("compute_user_features: %d utenti, %d feature.", len(rfm), rfm.shape[1])
    return rfm


def select_user_modeling_features(df: pd.DataFrame) -> list[str]:
    """Colonne numeriche da usare nel clustering (esclude flag binarie sole)."""
    return [
        "recency_days", "frequency", "monetary",
        "n_view", "n_add_to_cart", "n_purchase",
        "conversion_rate", "cart_to_purchase_rate",
        "n_categories_viewed", "n_products_viewed",
        "avg_price_viewed", "avg_price_purchased", "price_sensitivity_ratio",
        "n_recent_interactions", "n_recent_purchases", "recent_to_total_ratio",
    ]


# ----- Feature prodotti (per cluster prodotto) -----


def compute_product_features(
    events_history: pd.DataFrame,
    products: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Feature per il clustering prodotti, calcolate alla `as_of_date`."""
    pid = "product_id"
    counts = events_history.pivot_table(
        index=pid, columns=ACTION_COL, values=USER_ID_COL,
        aggfunc="count", fill_value=0,
    ).rename(columns=lambda c: f"n_{c}")

    for c in ("n_view", "n_add_to_cart", "n_purchase"):
        if c not in counts.columns:
            counts[c] = 0

    interactions = counts["n_view"] + counts["n_add_to_cart"] + counts["n_purchase"]
    counts["conversion_rate"] = (
        counts["n_purchase"] / interactions.replace(0, np.nan)
    ).fillna(0.0)

    avg_price = events_history.groupby(pid)[PRICE_COL].mean().rename("avg_price")
    n_unique_users = events_history.groupby(pid)[USER_ID_COL].nunique().rename("n_unique_users")

    out = counts.join(avg_price, how="left").join(n_unique_users, how="left")

    # Allinea con il catalogo prodotti (alcuni prodotti possono non avere eventi).
    out = products.set_index(pid).join(out, how="left").fillna({
        "n_view": 0, "n_add_to_cart": 0, "n_purchase": 0,
        "conversion_rate": 0.0, "avg_price": 0.0, "n_unique_users": 0,
    })
    out["n_unique_users"] = out["n_unique_users"].astype(int)
    return out.reset_index()


def select_product_modeling_features(df: pd.DataFrame) -> list[str]:
    return [
        "n_view", "n_add_to_cart", "n_purchase",
        "conversion_rate", "avg_price", "n_unique_users",
        "base_price", "quality_score", "base_conversion", "base_popularity",
    ]


__all__ = [
    "compute_user_features",
    "select_user_modeling_features",
    "compute_product_features",
    "select_product_modeling_features",
]
