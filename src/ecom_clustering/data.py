"""Caricamento dei 5 CSV del dataset e-commerce.

I dati sono organizzati come:
- events.csv:    500k eventi (timestamp, user_id, product_id, category_id, price, action, ...).
- users.csv:     1200 utenti con attributi statici.
- products.csv:  3500 prodotti.
- categories.csv: 15 categorie.
- promotions_daily.csv: sconti giornalieri per categoria.

Tutto qui è I/O. Le aggregazioni e il calcolo feature stanno in `rfm.py`/`features.py`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    CATEGORIES_FILE,
    EVENTS_FILE,
    EVENT_TIME_COL,
    PRODUCTS_FILE,
    PROMOTIONS_FILE,
    RAW_DIR,
    USERS_FILE,
)

logger = logging.getLogger(__name__)


@dataclass
class EcomData:
    """Container pulito per i 5 DataFrame caricati."""
    events: pd.DataFrame
    users: pd.DataFrame
    products: pd.DataFrame
    categories: pd.DataFrame
    promotions: pd.DataFrame

    @property
    def time_min(self) -> pd.Timestamp:
        return self.events[EVENT_TIME_COL].min()

    @property
    def time_max(self) -> pd.Timestamp:
        return self.events[EVENT_TIME_COL].max()

    def summary(self) -> str:
        return (
            f"events={len(self.events):,} | users={len(self.users):,} | "
            f"products={len(self.products):,} | categories={len(self.categories)} | "
            f"promotions={len(self.promotions)} giorni\n"
            f"period: {self.time_min} → {self.time_max}"
        )


def load_raw(raw_dir: Path = RAW_DIR) -> EcomData:
    """Carica tutti i CSV in DataFrame validati e ordinati."""
    events_path = raw_dir / EVENTS_FILE
    if not events_path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {events_path}. Atteso `events.csv` in `data/raw/`."
        )

    events = pd.read_csv(events_path, parse_dates=[EVENT_TIME_COL])
    events = events.sort_values(EVENT_TIME_COL).reset_index(drop=True)

    users = pd.read_csv(raw_dir / USERS_FILE)
    products = pd.read_csv(raw_dir / PRODUCTS_FILE)
    categories = pd.read_csv(raw_dir / CATEGORIES_FILE)
    promotions = pd.read_csv(raw_dir / PROMOTIONS_FILE, parse_dates=["date"])

    logger.info(
        "Caricati: events=%d, users=%d, products=%d, categories=%d, promotions=%d.",
        len(events), len(users), len(products), len(categories), len(promotions),
    )
    return EcomData(events, users, products, categories, promotions)


def filter_events_until(
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    inclusive: bool = False,
) -> pd.DataFrame:
    """Restituisce solo gli eventi precedenti a `as_of`.

    `inclusive=False` di default: non includere il timestamp stesso, evita ambiguità.

    Punto critico anti-leakage: questo filtro è il "barrier temporale" — tutto ciò
    che le feature aggregate calcolano dopo questo step usa SOLO eventi storici.
    """
    op = events[EVENT_TIME_COL] <= as_of if inclusive else events[EVENT_TIME_COL] < as_of
    return events.loc[op].copy()


def filter_events_between(
    events: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> pd.DataFrame:
    """Eventi in [t_start, t_end). Usato per la finestra futura della label."""
    mask = (events[EVENT_TIME_COL] >= t_start) & (events[EVENT_TIME_COL] < t_end)
    return events.loc[mask].copy()


__all__ = [
    "EcomData", "load_raw",
    "filter_events_until", "filter_events_between",
]
