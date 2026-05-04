"""Configurazione globale: path, costanti, iperparametri."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = REPORTS_DIR / "models"

# Filenames (5 CSV forniti col PW)
EVENTS_FILE: str = "events.csv"
USERS_FILE: str = "users.csv"
PRODUCTS_FILE: str = "products.csv"
CATEGORIES_FILE: str = "categories.csv"
PROMOTIONS_FILE: str = "promotions_daily.csv"

# Schema events
EVENT_TIME_COL: str = "timestamp"
USER_ID_COL: str = "user_id"
PRODUCT_ID_COL: str = "product_id"
CATEGORY_ID_COL: str = "category_id"
PRICE_COL: str = "price"
ACTION_COL: str = "action"      # view | add_to_cart | purchase
SESSION_COL: str = "session_id"

# Costanti dominio
ACTIONS: tuple[str, ...] = ("view", "add_to_cart", "purchase")
ACTION_VIEW: str = "view"
ACTION_CART: str = "add_to_cart"
ACTION_PURCHASE: str = "purchase"

RANDOM_STATE: int = 42

# Range K per la selezione automatica via silhouette.
KMEANS_K_RANGE: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
KMEANS_K_DEFAULT: int = 5

# Finestra "recente" per le feature comportamentali (giorni).
RECENT_WINDOW_DAYS: int = 14

# Orizzonte temporale per la label futura del classificatore (giorni).
FUTURE_HORIZON_DAYS: int = 30


@dataclass(frozen=True)
class PipelineConfig:
    random_state: int = RANDOM_STATE
    recent_window_days: int = RECENT_WINDOW_DAYS
    future_horizon_days: int = FUTURE_HORIZON_DAYS
    kmeans_k_user: int | None = None       # None → autoscelta silhouette
    kmeans_k_product: int | None = None
    n_jobs: int = -1
    verbose: int = 1


DEFAULT_CONFIG: PipelineConfig = PipelineConfig()


__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "PROCESSED_DIR",
    "REPORTS_DIR", "FIGURES_DIR", "MODELS_DIR",
    "EVENTS_FILE", "USERS_FILE", "PRODUCTS_FILE",
    "CATEGORIES_FILE", "PROMOTIONS_FILE",
    "EVENT_TIME_COL", "USER_ID_COL", "PRODUCT_ID_COL", "CATEGORY_ID_COL",
    "PRICE_COL", "ACTION_COL", "SESSION_COL",
    "ACTIONS", "ACTION_VIEW", "ACTION_CART", "ACTION_PURCHASE",
    "RANDOM_STATE", "KMEANS_K_RANGE", "KMEANS_K_DEFAULT",
    "RECENT_WINDOW_DAYS", "FUTURE_HORIZON_DAYS",
    "PipelineConfig", "DEFAULT_CONFIG",
]
