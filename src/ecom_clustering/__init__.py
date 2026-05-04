"""Pipeline ML per clustering utenti/prodotti e-commerce + classificatore supervisionato.

API pubbliche principali:

    from ecom_clustering.pipeline import run_full_pipeline
    from ecom_clustering.inference import predict_user_cluster
    from ecom_clustering.data import load_raw

Vedi docs/ per dettagli teorici e architetturali.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
