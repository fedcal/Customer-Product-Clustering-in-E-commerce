"""Orchestratore end-to-end: clustering utenti + clustering prodotti + classificatore futuro.

Esegue:

1. Caricamento 5 CSV.
2. Definizione `as_of_train` (snapshot per training del classificatore) e
   `as_of_test` (snapshot per il test). Tipicamente: train @ T-2H, test @ T-H,
   con H = `future_horizon_days`.
3. Calcolo feature utente e prodotto sui due snapshot.
4. Training del clustering KMeans sul `as_of_train`.
5. Generazione label future (cluster a `as_of_train + H`) → train classifier.
6. Generazione label per il test (cluster a `as_of_test + H`) → valutazione.
7. Persistenza artefatti.

Eseguibile come modulo:

    ecom-cluster              # full run
    ecom-cluster --quick      # K=5 fisso, no GMM
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .clustering import compare_kmeans_vs_gmm, fit_kmeans, select_k
from .config import (
    DEFAULT_CONFIG,
    FIGURES_DIR,
    KMEANS_K_DEFAULT,
    KMEANS_K_RANGE,
    MODELS_DIR,
    REPORTS_DIR,
    PipelineConfig,
)
from .data import EcomData, filter_events_until, load_raw
from .evaluation import (
    cluster_profile_heatmap,
    cluster_profile_table,
    cluster_size_plot,
    plot_confusion_matrix,
    plot_feature_importance,
)
from .features import (
    compute_product_features,
    compute_user_features,
    select_product_modeling_features,
    select_user_modeling_features,
)
from .labeling import make_future_user_clusters
from .supervised import (
    cross_val_macro_f1,
    evaluate_classifier,
    feature_importances,
    fit_random_forest,
    fit_xgboost,
)

logger = logging.getLogger(__name__)


def _pick_snapshots(ecom: EcomData, horizon: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Sceglie due snapshot temporali separati di `horizon` giorni.

    `as_of_train`: snapshot per training del classifier (= time_max - 2H).
    `as_of_test`:  snapshot per evaluation     (= time_max - H).

    Le label future (cluster a snapshot + H) cadono entrambe entro `time_max`.
    """
    t_max = ecom.time_max
    as_of_test = t_max - pd.Timedelta(days=horizon)
    as_of_train = as_of_test - pd.Timedelta(days=horizon)
    return as_of_train, as_of_test


def run_full_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
    quick: bool = False,
) -> dict:
    logger.info("=" * 70)
    logger.info("Pipeline ecommerce clustering (quick=%s)", quick)
    logger.info("=" * 70)

    # --- 1. Load ---
    ecom = load_raw()
    logger.info(ecom.summary())

    # --- 2. Snapshots ---
    as_of_train, as_of_test = _pick_snapshots(ecom, config.future_horizon_days)
    logger.info("Snapshots: train=%s, test=%s, horizon=%dd",
                as_of_train, as_of_test, config.future_horizon_days)

    # --- 3. Feature utenti @ as_of_train ---
    events_train_history = filter_events_until(ecom.events, as_of_train)
    user_feats_train = compute_user_features(
        events_train_history, as_of_train,
        recent_window_days=config.recent_window_days,
        user_universe=ecom.users["user_id"],
    )
    user_cols = select_user_modeling_features(user_feats_train)
    X_user_train = user_feats_train[user_cols].to_numpy()

    # --- 4. Scaling + clustering utenti ---
    user_scaler = StandardScaler().fit(X_user_train)
    X_user_train_s = user_scaler.transform(X_user_train)

    if config.kmeans_k_user is not None:
        best_k_user = config.kmeans_k_user
    elif quick:
        best_k_user = KMEANS_K_DEFAULT
        logger.info("Quick: K_user=%d default", best_k_user)
    else:
        logger.info("Selezione K_user via silhouette...")
        best_k_user, _ = select_k(X_user_train_s, k_range=KMEANS_K_RANGE,
                                   random_state=config.random_state)

    cluster_user = fit_kmeans(X_user_train_s, k=best_k_user, random_state=config.random_state)
    logger.info("[USER] %s → silhouette=%.4f, inertia=%.0f",
                cluster_user.name, cluster_user.silhouette, cluster_user.inertia)

    # Confronto KMeans vs GMM (skip in quick).
    if not quick:
        cmp_df = compare_kmeans_vs_gmm(X_user_train_s, k=best_k_user,
                                        random_state=config.random_state)
        logger.info("\n[USER] confronto KMeans vs GMM:\n%s", cmp_df.to_string(index=False))

    # --- 5. Feature prodotti @ as_of_train + clustering ---
    product_feats = compute_product_features(events_train_history, ecom.products, as_of_train)
    prod_cols = select_product_modeling_features(product_feats)
    X_prod = product_feats[prod_cols].fillna(0.0).to_numpy()
    prod_scaler = StandardScaler().fit(X_prod)
    X_prod_s = prod_scaler.transform(X_prod)

    if config.kmeans_k_product is not None:
        best_k_prod = config.kmeans_k_product
    elif quick:
        best_k_prod = 4
    else:
        best_k_prod, _ = select_k(X_prod_s, k_range=KMEANS_K_RANGE,
                                   random_state=config.random_state)

    cluster_prod = fit_kmeans(X_prod_s, k=best_k_prod, random_state=config.random_state)
    logger.info("[PROD] %s → silhouette=%.4f", cluster_prod.name, cluster_prod.silhouette)

    # --- 6. Future labels (cluster a as_of_train + H) per training del classifier ---
    y_train = make_future_user_clusters(
        ecom, as_of=as_of_train, horizon_days=config.future_horizon_days,
        cluster_model=cluster_user.model, scaler=user_scaler,
        user_universe=ecom.users["user_id"],
    )
    # Allinea train features ↔ train labels su user_id comuni.
    common_train_users = user_feats_train.index.intersection(y_train.index)
    X_train_clf = user_scaler.transform(user_feats_train.loc[common_train_users, user_cols].to_numpy())
    y_train_clf = y_train.loc[common_train_users].to_numpy()

    # Snapshot test: feature @ as_of_test + label @ as_of_test+H.
    events_test_history = filter_events_until(ecom.events, as_of_test)
    user_feats_test = compute_user_features(
        events_test_history, as_of_test,
        recent_window_days=config.recent_window_days,
        user_universe=ecom.users["user_id"],
    )
    y_test = make_future_user_clusters(
        ecom, as_of=as_of_test, horizon_days=config.future_horizon_days,
        cluster_model=cluster_user.model, scaler=user_scaler,
        user_universe=ecom.users["user_id"],
    )
    common_test_users = user_feats_test.index.intersection(y_test.index)
    X_test_clf = user_scaler.transform(user_feats_test.loc[common_test_users, user_cols].to_numpy())
    y_test_clf = y_test.loc[common_test_users].to_numpy()

    # --- 7. Train classifier (RF + XGB) ---
    n_classes = int(max(y_train_clf.max(), y_test_clf.max()) + 1)
    rf = fit_random_forest(X_train_clf, y_train_clf, random_state=config.random_state)
    xgb = fit_xgboost(X_train_clf, y_train_clf, n_classes=n_classes,
                       random_state=config.random_state)

    rf_eval = evaluate_classifier(rf, X_test_clf, y_test_clf)
    xgb_eval = evaluate_classifier(xgb, X_test_clf, y_test_clf)

    logger.info("\n[CLF] RandomForest test: macro_F1=%.3f, balanced_acc=%.3f, acc=%.3f",
                rf_eval["macro_f1"], rf_eval["balanced_accuracy"], rf_eval["accuracy"])
    logger.info("[CLF] XGBoost      test: macro_F1=%.3f, balanced_acc=%.3f, acc=%.3f",
                xgb_eval["macro_f1"], xgb_eval["balanced_accuracy"], xgb_eval["accuracy"])

    best_classifier = xgb if xgb_eval["macro_f1"] >= rf_eval["macro_f1"] else rf
    best_name = "XGBoost" if best_classifier is xgb else "RandomForest"
    best_eval = xgb_eval if best_classifier is xgb else rf_eval
    logger.info(">>> Miglior classificatore: %s (macro_F1=%.3f)", best_name, best_eval["macro_f1"])

    # --- 8. Plot diagnostici (best-effort) ---
    try:
        cluster_size_plot(cluster_user.labels, title="Distribuzione cluster utenti",
                          save_path=FIGURES_DIR / "user_cluster_size.png")
        profile = cluster_profile_table(user_feats_train, cluster_user.labels, user_cols)
        cluster_profile_heatmap(profile, save_path=FIGURES_DIR / "user_cluster_profile.png")
        plot_confusion_matrix(np.array(best_eval["confusion_matrix"]),
                               class_names=[f"C{i}" for i in range(n_classes)],
                               title=f"{best_name}: confusion matrix (test)",
                               save_path=FIGURES_DIR / f"{best_name.lower()}_confmat.png")
        imp = feature_importances(best_classifier, user_cols, top_n=15)
        plot_feature_importance(imp, save_path=FIGURES_DIR / f"{best_name.lower()}_importance.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plot saltati: %s", exc)

    # --- 9. Persist artefatti ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    user_pipeline_path = MODELS_DIR / "user_pipeline.joblib"
    joblib.dump({
        "cluster_model": cluster_user.model,
        "scaler": user_scaler,
        "feature_columns": user_cols,
        "k": best_k_user,
        "as_of": str(as_of_train),
    }, user_pipeline_path)

    prod_pipeline_path = MODELS_DIR / "product_pipeline.joblib"
    joblib.dump({
        "cluster_model": cluster_prod.model,
        "scaler": prod_scaler,
        "feature_columns": prod_cols,
        "k": best_k_prod,
        "as_of": str(as_of_train),
    }, prod_pipeline_path)

    classifier_path = MODELS_DIR / "future_cluster_classifier.joblib"
    joblib.dump({
        "classifier": best_classifier,
        "scaler": user_scaler,
        "feature_columns": user_cols,
        "horizon_days": config.future_horizon_days,
        "n_classes": n_classes,
        "name": best_name,
    }, classifier_path)

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({
        "as_of_train": str(as_of_train),
        "as_of_test": str(as_of_test),
        "horizon_days": config.future_horizon_days,
        "user_clustering": {
            "k": best_k_user,
            "silhouette": cluster_user.silhouette,
            "inertia": cluster_user.inertia,
        },
        "product_clustering": {
            "k": best_k_prod,
            "silhouette": cluster_prod.silhouette,
            "inertia": cluster_prod.inertia,
        },
        "classifier_test": {
            "best_model": best_name,
            "rf": _slim_metrics(rf_eval),
            "xgb": _slim_metrics(xgb_eval),
        },
    }, indent=2, default=float))

    return {
        "best_k_user": best_k_user,
        "best_k_product": best_k_prod,
        "user_cluster_result": cluster_user,
        "product_cluster_result": cluster_prod,
        "rf": rf, "xgb": xgb, "best_classifier_name": best_name,
        "rf_eval": rf_eval, "xgb_eval": xgb_eval,
        "user_feats_train": user_feats_train,
        "user_feats_test": user_feats_test,
    }


def _slim_metrics(d: dict) -> dict:
    return {k: d[k] for k in ("macro_f1", "weighted_f1", "balanced_accuracy", "accuracy")}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E-commerce clustering pipeline.")
    p.add_argument("--quick", action="store_true", help="Smoke test (K=5, no GMM).")
    p.add_argument("--horizon", type=int, default=DEFAULT_CONFIG.future_horizon_days,
                   help="Orizzonte label futura in giorni (default 30).")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stdout)
    args = _build_arg_parser().parse_args()
    cfg = PipelineConfig(
        random_state=DEFAULT_CONFIG.random_state,
        recent_window_days=DEFAULT_CONFIG.recent_window_days,
        future_horizon_days=args.horizon,
        kmeans_k_user=DEFAULT_CONFIG.kmeans_k_user,
        kmeans_k_product=DEFAULT_CONFIG.kmeans_k_product,
        n_jobs=DEFAULT_CONFIG.n_jobs, verbose=DEFAULT_CONFIG.verbose,
    )
    run_full_pipeline(config=cfg, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_full_pipeline"]
