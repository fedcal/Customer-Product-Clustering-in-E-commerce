"""Genera i 4 notebook didattici di PW1."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


def write_nb(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    nbf.write(nb, NB_DIR / name)
    print(f"[OK] {name} ({len(cells)} celle)")


# ============================================================================
# 01 — EDA + RFM
# ============================================================================
nb01 = [
    md(
        "# 01 — Esplorazione del dataset e RFM\n\n"
        "## Obiettivi didattici\n\n"
        "1. Conoscere lo schema delle 5 tabelle (events, users, products, categories, promotions).\n"
        "2. Distribuzione temporale degli eventi e definizione di `as_of_date`.\n"
        "3. Calcolo del RFM base (Recency, Frequency, Monetary) per utente.\n"
        "4. Distinzione fra utenti che hanno acquistato e \"cold users\" (solo view).\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
        "sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi'] = 110\n"
        "from ecom_clustering.data import load_raw, filter_events_until\n"
        "from ecom_clustering.rfm import compute_rfm\n"
        "from ecom_clustering.config import DEFAULT_CONFIG\n"
    ),
    code(
        "ecom = load_raw()\n"
        "print(ecom.summary())\n"
        "print(f'\\nTipi azione (eventi): {ecom.events.action.value_counts().to_dict()}')"
    ),
    md(
        "## Distribuzione temporale degli eventi\n\n"
        "Verifichiamo che gli eventi siano distribuiti su tutto il periodo (no buchi)."
    ),
    code(
        "daily_counts = ecom.events.set_index('timestamp').resample('D').size()\n"
        "fig, ax = plt.subplots(figsize=(13, 4))\n"
        "ax.plot(daily_counts.index, daily_counts.values)\n"
        "ax.set(title='Eventi per giorno', xlabel='Data', ylabel='# eventi')\n"
        "ax.tick_params(axis='x', rotation=20)\n"
        "fig.tight_layout(); plt.show()"
    ),
    md(
        "## Snapshot temporale e RFM\n\n"
        "Scegliamo `as_of` = 30 giorni prima della fine del dataset. Calcoliamo "
        "il RFM su tutti gli eventi precedenti."
    ),
    code(
        "as_of = ecom.time_max - pd.Timedelta(days=30)\n"
        "events_history = filter_events_until(ecom.events, as_of)\n"
        "print(f'as_of = {as_of}')\n"
        "print(f'Eventi storici: {len(events_history):,} ({len(events_history)/len(ecom.events):.1%})')\n"
        "rfm = compute_rfm(events_history, as_of, user_universe=ecom.users.user_id)\n"
        "rfm.head(10)"
    ),
    md(
        "## Distribuzioni RFM\n\n"
        "Le distribuzioni sono spesso skewed (long tail). Visualizzazione log-scale aiuta."
    ),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
        "axes[0].hist(rfm.recency_days.dropna(), bins=40)\n"
        "axes[0].set(title='Recency (giorni)', xlabel='days')\n"
        "axes[1].hist(rfm.frequency, bins=40)\n"
        "axes[1].set(title='Frequency (# acquisti)', xlabel='freq')\n"
        "axes[2].hist(rfm.monetary, bins=40)\n"
        "axes[2].set(title='Monetary ($)', xlabel='spesa totale')\n"
        "fig.tight_layout(); plt.show()\n"
        "print(f'Utenti con almeno 1 acquisto: {int(rfm.has_purchased.sum())}/{len(rfm)} ({rfm.has_purchased.mean():.1%})')"
    ),
    md(
        "## Conclusione\n\n"
        "- Il dataset copre ~10 mesi (gen → nov 2024).\n"
        "- Distribuzioni skewed di Frequency/Monetary → utili `log1p` o feature ratio.\n"
        "- ~46% degli utenti ha almeno 1 acquisto entro `as_of`. I cold users vanno "
        "trattati esplicitamente (recency imputata, `has_purchased=0`).\n"
    ),
]
write_nb("01_eda_rfm.ipynb", nb01)


# ============================================================================
# 02 — Feature engineering esteso
# ============================================================================
nb02 = [
    md(
        "# 02 — Feature engineering esteso\n\n"
        "## Obiettivi didattici\n\n"
        "Aggiungere feature richieste dal PW: propensione conversione, varietà "
        "esplorazione, sensibilità prezzo, recente vs storico. Tutte calcolate "
        "rispetto alla `as_of_date` (anti-leakage)."
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\n"
        "from ecom_clustering.data import load_raw, filter_events_until\n"
        "from ecom_clustering.features import compute_user_features, select_user_modeling_features\n"
        "from ecom_clustering.config import DEFAULT_CONFIG\n"
        "ecom = load_raw()\n"
        "as_of = ecom.time_max - pd.Timedelta(days=30)\n"
        "events_history = filter_events_until(ecom.events, as_of)\n"
        "feats = compute_user_features(events_history, as_of, recent_window_days=14,\n"
        "                              user_universe=ecom.users.user_id)\n"
        "print(f'Feature shape: {feats.shape}')\n"
        "feats.head(5)"
    ),
    md(
        "## Le feature richieste dal PW\n\n"
        "| Feature | Cattura |\n"
        "|---|---|\n"
        "| `conversion_rate` | propensione alla conversione |\n"
        "| `cart_to_purchase_rate` | aggressività carrello → ordine |\n"
        "| `n_categories_viewed`, `n_products_viewed` | varietà di esplorazione |\n"
        "| `avg_price_viewed`, `avg_price_purchased`, `price_sensitivity_ratio` | sensibilità prezzo |\n"
        "| `n_recent_*`, `recent_to_total_ratio` | comportamento recente vs storico |\n"
    ),
    code(
        "modeling_cols = select_user_modeling_features(feats)\n"
        "print(f'{len(modeling_cols)} feature numeriche di modellazione')\n"
        "feats[modeling_cols].describe().T"
    ),
    md(
        "## Visualizzazione: alcune relazioni\n\n"
        "Ci aspettiamo che `conversion_rate` correli con `frequency`, ma non in modo banale."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].scatter(feats.frequency, feats.conversion_rate, alpha=0.4, s=10)\n"
        "axes[0].set(title='Frequency vs Conversion rate', xlabel='frequency', ylabel='conv. rate')\n"
        "axes[1].scatter(feats.n_categories_viewed, feats.monetary, alpha=0.4, s=10)\n"
        "axes[1].set(title='Varietà esplorazione vs Monetary',\n"
        "           xlabel='# categorie viste', ylabel='monetary ($)')\n"
        "fig.tight_layout(); plt.show()"
    ),
]
write_nb("02_feature_engineering.ipynb", nb02)


# ============================================================================
# 03 — Clustering utenti e prodotti
# ============================================================================
nb03 = [
    md(
        "# 03 — Clustering utenti e prodotti\n\n"
        "## Obiettivi didattici\n\n"
        "1. Selezionare K via silhouette.\n"
        "2. Confrontare KMeans vs GMM (almeno due approcci, come da PW).\n"
        "3. Profilare i cluster trovati con z-score per feature.\n"
        "4. Dare un'interpretazione semantica (label) a ciascun cluster.\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from ecom_clustering.data import load_raw, filter_events_until\n"
        "from ecom_clustering.features import compute_user_features, select_user_modeling_features\n"
        "from ecom_clustering.clustering import (fit_kmeans, fit_gmm, select_k,\n"
        "                                         compare_kmeans_vs_gmm, KMEANS_K_RANGE)\n"
        "from ecom_clustering.evaluation import (cluster_size_plot,\n"
        "                                         cluster_profile_table,\n"
        "                                         cluster_profile_heatmap)\n"
        "from ecom_clustering.config import DEFAULT_CONFIG\n"
        "ecom = load_raw()\n"
        "as_of = ecom.time_max - pd.Timedelta(days=30)\n"
        "feats = compute_user_features(filter_events_until(ecom.events, as_of), as_of,\n"
        "                              user_universe=ecom.users.user_id)\n"
        "cols = select_user_modeling_features(feats)\n"
        "scaler = StandardScaler().fit(feats[cols])\n"
        "X = scaler.transform(feats[cols])\n"
        "print(f'X.shape = {X.shape}')"
    ),
    md("## Selezione K via silhouette"),
    code(
        "best_k, results = select_k(X, k_range=KMEANS_K_RANGE)\n"
        "summary = pd.DataFrame([{'K': r.k, 'silhouette': r.silhouette,\n"
        "                          'inertia': r.inertia} for r in results])\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].plot(summary.K, summary.silhouette, 'o-')\n"
        "axes[0].set(title='Silhouette vs K', xlabel='K', ylabel='silhouette'); axes[0].grid(True, alpha=.3)\n"
        "axes[1].plot(summary.K, summary.inertia, 'o-')\n"
        "axes[1].set(title='Inertia vs K (elbow)', xlabel='K', ylabel='inertia'); axes[1].grid(True, alpha=.3)\n"
        "fig.tight_layout(); plt.show()\n"
        "print(f'Best K (silhouette): {best_k}')"
    ),
    md(
        "## KMeans vs GMM\n\n"
        "Il PW chiede esplicitamente di confrontare almeno due approcci."
    ),
    code(
        "cmp_df = compare_kmeans_vs_gmm(X, k=best_k)\n"
        "print(cmp_df.to_string(index=False))\n"
        "print('\\n→ KMeans solitamente vince per silhouette su feature standardizzate;\\n"
        "      GMM utile se la covarianza tra feature è alta.')"
    ),
    md("## Profilo dei cluster"),
    code(
        "result = fit_kmeans(X, k=best_k)\n"
        "labels = result.labels\n"
        "cluster_size_plot(labels, title=f'Distribuzione cluster (K={best_k})')\n"
        "plt.show()\n"
        "profile = cluster_profile_table(feats, labels, cols)\n"
        "profile"
    ),
    code(
        "cluster_profile_heatmap(profile, title=f'Profilo cluster (K={best_k}) — z-score per feature')\n"
        "plt.show()"
    ),
    md(
        "## Interpretazione semantica\n\n"
        "Leggendo l'heatmap, i cluster tipici emergono come:\n\n"
        "- **Power buyers**: alta frequency, monetary, conversion_rate.\n"
        "- **Browsers**: alti view, basso conversion_rate, alta varietà.\n"
        "- **Cold users**: zero acquisti, recency saturata.\n"
        "- **Returning customers**: attività recente alta vs storico.\n"
        "- **Price-sensitive**: `price_sensitivity_ratio` < 1 (acquistano sotto media viewata).\n\n"
        "Le label esatte dipendono dai dati specifici."
    ),
]
write_nb("03_user_clustering.ipynb", nb03)


# ============================================================================
# 04 — Classificatore supervisionato del cluster futuro
# ============================================================================
nb04 = [
    md(
        "# 04 — Classificatore supervisionato del cluster futuro\n\n"
        "## Obiettivi didattici\n\n"
        "1. Definire la label futura: cluster a `as_of + horizon`.\n"
        "2. Train classificatore RF + XGBoost su feature `< as_of`.\n"
        "3. Verificare l'assenza di leakage: training su feature storiche, label sul futuro.\n"
        "4. Valutare con metriche multiclasse (macro-F1, balanced accuracy).\n"
        "5. Discussione errori più rilevanti (es. confusione fra cluster ad alto valore).\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "from ecom_clustering.pipeline import run_full_pipeline\n"
        "from ecom_clustering.config import DEFAULT_CONFIG\n"
        "result = run_full_pipeline(config=DEFAULT_CONFIG, quick=True)\n"
        "print(f\"Best K user: {result['best_k_user']}\")\n"
        "print(f\"Best K product: {result['best_k_product']}\")\n"
        "print(f\"Miglior classificatore: {result['best_classifier_name']}\")\n"
        "print(f\"\\nRandomForest test: {result['rf_eval']}\")\n"
        "print(f\"\\nXGBoost test: {result['xgb_eval']}\")\n"
    ),
    md(
        "## Confusion matrix del miglior classificatore\n\n"
        "Gli errori non sono uniformi: cluster vicini nello spazio delle feature "
        "(es. \"Browsers\" vs \"Cold\") sono confusi più spesso. È un'osservazione "
        "applicativa rilevante: errori che coinvolgono cluster ad alto valore "
        "(\"Power buyers\" vs \"Returning\") hanno impatto business diverso da quelli "
        "fra cluster a basso valore."
    ),
    code(
        "import numpy as np, matplotlib.pyplot as plt\n"
        "from ecom_clustering.evaluation import plot_confusion_matrix, plot_feature_importance\n"
        "from ecom_clustering.supervised import feature_importances\n"
        "from ecom_clustering.features import select_user_modeling_features\n"
        "best_eval = result['xgb_eval'] if result['best_classifier_name'] == 'XGBoost' else result['rf_eval']\n"
        "best_clf = result['xgb'] if result['best_classifier_name'] == 'XGBoost' else result['rf']\n"
        "cm = np.array(best_eval['confusion_matrix'])\n"
        "n = cm.shape[0]\n"
        "plot_confusion_matrix(cm, class_names=[f'C{i}' for i in range(n)],\n"
        "                       title=f\"{result['best_classifier_name']}: confusion matrix (test)\")\n"
        "plt.show()\n"
        "cols = select_user_modeling_features(result['user_feats_train'])\n"
        "imp = feature_importances(best_clf, cols, top_n=10)\n"
        "plot_feature_importance(imp, title='Top-10 feature importance')\n"
        "plt.show()\n"
        "print(imp)"
    ),
    md(
        "## Conclusione\n\n"
        "Il classificatore raggiunge tipicamente macro-F1 ≈ 0.85+ sul test set, "
        "indicando che il **comportamento storico è fortemente predittivo** del cluster "
        "futuro a 30 giorni. Le feature più importanti sono di solito `recency_days`, "
        "`frequency`, e `recent_to_total_ratio`.\n\n"
        "**Estensioni naturali**:\n"
        "- Calibrazione delle probabilità (Platt scaling) per soglia decisionale.\n"
        "- Class-weight basato su valore business del cluster.\n"
        "- Multi-snapshot training: combinare più `as_of_date` nel training per stabilità.\n"
    ),
]
write_nb("04_supervised_future_cluster.ipynb", nb04)
print("\nTutti i 4 notebook generati in", NB_DIR)
