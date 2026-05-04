---
layout: default
title: Architettura
parent: Scelte tecniche
nav_order: 1
description: >-
  Layout del repository, principi di design (separazione src/notebooks,
  configurazione centralizzata, two-snapshot temporale), responsabilità
  dei moduli, interfacce pubbliche e punti di estensione.
---

# Architettura del progetto
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Layout

```
ecom-customer-product-clustering/
├── README.md                    Quick-start, traccia PW, risultati.
├── LICENSE                      MIT.
├── pyproject.toml               Build config + dipendenze + entry point CLI.
├── requirements.txt             Lock approssimativo (alternativa pip install -e).
├── mkdocs.yml                   Config sito documentazione.
│
├── src/ecom_clustering/         Codice sorgente (installabile via `pip install -e .`)
│   ├── __init__.py
│   ├── config.py                Path, costanti dominio, iperparametri, PipelineConfig.
│   ├── data.py                  Caricamento 5 CSV + `filter_events_until` (barriera anti-leakage).
│   ├── rfm.py                   `compute_rfm` + gestione cold users.
│   ├── features.py              Feature engineering esteso (16 feature utente, 10 prodotto).
│   ├── clustering.py            KMeans, GMM, `select_k` via silhouette, `compare_kmeans_vs_gmm`.
│   ├── labeling.py              `make_future_user_clusters` — costruzione label future (anti-leakage).
│   ├── supervised.py            RandomForest + XGBoost, `evaluate_classifier`, CV macro-F1.
│   ├── evaluation.py            Plot diagnostici (cluster size, profile heatmap, confmat, importance).
│   ├── inference.py             `predict_user_cluster` + `predict_future_cluster_supervised`.
│   └── pipeline.py              Orchestratore end-to-end + CLI `ecom-cluster`.
│
├── notebooks/                   Documentazione esecutiva (4 notebook).
│   ├── 01_eda_rfm.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_user_clustering.ipynb
│   └── 04_supervised_future_cluster.ipynb
│
├── data/                        Dataset.
│   ├── raw/                     5 CSV originali (~30 MB, COMMITTATI).
│   │   ├── events.csv           500k eventi (timestamp, user, product, action, price, ...).
│   │   ├── users.csv            1200 utenti.
│   │   ├── products.csv         3500 prodotti.
│   │   ├── categories.csv       15 categorie.
│   │   └── promotions_daily.csv Sconti giornalieri per categoria.
│   └── processed/               Output preprocessing (gitignored).
│
├── reports/
│   ├── figures/                 PNG dei plot diagnostici.
│   ├── models/                  Modelli serializzati (.joblib) — gitignored.
│   └── metrics.json             Output metriche pipeline.
│
├── scripts/
│   └── build_notebooks.py       Genera i 4 notebook da sorgente Python.
│
├── tests/                       Smoke test (placeholder).
│
├── docs/
│   ├── index.md                 Home del sito.
│   ├── teoria/                  5 file Markdown didattici.
│   ├── scelte_tecniche/         Documenti di design (questo file + scelte_modello.md).
│   └── stylesheets/extra.css    Custom styling Material.
│
├── .github/workflows/docs.yml   GitHub Actions per build + deploy GitHub Pages.
└── venv/                        Virtual environment (gitignored).
```

## 2. Principi di design

### 2.1 Separazione `src/` vs `notebooks/`

Il **codice riutilizzabile** vive in `src/ecom_clustering/`. I **notebook** sono solo presentazione: importazioni, chiamate alle funzioni di `src/`, narrazione didattica.

Vantaggi:

- Modifiche al codice riflesse automaticamente nei notebook (basta riavviare il kernel).
- Niente duplicazione: la logica esiste una sola volta.
- Test unitari possibili sul codice in `src/` (i notebook non sono test-friendly).
- Diff Git puliti su `src/` (i notebook hanno output cells e metadata che inquinano i diff).

### 2.2 Notebook generati da script

I 4 notebook sono prodotti da `scripts/build_notebooks.py`. Per modificare un notebook si edita lo script e si rilancia. Vantaggi:

- **Riproducibilità**: chiunque può rigenerare i notebook identici.
- **Sorgente in formato testuale**: lo script `.py` è searchable/grep-abile.
- **Diff puliti**: niente cambi spuri di metadata o kernel hash.

### 2.3 Two-snapshot temporale

Il PW richiede esplicitamente split temporali e validazione su periodi futuri. Il design "two-snapshot" è la traduzione architetturale:

- `as_of_train = t_max − 2H`: snapshot training del classificatore (feature → label a `t_max − H`).
- `as_of_test = t_max − H`: snapshot test (feature → label a `t_max`).

Con $H$ = `future_horizon_days` (default 30 giorni). Implementato in `pipeline._pick_snapshots`. Vedi `docs/teoria/03_split_temporali_e_no_leakage.md` per i dettagli.

### 2.4 Configurazione centralizzata

Tutti i path, le costanti dominio, le grid di iperparametri vivono in `config.py`. Niente magic number sparsi.

`PipelineConfig` è un `@dataclass(frozen=True)`: ogni esperimento crea un proprio config immutabile, nessuna mutazione accidentale durante la pipeline.

```python
@dataclass(frozen=True)
class PipelineConfig:
    random_state: int = 42
    recent_window_days: int = 14
    future_horizon_days: int = 30
    kmeans_k_user: int | None = None       # None → autoscelta silhouette
    kmeans_k_product: int | None = None
    n_jobs: int = -1
    verbose: int = 1
```

### 2.5 Persistenza esplicita scaler + modello

Invece di usare `sklearn.pipeline.Pipeline`, salviamo **scaler e cluster model separati** in un dizionario serializzato:

```python
joblib.dump({
    "cluster_model":   cluster_user.model,
    "scaler":          user_scaler,
    "feature_columns": user_cols,
    "k":               best_k_user,
    "as_of":           str(as_of_train),
}, user_pipeline_path)
```

Ragione: per il labeling futuro (`make_future_user_clusters`) abbiamo bisogno di accedere a scaler e modello **separatamente** (lo scaler standardizza nuove feature, il modello le classifica). Una `Pipeline` sklearn renderebbe il flusso più opaco.

Trade-off: meno "elegante" della Pipeline, ma più chiaro nel contesto temporale del PW.

## 3. Responsabilità dei moduli

| Modulo | Responsabilità | Dipendenze interne |
|---|---|---|
| `config` | Path, costanti, iperparametri | nessuna |
| `data` | I/O CSV, filtri temporali | `config` |
| `rfm` | Calcolo RFM base + cold users | `config` |
| `features` | Feature engineering esteso | `config`, `rfm` |
| `clustering` | KMeans, GMM, selezione K | `config` |
| `labeling` | Costruzione label future no-leakage | `config`, `data`, `features` |
| `supervised` | Classificatori RF + XGB, metriche | `config` |
| `evaluation` | Plot diagnostici | nessuna interna |
| `inference` | API inferenza utente cluster | `config` |
| `pipeline` | Orchestratore end-to-end | tutti gli altri |

Grafico delle dipendenze (semplificato):

```
config ─┬─→ data ──┬─→ features ──┬─→ pipeline
        │          │              │
        ├─→ rfm ──┘              │
        │                          │
        ├─→ clustering ────────────┤
        │                          │
        ├─→ supervised ────────────┤
        │                          │
        └─→ inference              │
                                   │
        evaluation ────────────────┘
        labeling ──────────────────┘
```

Tutte le frecce vanno verso `pipeline`. Nessuna dipendenza ciclica.

## 4. Punti di estensione

### 4.1 Aggiungere una feature derivata

Modificare `compute_user_features` in `features.py` (o `compute_product_features`). Aggiungere il nome della feature in `select_user_modeling_features` per includerla nel clustering.

Esempio: aggiungere `avg_session_length`:

```python
session_lengths = (
    events_history.groupby([USER_ID_COL, "session_id"])[EVENT_TIME_COL]
    .agg(lambda s: (s.max() - s.min()).total_seconds())
)
avg_session = session_lengths.groupby(level=0).mean().rename("avg_session_seconds")
rfm = rfm.join(avg_session, how="left").fillna({"avg_session_seconds": 0.0})
```

E poi in `select_user_modeling_features`:

```python
return [
    "recency_days", ..., "avg_session_seconds",   # ← aggiunta
]
```

### 4.2 Aggiungere un nuovo algoritmo di clustering

Implementare una funzione `fit_<nome>` in `clustering.py` con la stessa firma di `fit_kmeans`:

```python
def fit_dbscan(X: np.ndarray, eps: float, random_state: int = RANDOM_STATE) -> ClusteringResult:
    model = DBSCAN(eps=eps, min_samples=5)
    labels = model.fit_predict(X)
    sil = _sample_silhouette(X, labels) if len(set(labels)) > 1 else float("nan")
    return ClusteringResult(name=f"DBSCAN(eps={eps})", model=model, k=len(set(labels)),
                             silhouette=sil, inertia=None, labels=labels)
```

Per usarlo nel `select_k`, basta passarlo come `fit_fn`. Nota: DBSCAN non ha `predict()`, quindi non si può usare per generare label future — sarebbe solo un EDA.

### 4.3 Aggiungere un nuovo classificatore

Implementare `fit_<nome>` in `supervised.py` con stessa firma di `fit_random_forest`. Aggiungere il caso in `pipeline.run_full_pipeline` nella sezione "Train classifier".

### 4.4 Cambiare l'orizzonte temporale

CLI: `ecom-cluster --horizon 14` (default 30).

In codice: `PipelineConfig(future_horizon_days=14)`.

### 4.5 Cambiare i CSV in input

Modificare le costanti in `config.py`:

```python
EVENTS_FILE: str = "events.csv"      # nome file
EVENT_TIME_COL: str = "timestamp"    # colonna timestamp
USER_ID_COL: str = "user_id"
...
```

## 5. CLI

L'unico entry point CLI definito in `pyproject.toml` è:

```toml
[project.scripts]
ecom-cluster = "ecom_clustering.pipeline:main"
```

Argomenti supportati:

| Flag | Default | Descrizione |
|---|---|---|
| `--quick` | False | Smoke test (K=5 fisso, no GMM, no select_k) |
| `--horizon N` | 30 | Orizzonte label futura in giorni |

Estensioni possibili (non implementate):

- `--k-user N` / `--k-product N`: forzare K specifici.
- `--as-of-train YYYY-MM-DD`: snapshot esplicito invece del default `t_max - 2H`.
- `--no-xgboost`: skip XGBoost (utile in ambienti senza supporto).

## 6. Trade-off espliciti

| Scelta | Pro | Contro |
|---|---|---|
| `Python 3.11+` come target | Type hints moderni, `match` statement | Esclude Python ≤ 3.10 |
| `xgboost>=2.1` come dipendenza | State-of-the-art tabular | ~50 MB di binari, dipendenza C++ |
| `sklearn>=1.6` | API moderne, output naming uniforme | Esclude utenti su versioni vecchie |
| Notebook generati da script | Diff Git puliti, riproducibilità | Editing meno comodo (no live cells) |
| Two-snapshot temporal split | No leakage by construction | Non è una stima "media" su walk-forward |
| `StandardScaler` (no Robust) | Semplice, default sano | Sensibile a outlier su `monetary` |
| K-Means + GMM (no DBSCAN/Hierarchical) | `predict()` per label future | Nessuna gestione esplicita outlier |
| `class_weight="balanced"` su RF | Gestisce squilibrio | XGBoost non lo supporta nativamente |
| Persistenza dict (no Pipeline) | Accesso separato a scaler/modello | Meno elegante della Pipeline sklearn |

## 7. Testing strategy

Il progetto **non implementa** una test suite completa — è didattico. Smoke test inclusi:

- `ecom-cluster --quick` esegue tutta la pipeline in <30s con K=5 fisso e senza GMM.
- Esecuzione dei 4 notebook end-to-end via `nbconvert --execute` (manuale).

Per produzione si dovrebbero aggiungere:

- **Unit test** su `compute_rfm`: cold users hanno `frequency=0`, recency = sentinella.
- **Unit test** su `filter_events_until`: nessun evento con `timestamp >= as_of`.
- **Property-based test** (Hypothesis) su `compute_user_features`: `recent_to_total_ratio` ∈ [0, 1].
- **Integration test** su `run_full_pipeline(quick=True)`: produce i 3 `.joblib` attesi.
- **Snapshot test** sulle metriche: dato un seed, macro-F1 deve restare entro tolleranza ±0.02.
- **Test di leakage** ("shuffle test"): label permutate → macro-F1 ≈ 1/K.
