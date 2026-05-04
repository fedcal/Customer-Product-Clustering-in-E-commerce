---
layout: default
title: Pipeline riproducibile & seed
parent: Teoria
nav_order: 5
math: mathjax
description: >-
  Riproducibilità in ML: seeding di numpy/sklearn/xgboost, ordering
  deterministico, versionamento di codice e dataset, struttura modulare
  con sklearn Pipeline e ColumnTransformer.
---

# Pipeline riproducibile, seed e versionamento
{: .no_toc }

> *"Riproducibilità non è 'mi ricordo come l'ho fatto'. È 'chiunque può rifarlo, ottenendo lo stesso identico risultato'."*

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 2. I tre livelli di riproducibilità

Esiste una gerarchia, dal più debole al più forte:

1. **Riproducibilità funzionale**: rieseguendo il codice ottengo metriche **simili** (entro tolleranza).
2. **Riproducibilità numerica**: rieseguendo il codice ottengo gli **stessi numeri** (bit per bit, o entro epsilon machine).
3. **Riproducibilità totale**: stesso ambiente, stessa versione di Python, stesso seed → stesso risultato sempre, su qualunque macchina dello stesso tipo.

Per il PW01 puntiamo al livello **2**: numeri identici se eseguito sulla stessa macchina con stessi seed. Il livello 3 (cross-platform, cross-Python-version) richiede contenitori (Docker) ed è fuori scope.

## 2. Seeding: dove e come

In Python ML "seriamo" almeno tre fonti di casualità:

### 2.1 NumPy

```python
import numpy as np

# Seed globale (sconsigliato in librerie, OK per script):
np.random.seed(42)

# Best practice moderna: generator locale
rng = np.random.default_rng(42)
sample = rng.choice(n, size=5000, replace=False)
```

Nel PW usiamo entrambi i pattern. `_sample_silhouette` ha `rng = np.random.default_rng(random_state)` — generator locale, non contamina lo stato globale.

### 2.2 Scikit-learn

Tutti gli stimatori che hanno casualità accettano `random_state`:

```python
KMeans(n_clusters=5, random_state=42, n_init=10)
RandomForestClassifier(random_state=42, ...)
XGBClassifier(random_state=42, ...)
StandardScaler()  # NO random_state — deterministico
```

Nel PW la costante è in `config.py`:

```python
RANDOM_STATE: int = 42
```

E viene propagata a tutti gli stimatori:

```python
def fit_kmeans(X, k, random_state=RANDOM_STATE):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=500)
    ...
```

### 2.3 XGBoost

XGBoost ha la propria fonte di casualità (sub-sampling, column sampling). `random_state` regola entrambe:

```python
XGBClassifier(
    random_state=random_state,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",   # deterministico se random_state fissato
    ...
)
```

!!! warning "GPU vs CPU"
    `tree_method="gpu_hist"` può dare risultati leggermente diversi tra esecuzioni (alcune somme floating-point sono non-deterministiche su GPU). Per il PW restiamo su `tree_method="hist"` (CPU).

## 3. Ordering deterministico dei dati

Anche con seed fissati, se l'ordine dei dati cambia, gli output cambiano. Esempio:

```python
events = pd.read_csv("events.csv")    # ordine non garantito
rfm = events.groupby("user_id").agg(...)   # OK, groupby è deterministico
```

Ma quando il dataset entra nel modello come `X.to_numpy()`, l'ordine delle righe conta:

```python
X = events.to_numpy()
# Se events è in ordine diverso, KMeans può convergere a un minimo diverso
# (con stesso seed ma stato iniziale diverso).
```

Soluzione adottata nel PW (`data.py::load_raw`):

```python
events = pd.read_csv(events_path, parse_dates=["timestamp"])
events = events.sort_values("timestamp").reset_index(drop=True)
```

L'ordinamento per `timestamp` (campo esterno, non dipendente dal contenuto) garantisce ordering deterministico.

## 4. Architettura modulare con sklearn

### 4.1 `Pipeline` per le trasformazioni

`sklearn.pipeline.Pipeline` incapsula una sequenza di step `(name, estimator)`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

clustering_pipeline = Pipeline([
    ("scaler",  StandardScaler()),
    ("kmeans",  KMeans(n_clusters=5, random_state=42, n_init=10)),
])

clustering_pipeline.fit(X_train)
labels = clustering_pipeline.predict(X_new)
```

Vantaggi:

- **Atomicità**: `fit` su tutta la pipeline → niente dimenticanze.
- **Persistenza**: `joblib.dump(clustering_pipeline, "model.joblib")` salva tutto.
- **No leakage** in CV: `cross_val_score(pipeline, X, y, cv=5)` rifitta la scaler ad ogni fold.

Nel PW01 abbiamo scelto un approccio **più esplicito**: scaler e clustering separati, salvati insieme nel `.joblib` come dict. Ragione: per il labeling futuro abbiamo bisogno di accedere al solo scaler e al solo cluster_model separatamente (vedi `make_future_user_clusters`):

```python
joblib.dump({
    "cluster_model":   cluster_user.model,
    "scaler":          user_scaler,
    "feature_columns": user_cols,
    "k":               best_k_user,
    "as_of":           str(as_of_train),
}, user_pipeline_path)
```

**Trade-off**: meno "elegante" della Pipeline sklearn, ma più chiaro nel flusso temporale del PW. In un progetto "regression-only" si userebbe Pipeline.

### 4.2 `ColumnTransformer` per feature eterogenee

Se ci fossero feature **categoriche** (es. country dell'utente), useremmo `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ("num", StandardScaler(),    numeric_cols),
    ("cat", OneHotEncoder(),     categorical_cols),
])
```

Nel PW01 le 16 feature utente sono **tutte numeriche** (dopo l'aggregazione), quindi `StandardScaler` da solo basta. Per estensioni (es. aggiungere `country` come feature) servirebbe ColumnTransformer.

## 5. Idempotenza

Una pipeline è **idempotente** se eseguirla 2 volte produce lo stesso output. Test:

```bash
ecom-cluster
sha256sum reports/models/user_pipeline.joblib > run1.sha
ecom-cluster
sha256sum reports/models/user_pipeline.joblib > run2.sha
diff run1.sha run2.sha   # atteso: nessuna differenza
```

Se differiscono:

- C'è una fonte di casualità non seeded.
- C'è un timestamp o ID univoco serializzato dentro il modello.
- L'ordine dei dati è cambiato.

!!! tip "joblib + protocollo deterministico"
    `joblib.dump(obj, path)` può aggiungere metadati di sistema (es. versione di scikit-learn). Per riproducibilità totale, fissa la versione in `pyproject.toml` (`scikit-learn>=1.6,<2.0`).

## 6. Versionamento

### 6.1 Codice

Git è ovvio. Ma serve un workflow disciplinato:

- Un commit = un'idea (clusterizzato, non "fix vari").
- Branch `main` sempre eseguibile (non rompere `ecom-cluster`).
- Tag per release importanti (`v0.1.0`).

### 6.2 Dati

Il dataset sintetico del PW (`data/raw/`, ~30 MB) è **committato** nel repository perché:

- È piccolo.
- È necessario per la riproducibilità (il sito doc cita risultati specifici).
- Non c'è una "fonte ufficiale" alternativa (è generato).

Per dataset più grandi useremmo:

- **Git LFS** (Large File Storage) per file binari grandi.
- **DVC** (Data Version Control) per pipeline ML data-aware.
- Hash del file (SHA-256) salvato in `config.py` per validazione.

### 6.3 Modelli

I `.joblib` sono **gitignored** (`.gitignore` esclude `reports/models/*`). Ragione: rigenerabili da `ecom-cluster`, occupano spazio, non servono nello storico.

In produzione si userebbe un **model registry** (MLflow, Weights & Biases) per tracciare metriche e artefatti.

### 6.4 Iperparametri e config

Tutti gli iperparametri sono in `config.py` (`RANDOM_STATE`, `KMEANS_K_RANGE`, `RECENT_WINDOW_DAYS`, `FUTURE_HORIZON_DAYS`, ...). Niente magic number sparsi nel codice.

`PipelineConfig` è un `@dataclass(frozen=True)` — **immutabile**:

```python
@dataclass(frozen=True)
class PipelineConfig:
    random_state: int = RANDOM_STATE
    recent_window_days: int = RECENT_WINDOW_DAYS
    future_horizon_days: int = FUTURE_HORIZON_DAYS
    kmeans_k_user: int | None = None
    kmeans_k_product: int | None = None
    n_jobs: int = -1
    verbose: int = 1
```

Vantaggi della immutabilità:

- Niente mutazione accidentale durante l'esecuzione.
- Si può loggare il config completo come "snapshot" dell'esperimento.
- Si può comparare due esperimenti con `==`.

## 7. Logging vs print

Nel PW usiamo `logging` invece di `print`:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("RFM computati per %d utenti.", len(rfm))
```

Vantaggi:

- **Livelli**: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Filtrabili in produzione.
- **Strutturato**: il timestamp è automatico, il nome del modulo è automatico.
- **Disabilitabile**: in test si può silenziare con `logging.disable(logging.CRITICAL)`.

In `pipeline.py::main`:

```python
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
```

## 8. Test smoke

Il PW01 non ha una test suite completa (didattico, non production). Ha però:

- **`ecom-cluster --quick`**: smoke test della pipeline in <30 secondi (K=5 fisso, no GMM, no plot).
- **`tests/`** directory: placeholder per test unitari futuri.

In produzione si dovrebbero aggiungere:

- Test su `compute_rfm`: proprietà "cold users hanno frequency=0".
- Test su `make_future_user_clusters`: proprietà "label sono in `[0, K)`".
- Test su `filter_events_until`: proprietà "nessun evento con `timestamp >= as_of`".
- Integration test su `run_full_pipeline(quick=True)`: deve completare e produrre i 3 `.joblib`.

## 9. Checklist pre-commit

Prima di committare modifiche al codice ML:

1. [ ] `ecom-cluster --quick` completa senza errori in <30s.
2. [ ] `ecom-cluster` completa senza errori in <5min.
3. [ ] Le metriche in `reports/metrics.json` sono ragionevoli (macro-F1 > baseline).
4. [ ] Non ho aggiunto magic number (tutti gli iperparametri in `config.py`).
5. [ ] Non ho introdotto seed non controllati (grep `random.` e `np.random.` per check).
6. [ ] Non ho aggiunto file grandi (`du -sh data/processed reports/models`).
7. [ ] La documentazione (`docs/`) è aggiornata se ho cambiato API pubbliche.

## 10. Riferimenti

- **Sklearn user guide**: [Glossary - random_state](https://scikit-learn.org/stable/glossary.html#term-random_state).
- **Pineau, J. et al.** (2021), *Improving Reproducibility in Machine Learning Research*, JMLR 22.
- **Sculley, D. et al.** (2015), *Hidden Technical Debt in Machine Learning Systems*, NeurIPS — debito tecnico in pipeline ML.
- **NumPy docs**: [Random Generator](https://numpy.org/doc/stable/reference/random/generator.html).
- **Joblib docs**: [Persistence](https://joblib.readthedocs.io/en/latest/persistence.html).
