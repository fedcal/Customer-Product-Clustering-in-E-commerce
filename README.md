# Customer & Product Clustering in E-commerce

 Progettazione e implementazione di una pipeline end-to-end per la segmentazione di utenti e prodotti in un contesto e-commerce, a partire da eventi temporali. Il progetto richiede la costruzione di feature aggregate coerenti nel tempo, l’addestramento di modelli di clustering e lo sviluppo di un modello supervisionato capace di predire l’appartenenza futura ai cluster, garantendo correttezza metodologica, assenza di leakage e riproducibilità del sistema.

## Dettagli

### Contesto

Lavorerai su un dataset di eventi e-commerce in cui ogni riga rappresenta un’azione utente su un prodotto a un certo timestamp. Le colonne includono identificativi di utente, prodotto e categoria, prezzo e tipo di azione (view, add_to_cart, purchase).
Il progetto richiede di progettare e implementare un sistema che costruisca cluster di utenti e prodotti a partire da questi eventi e che sia in grado di stimare l’assegnazione futura ai cluster su orizzonti temporali definiti.

### Obiettivo del progetto

L’obiettivo principale è costruire una pipeline completa e corretta dal punto di vista metodologico per il clustering di utenti e prodotti, partendo da dati temporali. Il focus è sulla struttura del sistema, sulla riproducibilità e sulla correttezza degli split temporali.
In secondo piano, ma obbligatorio, dovrai implementare un modello supervisionato che predica l’appartenenza futura ai cluster, rispettando rigorosamente il vincolo temporale.

### Attività richieste

Dovrai progettare una pipeline end-to-end che includa ingestione dei dati, pulizia, costruzione delle feature, training del clustering e assegnazione dei cluster. La pipeline deve essere modulare e facilmente rieseguibile cambiando la data di riferimento o i parametri principali.

La costruzione delle feature è parte integrante del lavoro. Dovrai implementare aggregazioni temporali corrette, assicurandoti che ogni feature utilizzi solo informazioni disponibili fino al tempo di riferimento. Le scelte di finestra temporale e di aggregazione devono essere esplicite.

Per gli utenti dovrai partire dall'analisi RFM ma arricchirlo con indicatori che descrivano propensione alla conversione, varietà di esplorazione, sensibilità al prezzo e differenze tra comportamento recente e storico. Le feature devono raccontare una storia sul comportamento, non limitarsi a contare eventi.

Per il clustering dovrai implementare almeno due approcci, confrontandoli tra di loro mettendo in risalto i punti positivi e negativi di entrambi e identificando quello più adatto allo scenario. L’attenzione è sulla coerenza dei risultati e sulla stabilità operativa, non sull’uso di tecniche complesse fine a sé stesse.

La previsione dell’appartenenza futura ai cluster deve essere trattata come un problema supervisionato. Dovrai definire chiaramente come viene costruita la label futura, implementare split temporali corretti e valutare il modello su periodi realmente futuri. È fondamentale dimostrare che non esiste leakage informativo.

La valutazione del modello di previsione deve includere metriche standard di classificazione multiclasse e una discussione sugli errori più rilevanti dal punto di vista applicativo, ad esempio errori che coinvolgono cluster ad alto valore.


### Deliverable

Dovrai consegnare un repository o progetto strutturato che contenga codice riutilizzabile per l’intera pipeline, insieme a un notebook in formato .ipynb con all'interno la documentazione con la descrizione dei vari passaggi effettuati ed i commenti alle informazioni principali estratte durante l'analisi. Il codice deve poter essere rieseguito per rigenerare feature, cluster e modelli senza interventi manuali.

### Criteri di valutazione

La valutazione premierà la correttezza metodologica, la gestione del tempo e degli split, la chiarezza della pipeline e la qualità del codice. Errori concettuali legati all’uso di informazioni future o a una definizione ambigua delle label avranno un impatto significativo sulla valutazione finale.

---

## Repository GitHub

> **Nome del repository pubblico (consigliato)**: `ecom-customer-product-clustering`
> URL: <https://github.com/fedcal/ecom-customer-product-clustering>

Il deploy della documentazione su GitHub Pages avviene automaticamente a ogni push su `main` direttamente dalla cartella [`docs/`](docs/) (Jekyll lato server). Per attivarlo: **Settings → Pages → Source = Deploy from a branch, Branch = `main`, Folder = `/docs`**. Nessun workflow GitHub Actions richiesto.

## Documentazione completa

Il sito di documentazione (Jekyll + tema **Just the Docs**, in italiano) è pubblicato a:

**<https://fedcal.github.io/ecom-customer-product-clustering/>**

Contiene:

- **Teoria**: RFM e feature point-in-time, K-Means vs alternative, split temporali e leakage, metriche multiclasse, riproducibilità.
- **Scelte tecniche**: architettura del repository, decisioni di modellazione, trade-off espliciti.
- **Quick start** e API di inferenza.

## Quick start

```bash
git clone https://github.com/fedcal/ecom-customer-product-clustering.git
cd ecom-customer-product-clustering
python3 -m venv venv && source venv/bin/activate
pip install -e ".[notebooks]"

# Esecuzione end-to-end (clustering utenti + prodotti + classificatore futuro)
ecom-cluster                 # full run con selezione K via silhouette
ecom-cluster --quick         # smoke test (K=5 fisso, no GMM)
ecom-cluster --horizon 14    # cambia l'orizzonte temporale della label
```

Output:

- `reports/models/user_pipeline.joblib` — clustering utenti + scaler.
- `reports/models/product_pipeline.joblib` — clustering prodotti + scaler.
- `reports/models/future_cluster_classifier.joblib` — miglior classificatore (RF o XGBoost).
- `reports/metrics.json` — silhouette, macro-F1, balanced accuracy, accuracy.
- `reports/figures/*.png` — distribuzione cluster, profilo, confusion matrix, importance.

### Inferenza

```python
import pandas as pd
from ecom_clustering.data import load_raw, filter_events_until
from ecom_clustering.features import compute_user_features
from ecom_clustering.inference import (
    predict_user_cluster,
    predict_future_cluster_supervised,
)

ecom = load_raw()
as_of = ecom.time_max - pd.Timedelta(days=30)
history = filter_events_until(ecom.events, as_of)

feats = compute_user_features(history, as_of, user_universe=ecom.users["user_id"])

cluster_now    = predict_user_cluster(feats)             # K-Means
cluster_future = predict_future_cluster_supervised(feats) # RF/XGBoost

print(cluster_now.value_counts().sort_index())
print(cluster_future.value_counts().sort_index())
```

### Notebook didattici

Quattro notebook in `notebooks/` mostrano il flusso passo passo (generati da `scripts/build_notebooks.py`):

1. `01_eda_rfm.ipynb` — EDA dataset + RFM base.
2. `02_feature_engineering.ipynb` — feature engineering esteso (16 feature utente).
3. `03_user_clustering.ipynb` — KMeans vs GMM, selezione K, profiling cluster.
4. `04_supervised_future_cluster.ipynb` — classificatore RF/XGB, metriche, confusion matrix.

## Struttura del repository

```
ecom-customer-product-clustering/
├── src/ecom_clustering/         Codice (pip install -e .)
├── notebooks/                   4 notebook didattici
├── data/raw/                    5 CSV (committati, ~30 MB)
├── reports/                     Output (figures, models, metrics.json)
├── scripts/build_notebooks.py   Generatore notebook
├── docs/                        Sito Jekyll + Just the Docs (GitHub Pages)
├── tests/                       Smoke test (placeholder)
├── pyproject.toml               Build + dipendenze + entry point CLI
├── LICENSE                      MIT
└── README.md                    Questo file
```

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di **DataMasters / Skiller**.

Per altri progetti e contatti: <https://federicocalo.dev>.

## Licenza

Distribuito sotto licenza [MIT](LICENSE). Copyright © 2026 Federico Calò.
