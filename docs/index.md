---
layout: home
title: Home
nav_order: 1
description: >-
  Pipeline ML end-to-end per clustering RFM utenti+prodotti su eventi
  e-commerce, con split temporale rigoroso (no leakage) e predizione del
  cluster futuro tramite classificatore multiclasse RandomForest / XGBoost.
permalink: /
---

<div class="hero-banner" markdown="0">
  <h1>Customer &amp; Product Clustering &mdash; E&#8209;commerce</h1>
  <p>
    Da eventi grezzi a <code>predict_user_cluster()</code>: feature RFM
    arricchite, split temporale rigoroso, K&#8209;Means vs GMM,
    classificatore multiclasse del cluster futuro. Riproducibile, modulare,
    GitHub Pages&#8209;ready.
  </p>
</div>

## In sintesi

Progetto di riferimento del percorso **Machine Learning Engineer** di
[DataMasters](https://datamasters.it/)/Skiller. Implementa l'intero flusso
di lavoro di un sistema di **segmentazione comportamentale** su eventi
e-commerce: dall'ingestione dei CSV grezzi alla predizione del cluster
futuro, con focus su **rigorosità metodologica**, **prevenzione del leakage
temporale** e **riproducibilità**.

<div class="kpi-grid" markdown="0">
  <div class="kpi-card">
    <div class="kpi-label">K cluster utenti</div>
    <div class="kpi-value">4&ndash;6</div>
    <div>selezione via silhouette</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">macro&#8209;F1 (XGBoost)</div>
    <div class="kpi-value">0.70&ndash;0.85</div>
    <div>cluster futuro, holdout</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">balanced accuracy</div>
    <div class="kpi-value">0.75&ndash;0.90</div>
    <div>holdout temporale</div>
  </div>
</div>

{: .note }
> I valori esatti dipendono dal random state e dal dataset sintetico
> versionato in `data/raw/`. Per i numeri puntuali della tua run, consulta
> `reports/metrics.json` dopo aver eseguito `ecom-cluster`.

## Repository GitHub

- **Nome del repository**: `ecom-customer-product-clustering`
- **URL**: [github.com/fedcal/ecom-customer-product-clustering](https://github.com/fedcal/ecom-customer-product-clustering)
- **Documentazione (questo sito)**: pubblicata via **GitHub Pages** dalla cartella
  [`/docs`](https://github.com/fedcal/ecom-customer-product-clustering/tree/main/docs).

{: .note }
> La documentazione viene servita direttamente dai file Markdown della cartella
> `docs/`, processati da Jekyll con il tema **Just the Docs**.
> Ogni push su `main` aggiorna automaticamente il sito.

## Quick start

```bash
git clone https://github.com/fedcal/ecom-customer-product-clustering.git
cd ecom-customer-product-clustering
python3 -m venv venv && source venv/bin/activate
pip install -e ".[notebooks]"

ecom-cluster                 # full run con selezione K via silhouette
ecom-cluster --quick         # smoke test (K=5 fisso, no GMM)
ecom-cluster --horizon 14    # cambia l'orizzonte temporale della label
```

Output principali in `reports/`:

- `models/user_pipeline.joblib` — clustering utenti + scaler.
- `models/product_pipeline.joblib` — clustering prodotti + scaler.
- `models/future_cluster_classifier.joblib` — miglior classificatore (RF / XGBoost).
- `metrics.json` — silhouette, macro-F1, balanced accuracy, accuracy.
- `figures/*.png` — distribuzione cluster, profilo, confusion matrix, feature importance.

## Inferenza programmatica

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

feats = compute_user_features(
    history, as_of, user_universe=ecom.users["user_id"]
)

# Cluster attuale (K-Means su feature standardizzate)
cluster_now = predict_user_cluster(feats)

# Cluster previsto a as_of + horizon (classificatore supervisionato)
cluster_future = predict_future_cluster_supervised(feats)

print(cluster_now.value_counts().sort_index())
print(cluster_future.value_counts().sort_index())
```

{: .warning }
> **Coerenza temporale**: `compute_user_features` usa SOLO eventi con
> `timestamp < as_of`. È la barriera anti-leakage del progetto: senza di essa,
> le feature "trapelerebbero" il futuro e le metriche del classificatore
> sarebbero artificialmente alte.

## Mappa della documentazione

### [Teoria](teoria/)

Fondamenti per leggere i risultati del progetto:

- [RFM & feature temporali](teoria/01_rfm_e_feature_temporali/) — Recency-Frequency-Monetary, propensione, varietà, point-in-time.
- [K-Means & alternative](teoria/02_clustering_kmeans_vs_alternative/) — scelta di K, GMM, DBSCAN, hierarchical.
- [Split temporali & no-leakage](teoria/03_split_temporali_e_no_leakage/) — hold-out cronologico, walk-forward, two-snapshot.
- [Classificazione multiclasse & metriche](teoria/04_classificazione_multiclasse_e_metriche/) — macro-F1, balanced accuracy, confusion matrix.
- [Pipeline riproducibile & seed](teoria/05_pipeline_riproducibile_e_seed/) — seeding deterministico, sklearn `Pipeline`, versionamento.

### [Scelte tecniche](scelte_tecniche/)

Decisioni architetturali e di modellazione:

- [Architettura](scelte_tecniche/architettura/) — moduli, two-snapshot temporale, CLI, flusso dati.
- [Scelte di modellazione](scelte_tecniche/scelte_modello/) — trade-off espliciti su clustering e classificatore.

## Stack tecnologico

| Layer | Tecnologie |
|:--|:--|
| Linguaggio | Python 3.11+ |
| ML | scikit-learn, xgboost |
| Data | pandas, numpy |
| Plotting | matplotlib, seaborn |
| Notebook | jupyter |
| Persistenza | joblib |
| Documentazione | Jekyll + Just the Docs |

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso
*Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti:
[**federicocalo.dev**](https://federicocalo.dev){: .btn .btn-purple }
