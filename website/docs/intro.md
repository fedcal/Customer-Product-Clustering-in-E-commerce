---
sidebar_position: 1
title: Introduzione
description: |
  Pipeline ML end-to-end per il clustering RFM di utenti e prodotti su eventi e-commerce, con split temporale rigoroso e predizione del cluster futuro tramite classificatore multiclasse.
slug: /intro
---

# Ecom Clustering — Customer & Product

Pipeline didattica e **production-friendly** che, a partire da un dataset di eventi e-commerce (view, add_to_cart, purchase), costruisce un sistema completo di **segmentazione comportamentale**: ingestione → feature RFM point-in-time → clustering K-Means/GMM → labeling temporale anti-leakage → classificatore multiclasse del cluster futuro → inferenza programmatica.

:::tip In una riga
*Da eventi grezzi a `predict_user_cluster()` con split temporale rigoroso e zero leakage.*
:::

## Repository GitHub

| Item | Link |
|---|---|
| Repo | [`fedcal/Customer-Product-Clustering-in-E-commerce`](https://github.com/fedcal/Customer-Product-Clustering-in-E-commerce) |
| Documentazione | [https://fedcal.github.io/Customer-Product-Clustering-in-E-commerce/](https://fedcal.github.io/Customer-Product-Clustering-in-E-commerce/) |
| Licenza | MIT |
| Stack docs | Docusaurus 3 + TypeScript + KaTeX |

## Obiettivi del progetto

1. **Clustering utenti e prodotti** a partire da feature RFM-extended (16 feature utente, 10 feature prodotto), calcolate in modo **point-in-time** per evitare data leakage.
2. **Selezione automatica di K** via silhouette score (range `K ∈ {3..8}`), con confronto K-Means vs Gaussian Mixture.
3. **Predizione del cluster futuro**: classificatore multiclasse (RandomForest + XGBoost) addestrato su due snapshot temporali disgiunti (two-snapshot anti-leakage).
4. **Riproducibilità totale**: random seed propagato a tutti gli stimatori, persistenza esplicita di scaler + modello, CLI deterministica.

## Risultati di riferimento (orientativi)

I valori esatti dipendono dal random state e dal dataset sintetico versionato in `data/raw/`. Per i numeri puntuali della tua run, consulta `reports/metrics.json` dopo aver eseguito `ecom-cluster`.

| Indicatore | Valore atteso | Note |
|:--|:--:|:--|
| K cluster utenti | 4–6 | selezione via silhouette |
| Silhouette utenti | 0.20–0.40 | dataset sintetico, RFM-extended |
| Macro-F1 cluster futuro | 0.70–0.85 | RF / XGBoost, holdout temporale |
| Balanced accuracy | 0.75–0.90 | holdout temporale |

## Quick start

```bash
git clone https://github.com/fedcal/Customer-Product-Clustering-in-E-commerce.git
cd Customer-Product-Clustering-in-E-commerce
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

:::warning Coerenza temporale
`compute_user_features` usa SOLO eventi con `timestamp < as_of`. È la **barriera anti-leakage** del progetto: senza di essa, le feature trapelerebbero il futuro e le metriche del classificatore sarebbero artificialmente alte.
:::

## Mappa della documentazione

### [Teoria](/docs/category/teoria)

Cinque articoli che costruiscono progressivamente i concetti necessari a leggere il progetto:

1. [RFM & feature temporali](./teoria/01-rfm-feature-temporali.md) — Recency-Frequency-Monetary, propensione, varietà di esplorazione, point-in-time.
2. [Clustering: KMeans vs alternative](./teoria/02-clustering-kmeans-vs-alternative.md) — scelta di K, GMM, DBSCAN, hierarchical.
3. [Split temporali & no leakage](./teoria/03-split-temporali-no-leakage.md) — hold-out cronologico, walk-forward, two-snapshot.
4. [Classificazione multiclasse & metriche](./teoria/04-classificazione-multiclasse-metriche.md) — macro-F1, balanced accuracy, confusion matrix.
5. [Pipeline riproducibile & seed](./teoria/05-pipeline-riproducibile-seed.md) — seeding deterministico, sklearn `Pipeline`, versionamento.

### [Scelte tecniche](/docs/category/scelte-tecniche)

Decisioni architetturali e razionali documentati:

- [Architettura](./scelte-tecniche/architettura.md) — moduli, two-snapshot temporale, CLI, flusso dati.
- [Scelte di modellazione](./scelte-tecniche/scelte-modello.md) — trade-off espliciti su clustering e classificatore.

## Stack tecnologico

| Layer | Tecnologie |
|:--|:--|
| Linguaggio | Python 3.11+ |
| ML | scikit-learn, xgboost |
| Data | pandas, numpy |
| Plotting | matplotlib, seaborn |
| Notebook | jupyter, jupytext |
| Persistenza | joblib |
| Documentazione | Docusaurus 3 + TypeScript + KaTeX |
| CI/CD | GitHub Actions + GitHub Pages |

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti: [**federicocalo.dev**](https://federicocalo.dev).
