---
layout: default
title: Teoria
nav_order: 2
has_children: true
permalink: /teoria/
description: >-
  Fondamenti teorici della pipeline E-commerce Clustering: RFM e feature
  point-in-time, K-Means vs alternative, split temporali, classificazione
  multiclasse e riproducibilità della pipeline.
---

# Teoria

I cinque articoli di questa sezione costruiscono progressivamente le basi
necessarie per leggere il progetto **Customer & Product Clustering in
E-commerce** e per ragionare in modo critico sui risultati.

## Percorso consigliato di lettura

| Capitolo | Titolo | Concetti chiave |
|:--|:--|:--|
| 1 | [RFM & feature temporali](01_rfm_e_feature_temporali/) | Recency-Frequency-Monetary, feature point-in-time, propensione, varietà, price sensitivity |
| 2 | [K-Means & alternative](02_clustering_kmeans_vs_alternative/) | K-Means, scelta di K (elbow + silhouette), GMM, DBSCAN, hierarchical |
| 3 | [Split temporali & no-leakage](03_split_temporali_e_no_leakage/) | Hold-out cronologico, walk-forward, leakage temporale, two-snapshot |
| 4 | [Classificazione multiclasse & metriche](04_classificazione_multiclasse_e_metriche/) | Accuracy, macro-F1, weighted-F1, balanced accuracy, confusion matrix |
| 5 | [Pipeline riproducibile & seed](05_pipeline_riproducibile_e_seed/) | Seeding di numpy/sklearn/xgboost, ordering deterministico, sklearn `Pipeline` |

{: .note }
> Ogni capitolo è autocontenuto: leggi nell'ordine se vuoi una progressione
> didattica, oppure salta direttamente al capitolo che ti serve.

## Riferimenti trasversali

- Hughes, A. M. (2005) — *Strategic Database Marketing*, McGraw-Hill (RFM).
- Kaufman et al. (2012) — *Leakage in Data Mining: Formulation, Detection,
  and Avoidance*, ACM TKDD 6(4).
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (2009).
