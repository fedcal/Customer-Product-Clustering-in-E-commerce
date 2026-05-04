---
sidebar_position: 1
title: Introduzione
description: |
  Pipeline end-to-end per il clustering RFM di utenti e prodotti, con predizione cluster futuro a horizon di 30/90 giorni.
slug: /intro
---

# Ecom Clustering — Customer & Product

Pipeline didattica e **production-friendly** che, a partire da eventi e-commerce (visite, carrello, acquisti), costruisce features RFM, addestra modelli di clustering non supervisionato (KMeans / GMM) e un classificatore supervisionato per predire il cluster futuro a 30/90 giorni: feature engineering temporale → train clustering → labeling → supervised → inferenza tramite `predict_future_cluster_supervised()`.

:::tip In una riga
*Da CSV grezzo a `predict_user_cluster()` e `predict_future_cluster_supervised()` con pochi comandi e zero leakage temporale.*
:::

## Repository GitHub

| Item | Link |
|---|---|
| Repo | [`fedcal/Customer-Product-Clustering-in-E-commerce`](https://github.com/fedcal/Customer-Product-Clustering-in-E-commerce) |
| Documentazione | [https://fedcal.github.io/Customer-Product-Clustering-in-E-commerce/](https://fedcal.github.io/Customer-Product-Clustering-in-E-commerce/) |
| Licenza | MIT |
| Stack docs | Docusaurus 3 + TypeScript + KaTeX |

## Mappa della documentazione

### [Teoria](/docs/category/teoria)

1. [RFM & feature temporali](./teoria/01-rfm-feature-temporali.md) — Recency, Frequency, Monetary su finestre temporali, feature di engagement, no-leakage.
2. [Clustering: KMeans vs alternative](./teoria/02-clustering-kmeans-vs-alternative.md) — KMeans, MiniBatch, GMM, HDBSCAN: quando preferirli, silhouette, davies-bouldin.
3. [Split temporali & no leakage](./teoria/03-split-temporali-no-leakage.md) — Time-aware split su eventi e-commerce, prevenzione del leakage temporale.
4. [Classificazione multiclasse & metriche](./teoria/04-classificazione-multiclasse-metriche.md) — Predizione cluster futuro come problema multiclass, macro-F1, balanced accuracy.
5. [Pipeline riproducibile & seed](./teoria/05-pipeline-riproducibile-seed.md) — Seed handling, version pinning, riproducibilità di pipeline ML stocastiche.

### [Scelte tecniche](/docs/category/scelte-tecniche)

- [Architettura del progetto](./scelte-tecniche/architettura.md) — Layout dei moduli ecom_clustering/: data, features, clustering, supervised, inference.
- [Scelte di modellazione: razionale](./scelte-tecniche/scelte-modello.md) — Razionale clustering KMeans + GMM, classificatore RandomForest, gestione classi sbilanciate.

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti: [**federicocalo.dev**](https://federicocalo.dev).
