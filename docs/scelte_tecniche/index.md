---
layout: default
title: Scelte tecniche
nav_order: 3
has_children: true
permalink: /scelte_tecniche/
description: >-
  Decisioni architetturali e di modellazione del progetto E-commerce
  Customer & Product Clustering, con trade-off espliciti e razionali
  documentati.
---

# Scelte tecniche

Questa sezione documenta **come** è costruito il progetto e **perché** ogni
componente è stata progettata in un certo modo. È pensata per chi vuole
estendere o adattare la pipeline a un dominio simile (segmentazione utenti,
predizione di cluster futuri su orizzonti temporali).

## Capitoli

| Capitolo | Titolo | Cosa contiene |
|:--|:--|:--|
| 1 | [Architettura](architettura/) | Layout repo, moduli `src/ecom_clustering`, two-snapshot temporale, CLI, dipendenze fra componenti. |
| 2 | [Scelte di modellazione](scelte_modello/) | K-Means come baseline, GMM come confronto, feature RFM-extended, classificatore RF + XGBoost, gestione sbilanciamento. |

{: .tip }
> Per la teoria sottostante (algoritmi, metriche, leakage temporale) consulta
> la sezione **[Teoria](../teoria/)**.
