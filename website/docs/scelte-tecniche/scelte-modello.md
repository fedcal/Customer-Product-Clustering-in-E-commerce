---
sidebar_position: 2
title: "Scelte di modellazione: razionale"
description: |
  Razionale delle decisioni: K-Means come modello primario, GMM come confronto, RandomForest e XGBoost per il cluster futuro, gestione dello sbilanciamento di classe.
---

Documenta le decisioni "perché così e non cosà" di livello modeling. Per dettagli teorici sulle tecniche, vedi la sezione [Teoria](/docs/category/teoria).

## 1. Set di feature scelte

### Perché 16 feature utente e non 3 (solo RFM)

Il PW richiede esplicitamente di "arricchire RFM con indicatori che descrivano propensione alla conversione, varietà di esplorazione, sensibilità al prezzo e differenze tra comportamento recente e storico".

Decisione: lista finale di 16 feature (vedi `select_user_modeling_features`):

| Gruppo | Feature | N |
|---|---|---|
| RFM base | `recency_days`, `frequency`, `monetary` | 3 |
| Conteggi azioni | `n_view`, `n_add_to_cart`, `n_purchase` | 3 |
| Conversione | `conversion_rate`, `cart_to_purchase_rate` | 2 |
| Esplorazione | `n_categories_viewed`, `n_products_viewed` | 2 |
| Price sensitivity | `avg_price_viewed`, `avg_price_purchased`, `price_sensitivity_ratio` | 3 |
| Trend recente | `n_recent_interactions`, `n_recent_purchases`, `recent_to_total_ratio` | 3 |

**Totale: 16 feature**.

**Perché non aggiungere altre feature**:

- `n_sessions`: ridondante con `n_view + n_cart + n_purchase` su questo dataset.
- `avg_basket_size`: i dati non hanno un concetto esplicito di "carrello finalizzato".
- One-hot dei top-K prodotti viewati: alta dimensionalità, basso segnale per il clustering.

**Perché non rimuovere feature**:

- `cart_to_purchase_rate` sembra ridondante con `conversion_rate`, ma cattura un comportamento diverso (utenti "carrellisti seriali" vs "decisivi").
- `price_sensitivity_ratio` è correlato con `avg_price_purchased / avg_price_viewed`, ma è normalizzato e robusto a utenti senza acquisti (dove sarebbe NaN).

### Feature prodotto (10 feature)

Per il clustering prodotti (vedi `select_product_modeling_features`):

```python
[
    "n_view", "n_add_to_cart", "n_purchase",
    "conversion_rate", "avg_price", "n_unique_users",
    "base_price", "quality_score", "base_conversion", "base_popularity",
]
```

Le ultime 4 sono attributi statici da `products.csv` (non dipendenti dal tempo). Le prime 6 sono aggregazioni point-in-time da `events`.

## 2. Famiglie di clustering scelte

### K-Means (modello primario)

**Perché**:

- Veloce ($O(n \cdot K \cdot p)$): su 1200 utenti × 16 feature, 1 secondo.
- Ha `predict()`: necessario per generare le label future applicando il modello fittato sul training a snapshot futuri.
- Pyramide di iperparametri minimale ($K$).

**Perché non DBSCAN**: nessun `predict()` per nuovi punti. Riapplicare DBSCAN sui dati futuri darebbe label diverse → impossibile costruire un classificatore che impari quali utenti sono "VIP".

**Perché non Hierarchical (Ward)**: $O(n^2)$ memoria, $O(n^3)$ tempo. Su 1200 utenti gestibile, ma sopra i 10k impraticabile. Inoltre niente `predict()` (idem DBSCAN).

### Gaussian Mixture Models (confronto)

**Perché**: il PW richiede "almeno due approcci confrontati". GMM è il candidato più ragionevole perché:

- Generalizza K-Means (covarianze non sferiche).
- Ha `predict()` per nuovi punti.
- È implementato in scikit-learn → niente dipendenze aggiuntive.

**Quando GMM batte K-Means**: cluster ellittici o sovrapposti. Su feature RFM-extended il pattern reale dipende dal dataset:

- Se i cluster sono "naturali" (es. VIP molto separati dagli occasionali), K-Means basta.
- Se ci sono transizioni continue (utenti "borderline"), GMM coglie meglio l'ambiguità.

**Confronto operativo**: `compare_kmeans_vs_gmm` in `clustering.py` riporta silhouette + inertia su entrambi. Se GMM ha silhouette nettamente superiore (>+0.05), va considerato come modello primario; altrimenti K-Means resta più semplice e veloce.

### Cosa NON ho fatto (e perché)

- **Ensemble di K-Means con K diversi**: complessità non giustificata su dataset di 1200 utenti.
- **Spectral clustering**: $O(n^3)$ per la decomposizione SVD, non scala. Da considerare se il segnale fosse non lineare e i cluster intrecciati.
- **HDBSCAN**: variante "smart" di DBSCAN, gestisce densità variabili. Promettente ma `predict()` ancora limitato (richiede `prediction_data=True` + più memoria).

## 3. Selezione di K

### Range esplorato

`KMEANS_K_RANGE = (3, 4, 5, 6, 7, 8)` definito in `config.py`.

**Limite inferiore $K = 3$**:

- $K = 2$ produce cluster troppo grossolani (es. solo "attivi" vs "cold").
- Per business segmentation servono almeno 3 gruppi (alto / medio / basso valore).

**Limite superiore $K = 8$**:

- Sopra $K = 8$ i cluster diventano troppo piccoli per essere interpretabili.
- Con 1200 utenti, $K = 10$ darebbe ~120 utenti per cluster — varianza dei centroidi alta, instabilità tra run.

### Criterio di scelta: silhouette

Implementato in `select_k`. Su sample di 5000 punti per efficienza. La scelta di silhouette su elbow:

- Più robusta (non richiede interpretazione visiva).
- Più stabile in CI/CD (la decisione è deterministica).

**Trade-off**: la silhouette può preferire $K$ più alti (cluster piccoli sono più "coesi"). Mitigazione: il range superiore di 8 limita questo bias.

### `n_init = 10`

KMeans esegue 10 inizializzazioni `k-means++` e mantiene quella con inertia minima. Costo lineare in `n_init`, ma drasticamente più stabile rispetto a `n_init=1`.

### `max_iter = 500`

Sufficiente per convergere su questo dataset. In dataset più grandi/rumorosi può servire alzare.

## 4. Famiglie di classificatori scelte

### RandomForest

**Perché**:

- **Baseline non lineare** robusta a multicollinearità (presente nelle 16 feature).
- **Class weights nativi**: `class_weight="balanced"` gestisce squilibrio classi (cluster di taglie diverse).
- **Feature importance** out-of-the-box (anche se sostituibile con permutation_importance per maggiore correttezza).
- **Robusto agli outlier** sulle feature.

**Iperparametri scelti** (in `fit_random_forest`):

```python
RandomForestClassifier(
    n_estimators=400,        # 400 alberi: ROI marginale oltre, costo lineare
    max_depth=None,          # alberi completi, decorrelazione + bagging compensano
    min_samples_split=2,     # default
    max_features="sqrt",     # classico Breiman
    n_jobs=-1,               # parallelo
    random_state=42,
    class_weight="balanced", # gestisce squilibrio classi
)
```

### XGBoost

**Perché**:

- **State-of-the-art** su tabular structured data.
- Spesso batte RandomForest di 1-3 punti su macro-F1.
- Confronto utile dal punto di vista didattico (RF mostra il guadagno del bagging; XGBoost mostra il guadagno aggiuntivo del boosting).

**Iperparametri scelti** (in `fit_xgboost`):

```python
XGBClassifier(
    n_estimators=400,        # con lr=0.05, sufficiente per convergenza
    max_depth=5,             # interazioni 4-5-way, no memorizzazione
    learning_rate=0.05,      # paradigma "small steps" di Friedman
    subsample=0.8,           # regolarizzazione stocastica
    colsample_bytree=0.8,    # idem
    objective="multi:softprob",
    num_class=n_classes,
    tree_method="hist",      # CPU deterministico
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)
```

**Cosa NON ho fatto**:

- **`scale_pos_weight`** (XGBoost equivalente di `class_weight`): è per binario, non multiclasse. Per multiclasse XGBoost richiederebbe `sample_weight` esplicito → estensione possibile.
- **Early stopping**: richiede un eval_set, che cambia la natura del training (dovremmo dividere il train in train+val). Per il PW didattico semplifichiamo.

### LogisticRegression

Considerata e scartata. Ragioni:

- Su 16 feature standardizzate con possibili relazioni non lineari (es. `recency × frequency`), una lineare con `class_weight="balanced"` darebbe macro-F1 sensibilmente più bassa di RF/XGBoost.
- Vantaggi (interpretabilità coefficienti) sono mitigati dalla disponibilità di feature importance per RF/XGB.

Se servisse interpretabilità lineare, baseline da aggiungere come terzo modello.

## 5. Selezione del miglior classificatore

Criterio adottato (in `pipeline.run_full_pipeline`):

```python
best_classifier = xgb if xgb_eval["macro_f1"] >= rf_eval["macro_f1"] else rf
```

**Se XGB ≥ RF su macro-F1, scelgo XGB**. Altrimenti RF.

Criteri alternativi non adottati (per semplicità ma rilevanti):

1. **Stabilità CV**: confrontare la std macro-F1 su K-fold, non solo la mean.
2. **Latenza inferenza**: RF (~50 ms) vs XGB (~5-10 ms). Su questo dataset trascurabile.
3. **Dimensione modello**: RF con 400 alberi pesa più di XGB.
4. **Errori costosi**: pesare gli errori VIP→occasionale di più. Richiede una matrice di costo.

In produzione, scelta combinata di queste 4 metriche → estensione possibile per il PW.

## 6. Gestione sbilanciamento classi

I cluster K-Means sono spesso sbilanciati: con 1200 utenti e $K=5$, è normale avere un cluster con 500 utenti e uno con 50.

### Cosa funziona

1. **`class_weight="balanced"` su RandomForest**: implementato. Modifica il peso delle classi nell'impurity gini → predizioni più eque.
2. **Macro-F1 come metrica primaria**: pesa ogni classe uguale, indipendente dalla frequenza.
3. **Confusion matrix nel report**: per identificare visivamente quali classi sono mal predette.

### Cosa non ho fatto (estensioni)

- **SMOTE**: oversampling sintetico delle classi minoritarie. Da applicare DENTRO la pipeline (no leakage). Tipicamente migliora macro-F1 di 1-3 punti.
- **`sample_weight`** custom in XGBoost: assegnare pesi inversamente proporzionali alla frequenza. ~equivalente a `class_weight="balanced"` di RF.
- **Cost-sensitive learning**: definire $c_{ij}$ esplicito (vedi `docs/teoria/04-classificazione-multiclasse-metriche.md`, sezione 4.1).

## 7. Riproducibilità

`RANDOM_STATE = 42` fissato in `config.py` e propagato a:

- `KMeans(random_state=...)`
- `GaussianMixture(random_state=...)`
- `RandomForestClassifier(random_state=...)`
- `XGBClassifier(random_state=...)`
- `_sample_silhouette` (numpy generator locale)

Vedi [`docs/teoria/05-pipeline-riproducibile-seed.md`](../teoria/05-pipeline-riproducibile-seed.md) per i dettagli su seeding e idempotenza.

## 8. Quale sarebbe il next step?

Per portare il modello in produzione:

1. **Tuning iperparametri**: GridSearchCV / Optuna su `n_estimators`, `max_depth`, `learning_rate` di XGBoost. Atteso miglioramento +1-3 pt macro-F1.
2. **Stacking** RF + XGB con meta-learner (LogReg). Atteso +1-2 pt.
3. **Drift detection**: monitor mensile sulle distribuzioni delle 16 feature (KS-test). Allerta se le distribuzioni cambiano significativamente.
4. **Retraining schedule**: re-fit ogni 30 giorni con nuovi snapshot. Walk-forward sulla performance.
5. **A/B test del classificatore** (vecchio vs nuovo) prima del rollout.
6. **API REST** che esponga `predict_user_cluster()` (FastAPI + Docker).
7. **Logging delle predizioni** con feature usate e timestamp, per debug e per costruire dataset di follow-up.
8. **Integrazione con un model registry** (MLflow): tracking metriche, artefatti, esperimenti.

Tutto fuori scope per il PW didattico, ma il codice è già strutturato per supportarlo (config centralizzata, seed fissati, persistenza esplicita di scaler+modello).
