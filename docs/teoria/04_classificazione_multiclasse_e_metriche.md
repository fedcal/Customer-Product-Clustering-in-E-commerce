---
layout: default
title: Classificazione multiclasse & metriche
parent: Teoria
nav_order: 4
math: mathjax
description: >-
  Accuracy, macro-F1, weighted-F1, balanced accuracy, log-loss e confusion
  matrix. Scelta della metrica con classi sbilanciate e impatto degli
  errori "costosi" nel contesto e-commerce.
---

# Classificazione multiclasse e metriche
{: .no_toc }

> *"Una metrica scelta male trasforma un modello mediocre in 'state-of-the-art' e un buon modello in spazzatura. La metrica è la prima decisione di modeling."*

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Setup del problema

Il PW01 è un problema di **classificazione multiclasse**:

- **Input** $\mathbf{x}_u$: 16 feature dell'utente $u$ alla data `as_of` (RFM-extended).
- **Output** $y_u \in \{0, 1, \dots, K-1\}$: cluster di appartenenza atteso a `as_of + horizon`, dove $K$ è il numero di cluster scelto dal K-Means.

Modelli candidati: `RandomForestClassifier` e `XGBClassifier` (vedi `src/ecom_clustering/supervised.py`). Entrambi gestiscono nativamente più di 2 classi.

## 2. Le 4 metriche fondamentali

### 2.1 Accuracy

$$
\text{Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\{ \hat{y}_i = y_i \}
$$

Frazione di predizioni corrette. **Inadatta con classi sbilanciate**: se il 80% degli utenti è nel cluster 0, predire sempre 0 dà accuracy 0.80 ma è un classificatore inutile.

### 2.2 Macro-F1

Calcola precision e recall per ogni classe, poi media:

$$
\text{F1}_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}
\quad,\quad
\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k
$$

dove $P_k$ e $R_k$ sono precision e recall della classe $k$. La media è **non pesata**: ogni classe pesa uguale, indipendentemente dalla sua frequenza.

**Caso d'uso**: quando ci interessa che il modello funzioni anche sulle classi minoritarie. È la **metrica primaria del PW**: cluster ad alto valore (VIP) sono numericamente piccoli ma hanno il massimo impatto business — non vogliamo che vengano "sciolti" nel cluster di base.

### 2.3 Weighted-F1

$$
\text{Weighted-F1} = \sum_{k=1}^K \frac{n_k}{n} \cdot \text{F1}_k
$$

Media pesata per la frequenza di ogni classe. Più "vicina" all'accuracy: dà importanza maggiore alle classi numerose.

**Caso d'uso**: quando ti interessa la performance "media" pesata sul mix reale di classi. Da riportare insieme a macro-F1, mai da solo.

### 2.4 Balanced accuracy

$$
\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^K \text{Recall}_k
$$

Media non pesata della recall per classe. È equivalente all'accuracy se tutte le classi avessero la stessa frequenza. **Robusta allo sbilanciamento**.

## 3. Perché macro-F1 è la metrica primaria del PW

Il PW richiede esplicitamente di "discutere gli errori più rilevanti dal punto di vista applicativo, ad esempio errori che coinvolgono cluster ad alto valore". In e-commerce:

- Cluster "VIP" → 5-10% degli utenti, generano 30-50% del fatturato.
- Cluster "occasionali" → 50-70% degli utenti, generano 20-30% del fatturato.

Una accuracy alta mascherata (predire sempre "occasionale") fallisce il test business. La **macro-F1** invece "punisce" il modello che ignora le classi minoritarie — perché la $\text{F1}_{\text{VIP}}$ entra nella media con peso 1, esattamente come la $\text{F1}_{\text{occasionali}}$.

Implementazione (`evaluate_classifier` in `supervised.py`):

```python
return {
    "macro_f1":           float(f1_score(y, y_pred, average="macro",    zero_division=0)),
    "weighted_f1":        float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    "balanced_accuracy":  float(balanced_accuracy_score(y, y_pred)),
    "accuracy":           float((y_pred == y).mean()),
    "confusion_matrix":   confusion_matrix(y, y_pred).tolist(),
    "classification_report": classification_report(y, y_pred, zero_division=0, output_dict=True),
}
```

## 4. La confusion matrix

Per un problema $K$-classi, è una matrice $K \times K$ dove l'elemento $(i, j)$ è il numero di esempi reali della classe $i$ predetti come $j$:

$$
M_{ij} = \left| \{ u : y_u = i \land \hat{y}_u = j \} \right|
$$

La diagonale ($i = j$) sono le predizioni corrette. Tutto il resto è errore.

### 4.1 Errori "costosi"

Non tutti gli errori sono uguali. Nel PW:

| Errore | Costo business |
|---|---|
| VIP → predetto VIP | 0 (corretto) |
| VIP → predetto occasionale | ALTO (perdiamo retention investment su cliente di valore) |
| Occasionale → predetto VIP | MEDIO (sprechiamo ads premium su utente di basso valore) |
| Occasionale → predetto occasionale | 0 (corretto) |
| Cold → predetto attivo | BASSO (al peggio mandiamo email a utente inattivo) |

In presenza di asimmetrie di costo, una metrica "monetaria" $\sum_{i,j} M_{ij} \cdot c_{ij}$ è più informativa di macro-F1. Per il PW non costruiamo $c_{ij}$ esplicito (richiede dati che non abbiamo), ma la **discussione qualitativa** della confusion matrix è obbligatoria nel report finale.

### 4.2 Visualizzare la confusion matrix

```python
from ecom_clustering.evaluation import plot_confusion_matrix

plot_confusion_matrix(
    cm=eval_result["confusion_matrix"],
    class_names=[f"C{i}" for i in range(n_classes)],
    title="XGBoost: confusion matrix (test)",
    save_path="reports/figures/xgb_confmat.png",
)
```

Letture chiave:

- **Diagonale dominante** → modello ragionevole.
- **Riga $k$ con valori non sulla diagonale** → modello fatica a riconoscere la classe $k$.
- **Colonna $k$ piena di valori** → il modello "imputa troppe cose" a $k$ (over-prediction).
- **Pattern triangolare superiore/inferiore** → confusione fra classi adiacenti (ordine semantico, comune con cluster derivati da scoring continuo).

## 5. Log-loss (cross-entropy)

Per modelli che predicono probabilità (`predict_proba`), la **log-loss** è una metrica più granulare:

$$
\text{LogLoss} = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K \mathbb{1}\{y_i = k\} \, \log p_{ik}
$$

Penalizza non solo le predizioni sbagliate, ma anche quelle "indecise" sulla classe corretta. Utile per:

- Calibrare il modello (se le probabilità non riflettono frequenze reali, log-loss è alta anche con accuracy buona).
- Confrontare modelli su cui si fanno operazioni di soft-decision (es. ranking di utenti per probabilità di essere VIP).

Nel PW non la includiamo nelle metriche primarie (la traccia non lo richiede), ma è facile da aggiungere con `sklearn.metrics.log_loss(y, y_proba)`.

## 6. Cross-validation per stime stabili

Una singola valutazione su test set ha **alta varianza**. Per stime più robuste, K-fold CV sul training:

```python
from ecom_clustering.supervised import cross_val_macro_f1

mean, std = cross_val_macro_f1(rf_model, X_train_clf, y_train_clf, cv=5)
print(f"macro-F1: {mean:.3f} ± {std:.3f}")
```

!!! warning "CV su problema temporale: vedi capitolo split"
    `cross_val_score(model, X, y, cv=5)` con default `KFold(shuffle=False)` su array che NON sono ordinati per tempo NON garantisce split temporali. Per il PW, la valutazione "ufficiale" è sull'holdout temporale `as_of_test`. La CV sul training è solo un sanity check sulla varianza interna del modello.

## 7. Class imbalance: cosa fare

Se i cluster sono fortemente sbilanciati (capita: alcuni cluster hanno 50 utenti, altri 500), opzioni:

### 7.1 `class_weight="balanced"` (RandomForest)

Ogni classe è pesata inversamente alla sua frequenza nel computo della loss:

$$
w_k = \frac{n}{K \cdot n_k}
$$

Implementato nel PW:

```python
RandomForestClassifier(
    ...,
    class_weight="balanced",   # gestisce squilibrio classi
)
```

### 7.2 Oversampling con SMOTE

Genera esempi sintetici della classe minoritaria. Da applicare **dentro la pipeline**, non sul dataset prima dello split (altrimenti, leakage). Libreria: `imbalanced-learn`.

Pro: spesso migliora macro-F1. Contro: dipende dalla qualità dell'interpolazione e può creare punti irrealistici nello spazio delle feature.

### 7.3 Threshold tuning (binario)

Per problemi binari si può abbassare la soglia di decisione (default 0.5) per favorire la recall della classe minoritaria. Per multiclasse è più complesso (serve un threshold per classe + tie-breaking) → fuori scope per il PW.

### 7.4 Cost-sensitive learning

Definire la matrice di costo $c_{ij}$ e ottimizzare la **loss attesa** invece di accuracy. Realistica in scenari con costi noti, ma richiede dati di business.

## 8. Riassunto per il PW

Segui questa checklist quando valuti il classificatore del cluster futuro:

1. [ ] Calcola **macro-F1** sull'holdout temporale `as_of_test`. Metrica primaria.
2. [ ] Calcola **balanced accuracy** e **weighted-F1**. Metriche secondarie.
3. [ ] Plot della **confusion matrix** con label leggibili (`C0`, `C1`, ...).
4. [ ] Identifica i **due cluster con macro-F1 più bassa** e discutili nel report.
5. [ ] Confronta con baseline "majority class" (predire sempre il cluster più popoloso): se il modello non batte questa baseline, è inutile.
6. [ ] (Opzionale) Riporta log-loss se hai `predict_proba`.

## 9. Riferimenti

- **Sokolova, M. & Lapalme, G.** (2009), *A systematic analysis of performance measures for classification tasks*, Information Processing & Management 45(4) — review esauriente delle metriche.
- **He, H. & Garcia, E. A.** (2009), *Learning from Imbalanced Data*, IEEE TKDE 21(9).
- **Sklearn user guide**: [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
- **Bishop, C. M.**, *Pattern Recognition and Machine Learning*, cap. 1.5 (decision theory) e cap. 4 (linear models for classification).
