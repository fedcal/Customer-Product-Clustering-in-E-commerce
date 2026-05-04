---
sidebar_position: 3
title: "Split temporali & no leakage"
description: |
  Hold-out cronologico, walk-forward validation, two-snapshot. Definizione di leakage temporale ed esempi pratici nel contesto del clustering RFM.
---

# Split temporali e prevenzione del data leakage

> *"In serie temporali, il futuro non è un campione casuale del passato."*

## 1. Il problema

Il PW01 ha una dimensione esplicitamente **temporale**: gli eventi sono datati e l'obiettivo è predire l'**appartenenza futura** ai cluster. Questo cambia tutto rispetto a un classificatore tradizionale:

- Non possiamo usare `train_test_split(events, ..., random_state=42)` — mescolerebbe passato e futuro.
- Non possiamo usare `KFold(shuffle=True)` — stessa ragione.
- La label non è una colonna data: la **dobbiamo costruire** noi a partire dagli eventi futuri rispetto allo snapshot.

## 2. Definizione formale di data leakage temporale

> **Leakage temporale**: l'utilizzo, in fase di training del modello, di informazioni che al tempo di osservazione $t$ NON sarebbero state disponibili.

In formula: una feature $\phi$ usata per predire al tempo $t$ ha leakage se esistono eventi $e$ con $\text{timestamp}(e) \geq t$ tali che $\phi$ dipende da $e$.

Esempi nel PW:

| Cosa | Leakage? | Perché |
|---|---|---|
| `recency_days(u, t) = t − last_purchase(u, before t)` | NO | usa solo eventi `< t` |
| `recency_days(u, t) = t − last_purchase(u, all events)` | SÌ | usa eventi futuri |
| `monetary(u, t) = sum(price, purchases before t)` | NO | corretto |
| `n_recent_interactions = events in [t-14, t)` | NO | finestra chiusa a destra esclusiva |
| `future_cluster(u, t) = cluster(features(u, t+H))` | È la LABEL | non è leakage: è il target |

La regola operativa: **filtrare gli eventi con `timestamp < as_of` PRIMA di calcolare qualsiasi feature**. Nel PW questo è imposto da `filter_events_until` in `data.py`:

```python
def filter_events_until(events, as_of, inclusive=False):
    op = events["timestamp"] <= as_of if inclusive else events["timestamp"] < as_of
    return events.loc[op].copy()
```

## 3. La struttura "due snapshot" del PW

Il PW richiede di valutare il classificatore su **periodi realmente futuri**. La soluzione adottata in `pipeline.py::_pick_snapshots`:

```
                 t_max - 2H    t_max - H        t_max
                      │             │             │
   ─────────────━━━━━━●━━━━━━━━━━━━●─────────────●  asse temporale
                      │             │             │
                      │             │             │
              as_of_train     as_of_test    horizon end
                  (1)             (2)            (3)
                      │             │             │
                      └─── H ──────┘             │
                                    └─── H ──────┘
```

Concretamente con $H = 30$ giorni:

1. **`as_of_train`** = `t_max − 2H`: snapshot a cui osserviamo le feature di training del classificatore.
2. **`as_of_test`** = `t_max − H`: snapshot a cui osserviamo le feature di test.
3. La **label** di ogni utente è il cluster a cui apparterrà dopo $H$ giorni:
   - Per il train: cluster a `as_of_train + H` = `t_max − H` (entro il dataset).
   - Per il test: cluster a `as_of_test + H` = `t_max` (limite ultimo del dataset).

Vantaggi di questa struttura:

- **Train e test sono temporalmente disgiunti**: nessuna sovrapposizione informativa.
- **Le feature sono point-in-time** rispetto al rispettivo snapshot.
- **Le label** sono calcolate applicando il modello di clustering (fittato sul training!) alle feature future.

Il flusso anti-leakage critico è in `labeling.py::make_future_user_clusters`:

```python
def make_future_user_clusters(ecom, as_of, horizon_days, cluster_model, scaler, ...):
    future_t = as_of + pd.Timedelta(days=horizon_days)
    events_future = filter_events_until(ecom.events, future_t)
    feats_future = compute_user_features(events_future, future_t, ...)
    X = scaler.transform(feats_future[cols].to_numpy())  # scaler dal TRAINING
    labels = cluster_model.predict(X)                    # K-Means dal TRAINING
    return pd.Series(labels, index=feats_future.index, ...)
```

:::danger[Sottiglia: il cluster_model è fittato sul training]

Le label future del **test set** vengono ottenute applicando il K-Means **fittato su `as_of_train`** alle feature di `as_of_test + H`. Questo è essenziale: se rifittassimo K-Means sui dati di test, l'identità dei cluster cambierebbe (label = 0 nel training potrebbe essere label = 2 nel test) e il classificatore non avrebbe nulla da imparare.

:::

## 4. Strategie di split temporale (panoramica)

### 4.1 Hold-out cronologico

Il più semplice: scegli una data $t^*$, tutto prima è train, tutto dopo è test.

```
events:  ──────●───────●───●─────●──────●───●──●────●─────
                                    ↑
                                   t*
              [────── train ────][────── test ──────]
```

Pro: semplicissimo, riproducibile.
Contro: una sola stima di performance → varianza alta della metrica.

**È quello che facciamo nel PW**, con la variante "due snapshot" descritta sopra.

### 4.2 Walk-forward (rolling)

Per stimare la performance media nel tempo:

```
fold 1:  [── train ──][test]
fold 2:  [─── train ────][test]
fold 3:  [──── train ─────][test]
...
```

Si addestra il modello a $K$ tempi diversi, si misura la performance media. È quello che useremmo se il PW richiedesse una stima robusta del classificatore (es. monitoraggio in produzione).

### 4.3 Time-series K-fold (sklearn `TimeSeriesSplit`)

Variante di walk-forward in cui i fold di train crescono progressivamente:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_temporal)):
    # train_idx ⊂ {0, 1, ..., t}
    # test_idx  ⊂ {t+1, ..., T}
    ...
```

Da NON usare con `shuffle=True`. Da usare quando il dataset è ordinato per tempo e non c'è una struttura "snapshot" come la nostra.

## 5. Errori comuni di leakage nel contesto del PW

### 5.1 Calcolare RFM su eventi non filtrati

```python
# WRONG
rfm_train = compute_rfm(ecom.events, as_of=as_of_train)   # usa anche eventi futuri!
rfm_test  = compute_rfm(ecom.events, as_of=as_of_test)
```

Anche se `compute_rfm` ha il parametro `as_of`, **non filtra automaticamente** gli eventi: si fida che chi chiama abbia già filtrato. Il pattern corretto (usato nella pipeline) è:

```python
# RIGHT
events_train_history = filter_events_until(ecom.events, as_of_train)
rfm_train = compute_rfm(events_train_history, as_of=as_of_train)
```

### 5.2 Standardizzare con statistiche globali

```python
# WRONG: scaler vede train+test
scaler = StandardScaler().fit(np.vstack([X_train, X_test]))
```

Il test set "dona" la sua media e std al training. Le metriche risultano gonfiate.

```python
# RIGHT
scaler = StandardScaler().fit(X_train)
X_test_s = scaler.transform(X_test)   # riusa mu/sigma del train
```

### 5.3 Generare label future con K-Means rifittato

```python
# WRONG: rifittare KMeans sui dati di test cambia l'identità dei cluster
kmeans_test = KMeans(n_clusters=K, random_state=42).fit(X_test)
y_test = kmeans_test.predict(X_test)
```

L'unico modo corretto: **riusare il modello di clustering fittato sul training** (vedi sezione 3).

### 5.4 Tunare iperparametri sul test set

Anche con uno split temporale corretto, se modifichi il modello dopo aver visto le metriche di test, il test set diventa implicitamente parte del training. Usa una **finestra di validation** intermedia (es. `as_of_val = t_max − 1.5H`) per il tuning, riserva il test per la valutazione finale.

## 6. Sanity check: lo "shuffle test"

Per verificare empiricamente l'assenza di leakage:

```python
# Mescola le label di training (rompe la relazione X → y)
y_train_shuffled = pd.Series(y_train.values).sample(frac=1, random_state=0).values

clf = fit_random_forest(X_train_clf, y_train_shuffled)
acc_shuffled = (clf.predict(X_test_clf) == y_test_clf).mean()

print(f"Accuracy con label shuffled: {acc_shuffled:.3f}")
# Atteso: ≈ 1/n_classes (random). Se > 1.5/n_classes → c'è leakage residuo.
```

Se ottieni accuracy molto sopra il random, una feature sta "trapelando" il target — tipicamente una feature derivata dalle label senza accorgersene.

## 7. Cosa NON è leakage (non confondere)

- **Avere le features di test in formato comparabile con quelle di train**: ovvio, deve essere così, basta non usare statistiche del test.
- **Ordinare gli eventi per timestamp**: ovvio, è un pre-processing semantico.
- **Conoscere la struttura del problema** (es. che esiste una colonna `purchase`): la conoscenza del dominio non è leakage.

Il leakage è specificamente l'uso di **valori** (statistiche aggregate, label, eventi futuri) che non sarebbero noti al tempo $t$.

## 8. Riferimenti

- **Kaufman, Rosset, Perlich, Stitelman** (2012), *Leakage in Data Mining: Formulation, Detection, and Avoidance*, ACM TKDD 6(4) — paper di riferimento, taxonomy del leakage.
- **Kapoor, A. & Narayanan, A.** (2023), *Leakage and the Reproducibility Crisis in ML-based Science*, Patterns 4(9).
- **Bergmeir, C. & Benítez, J. M.** (2012), *On the use of cross-validation for time series predictor evaluation*, Information Sciences 191.
- **Sklearn user guide**: [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- **Hyndman, R. J. & Athanasopoulos, G.**, *Forecasting: Principles and Practice* — capitolo 5 sul time series cross-validation.
