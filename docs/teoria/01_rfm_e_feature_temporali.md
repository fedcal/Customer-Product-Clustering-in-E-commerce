---
layout: default
title: RFM & feature temporali
parent: Teoria
nav_order: 1
math: mathjax
description: >-
  Analisi RFM (Recency-Frequency-Monetary), arricchita con propensione,
  varietà di esplorazione e price sensitivity. Feature point-in-time
  calcolate "as-of" per evitare data leakage temporale.
---

# RFM e feature temporali point-in-time
{: .no_toc }

> *"Il segnale non è quanto un utente ha comprato in totale: è quanto ha comprato fino a ieri."*

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Cos'è l'analisi RFM e perché funziona

La **RFM analysis** (Recency, Frequency, Monetary) nasce nel direct marketing degli anni '90 — prima dei sistemi di raccomandazione moderni — come metodo semplice ma efficace per segmentare la base clienti. Su tre numeri costruisce una caratterizzazione potente:

- **Recency** ($R$): giorni dall'ultimo acquisto. Più $R$ è alto, più l'utente è "freddo".
- **Frequency** ($F$): numero di acquisti nello storico. Più $F$ è alto, più l'utente è abituale.
- **Monetary** ($M$): spesa totale nello storico. Più $M$ è alto, più l'utente è di valore.

Per il PW, calcoliamo RFM per ogni `user_id` rispetto a uno snapshot temporale `as_of`:

$$
\begin{aligned}
R_u(t) &= t - \max_{e \in E_u^{(<t)}} \text{timestamp}(e) \\
F_u(t) &= \left| \{ e \in E_u^{(<t)} \mid \text{action}(e) = \text{purchase} \} \right| \\
M_u(t) &= \sum_{e \in E_u^{(<t)},\, \text{action}=\text{purchase}} \text{price}(e)
\end{aligned}
$$

dove $E_u^{(<t)}$ è l'insieme degli eventi dell'utente $u$ con `timestamp < t`. La condizione $< t$ è la **barriera anti-leakage**: tutto ciò che accade dopo $t$ NON entra nelle feature.

Implementazione in `src/ecom_clustering/rfm.py`:

```python
def compute_rfm(events_history, as_of, user_universe=None):
    purchases = events_history[events_history["action"] == "purchase"]
    grp = purchases.groupby("user_id")
    last_purchase = grp["timestamp"].max()
    rfm = pd.DataFrame({
        "recency_days": (as_of - last_purchase).dt.total_seconds() / 86400.0,
        "frequency":    grp.size(),
        "monetary":     grp["price"].sum(),
    })
    rfm["has_purchased"] = 1
    # ... reindex + fill cold users ...
    return rfm
```

!!! warning "Cold users"
    Gli utenti senza acquisti nello storico (cold) hanno $R$ indefinita. Imputiamo con $\max(R) + 1$ (sentinella esplicita): "non hanno mai acquistato, sono ancora più vecchi del più vecchio noto". Il segnale "cold" è codificato anche da `has_purchased = 0`.

## 2. Perché RFM da solo non basta

Il PW richiede esplicitamente di **arricchire** RFM con feature comportamentali. La ragione è semplice: tre numeri non bastano a descrivere comportamenti complessi. Due utenti con lo stesso $(R, F, M)$ possono avere profili radicalmente diversi:

- Uno guarda 100 prodotti diversi prima di acquistare → **explorer**.
- L'altro guarda 5 prodotti e compra subito → **decisivo**.

La RFM non distingue questi due, ma il classificatore di cluster futuro lo deve fare. Aggiungiamo quindi:

### 2.1 Propensione alla conversione

Rapporto fra acquisti e interazioni totali:

$$
\text{conv\_rate}_u = \frac{n_{\text{purchase}}}{n_{\text{view}} + n_{\text{cart}} + n_{\text{purchase}}}
$$

Un utente con `conv_rate ≈ 0.05` è un browser; con `conv_rate ≈ 0.30` è un acquirente determinato. Variante più stringente: `cart_to_purchase_rate = n_purchase / n_add_to_cart`, che misura "quanti carrelli si chiudono".

### 2.2 Varietà di esplorazione

Numero di **categorie** e **prodotti distinti** viewati:

$$
\text{n\_cat\_viewed}_u = \left| \{ c \mid \exists e \in E_u^{(<t)}: \text{category}(e) = c \land \text{action} = \text{view} \} \right|
$$

Un utente che vede 12 categorie diverse è un **explorer** (utile per ads/cross-sell); uno che vede 1–2 categorie è un **focused buyer** (utile per email transazionali).

### 2.3 Sensibilità al prezzo

Rapporto fra prezzo medio dei prodotti **acquistati** e prezzo medio dei prodotti **viewati**:

$$
\text{price\_sensitivity}_u = \frac{\overline{\text{price}}_{\text{purchase}}}{\overline{\text{price}}_{\text{view}}}
$$

- $< 1$ → l'utente acquista in media più economico di quello che guarda → **caccia all'occasione**.
- $\approx 1$ → coerente: compra ciò che guarda.
- $> 1$ → premium buyer: guarda l'economico ma compra il premium (es. dopo riflessione).

### 2.4 Recente vs storico

Per cogliere **trend** (utente in crescita o in calo) calcoliamo le interazioni nella finestra recente (default 14 giorni) e il rapporto con il totale:

$$
\text{recent\_to\_total}_u = \frac{n_{\text{interactions}}^{[t-W,\, t)}}{n_{\text{interactions}}^{(<t)}}
$$

dove $W = 14$ giorni. Questo rapporto è in $[0, 1]$:

- $\approx 1$ → tutta l'attività è recente, utente "appena attivato".
- $< 0.1$ → attivo storicamente ma in fase di abbandono.

Implementazione completa in `src/ecom_clustering/features.py::compute_user_features`. La lista finale di colonne usate per il clustering è 16 feature numeriche (vedi `select_user_modeling_features`).

## 3. Point-in-time: la regola d'oro

Una **feature point-in-time** è una feature il cui valore al tempo $t$ può essere calcolato usando solo eventi con `timestamp < t`. Tutte le 16 feature utente del PW sono point-in-time **per costruzione**, perché entrano dentro `compute_user_features` solo dopo che è stato applicato:

```python
events_history = filter_events_until(events, as_of)   # timestamp < as_of
feats = compute_user_features(events_history, as_of, ...)
```

!!! danger "Esempio di feature NON point-in-time"
    Calcolare `monetary_total` come somma su TUTTI gli eventi del dataset, indipendentemente da `as_of`, è leakage: stiamo dicendo al modello "questo utente alla fine spenderà tot", ma quel valore include eventi futuri.

    Il bug è subdolo perché in fase di training/test il modello ottiene metriche eccellenti — ma in produzione, dove "il futuro" non è disponibile, le metriche crollano.

## 4. Feature prodotto

Per il clustering prodotti applichiamo la stessa logica:

| Feature | Calcolo |
|---|---|
| `n_view`, `n_add_to_cart`, `n_purchase` | conteggio per `product_id` |
| `conversion_rate` | $n_{\text{purchase}} / n_{\text{interactions}}$ |
| `avg_price` | media `price` osservata |
| `n_unique_users` | distinct di `user_id` che hanno interagito |
| `base_price`, `quality_score`, `base_conversion`, `base_popularity` | attributi statici da `products.csv` |

Anche qui, `events_history` è filtrato a $<$ `as_of`. I prodotti senza eventi pre-`as_of` ottengono 0 (cold start). Vedi `compute_product_features` in `features.py`.

## 5. Standardizzazione

Le 16 feature utente hanno scale molto diverse:

- `recency_days` ∈ [0, 365]
- `monetary` ∈ [0, 10000+]
- `conversion_rate` ∈ [0, 1]

Senza standardizzazione, K-Means (basato su distanza euclidea) si fa dominare dalle feature con varianza più alta — `monetary` schiaccia `conversion_rate`. Applichiamo `StandardScaler`:

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

dove $\mu_j, \sigma_j$ sono media e deviazione standard della feature $j$ calcolate sul training set. In produzione si riusano le stesse statistiche del training (vedi `joblib.dump({"scaler": user_scaler, ...})`).

## 6. Cosa NON ho fatto (e perché)

- **Log-transform di `monetary`**: ridurrebbe la skewness, ma con 1200 utenti la varianza dovuta a outlier è gestita dallo scaler. In dataset più grandi sarebbe consigliabile.
- **Target encoding di `category_id`**: per il clustering non c'è target, e per il classificatore le categorie viewate sono già aggregate in `n_categories_viewed`. Estensione possibile per dataset più ricchi.
- **Embedding di `product_id`**: irrilevante per RFM, utile per modelli di raccomandazione (collaborative filtering, two-tower). Fuori scope.

## 7. Riferimenti

- **Bult, J. R. & Wansbeek, T.** (1995), *Optimal Selection for Direct Mail*, Marketing Science 14(4) — formulazione storica della RFM.
- **Hughes, A. M.** (2005), *Strategic Database Marketing*, McGraw-Hill — punto di riferimento pratico.
- **Sklearn user guide**: [`StandardScaler`](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling).
- **Kaufman et al.** (2012), *Leakage in Data Mining: Formulation, Detection, and Avoidance*, ACM TKDD 6(4) — taxonomy del leakage temporale.
