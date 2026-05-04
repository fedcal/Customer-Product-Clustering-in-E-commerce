---
layout: default
title: K-Means & alternative
parent: Teoria
nav_order: 2
math: mathjax
description: >-
  K-Means come baseline, scelta di K via elbow e silhouette, confronto con
  Gaussian Mixture, DBSCAN e Hierarchical. Limiti su feature non-gaussiane
  e ruolo della standardizzazione.
---

# Clustering: K-Means e le sue alternative
{: .no_toc }

> *"K-Means è veloce, semplice e quasi sempre il primo modello che provi. È anche quasi sempre l'ultimo che ti convince davvero."*

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Cos'è il clustering e perché lo usiamo

Il **clustering** è il problema di partizionare $n$ punti in $\mathbb{R}^p$ in $K$ gruppi disgiunti tali che punti dentro lo stesso gruppo siano "simili" e punti in gruppi diversi siano "dissimili". È un task **non supervisionato**: non c'è una label di riferimento, la qualità è un giudizio strutturale (silhouette, inertia, separazione).

Nel PW lo usiamo per **etichettare** gli utenti (e i prodotti) con un gruppo discreto, in modo che:

1. Si possano profilare i gruppi in linguaggio business ("VIP", "occasionali", "cold", ...).
2. Si possa addestrare un classificatore supervisionato a predire l'appartenenza futura al gruppo.

## 2. K-Means: l'algoritmo di Lloyd

Dato un numero $K$ di cluster, K-Means risolve:

$$
\min_{\{C_k\}_{k=1}^K,\, \{\boldsymbol{\mu}_k\}} \sum_{k=1}^K \sum_{\mathbf{x} \in C_k} \| \mathbf{x} - \boldsymbol{\mu}_k \|_2^2
$$

cioè minimizza la somma delle distanze quadratiche dei punti dal centroide del proprio cluster (la **inertia**, in scikit-learn). L'algoritmo iterativo di Lloyd:

1. Inizializza $K$ centroidi (es. con `k-means++`).
2. Assegna ogni punto al centroide più vicino (E-step).
3. Ricalcola ogni centroide come media dei punti assegnati (M-step).
4. Ripeti finché non convergi.

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500)
labels = model.fit_predict(X_standardized)
```

!!! note "Perché `n_init=10`"
    L'algoritmo è sensibile all'inizializzazione e converge a un **minimo locale**. `n_init=10` esegue 10 inizializzazioni random e mantiene quella con inertia minima. È un costo lineare in `n_init` ma drasticamente più stabile.

### Punti di forza

- **Veloce**: $O(n \cdot K \cdot p \cdot \text{iter})$. Su 1200 utenti × 16 feature, una run è < 1 secondo.
- **Semplice**: 1 iperparametro ($K$).
- **Riproducibile** con `random_state` fissato.
- **Predict per nuovi punti**: $\arg\min_k \| \mathbf{x}_{\text{new}} - \boldsymbol{\mu}_k \|$ — utile per la fase di label-future del PW.

### Punti deboli

- Assume **cluster sferici** ed **equa varianza**. Se i cluster reali hanno forma allungata o curvilinea, K-Means li frammenta.
- Sensibile agli **outlier**: la media è sbilanciata da pochi punti estremi.
- Richiede di scegliere $K$ **a priori**.
- Distanza euclidea → richiede **feature standardizzate** (vedi sezione 5).

## 3. Scelta di K: elbow e silhouette

### 3.1 Metodo elbow (gomito)

Si fitta K-Means per vari $K$ e si plotta l'inertia:

```python
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42).fit(X)
    print(k, model.inertia_)
```

L'inertia decresce monotonicamente con $K$ (più centroidi → meno distanza media). Si cerca il **"gomito"** della curva: il punto oltre il quale aggiungere cluster smette di valere la pena.

Limite: il gomito spesso è **soggettivo**. Su dati continui può non esserci un gomito chiaro.

### 3.2 Silhouette score

Per ogni punto $i$ nel cluster $C_a$:

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
$$

dove:

- $a(i)$ = distanza media di $i$ dagli altri punti dello stesso cluster (coesione).
- $b(i) = \min_{C_b \neq C_a} \overline{d}(i, C_b)$ = distanza media dal cluster più vicino diverso dal proprio (separazione).

$s(i) \in [-1, 1]$:

- $s \approx 1$ → punto ben classificato.
- $s \approx 0$ → al confine fra due cluster.
- $s < 0$ → assegnato al cluster sbagliato.

Lo **silhouette score globale** è la media di $s(i)$. Si sceglie $K$ che la massimizza.

Implementazione nel PW (`src/ecom_clustering/clustering.py::select_k`):

```python
def select_k(X, k_range=(3,4,5,6,7,8), fit_fn=fit_kmeans, random_state=42):
    results = []
    for k in k_range:
        r = fit_fn(X, k=k, random_state=random_state)
        results.append(r)   # contiene .silhouette
    best = max(results, key=lambda r: r.silhouette)
    return best.k, results
```

!!! tip "Silhouette su sample"
    Il calcolo della silhouette è $O(n^2)$ — pesante su milioni di punti.
    Usiamo `_sample_silhouette` con campionamento di 5000 punti: trade-off
    fra costo e stabilità della stima.

### 3.3 Quando l'elbow e la silhouette discordano

Capita spesso. Caso tipico:

- Elbow → $K=3$ (poi inertia decresce poco).
- Silhouette → $K=5$ (cluster più piccoli sono più coesi).

Nel PW il PW ammette $K \in \{3, ..., 8\}$ e usiamo la silhouette come criterio finale (più robusta del gomito visivo). Se la silhouette è sotto 0.20, il segnale di clustering è debole — il dato è "continuo", non si divide naturalmente in gruppi netti.

## 4. Alternative al K-Means

### 4.1 Gaussian Mixture Models (GMM)

Generalizzazione probabilistica di K-Means: invece di assegnare un punto a un cluster, ne stima la **probabilità di appartenenza** a ciascuno. Modello generativo:

$$
p(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

Fit con EM (Expectation-Maximization). Vantaggi:

- **Cluster ellittici** (non solo sferici): `covariance_type="full"` permette covarianze arbitrarie per cluster.
- **Soft assignment**: utile quando i cluster si sovrappongono.

Svantaggi:

- Più parametri da stimare → richiede più dati.
- Più lento di K-Means.
- Sensibile all'inizializzazione (come K-Means).

Nel PW confrontiamo K-Means vs GMM con stesso $K$ (`compare_kmeans_vs_gmm`):

```python
def compare_kmeans_vs_gmm(X, k, random_state=42):
    rk = fit_kmeans(X, k=k, random_state=random_state)
    rg = fit_gmm(X,    k=k, random_state=random_state)
    return pd.DataFrame([
        {"model": rk.name, "silhouette": rk.silhouette, "inertia": rk.inertia},
        {"model": rg.name, "silhouette": rg.silhouette, "inertia": None},
    ])
```

### 4.2 DBSCAN

**Density-Based Spatial Clustering of Applications with Noise**. Non richiede $K$: trova "regioni dense" (più di `min_samples` punti entro raggio `eps`) e considera tutto il resto rumore.

Vantaggi:

- Non assume forma dei cluster (qualsiasi forma curvilinea va bene).
- Ha una classe naturale per gli **outlier** (label `-1`).

Svantaggi:

- Sensibile ai parametri `eps` e `min_samples` (e non c'è un metodo standard per sceglierli).
- In alta dimensione la "densità" perde significato (curse of dimensionality).
- Non ha `predict()` per nuovi punti — non si applica direttamente al label-future del PW.

**Verdetto per il nostro caso**: scartato in produzione perché il clustering deve essere applicabile a snapshot futuri (servirebbe rifittarlo ogni volta con nuovi parametri). Utile come EDA esplorativa.

### 4.3 Hierarchical Clustering (Ward)

Costruisce un dendrogramma agglomerativo: ogni punto parte come cluster singolo, si fondono i due più vicini (criterio Ward = minima incrementi di varianza), si itera fino ad avere 1 solo cluster. Si "taglia" il dendrogramma a un'altezza scelta.

Vantaggi:

- Visualizzazione gerarchica intuitiva.
- Non richiede $K$ (lo si decide tagliando).

Svantaggi:

- $O(n^2)$ in memoria, $O(n^3)$ in tempo per Ward → impraticabile oltre 10k punti.
- `predict()` non esiste — stesso problema di DBSCAN per il PW.

## 5. Standardizzazione: NON è opzionale

K-Means usa la distanza euclidea. Senza standardizzazione, le feature con varianza più alta dominano la distanza. Esempio: nel PW abbiamo `monetary ∈ [0, 10000]` e `conversion_rate ∈ [0, 1]`.

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \dots}
$$

Se $x_1$ è `monetary`, una differenza di $1000\$$ contribuisce per $10^6$ alla distanza; una differenza di $0.5$ in `conversion_rate` contribuisce per $0.25$. Il clustering ignora di fatto `conversion_rate`.

Soluzione standard: **z-score** (`StandardScaler`):

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
# In produzione/test riusiamo lo stesso scaler:
X_test_s = scaler.transform(X_test)
```

Tutte le feature post-scaling hanno media 0 e varianza 1 → contribuiscono **equamente** alla distanza.

!!! warning "Salvare lo scaler"
    Non basta standardizzare in fase di training: lo scaler ($\mu, \sigma$ del training) va **persistito** insieme al modello di clustering. Vedi `joblib.dump({"cluster_model": ..., "scaler": user_scaler, ...})` in `pipeline.py`. In inferenza si applica `scaler.transform(X_new)` con le statistiche del training, mai si ri-fittano sul nuovo dato.

### Alternative allo z-score

- **MinMaxScaler** (`[0, 1]`): utile se le feature hanno una scala "naturale" e non gaussiana, ma sensibile agli outlier.
- **RobustScaler** (mediana + IQR): robusto agli outlier. Buona scelta per `monetary` se il dataset ha pochi utenti VIP che spendono molto. Da considerare come miglioramento per il PW.
- **PowerTransformer (Yeo-Johnson)**: rende la distribuzione approssimativamente gaussiana. Buon match per K-Means che assume cluster sferici, ma costa interpretabilità.

Nel PW usiamo `StandardScaler` per semplicità e perché il dataset sintetico è ragionevolmente pulito.

## 6. Riassunto operativo

| Algoritmo | Quando | Pro | Contro |
|---|---|---|---|
| **K-Means** | baseline, dati continui, cluster sferici, serve `predict()` | veloce, semplice, riproducibile | assume sfericità, sensibile a outlier |
| **GMM** | cluster ellittici, soft assignment | più flessibile di K-Means | più lento, richiede più dati |
| **DBSCAN** | EDA, cluster di forma arbitraria, identificazione outlier | nessun $K$, robusto | no `predict()`, sensibile a `eps` |
| **Hierarchical** | $n < 10k$, visualizzazione gerarchica | intuitivo, no $K$ | $O(n^3)$, no `predict()` |

**Per il PW01**: K-Means come modello primario (serve `predict` per generare label future), GMM come confronto "almeno due approcci" richiesto dalla traccia.

## 7. Riferimenti

- **Lloyd, S.** (1982), *Least Squares Quantization in PCM*, IEEE TIT 28(2) — algoritmo originale.
- **Arthur, D. & Vassilvitskii, S.** (2007), *k-means++: The Advantages of Careful Seeding*, SODA — inizializzazione moderna.
- **Rousseeuw, P. J.** (1987), *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*, J. Comp. Appl. Math. 20.
- **Sklearn user guide**: [Clustering](https://scikit-learn.org/stable/modules/clustering.html).
- **Hastie, Tibshirani, Friedman**, *The Elements of Statistical Learning*, cap. 14 (unsupervised learning).
