# 📈 Prédiction du Prix de l'Action NVIDIA — LSTM & Bi-LSTM

> Projet de prévision de séries temporelles financières — Comparaison de modèles classiques et de réseaux de neurones récurrents (LSTM vs Bi-LSTM) pour prédire le prix de clôture de l'action NVIDIA (2004–2023).

---

## 🎯 Problématique

L'objectif est de construire une pipeline complète de data science sur les données boursières historiques de NVIDIA, en comparant des **régressions classiques** (Linear Regression, Random Forest, SVR, Gradient Boosting) aux **réseaux de neurones récurrents** (LSTM et Bi-LSTM), qui sont spécialement conçus pour capturer les dépendances temporelles dans les séries financières.

---

## 🗂️ Structure du projet

```
📁 nvidia_prediction/
│
├── 📓 nvidia_stock_prediction.ipynb    # Notebook principal — Pipeline complet
├── 🐍 nvidia_prediction.py             # Script Python
│
├── 📊 nvidia_stock_prices.csv          # Dataset principal (2004–2023)
│
└── 📁 images/
    ├── comparaison_metriques.png       # Comparaison MAE / RMSE / R² (LSTM vs Bi-LSTM)
    └── comparaison_predictions.png     # Courbes prix réel vs LSTM vs Bi-LSTM
```

📥 **Dataset :** [`nvidia_stock_prices.csv`](https://github.com/sarra725/nvidia-stock-prediction/blob/main/01_data/nvidia_data.csv)

| Colonne | Description |
|---|---|
| `Date` | Date de la transaction (2004–2023) |
| `Open` | Prix d'ouverture |
| `High` / `Low` | Prix max / min de la journée |
| `Close` | Prix de clôture (**cible à prédire**) |
| `Volume` | Nombre de transactions |

---

## ⚙️ Pipeline

### 1️⃣ Exploration & Visualisation (EDA)
- Prix de clôture et volume sur toute la période
- Zoom 2022–2023 avec bandes OHLC
- Moyennes mobiles : MA 20j, MA 50j, MA 200j
- Distribution des rendements journaliers
- Boxplot des prix par année
- Matrice de corrélation (Open, High, Low très corrélés à Close → exclus du modèle)

### 2️⃣ Feature Engineering
Features basées uniquement sur le **passé** pour éviter le data leakage :

| Feature | Description |
|---|---|
| `Close_lag1/2/5/10` | Prix de clôture à J-1, J-2, J-5, J-10 |
| `Volume_lag1` | Volume à J-1 |
| `Rendement_lag1` | Rendement journalier à J-1 |
| `Volatilite_7j` | Volatilité sur 7 jours glissants |
| `MA_5/20/50_feat` | Moyennes mobiles décalées |

### 3️⃣ Modélisation
- **Split temporel 80/20** (pas de shuffle pour préserver l'ordre chronologique)
- **Séquences de 60 jours** pour LSTM et Bi-LSTM (`look_back = 60`)
- **Normalisation MinMaxScaler** avant les modèles deep learning
- **Optimiseur Adam** — Loss MSE — 100 epochs

---

## 🤖 Modèles implémentés

| Modèle | Type |
|---|---|
| Linear Regression | Régression classique (baseline) |
| Random Forest | Ensemble d'arbres |
| SVR | Support Vector Regression |
| Gradient Boosting | Boosting classique |
| **LSTM** | Réseau récurrent — mémoire unidirectionnelle |
| **Bi-LSTM** | Réseau récurrent — mémoire bidirectionnelle |

📄 [`nvidia_prediction.py`](https://github.com/sarra725/nvidia-stock-prediction/blob/main/02_scripts/nvidia.ipynb)

---

## 📊 Résultats & Comparaison LSTM vs Bi-LSTM

| Modèle | MAE ↓ | RMSE ↓ | R² ↑ |
|:---|:---:|:---:|:---:|
| LSTM | 2.576 | 3.912 | 0.885 |
| 🥇 **Bi-LSTM** | **2.409** | **3.761** | **0.893** |

> 💡 **Conclusion :** Le **Bi-LSTM** surpasse le LSTM sur toutes les métriques grâce à sa capacité à analyser la séquence dans les deux sens (passé et futur). L'erreur est réduite de ~6% sur le MAE et ~4% sur le RMSE.

### Exemple de prédiction sur un point réel (jour 50 du test set)

| | Valeur |
|---|---|
| Prix réel | $8.5253 |
| Prédiction Bi-LSTM | $8.2787 (erreur : 2.89%) |
| Prédiction LSTM | $8.0988 (erreur : 5.00%) |

---

## 📈 Visualisations

### Comparaison des métriques (MAE / RMSE / R²)
![Comparaison LSTM vs Bi-LSTM — MAE, RMSE, R²](https://github.com/sarra725/nvidia-stock-prediction/blob/main/03_images/comparaison_metriques.png)

### Prédictions vs Prix réel sur l'ensemble de test
![Comparaison LSTM vs Bi-LSTM vs Prix Réel](https://github.com/sarra725/nvidia-stock-prediction/blob/main/03_images/comparaison_predictions.png)

> *Les deux modèles suivent bien la tendance générale, mais divergent lors des hausses récentes brutales (2022–2023), typiques des envolées boursières liées à l'IA.*

---

## 🚀 Installation & Utilisation

### Prérequis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch xgboost
```

### Lancer le script

```bash
python nvidia_prediction.py
```

### Lancer le notebook

```bash
jupyter notebook nvidia_stock_prediction.ipynb
```

> ⚠️ **Important :** Placer `nvidia_stock_prices.csv` à la racine du projet ou adapter le chemin :
> ```python
> df = pd.read_csv("nvidia_stock_prices.csv", parse_dates=['Date'])
> ```

---

## 📦 Dépendances

| Bibliothèque | Usage |
|---|---|
| `pandas` / `numpy` | Manipulation et feature engineering |
| `scikit-learn` | Modèles classiques, métriques, preprocessing |
| `torch` (PyTorch) | Architecture LSTM et Bi-LSTM |
| `matplotlib` / `seaborn` | Visualisations et courbes |

---

## 👩‍💻 Auteure

**Sarra** — Étudiante en Deep Learning & Séries Temporelles  
Projet réalisé dans le cadre d'un cours de prévision financière par réseaux de neurones récurrents.
