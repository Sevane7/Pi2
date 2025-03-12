# Analyse de la Dépendance Sérielle et Stratégie Systématique

## 1. Introduction
Ce projet vise à analyser la dépendance sérielle des prix en utilisant le paramètre de Hurst estimé via la méthode du log-périodogramme. Ensuite, nous exploitons cette information pour réaliser un forecasting basé sur la covariance des log-prix et la formule de covariance des fBm. Enfin, nous effectuons un backtest en évaluant différentes métriques de performance.

## 2. Téléchargement des Données
Les données haute fréquence (1min) et daily sont récupérées via Bloomberg.

## 3. Estimation du Paramètre de Hurst
L'estimation est réalisée grâce à la méthode du log-périodogramme, qui applique une régression linéaire en échelle log-log sur le spectre de puissance du fractional Gaussian noise (fGn) pour en déduire le paramètre de Hurst. Le fGn est obtenu en prenant les différences successives du fBm, ce qui permet d'obtenir un processus stationnaire.

## 4. Forecasting avec la Covariance des Log-Prix
L’étape suivante consiste à prédire les mouvements de prix en exploitant la covariance des log-prix et la covariance des fBm.

## 5. Backtest et Évaluation des Performances
Nous testons la stratégie en mesurant le **hit ratio** sur différentes combinaisons de prévisions (taille de matrice, horizon de forecast). Ensuite, nous calculons les métriques classiques de performance :

- **Sharpe ratio**
- **Volatilité**
- **Max Drawdown (MDD)**
- **Profit & Loss (P&L)**

## 6. Conclusion
Ce projet permet d'exploiter la dépendance sérielle des prix pour construire une stratégie systématique et l'évaluer rigoureusement via un backtest.
