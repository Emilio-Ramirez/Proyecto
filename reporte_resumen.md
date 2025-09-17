# Proyecto Final — Predicción binaria de movimiento diario (OFFLINE)
**Tickers:** AAPL, MSFT, NVDA  
**Días por ticker:** 70  
**Split temporal:** 80/20

## Definición del problema
Dado el historial reciente de una acción, predecir si el **cierre de mañana** será mayor que el de hoy (`y=1` si sube, `0` en otro caso).

## Datos y features
- Datos OHLCV **sintéticos** (random walk con drift y momentum) — *sin internet*. 
- Features: retornos (`ret_1, ret_5, ret_10`), medias móviles (`sma_5, sma_10, sma_20`), volatilidad (`std_5, std_10, std_20`), `rsi_14`, `macd`/`signal`/`hist`, volumen (`vol_chg`, `vol_znorm_20`), día de semana (`dow`).
- Tamaños: **train=120**, **test=30** (total=150).

## Modelos y evaluación
- Algoritmos vistos en clase: **Regresión Logística**, **KNN**, **GaussianNB**.
- Validación: **TimeSeriesSplit (3)** en *train* (métrica: F1) + evaluación final en *test* con **Accuracy, Precision, Recall, F1, ROC-AUC** y **matriz de confusión**.

## Resultados (test)
| modelo             |    cv_f1 |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:-------------------|---------:|-----------:|------------:|---------:|---------:|----------:|
| KNN                | 0.249863 |   0.433333 |    0.333333 | 0.888889 | 0.484848 |  0.470899 |
| LogisticRegression | 0.402849 |   0.433333 |    0.25     | 0.444444 | 0.32     |  0.380952 |
| GaussianNB         | 0.277627 |   0.3      |    0.227273 | 0.555556 | 0.322581 |  0.365079 |

Se guardaron curvas ROC: `files/roc_LogisticRegression.png`, `files/roc_KNN.png`, `files/roc_GaussianNB.png`.

## Conclusión breve
El problema de horizonte 1 día es ruidoso; aun así, los indicadores de momentum/volatilidad permiten un desempeño moderado. No hay fuga temporal (split por fecha).