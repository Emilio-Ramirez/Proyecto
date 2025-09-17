# proyecto_final_bolsa_v2.py
# ------------------------------------------------------------
# Proyecto final (OFFLINE) - Clasificación binaria “¿mañana sube?”
# Estilo tareas: funciones separadas, evaluación con métricas,
# validación temporal (train/test 80/20), CV (TimeSeriesSplit),
# guardado de ROC, pipelines .joblib y reporte Markdown.
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# -----------------------------
# Configuración por defecto (~150 filas)
# -----------------------------
TICKERS_DEFAULT = ["AAPL", "MSFT", "NVDA"]  # 3 tickers
DAYS_DEFAULT = 70                           # 70 días -> ~50 útiles/ticker
TRAIN_FRACTION = 0.80
RANDOM_STATE = 42

# “Micro-tendencias” y volatilidades por ticker (para que sea aprendible)
DRIFT = {"AAPL":0.0004,"MSFT":0.0003,"NVDA":0.0008,"AMZN":0.0005,"GOOGL":0.0002,"META":0.0006,"TSLA":0.0007,"JPM":0.0001,"XOM":0.0001,"KO":0.00005}
SIGMA = {"AAPL":0.012,"MSFT":0.011,"NVDA":0.020,"AMZN":0.015,"GOOGL":0.012,"META":0.016,"TSLA":0.022,"JPM":0.009,"XOM":0.010,"KO":0.007}
BASEP = {"AAPL":190,"MSFT":420,"NVDA":1200,"AMZN":170,"GOOGL":140,"META":470,"TSLA":200,"JPM":210,"XOM":110,"KO":62}

@dataclass
class Results:
    metrics_df: pd.DataFrame
    best_model_name: str

# ============================================================
# 1) Generación de datos OHLCV sintéticos (100% offline)
# ============================================================
def gen_business_days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-01", periods=n)

def generate_ohlcv(ticker: str, n_days: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Random walk con drift y momentum leve. Devuelve DataFrame con columnas OHLCV."""
    rng = np.random.default_rng(seed + (abs(hash(ticker)) % 10**6))
    mu, s, p0 = DRIFT.get(ticker, 0.0), SIGMA.get(ticker, 0.012), BASEP.get(ticker, 100.0)

    rets = np.zeros(n_days, dtype=float)
    for t in range(n_days):
        eps = rng.normal()
        mom = 0.15 * (rets[t-1] if t else 0.0)   # momentum suave
        rets[t] = mu + s*eps + mom

    close = p0 * np.cumprod(1.0 + rets)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.003, size=n_days)) + 0.002
    high = np.maximum(open_, close) * (1 + spread)
    low  = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(mean=15, sigma=0.35, size=n_days).astype(int)

    return pd.DataFrame(
        {"Open":open_, "High":high, "Low":low, "Close":close, "Adj Close":close, "Volume":volume},
        index=gen_business_days(n_days)
    )

# ============================================================
# 2) Ingeniería de características y etiqueta y(t)=1[Close_{t+1}>Close_t]
# ============================================================
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores en t y etiqueta usando t+1 (sin fuga)."""
    close, vol = df["Close"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    # Señales de momentum / medias / volatilidad
    out["ret_1"]  = close.pct_change(1)
    out["ret_5"]  = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    out["sma_5"]  = close.rolling(5).mean()
    out["sma_10"] = close.rolling(10).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["sma5_gt_sma20"] = (out["sma_5"] > out["sma_20"]).astype(int)

    out["std_5"]  = close.pct_change().rolling(5).std()
    out["std_10"] = close.pct_change().rolling(10).std()
    out["std_20"] = close.pct_change().rolling(20).std()

    out["rsi_14"] = compute_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    out["vol_chg"] = vol.pct_change()
    out["vol_znorm_20"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std())

    out["dow"] = out.index.dayofweek  # 0..6

    # Etiqueta binaria
    out["y"] = (close.shift(-1) > close).astype(int)

    return out.dropna()

# ============================================================
# 3) Construcción del dataset (split temporal 80/20 por ticker)
# ============================================================
def build_datasets(tickers: List[str], days_per_ticker: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for tk in tickers:
        raw = generate_ohlcv(tk, days_per_ticker)
        feats = build_features(raw)
        feats["ticker"] = tk
        feats["date"] = feats.index

        n = len(feats); ntr = int(n * TRAIN_FRACTION)
        train_parts.append(feats.iloc[:ntr].copy())
        test_parts .append(feats.iloc[ntr:].copy())
        print(f"[{tk}] total={n}  train={len(feats.iloc[:ntr])}  test={len(feats.iloc[ntr:])}")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)

    print(f"[INFO] train={len(train_df)}  test={len(test_df)}  total={len(train_df)+len(test_df)}")
    assert len(train_df) >= 100 and len(test_df) >= int(0.2*len(train_df)), \
        "No se cumple el mínimo (≥100 train y ≥20% test). Sube --days o añade tickers."

    # Guardar datasets (útiles para el reporte)
    train_df.to_csv("files/dataset_train.csv", index=False)
    test_df .to_csv("files/dataset_test.csv", index=False)
    print("[OK] Guardado: files/dataset_train.csv y files/dataset_test.csv")
    return train_df, test_df

# ============================================================
# 4) Preprocesamiento y modelos (pipelines)
# ============================================================
def make_ohe_dense():
    """Compatibilidad entre versiones de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # <=1.1

def split_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    drop = ["y","date"]
    X = df[[c for c in df.columns if c not in drop]].copy()
    y = df["y"].astype(int).values
    cat_cols = ["ticker","dow"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    return ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", make_ohe_dense(), cat_cols)],
        remainder="drop"
    )

def build_models() -> Dict[str, object]:
    """Modelos vistos en clase."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB()
    }

# ============================================================
# 5) Validación (CV en train) + Entrenamiento final + Evaluación en test
# ============================================================
def crossval_on_train(name: str, pipe: Pipeline, Xtr: pd.DataFrame, ytr: np.ndarray) -> float:
    """CV temporal (TimeSeriesSplit) para estimar desempeño en train (métrica: F1)."""
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(pipe, Xtr, ytr, cv=tscv, scoring="f1")
    print(f"[CV] {name}: F1 mean={scores.mean():.4f} (splits={len(scores)})")
    return float(scores.mean())

def evaluate_on_test(name: str, pipe: Pipeline, Xte: pd.DataFrame, yte: np.ndarray) -> Dict[str, float]:
    ypred = pipe.predict(Xte)
    yprob = pipe.predict_proba(Xte)[:,1] if hasattr(pipe, "predict_proba") else None

    acc  = accuracy_score(yte, ypred)
    prec = precision_score(yte, ypred)
    rec  = recall_score(yte, ypred)
    f1   = f1_score(yte, ypred)
    auc  = roc_auc_score(yte, yprob) if yprob is not None else np.nan
    cm   = confusion_matrix(yte, ypred)

    print(f"\n=== {name} (TEST) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print("Matriz de confusión (rows=verdad, cols=pred):\n", cm)
    print("\nClassification report:\n", classification_report(yte, ypred, digits=4))

    if yprob is not None:
        fpr, tpr, _ = roc_curve(yte, yprob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"files/roc_{name}.png", dpi=160)
        plt.close()
        print(f"[OK] Guardado: files/roc_{name}.png")

    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "roc_auc":auc}

def train_cv_eval(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Results:
    Xtr, ytr, num_cols, cat_cols = split_Xy(train_df)
    Xte, yte, _, _ = split_Xy(test_df)

    pre = build_preprocessor(num_cols, cat_cols)
    models = build_models()

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        # CV en train (estilo tareas anteriores)
        cv_f1 = crossval_on_train(name, pipe, Xtr, ytr)
        # Entrenamiento final y prueba en test
        pipe.fit(Xtr, ytr)
        test_metrics = evaluate_on_test(name, pipe, Xte, yte)
        joblib.dump(pipe, f"files/pipeline_{name}.joblib")
        rows.append({"modelo":name, "cv_f1":cv_f1, **test_metrics})

    metrics_df = pd.DataFrame(rows).set_index("modelo").sort_values("roc_auc", ascending=False)
    metrics_df.to_csv("files/metrics.csv")
    print("\n[RESUMEN MÉTRICAS TEST]\n", metrics_df, "\n[OK] Guardado: files/metrics.csv")

    best_model_name = metrics_df.index[0]
    print(f"[MEJOR MODELO] {best_model_name}")
    return Results(metrics_df=metrics_df, best_model_name=best_model_name)

# ============================================================
# 6) Predicción final sobre la última fila de test de un ticker
# ============================================================
def predict_last_test_row(ticker: str, best_model_name: str, test_df: pd.DataFrame):
    sub = test_df[test_df["ticker"] == ticker]
    if sub.empty:
        print(f"[ERROR] No hay filas de test para {ticker}.")
        return
    x = sub.iloc[-1].drop(labels=["y","date"])
    pipe = joblib.load(f"files/pipeline_{best_model_name}.joblib")
    pred = int(pipe.predict(pd.DataFrame([x]))[0])
    prob = float(pipe.predict_proba(pd.DataFrame([x]))[0][1]) if hasattr(pipe, "predict_proba") else np.nan
    print(f"\n[PREDICCIÓN - {ticker}] Modelo: {best_model_name}")
    print(f"¿Mañana sube? -> {pred}  (1=sube, 0=no sube)")
    if not np.isnan(prob):
        print(f"Prob(Sube): {prob:.3f}")
    print("Nota: educativo; no es consejo financiero.")

# ============================================================
# 7) Reporte Markdown autogenerado (resumen para entregar)
# ============================================================
def write_markdown_report(tickers: List[str], days: int, train_df: pd.DataFrame, test_df: pd.DataFrame, results: Results):
    md = []
    md.append("# Proyecto Final — Predicción binaria de movimiento diario (OFFLINE)")
    md.append(f"**Tickers:** {', '.join(tickers)}  \n**Días por ticker:** {days}  \n**Split temporal:** {int(TRAIN_FRACTION*100)}/{int(100-TRAIN_FRACTION*100)}")
    md.append("")
    md.append("## Definición del problema")
    md.append("Dado el historial reciente de una acción, predecir si el **cierre de mañana** será mayor que el de hoy (`y=1` si sube, `0` en otro caso).")
    md.append("")
    md.append("## Datos y features")
    md.append("- Datos OHLCV **sintéticos** (random walk con drift y momentum) — *sin internet*. ")
    md.append("- Features: retornos (`ret_1, ret_5, ret_10`), medias móviles (`sma_5, sma_10, sma_20`), volatilidad (`std_5, std_10, std_20`), `rsi_14`, `macd`/`signal`/`hist`, volumen (`vol_chg`, `vol_znorm_20`), día de semana (`dow`).")
    md.append(f"- Tamaños: **train={len(train_df)}**, **test={len(test_df)}** (total={len(train_df)+len(test_df)}).")
    md.append("")
    md.append("## Modelos y evaluación")
    md.append("- Algoritmos vistos en clase: **Regresión Logística**, **KNN**, **GaussianNB**.")
    md.append("- Validación: **TimeSeriesSplit (3)** en *train* (métrica: F1) + evaluación final en *test* con **Accuracy, Precision, Recall, F1, ROC-AUC** y **matriz de confusión**.")
    md.append("")
    md.append("## Resultados (test)")
    md.append(results.metrics_df.reset_index().to_markdown(index=False))
    md.append("")
    md.append("Se guardaron curvas ROC: `files/roc_LogisticRegression.png`, `files/roc_KNN.png`, `files/roc_GaussianNB.png`.")
    md.append("")
    md.append("## Conclusión breve")
    md.append("El problema de horizonte 1 día es ruidoso; aun así, los indicadores de momentum/volatilidad permiten un desempeño moderado. No hay fuga temporal (split por fecha).")
    with open("reporte_resumen.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("[OK] Guardado: reporte_resumen.md")

# ============================================================
# 8) Main (CLI)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Proyecto Final (OFFLINE) — Clasificación binaria “¿mañana sube?”")
    parser.add_argument("--tickers", type=str, default=",".join(TICKERS_DEFAULT), help="Ej.: AAPL,MSFT,NVDA")
    parser.add_argument("--days", type=int, default=DAYS_DEFAULT, help="Días por ticker (>=40 recomendado)")
    parser.add_argument("--predict", type=str, default=None, help="Ticker para predecir al final (ej.: --predict AAPL)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    days = args.days

    # 0) Create files directory if it doesn't exist
    os.makedirs("files", exist_ok=True)

    # 1) Dataset (train/test temporal por ticker)
    train_df, test_df = build_datasets(tickers, days)

    # 2) Entrenar + CV + evaluar en test
    results = train_cv_eval(train_df, test_df)

    # 3) Reporte Markdown (resumen)
    write_markdown_report(tickers, days, train_df, test_df, results)

    # 4) Predicción opcional
    if args.predict:
        tk = args.predict.upper()
        if tk in tickers:
            predict_last_test_row(tk, results.best_model_name, test_df)
        else:
            print(f"[ERROR] {tk} no está en {tickers}")

if __name__ == "__main__":
    main()
