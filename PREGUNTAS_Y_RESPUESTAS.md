# Preguntas y Respuestas - Proyecto Final: Predicción de Acciones

## Tabla de Contenidos
1. [Preguntas Generales del Proyecto](#preguntas-generales-del-proyecto)
2. [Decisiones de Datos y Features](#decisiones-de-datos-y-features)
3. [Selección y Comparación de Algoritmos](#selección-y-comparación-de-algoritmos)
4. [Validación y Evaluación](#validación-y-evaluación)
5. [Implementación Técnica](#implementación-técnica)
6. [Limitaciones y Mejoras](#limitaciones-y-mejoras)
7. [Preguntas Teóricas Profundas](#preguntas-teóricas-profundas)

---

## Preguntas Generales del Proyecto

### ❓ **¿Por qué eligieron el problema de predicción de acciones?**

**Respuesta:**
Elegimos este problema por múltiples razones estratégicas:

1. **Complejidad Apropiada**: Es lo suficientemente complejo para demostrar múltiples conceptos de ML, pero manejable con las técnicas vistas en clase
2. **Relevancia Práctica**: Las finanzas son un dominio donde ML tiene aplicaciones reales y medibles
3. **Datos Ricos**: Los mercados financieros generan series temporales densas ideales para ML
4. **Evaluación Clara**: Las métricas binarias son fáciles de interpretar y comparar
5. **Desafío Temporal**: Permite demostrar comprensión de validación temporal y data leakage

### ❓ **¿Por qué clasificación binaria y no regresión?**

**Respuesta:**
La clasificación binaria es más apropiada porque:

- **Objetivo Práctico**: En trading, la decisión es "comprar/no comprar", no predecir precio exacto
- **Robustez**: Menos sensible a outliers que regresión de precios
- **Interpretabilidad**: "¿Sube mañana?" es más actionable que "precio = $X"
- **Métricas**: Precision/Recall tienen significado directo en finanzas (falsos positivos = pérdidas)
- **Datos Sintéticos**: Es más fácil generar patrones binarios realistas que precios exactos

### ❓ **¿Por qué horizonte de 1 día y no más largo?**

**Respuesta:**
**Ventajas del horizonte diario:**
- **Datos Suficientes**: Con 70 días generamos ~50 ejemplos útiles por ticker
- **Menor Complejidad**: Factores de corto plazo son más predecibles
- **Menos Noise**: Eventos macro de largo plazo no dominan la señal

**Desventajas reconocidas:**
- **Mayor Ruido**: Movimientos diarios tienen más randomness
- **Menor Persistencia**: Patrones de 1 día son menos estables

**¿Qué pasaría con horizonte más largo?**
- **Semanas/Meses**: Necesitaríamos más datos históricos y factores macro
- **Horas**: Requerirían datos intraday y microestructura de mercado

---

## Decisiones de Datos y Features

### ❓ **¿Por qué datos sintéticos en lugar de datos reales?**

**Respuesta Detallada:**

**Ventajas de Datos Sintéticos:**
1. **Reproducibilidad**: Mismos resultados en cualquier máquina/tiempo
2. **Control Total**: Podemos ajustar complejidad y signal-to-noise ratio
3. **Sin Dependencias**: No requiere APIs, claves, o conexión internet
4. **Experimentación**: Podemos probar diferentes regímenes de mercado
5. **Realismo Controlado**: Implementamos características conocidas (momentum, volatilidad)

**Limitaciones:**
1. **Factores Externos**: No captura noticias, eventos geopolíticos
2. **Microestructura**: No incluye bid-ask spreads, liquidez
3. **Regímenes**: No simula crisis, bubbles, cambios estructurales
4. **Correlaciones**: Patrones inter-activos simplificados

**¿Qué haríamos con datos reales?**
```python
# Integraríamos APIs como:
import yfinance as yf
import alpha_vantage
# Pero perderíamos reproducibilidad y control
```

### ❓ **¿Por qué eligieron esas features específicas?**

**Respuesta por Categoría:**

#### **Momentum (ret_1, ret_5, ret_10)**
- **Justificación**: Capturan tendencias recientes
- **Teoría**: Momentum effect es una anomalía documentada en finanzas
- **Implementación**: `close.pct_change(n)`

#### **Medias Móviles (SMA 5, 10, 20)**
- **Justificación**: Indicadores de tendencia más utilizados
- **Teoría**: Suavizan ruido y muestran dirección
- **Señal**: `sma5_gt_sma20` captura crossovers bullish/bearish

#### **Volatilidad (std_5, std_10, std_20)**
- **Justificación**: Mide incertidumbre y riesgo
- **Teoría**: Volatility clustering - períodos volátiles tienden a continuar
- **Implementación**: Rolling standard deviation de retornos

#### **RSI (14 períodos)**
- **Justificación**: Indicador de momentum más popular
- **Teoría**: Identifica condiciones de sobrecompra/sobreventa
- **Rango**: 0-100, <30 = oversold, >70 = overbought

#### **MACD**
- **Justificación**: Combina momentum y trend-following
- **Componentes**: MACD line, Signal line, Histogram
- **Señales**: Crossovers y divergencias

### ❓ **¿Consideraron features adicionales? ¿Por qué no las incluyeron?**

**Features Consideradas pero No Incluidas:**

1. **Bollinger Bands**
   - **Por qué no**: Correlacionado con volatilidad ya incluida
   - **Cuándo usar**: Si quisiéramos señales de reversión específicas

2. **Stochastic Oscillator**
   - **Por qué no**: Similar a RSI, evitamos redundancia
   - **Trade-off**: Prefirimos simplicidad vs. múltiples osciladores

3. **Volume Indicators (OBV, VWAP)**
   - **Por qué no**: Datos sintéticos de volumen son menos realistas
   - **En datos reales**: Serían muy valiosos

4. **Features Lag**
   - **Por qué no**: Con horizonte 1-día, lags largos reducen datos útiles
   - **Consideración**: `ret_5, ret_10` ya capturan historia reciente

**¿Qué pasaría si las incluyéramos?**
- **Más features**: Riesgo de overfitting con dataset pequeño (~150 samples)
- **Multicolinealidad**: Muchos indicadores técnicos están correlacionados
- **Curse of dimensionality**: Especialmente problemático para KNN

### ❓ **¿Cómo evitaron data leakage?**

**Respuesta Técnica Crítica:**

**1. Construcción de Features:**
```python
# ✅ CORRECTO - Feature en t, label en t+1
out["ret_1"] = close.pct_change(1)  # Usa precio hasta hoy
out["y"] = (close.shift(-1) > close).astype(int)  # Usa precio de mañana
```

**2. Split Temporal:**
```python
# ✅ CORRECTO - Respeta orden temporal
n = len(feats); ntr = int(n * TRAIN_FRACTION)
train = feats.iloc[:ntr]  # Primeros 80%
test = feats.iloc[ntr:]   # Últimos 20%
```

**3. Preprocessing:**
```python
# ✅ CORRECTO - Scaler se ajusta solo en train
pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
pipe.fit(X_train, y_train)  # Scaler ve solo train
```

**4. Cross-Validation:**
```python
# ✅ CORRECTO - TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=3)
# ❌ INCORRECTO - KFold shuffled
cv = KFold(shuffle=True)  # ¡Mezcla temporalidad!
```

**¿Qué pasaría con data leakage?**
- **Overfitting temporal**: Métricas artificialmente altas
- **Falla en producción**: Modelo inútil con datos nuevos
- **Invalidez científica**: Conclusiones erróneas

---

## Selección y Comparación de Algoritmos

### ❓ **¿Por qué eligieron esos tres algoritmos específicamente?**

**Respuesta Estratégica:**

#### **Regresión Logística**
**Por qué incluir:**
- **Baseline**: Estándar de industria para clasificación binaria
- **Interpretabilidad**: Coeficientes muestran importancia de features
- **Probabilidades calibradas**: Útil para decisiones con incertidumbre
- **Eficiencia**: Rápida para entrenamiento y predicción
- **Robustez**: Menos propensa a overfitting

**Cuándo fallaría:**
- Relaciones fuertemente no-lineales
- Interacciones complejas entre features

#### **K-Nearest Neighbors**
**Por qué incluir:**
- **No-paramétrico**: No asume forma funcional específica
- **Flexibilidad**: Puede capturar patrones locales complejos
- **Simplicidad conceptual**: Fácil de entender y explicar
- **Benchmark**: Buen punto de comparación

**Limitaciones esperadas:**
- **Curse of dimensionality**: Performance degrada con muchas features
- **Sensibilidad al ruido**: Outliers afectan vecindarios
- **Computacionalmente caro**: O(n) para cada predicción

#### **Gaussian Naive Bayes**
**Por qué incluir:**
- **Velocidad**: Extremadamente rápido
- **Pocos datos**: Funciona bien con samples limitados
- **Probabilístico**: Proporciona incertidumbre
- **Baseline probabilístico**: Asunción fuerte de independencia

**Cuándo esperamos que falle:**
- Features correlacionadas (violación de independencia)
- Distribuciones no-gaussianas

### ❓ **¿Por qué no usaron Random Forest o XGBoost?**

**Respuesta:**
1. **Restricción del curso**: Debemos usar algoritmos enseñados en clase
2. **Comparabilidad**: Queremos comparar paradigmas básicos diferentes
3. **Interpretabilidad**: RF/XGB son "black boxes" más complejas
4. **Datos limitados**: ~150 samples, modelos complejos harían overfitting

**¿Qué esperaríamos con RF/XGB?**
- **Mejores métricas**: Probablemente ROC-AUC > 0.7
- **Overfitting**: Especialmente con validation tan pequeña
- **Menos interpretabilidad**: Harder to explain decisions

### ❓ **¿Cómo decidieron los hiperparámetros?**

**Respuesta Detallada:**

#### **Regresión Logística**
```python
LogisticRegression(max_iter=1000, random_state=42)
```
- **max_iter=1000**: Asegura convergencia con datasets pequeños
- **Regularización**: Usamos L2 por defecto (apropiado para features correlacionadas)
- **Solver**: 'lbfgs' por defecto (bueno para datasets pequeños)

#### **KNN**
```python
KNeighborsClassifier(n_neighbors=5)
```
- **k=5**: Balance entre bias (k muy alto) y variance (k muy bajo)
- **Rule of thumb**: k ≈ sqrt(n_samples) = sqrt(120) ≈ 11, pero 5 es más conservador
- **Distance**: Euclidean por defecto (apropiado con StandardScaler)

#### **Naive Bayes**
```python
GaussianNB()  # Solo parámetros por defecto
```
- **Sin hiperparámetros**: NB tiene pocos parámetros ajustables
- **Smoothing**: Usa Laplace smoothing automáticamente

**¿Por qué no Grid Search?**
- **Datos limitados**: Cross-validation con 3 splits ya reduce mucho los datos
- **Scope del proyecto**: Focus en comparación de algoritmos, no optimización
- **Riesgo de overfitting**: Con ~40 test samples, fácil overfit to validation

### ❓ **¿Esperaban que Logistic Regression fuera el mejor?**

**Respuesta Analítica:**

**Por qué tenía sentido:**
1. **Features lineales**: Muchos indicadores técnicos tienen relaciones aproximadamente lineales
2. **Dimensionalidad**: ~15 features es manageable para LR
3. **Regularización**: L2 penalty ayuda con multicolinealidad de indicators
4. **Datos balanceados**: LR funciona bien cuando clases están balanceadas

**Sorpresas en resultados:**
- **KNN peor de lo esperado**: Curse of dimensionality con 15 features
- **NB competitive**: A pesar de features claramente correlacionadas
- **Margen pequeño**: Diferencias entre modelos menores de lo esperado

**¿Qué habría cambiado el ranking?**
1. **Más datos**: KNN mejoraría con más samples
2. **Feature selection**: Podría ayudar especialmente a NB
3. **Feature engineering**: Interacciones podrían favorecer a modelos no-lineales

---

## Validación y Evaluación

### ❓ **¿Por qué TimeSeriesSplit en lugar de validación cruzada normal?**

**Respuesta Fundamental:**

**El problema con KFold normal:**
```python
# ❌ PELIGROSO en series temporales
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True)

# Problema: usa datos del futuro para predecir el pasado
# Train: [Jan, Mar, May] → Test: [Feb, Apr]
# ¡Imposible en la realidad!
```

**La solución correcta:**
```python
# ✅ RESPETA la temporalidad
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=3)

# Split 1: Train[1:30] → Test[31:40]
# Split 2: Train[1:40] → Test[41:50]
# Split 3: Train[1:50] → Test[51:60]
```

**¿Por qué es crítico?**
1. **Realismo**: En trading real, solo tienes datos históricos
2. **Data leakage**: CV normal infla artificialmente el performance
3. **Validez**: Resultados reflejan performance real esperada

**¿Qué pasaría con CV normal?**
- **Métricas infladas**: ROC-AUC podría ser >0.8
- **Overconfidence**: Falsa sensación de modelo exitoso
- **Falla en producción**: Modelo inútil con datos nuevos

### ❓ **¿Por qué eligieron F1 como métrica principal para CV?**

**Respuesta Estratégica:**

**F1 Score en contexto financiero:**
- **Balance**: Combina precision (evitar falsos positivos = pérdidas) y recall (capturar oportunidades)
- **Datos desbalanceados**: Maneja mejor que accuracy cuando clases no están 50/50
- **Interpretabilidad**: Una métrica en lugar de dos separadas

**Alternativas consideradas:**

#### **Accuracy**
- **Pro**: Fácil de interpretar
- **Con**: Puede engañar con datos desbalanceados
- **En nuestro caso**: Clases relativamente balanceadas, así que viable

#### **ROC-AUC**
- **Pro**: Independiente del threshold
- **Con**: Puede ser optimista con datos desbalanceados
- **En nuestro caso**: Usamos para evaluación final, no CV

#### **Precision**
- **Pro**: Minimiza falsos positivos (pérdidas)
- **Con**: Puede ser conservador en exceso
- **Uso**: Importante para estrategias de bajo riesgo

#### **Recall**
- **Pro**: Captura todas las oportunidades
- **Con**: Muchos falsos positivos
- **Uso**: Importante para estrategias agresivas

**¿Qué pasaría con otras métricas?**
- **Precision**: Favorecería modelos conservadores (menos trades)
- **Recall**: Favorecería modelos agresivos (más trades)
- **ROC-AUC**: Podría dar rankings diferentes por optimización de threshold

### ❓ **¿Cómo interpretan las métricas obtenidas?**

**Análisis Detallado de Resultados:**

#### **ROC-AUC = 0.63 (Logistic Regression)**
- **Interpretación**: 63% probabilidad de que modelo rank positivos > negativos
- **Benchmark**: Random = 0.5, Perfect = 1.0
- **En finanzas**: >0.6 se considera decent para predicciones diarias
- **Realista**: Mercados eficientes hacen predicción muy difícil

#### **Accuracy = 0.625**
- **Interpretación**: Correcto 62.5% de las veces
- **Contexto**: Buy-and-hold (siempre predecir "sube") tendría ~55% en mercados alcistas
- **Valor agregado**: +7.5 puntos porcentuales sobre naive strategy

#### **Precision = 0.615**
- **Interpretación**: Cuando predice "sube", es correcto 61.5% de las veces
- **Trading**: De 100 señales de compra, ~38 serían pérdidas
- **Risk management**: Necesitaríamos stop-losses y position sizing

#### **Recall = 0.727**
- **Interpretación**: Captura 72.7% de los movimientos alcistas reales
- **Trading**: No perderíamos 27.3% de oportunidades
- **Balance**: Good recall significa no dejar money en la mesa

**¿Son buenos estos resultados?**
- **Para académico**: Excelentes, demuestran signal > noise
- **Para trading real**: Moderados, necesitarían refinamiento
- **Comparado con literatura**: Typical para prediction de 1 día

### ❓ **¿Por qué usaron matriz de confusión?**

**Valor Específico en Finanzas:**

```
Matriz de Confusión:
                 Predicción
                  0    1
Realidad   0    [TN] [FP]  ← Falsos positivos = pérdidas
           1    [FN] [TP]  ← Falsos negativos = oportunidades perdidas
```

**Interpretación de cada cuadrante:**
- **True Positives (TP)**: Predicciones correctas de subida → Ganancias
- **True Negatives (TN)**: Predicciones correctas de bajada → Evitar pérdidas
- **False Positives (FP)**: Predicciones incorrectas de subida → Pérdidas reales
- **False Negatives (FN)**: Predicciones incorrectas de bajada → Oportunidades perdidas

**¿Por qué importa cada tipo de error?**
1. **FP son costosos**: Pérdidas reales de dinero
2. **FN son oportunidad perdida**: No ganamos, pero no perdemos
3. **Asymmetric cost**: En trading, FP puede ser más caro que FN

---

## Implementación Técnica

### ❓ **¿Por qué usaron Pipelines de scikit-learn?**

**Respuesta Técnica:**

**Ventajas principales:**
1. **Prevención de data leakage**: Preprocessing se aplica correctamente
2. **Reproducibilidad**: Todo el flujo encapsulado
3. **Deployment**: Un objeto completo para guardar/cargar
4. **Mantenibilidad**: Código más limpio y modular

**Ejemplo del problema sin Pipeline:**
```python
# ❌ RIESGO de data leakage
scaler = StandardScaler()
scaler.fit(X_train)  # ✅ Bien hasta aquí

# En CV, necesitamos re-escalar para cada fold
# Fácil olvidar y causar leakage
X_val_scaled = scaler.transform(X_val)  # ¿Scaler de qué datos?
```

**Solución con Pipeline:**
```python
# ✅ SEGURO - encapsula todo
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
cross_val_score(pipe, X, y, cv=tscv)  # Automáticamente maneja scaler
```

### ❓ **¿Por qué ColumnTransformer?**

**Problema que resuelve:**
Tenemos features mixtas:
- **Numéricas**: ret_1, sma_5, rsi_14, etc. → Necesitan StandardScaler
- **Categóricas**: ticker, dow → Necesitan OneHotEncoder

**Solución elegante:**
```python
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="drop")
```

**Alternativas consideradas:**
1. **Preprocessing manual**: Más código, más errores
2. **Todo numérico**: Perderíamos información categórica
3. **LabelEncoder**: Implica orden ordinal que no existe

### ❓ **¿Por qué guardaron modelos en .joblib?**

**Justificación práctica:**
1. **Persistencia**: Reutilizar modelos entrenados
2. **Deployment**: Cargar modelo para predicciones nuevas
3. **Reproducibilidad**: Mismo modelo para evaluación posterior
4. **Eficiencia**: No re-entrenar para cada predicción

**Alternativas evaluadas:**
- **Pickle**: Menos eficiente para arrays grandes
- **SaveModel custom**: Más trabajo, menos standard
- **JSON**: No maneja objetos complejos como Pipelines

### ❓ **¿Cómo manejaron la compatibilidad de versiones?**

**Problema específico:**
OneHotEncoder cambió API entre scikit-learn 1.1 y 1.2:
- **v1.1**: `sparse=False`
- **v1.2**: `sparse_output=False`

**Solución implementada:**
```python
def make_ohe_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # <=1.1
```

**Por qué es importante:**
- **Portabilidad**: Funciona en diferentes entornos
- **Robustez**: No falla por versión de sklearn
- **Profesionalismo**: Código production-ready

---

## Limitaciones y Mejoras

### ❓ **¿Cuáles son las principales limitaciones de su enfoque?**

**Respuesta Crítica y Honesta:**

#### **1. Limitaciones de Datos**
- **Sintéticos**: No capturan complejidad real de mercados
- **Tamaño**: ~150 samples es pequeño para ML robusto
- **Scope**: Solo 3 tickers, 70 días cada uno
- **Missing factors**: No news, sentiment, macro indicators

#### **2. Limitaciones de Features**
- **Technical only**: Solo indicadores técnicos, no fundamentals
- **Horizon limitation**: Optimizados para 1 día, no multiperiod
- **No regime detection**: No detecta cambios estructurales de mercado
- **Stationarity**: Asume patrones estables en el tiempo

#### **3. Limitaciones de Modelos**
- **Linear bias**: Logistic Regression asume relationships lineales
- **Algorithm constraint**: Limitados a algoritmos de clase
- **No ensembles**: Podríamos combinar múltiples modelos
- **Static**: No hay online learning o adaptation

#### **4. Limitaciones de Evaluación**
- **Single metric focus**: F1 puede no ser optimal para trading
- **No transaction costs**: Ignora spreads, commissions, slippage
- **No risk metrics**: No Sharpe ratio, max drawdown, VaR
- **Temporal stability**: No evaluamos degradación over time

### ❓ **¿Qué harían diferente si tuvieran más tiempo/recursos?**

**Roadmap de Mejoras:**

#### **Datos y Features (Prioridad Alta)**
1. **Real market data**: Yahoo Finance, Alpha Vantage APIs
2. **More tickers**: Include different sectors, market caps
3. **Longer history**: 2-5 años para más samples
4. **Higher frequency**: Hourly o intraday data
5. **Alternative data**: News sentiment, social media, options flow

#### **Feature Engineering (Prioridad Alta)**
1. **Fundamental features**: P/E, revenue growth, debt ratios
2. **Market microstructure**: Bid-ask spreads, order flow
3. **Cross-asset**: VIX, bond yields, currency rates
4. **Regime indicators**: Volatility regimes, trend classification
5. **Interaction terms**: Non-linear combinations

#### **Modelos (Prioridad Media)**
1. **Tree-based**: Random Forest, XGBoost, LightGBM
2. **Deep learning**: LSTM, GRU for sequential patterns
3. **Ensemble methods**: Voting, stacking, blending
4. **Online learning**: Adaptive models que se actualizan
5. **Bayesian**: Uncertainty quantification

#### **Evaluación (Prioridad Alta)**
1. **Financial metrics**: Sharpe ratio, max drawdown, Calmar ratio
2. **Transaction costs**: Realistic trading simulation
3. **Risk management**: Position sizing, stop losses
4. **Regime analysis**: Performance across different market conditions
5. **Backtesting**: Out-of-sample rolling window validation

### ❓ **¿Qué riesgos ven en aplicar esto en trading real?**

**Análisis de Riesgos Crítico:**

#### **Riesgos del Modelo**
1. **Overfitting**: Modelo funciona en backtest pero falla live
2. **Market regime change**: Patrones cambian, modelo se vuelve obsoleto
3. **Data quality**: Bad data → bad predictions
4. **Latency**: Delays en datos/ejecución afectan performance

#### **Riesgos Financieros**
1. **Drawdowns**: Consecutive losses pueden ser severas
2. **Leverage**: Amplifica tanto gains como losses
3. **Liquidity**: Model assumes perfect execution
4. **Black swan events**: Extreme moves no capturados en training

#### **Riesgos Operacionales**
1. **System failures**: Technology glitches
2. **Regulatory**: Compliance con trading rules
3. **Slippage**: Real execution price ≠ predicted price
4. **Position sizing**: How much capital to risk per trade

#### **Riesgos de Comportamiento**
1. **Overconfidence**: Good backtest ≠ guaranteed profits
2. **Emotional trading**: Manual overrides destroy systematic edge
3. **Parameter tuning**: Constant tweaking destroys strategy
4. **Survivorship bias**: Only successful strategies get published

**Mitigaciones recomendadas:**
- **Paper trading**: Test with virtual money first
- **Position sizing**: Never risk more than 1-2% per trade
- **Stop losses**: Predefined exit rules
- **Diversification**: Multiple strategies, timeframes, assets
- **Monitoring**: Continuous performance tracking
- **Kill switches**: Automatic shutdown if losses exceed threshold

### ❓ **¿Consideraron análisis de sentiment o noticias?**

**Respuesta:**

**Por qué no lo incluimos:**
1. **Scope limitations**: Focus en technical analysis solamente
2. **Data complexity**: News data requires NLP expertise
3. **Real-time challenges**: News impact es inmediato, hard to capture
4. **Synthetic data**: No hay synthetic news fácil de generar

**¿Cómo lo integraríamos?**

#### **Sentiment Analysis**
```python
# Hypothetical pipeline
from textblob import TextBlob
from newsapi import NewsApiClient

def get_sentiment_score(ticker, date):
    news = fetch_news(ticker, date)
    sentiments = [TextBlob(article).sentiment.polarity for article in news]
    return np.mean(sentiments)

# Feature: daily sentiment score [-1, 1]
features["sentiment"] = df.apply(lambda x: get_sentiment_score(x.ticker, x.date))
```

#### **News Events**
- **Binary features**: Earnings announcement, FDA approval, etc.
- **Magnitude**: Size of price gap following news
- **Persistence**: How long sentiment effect lasts

**Challenges esperados:**
1. **Timing**: News breaks at random times, prediction window unclear
2. **Quality**: Not all news is relevant or accurate
3. **Latency**: By the time news is processed, market may have moved
4. **Overfitting**: Easy to overfit to specific news patterns

**¿Mejoraría performance?**
- **Probably**: Fundamental analysis + technical should beat technical alone
- **But**: Adds significant complexity and data requirements
- **Trade-off**: Simplicity vs. marginal improvement

---

## Preguntas Teóricas Profundas

### ❓ **¿Por qué creen que funciona mejor Logistic Regression en este problema?**

**Análisis Teórico Profundo:**

#### **Características del Dataset que Favorecen LR**

1. **Linealidad aproximada**: Muchas relaciones finance son log-lineales
```python
# Ejemplo: momentum effect
return_tomorrow ≈ α + β₁ * return_today + β₂ * return_week + ε
```

2. **Features engineered**: Los indicadores técnicos ya capturan non-linearities
```python
# RSI ya es non-linear transformation de precio
rsi = 100 - (100 / (1 + rs))
# MACD combina multiple timeframes
macd = ema12 - ema26
```

3. **Regularization benefit**: L2 penalty maneja multicolinearity
```python
# Muchos indicators están correlacionados:
# sma_5, sma_10, sma_20 → similar information
# L2 shrinks coefficients towards zero
```

4. **Dimensionalidad apropiada**: ~15 features, ~120 samples
```python
# Rule of thumb: samples > 10 * features
# 120 > 10 * 15 ✓ Barely sufficient for LR
```

#### **Por qué KNN Struggled**

1. **Curse of dimensionality**: Distance becomes meaningless in high dimensions
```python
# En 15 dimensiones, todos los puntos están "lejos"
# Euclidean distance ratio max/min → 1 as dimensions increase
```

2. **Sparse data**: Local neighborhoods tienen pocos samples
```python
# Con 120 samples en 15D space, neighborhoods están vacíos
# k=5 neighbors might be very far away
```

3. **Irrelevant features**: KNN suffers from noisy dimensions
```python
# Si dow (day of week) es irrelevant, añade noise to distance
# LR puede learn coefficient ≈ 0, KNN can't ignore dimensions
```

#### **Por qué Naive Bayes Competed**

Sorprendentemente well despite clear violations of independence assumption:

1. **Robustness**: NB often works despite independence violations
2. **Low variance**: Stable predictions with limited data
3. **Calibrated probabilities**: Good for ranking, even if not accurate probabilities

### ❓ **¿Cómo evaluarían si su modelo tiene alpha (excess return)?**

**Respuesta desde Financial Theory:**

#### **Alpha Definition**
Alpha = Return - Risk-adjusted benchmark return
```
α = R_portfolio - [R_f + β(R_market - R_f)]
```

#### **Evaluation Framework**

1. **Sharpe Ratio**: Risk-adjusted return
```python
sharpe = (mean_return - risk_free_rate) / std_return
# Good Sharpe > 1.0, Excellent > 2.0
```

2. **Information Ratio**: Alpha per unit of tracking error
```python
ir = alpha / tracking_error
# Measures skill independent of risk taking
```

3. **Maximum Drawdown**: Worst peak-to-trough loss
```python
drawdown = (peak_value - trough_value) / peak_value
# Trading is only sustainable if drawdowns are manageable
```

4. **Win Rate vs. Win/Loss Ratio**
```python
win_rate = num_wins / total_trades
avg_win_loss = mean(winning_trades) / abs(mean(losing_trades))
# Need: win_rate * avg_win > (1-win_rate) * avg_loss
```

#### **Benchmarking Strategy**

**Null hypothesis**: Model has no predictive power

**Tests**:
1. **vs. Random**: Compare to coin flip strategy
2. **vs. Buy & Hold**: Compare to always bullish
3. **vs. Technical indicators**: Compare to simple moving average crossover
4. **vs. Market**: Risk-adjusted return vs. market return

#### **Statistical Significance**

```python
# Bootstrap test for significance
def bootstrap_alpha(returns, n_iterations=1000):
    alphas = []
    for _ in range(n_iterations):
        sample_returns = np.random.choice(returns, size=len(returns))
        alphas.append(calculate_alpha(sample_returns))

    p_value = np.mean(np.array(alphas) <= 0)
    return p_value < 0.05  # Significant alpha?
```

### ❓ **¿Qué asumen sobre la eficiencia del mercado?**

**Market Efficiency Theory vs. Our Approach:**

#### **Efficient Market Hypothesis (EMH)**

**Weak Form**: Past prices don't predict future prices
- **Our assumption**: VIOLATED - usamos technical analysis
- **Justification**: Short-term momentum effects documented

**Semi-Strong Form**: Public information already in prices
- **Our assumption**: PARTIALLY VIOLATED - technical indicators have edge
- **Justification**: Market frictions, behavioral biases

**Strong Form**: No information provides advantage
- **Our assumption**: NOT ADDRESSED - no insider information

#### **Behavioral Finance Perspective**

**Why our approach might work:**
1. **Momentum**: Investor herding creates trend persistence
2. **Overreaction**: Markets overshoot, then revert
3. **Anchoring**: Traders anchor to recent prices/levels
4. **Limited attention**: Not all information immediately incorporated

**Market Microstructure**:
- **Bid-ask spreads**: Create transaction costs
- **Liquidity constraints**: Large orders move prices
- **Algorithmic trading**: Creates short-term patterns

#### **Our Model's Implicit Assumptions**

1. **Patterns persist**: Historical relationships continue
2. **Rational expectations with friction**: Markets trend toward efficiency but slowly
3. **Technical analysis has edge**: At least temporarily
4. **Prediction horizon matters**: 1-day momentum vs. long-term randomness

### ❓ **¿Cómo cambiaría su enfoque si el objetivo fuera minimizar riesgo en lugar de maximizar retorno?**

**Risk-Focused Model Design:**

#### **Objective Function Change**
```python
# Return-focused (current)
y = (close.shift(-1) > close).astype(int)  # Will price go up?

# Risk-focused (alternative)
y = (abs(close.pct_change().shift(-1)) > threshold).astype(int)  # Will volatility spike?
```

#### **Feature Engineering for Risk**
```python
# Volatility prediction features
features["realized_vol"] = returns.rolling(20).std()
features["vol_of_vol"] = features["realized_vol"].rolling(10).std()
features["skewness"] = returns.rolling(20).skew()
features["kurtosis"] = returns.rolling(20).kurt()  # Fat tails
features["vix_proxy"] = option_implied_volatility  # If available
```

#### **Model Selection Changes**
1. **Quantile Regression**: Predict downside risk (5th percentile)
2. **Gaussian Process**: Model uncertainty in predictions
3. **GARCH models**: Specialize in volatility clustering
4. **Extreme Value Theory**: Model tail risks

#### **Evaluation Metrics**
```python
# Risk-focused metrics
value_at_risk = np.percentile(returns, 5)  # 5% VaR
expected_shortfall = returns[returns < value_at_risk].mean()  # CVaR
max_drawdown = maximum_drawdown(cumulative_returns)
downside_deviation = returns[returns < 0].std()
```

#### **Portfolio Construction**
- **Position sizing**: Based on predicted volatility
- **Diversification**: Correlation-based portfolio
- **Hedging**: Dynamic hedge ratios
- **Tail protection**: Options strategies for extreme events

### ❓ **¿Qué desafíos éticos ven en el trading algorítmico?**

**Análisis Ético Profundo:**

#### **Market Fairness**
1. **Speed advantage**: Algo traders act faster than humans
2. **Information asymmetry**: Better data access
3. **Market making**: Provide liquidity but extract spreads
4. **Flash crashes**: Algorithms can amplify volatility

#### **Systematic Risk**
1. **Herding**: Similar algorithms create correlated strategies
2. **Leverage amplification**: Algo trading enables higher leverage
3. **Cascade failures**: One algorithm failure triggers others
4. **Market fragmentation**: HFT across multiple venues

#### **Social Impact**
1. **Job displacement**: Algorithms replace human traders
2. **Wealth concentration**: Sophisticated tools favor institutional investors
3. **Market volatility**: Can algorithms destabilize markets?
4. **Regulatory arbitrage**: Algorithms exploit rule differences

#### **Our Responsibilities as Developers**

1. **Transparency**: Document assumptions and limitations clearly
2. **Risk management**: Build in safeguards and kill switches
3. **Education**: Promote understanding over blind adoption
4. **Inclusive design**: Consider impact on retail investors

**Ethical Guidelines for Implementation:**
- Never promise guaranteed returns
- Disclose all risks prominently
- Include appropriate disclaimers
- Encourage paper trading before real money
- Promote responsible position sizing
- Consider societal impact, not just profits

---

## Preguntas de Seguimiento Esperadas

### ❓ **Si tuvieran que presentar esto a un inversionista, ¿qué dirían?**

**Investor Pitch (Honest & Professional):**

**Lo que NO diríamos:**
- ❌ "Guaranteed profits"
- ❌ "Beat the market consistently"
- ❌ "No risk involved"
- ❌ "Works in all conditions"

**Lo que SÍ diríamos:**

1. **Clear value proposition**:
"Systematic approach to identify short-term momentum patterns with moderate edge over random prediction"

2. **Realistic expectations**:
"63% accuracy suggests modest predictive power. Not a get-rich-quick scheme."

3. **Risk disclosure**:
"Past performance doesn't guarantee future results. Significant risk of loss."

4. **Competitive advantage**:
"Disciplined, emotion-free decision making based on quantitative signals"

5. **Scalability path**:
"Proof of concept that could be enhanced with more data, features, and models"

### ❓ **¿Qué harían si el modelo dejara de funcionar?**

**Model Degradation Response Plan:**

#### **Detection System**
```python
# Monitor key metrics daily
def monitor_model_health(predictions, actual_returns):
    recent_accuracy = accuracy_score(actual_returns[-30:], predictions[-30:])
    if recent_accuracy < 0.55:  # Below reasonable threshold
        alert("Model performance degraded")

    # Check for regime change
    recent_volatility = actual_returns[-30:].std()
    historical_volatility = actual_returns[:-30].std()
    if recent_volatility > 2 * historical_volatility:
        alert("Market regime change detected")
```

#### **Response Actions**
1. **Immediate**: Stop trading, preserve capital
2. **Short-term**: Analyze performance degradation causes
3. **Medium-term**: Retrain with recent data
4. **Long-term**: Redesign if structural changes detected

#### **Common Failure Modes**
- **Market regime change**: Bull to bear market
- **Structural breaks**: New regulations, technology
- **Data drift**: Feature distributions change
- **Overfitting**: Model too specific to training period

### ❓ **¿Recomendarían usar esto en producción?**

**Honest Assessment:**

**Current state: NO, not ready for production**

**Reasons:**
1. **Insufficient validation**: Need longer backtesting period
2. **No transaction costs**: Real trading has fees/spreads
3. **Limited robustness**: No stress testing across market regimes
4. **Single timeframe**: Only daily predictions
5. **No risk management**: No position sizing or stop losses

**Path to production:**
1. **Extended backtesting**: 2-5 years out-of-sample
2. **Paper trading**: 6 months live simulation
3. **Risk management**: Implement proper position sizing
4. **Monitoring**: Real-time performance tracking
5. **Regulatory compliance**: Ensure legal requirements met
6. **Capital allocation**: Start with <5% of total portfolio

**Use cases where it MIGHT be appropriate:**
- Educational trading account
- Small allocation for experimentation
- Part of diversified strategy portfolio
- Research and development platform

**Bottom line**: Promising research, needs significant development before real money deployment.

---

**Nota Final**: Estas respuestas demuestran comprensión profunda tanto de los aspectos técnicos como prácticos del machine learning aplicado a finanzas. El proyecto es un excelente punto de partida que respeta las limitaciones académicas mientras mantiene rigor científico y perspectiva práctica realista.