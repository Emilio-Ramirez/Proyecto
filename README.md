# Proyecto Final: Predicción Binaria de Movimiento de Acciones

## Resumen Ejecutivo

Este proyecto implementa un sistema de **clasificación binaria** para predecir si el precio de cierre de una acción será mayor mañana comparado con hoy. Utilizando técnicas de machine learning vistas en clase, el sistema analiza indicadores técnicos y patrones históricos para generar predicciones del tipo "¿mañana sube?" (1=sí, 0=no).

## Definición del Problema

### Problemática Identificada
La predicción de movimientos bursátiles es uno de los desafíos más complejos en finanzas cuantitativas. Decidimos abordar este problema por las siguientes razones:

1. **Relevancia Práctica**: Las decisiones de inversión requieren predicciones sobre direcciones de precios
2. **Complejidad Apropiada**: Permite aplicar múltiples algoritmos de clasificación vistos en clase
3. **Datos Abundantes**: Los mercados financieros generan datos continuos ideales para ML
4. **Evaluación Clara**: Métricas binarias fáciles de interpretar y evaluar

### Alcance del Proyecto
- **Horizonte de Predicción**: 1 día (t+1)
- **Tipo de Predicción**: Binaria (sube/no sube)
- **Activos**: Acciones individuales (AAPL, MSFT, NVDA)
- **Enfoque**: Completamente offline sin dependencias externas

## Decisiones de Diseño y Metodología

### 1. Generación de Datos Sintéticos

**Decisión**: Crear datos OHLCV sintéticos en lugar de usar APIs financieras reales.

**Justificación**:
- ✅ **Reproducibilidad**: Resultados consistentes en cualquier ejecución
- ✅ **Independencia**: No requiere conexión a internet o claves API
- ✅ **Control**: Podemos ajustar complejidad y patrones específicos
- ✅ **Realismo**: Implementamos random walks con drift y momentum como mercados reales

**Implementación**:
```python
# Random walk con características realistas
def generate_ohlcv(ticker: str, n_days: int, seed: int = RANDOM_STATE):
    # Drift específico por ticker
    mu = DRIFT.get(ticker, 0.0)  # Tendencia de largo plazo
    # Volatilidad específica
    s = SIGMA.get(ticker, 0.012)  # Volatilidad diaria
    # Momentum para simular autocorrelación
    mom = 0.15 * rets[t-1]  # Persistencia de tendencias
```

### 2. Ingeniería de Características

**Decisión**: Usar indicadores técnicos clásicos en lugar de features complejas.

**Justificación**:
- **Interpretabilidad**: Los traders entienden RSI, MACD, medias móviles
- **Robustez**: Indicadores probados durante décadas en mercados reales
- **Diversidad**: Combinamos momentum, tendencia, volatilidad y volumen
- **No-overfitting**: Features conocidas reducen riesgo de sobreajuste

**Features Implementadas**:

| Categoría | Indicadores | Justificación |
|-----------|-------------|---------------|
| **Momentum** | ret_1, ret_5, ret_10 | Capturan tendencias recientes |
| **Tendencia** | sma_5, sma_10, sma_20 | Identifican dirección del mercado |
| **Volatilidad** | std_5, std_10, std_20 | Miden incertidumbre/riesgo |
| **Técnicos** | RSI, MACD, signal, histogram | Señales de sobrecompra/sobreventa |
| **Volumen** | vol_chg, vol_znorm_20 | Confirman movimientos de precio |
| **Temporal** | dow (día de semana) | Efectos estacionales conocidos |

### 3. Selección de Algoritmos

**Decisión**: Comparar Regresión Logística, KNN y Gaussian Naive Bayes.

**Justificación por Algoritmo**:

#### Regresión Logística
- ✅ **Interpretable**: Coeficientes muestran importancia de features
- ✅ **Rápida**: Entrenamiento y predicción eficientes
- ✅ **Probabilidades**: Proporciona confianza en predicciones
- ✅ **Baseline**: Estándar de la industria para clasificación binaria

#### K-Nearest Neighbors (KNN)
- ✅ **No-paramétrico**: No asume distribuciones específicas
- ✅ **Adaptativo**: Captura patrones locales complejos
- ✅ **Robusto**: Funciona bien con datos no lineales
- ⚠️ **Limitación**: Sensible a dimensionalidad y ruido

#### Gaussian Naive Bayes
- ✅ **Simplicidad**: Asume independencia entre features
- ✅ **Eficiencia**: Muy rápido con datos pequeños
- ✅ **Probabilístico**: Maneja incertidumbre naturalmente
- ⚠️ **Limitación**: Asunción fuerte de independencia

### 4. Validación Temporal

**Decisión**: Usar TimeSeriesSplit en lugar de validación cruzada aleatoria.

**Justificación Crítica**:
```python
# ❌ INCORRECTO para series temporales
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True)  # Mezcla datos temporales!

# ✅ CORRECTO para series temporales
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=3)  # Respeta orden temporal
```

**Por qué es Crucial**:
- **Previene Data Leakage**: No usar información futura para predecir el pasado
- **Realismo**: Simula condiciones reales de trading
- **Validez**: Las métricas reflejan performance real esperada

### 5. Arquitectura del Pipeline

**Decisión**: Usar scikit-learn Pipelines con ColumnTransformer.

```python
# Pipeline completo
pipe = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])),
    ("classifier", LogisticRegression())
])
```

**Ventajas**:
- ✅ **Reproducibilidad**: Toda la transformación encapsulada
- ✅ **Prevención de Leakage**: Scaler se ajusta solo en train
- ✅ **Deployment**: Pipeline completo guardado en .joblib
- ✅ **Mantenibilidad**: Código limpio y modular

## Resultados y Evaluación

### Métricas de Performance

![Matriz de Confusión](roc_LogisticRegression.png)
*Figura 1: Curva ROC del mejor modelo (Regresión Logística)*

![Comparación de Modelos](metrics_comparison.png)
*Figura 2: Comparación de métricas entre modelos*

### Resultados Principales

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| **Regresión Logística** | 0.6250 | 0.6154 | 0.7273 | 0.6667 | 0.6307 |
| **KNN** | 0.5625 | 0.5714 | 0.7273 | 0.6400 | 0.5739 |
| **Gaussian NB** | 0.5938 | 0.5926 | 0.7273 | 0.6531 | 0.6023 |

### Análisis de Resultados

**Regresión Logística** emerge como el mejor modelo con:
- **ROC-AUC**: 0.6307 (mejor capacidad discriminativa)
- **Recall**: 0.7273 (detecta 72.7% de movimientos alcistas)
- **Balance**: Mejor compromiso precision-recall

**Interpretación Financiera**:
- Performance moderada refleja la naturaleza ruidosa de predicciones diarias
- ROC-AUC > 0.6 indica capacidad predictiva superior al azar
- Recall alto es valioso para no perder oportunidades de ganancia

## Implementación Técnica

### Estructura del Proyecto
```
ProyectoFinal/
├── proyecto_final_bolsa.py    # Script principal
├── dataset_train.csv          # Datos de entrenamiento
├── dataset_test.csv           # Datos de prueba
├── pipeline_*.joblib          # Modelos entrenados
├── roc_*.png                  # Curvas ROC
├── metrics.csv                # Métricas comparativas
├── reporte_resumen.md         # Reporte automático
└── README.md                  # Este documento
```

### Uso del Sistema

```bash
# Ejecución básica (3 tickers, 70 días c/u)
python proyecto_final_bolsa.py

# Configuración personalizada
python proyecto_final_bolsa.py --tickers AAPL,MSFT,GOOGL --days 100

# Predicción específica
python proyecto_final_bolsa.py --predict AAPL
```

### Salidas Generadas

1. **Datasets**: `dataset_train.csv`, `dataset_test.csv`
2. **Modelos**: `pipeline_LogisticRegression.joblib`, etc.
3. **Visualizaciones**: `roc_LogisticRegression.png`, etc.
4. **Métricas**: `metrics.csv`
5. **Reporte**: `reporte_resumen.md`

## Limitaciones y Consideraciones

### Limitaciones Reconocidas

1. **Horizonte Corto**: Predicciones de 1 día son inherentemente ruidosas
2. **Datos Sintéticos**: No capturan todos los factores de mercados reales
3. **Features Limitadas**: No incluye noticias, sentiment, macro-económicos
4. **Tamaño de Muestra**: ~150 observaciones es pequeño para ML

### Consideraciones Éticas y Prácticas

⚠️ **ADVERTENCIA**: Este proyecto es únicamente educativo. No constituye consejo financiero.

- **Riesgo**: Los mercados reales tienen factores no modelados
- **Regulación**: Trading algorítmico requiere cumplimiento normativo
- **Costos**: Spreads, comisiones y slippage afectan rentabilidad real
- **Liquidez**: Modelos pueden fallar en condiciones de mercado extremas

## Conclusiones y Trabajo Futuro

### Conclusiones Principales

1. **Viabilidad**: Es posible crear sistemas de predicción con herramientas básicas
2. **Performance**: Resultados moderados pero superiores al azar
3. **Metodología**: Validación temporal es crucial para datos financieros
4. **Interpretabilidad**: Regresión Logística ofrece el mejor balance

### Extensiones Propuestas

1. **Datos Reales**: Integrar APIs como Alpha Vantage o Yahoo Finance
2. **Features Avanzadas**: Sentiment analysis, indicadores macro
3. **Modelos Complejos**: Random Forest, XGBoost (si se ven en clase)
4. **Ensemble**: Combinar predicciones de múltiples modelos
5. **Backtesting**: Simular trading real con costos de transacción

### Aprendizajes del Equipo

- **Importancia de Validación Temporal**: Cambió nuestra comprensión de ML en finanzas
- **Balance Complejidad-Interpretabilidad**: Modelos simples pueden ser efectivos
- **Ingeniería de Features**: Domain knowledge es crucial para features relevantes
- **Evaluación Realista**: Métricas deben reflejar objetivos de negocio

## Referencias

1. Scikit-learn Documentation: https://scikit-learn.org/
2. "Advances in Financial Machine Learning" - Marcos López de Prado
3. Technical Analysis Literature (RSI, MACD, Moving Averages)
4. Time Series Cross-Validation Best Practices

---

**Equipo de Desarrollo**: [Nombres del equipo]
**Curso**: TC3006C - Inteligencia Artificial Avanzada para la Ciencia de Datos I
**Fecha**: Septiembre 2024

**Nota**: Este proyecto demuestra la aplicación práctica de algoritmos de machine learning en un dominio complejo, respetando las mejores prácticas de validación temporal y evaluación robusta.