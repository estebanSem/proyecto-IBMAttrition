[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QC8gg3LY)
Para un "preprocesamiento complejo" (donde mezclas columnas numéricas, categóricas y reglas de negocio propias), la forma más robusta y profesional de hacerlo es combinando un `ColumnTransformer` (para tratar tipos de datos distintos) con una **clase personalizada**.

Aquí tienes el código completo. Este archivo `.pkl` será "autónomo": incluirá tu lógica, tus transformaciones y el modelo.

### 1. Definir la lógica personalizada y el Pipeline

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# --- PASO 1: Tu lógica personalizada ---
# Creamos una clase para "limpieza a medida" (ej: crear una columna de ratio)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Ejemplo: Crear una variable compleja de ratio
        if 'ingresos' in X.columns and 'gastos' in X.columns:
            X['ratio_ahorro'] = X['ingresos'] / (X['gastos'] + 1)
        return X

# --- PASO 2: Definir qué hacer con cada tipo de columna ---
col_numericas = ['edad', 'ingresos', 'gastos']
col_categoricas = ['ciudad', 'tipo_cliente']

# Transformador para números: Imputar nulos con mediana + Escalar
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Transformador para categorías: Imputar con el más frecuente + OneHotEncoding
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'ignore' es vital para test ciegos
])

# Combinar todo el preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, col_numericas),
        ('cat', cat_transformer, col_categoricas)
    ])

# --- PASO 3: Crear el Pipeline Final ---
full_pipeline = Pipeline(steps=[
    ('custom_fe', FeatureEngineer()), # Primero tu lógica propia
    ('preprocessor', preprocessor),   # Luego el tratamiento estándar
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- PASO 4: Entrenar y Persistir ---
# full_pipeline.fit(X_train, y_train)
joblib.dump(full_pipeline, 'modelo_complejo_rf.pkl')

```

---

### 2. ¿Cómo lo evaluará tu profesor? (El test ciego)

En el ordenador de tu profesor, solo necesitará este código. Al haber usado un `Pipeline`, los datos nuevos se transformarán automáticamente siguiendo las reglas que tú definiste arriba.

```python
import joblib

# El profesor carga el modelo
model = joblib.load('modelo_complejo_rf.pkl')

# Supongamos que le llegan datos nuevos "sucios"
datos_ciegos = pd.DataFrame({
    'edad': [25, np.nan], 
    'ingresos': [3000, 4500],
    'gastos': [1500, 2000],
    'ciudad': ['Madrid', 'Sevilla'],
    'tipo_cliente': ['Premium', 'Básico']
})

# El pipeline hace la magia: crea el ratio, imputa el nulo, escala y predice
predicciones = model.predict(datos_ciegos)
print(predicciones)

```

---

### ¿Por qué este código es "blind-test proof"?

1. **`handle_unknown='ignore'`**: Si en el test ciego aparece una ciudad que no estaba en tus datos (ej: "Valencia"), el modelo no fallará, simplemente ignorará esa categoría. Sin esto, el código daría error.
2. **`SimpleImputer`**: Si los datos del profesor traen valores vacíos (NaN), tu modelo los rellenará con la mediana que *tú* aprendiste en el entrenamiento.
3. **`FeatureEngineer`**: Cualquier cálculo de columnas nuevas se replica exactamente igual.

**Nota importante:** Para que el profesor pueda cargar el archivo, él debe tener definida la clase `FeatureEngineer` en su script o tú debes entregársela en un archivo `.py` que él importe. Es el único requisito de Python para deserializar objetos personalizados.

¿Quieres que te explique cómo estructurar el archivo para que tu profesor no tenga que copiar y pegar tu clase personalizada?
