# 🧑‍💼 IBM HR Attrition — Predicción de Rotación de Empleados

Proyecto de Machine Learning para predecir qué empleados tienen mayor probabilidad de abandonar la empresa, usando el dataset público **IBM HR Analytics Employee Attrition & Performance**.

---

## 📋 Descripción

La rotación de empleados (*attrition*) es uno de los mayores costes ocultos para las organizaciones. Este proyecto entrena y evalúa modelos de clasificación para identificar los factores clave que llevan a un empleado a dejar la empresa, permitiendo a RRHH actuar de forma preventiva.

Se comparan dos aproximaciones:
- **Random Forest** — modelo de ensamble robusto ante variables irrelevantes y no linealidades.
- **Regresión Logística** — modelo interpretable, útil para entender el peso de cada variable.

---

## 🗂️ Estructura del proyecto

```
proyecto-IBMAttrition/
│
├── data/                   # Dataset original y versiones procesadas
├── models/                 # Modelos entrenados serializados (.pkl / .joblib)
├── src/                    # Código fuente modular
│   ├── data_loader.py           # Lectura y limpieza de datos
│   ├── evaluation.py            # Funciones para métricas y gráficos
│   ├── extractors.py            # Class extractor 
│   ├── feature_engeneering.py   # Creacion de variables nuevas
│   ├── loader.py                # Class load
│   ├── parser.py                # Parseador del config.yaml
│   ├── processing.py            # Preprocesado de los datos
│   └── train.py                 # entrenamiento para los dos modelos
│
├── 01_eda.ipynb            # Análisis exploratorio de datos (EDA)
├── 02_entreno_rf.ipynb     # Entrenamiento y evaluación — Random Forest
├── 03_entreno_rl.ipynb     # Entrenamiento y evaluación — Regresión Logística
│
├── main.py                 # Script principal — ejecuta el pipeline completo
├── config.yaml             # Configuración centralizada (rutas, hiperparámetros)
├── pyproject.toml          # Dependencias del proyecto (gestionadas con uv)
└── uv.lock                 # Lockfile de dependencias
```

> **Nota de diseño:** Los notebooks (`01`, `02`, `03`) están pensados como herramienta de análisis visual y exploración. El entrenamiento real del modelo se ejecuta mediante un pipeline , que importa las funciones de `src/` y lee la configuración de `config.yaml`.

---

## 🚀 Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/estebanSem/proyecto-IBMAttrition.git
cd proyecto-IBMAttrition
```

### 2. Instalar dependencias

Este proyecto usa [uv](https://github.com/astral-sh/uv) como gestor de paquetes:

```bash
pip install uv
uv sync
```

### 3. Ejecutar el pipeline completo

```
Seguir los notebooks
```

Esto leerá la configuración de `config.yaml`, procesará los datos, entrenará los modelos y guardará los resultados en `models/`.

### 4. Explorar los notebooks (opcional)

```bash
jupyter notebook
```

| Notebook | Descripción |
|---|---|
| `01_eda.ipynb` | Análisis exploratorio: distribuciones, correlaciones, clase desbalanceada |
| `02_entreno_rf.ipynb` | Entrenamiento y métricas del modelo Random Forest |
| `03_entreno_rl.ipynb` | Entrenamiento y métricas del modelo Regresión Logística |

---

## 📊 Dataset

**IBM HR Analytics Employee Attrition & Performance**

- **Fuente:** [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Registros:** 1.470 empleados
- **Target:** `Attrition` (Yes / No) — variable binaria
- **Features:** 35 variables entre demográficas, laborales y de satisfacción (edad, departamento, horas extra, nivel de satisfacción, salario, etc.)

> ⚠️ El dataset presenta **desbalance de clases** (~84% No attrition vs ~16% Yes). Esto se tiene en cuenta durante el entrenamiento.

---

## 🤖 Modelos

### Random Forest
- Modelo de ensamble basado en árboles de decisión
- Robusto ante outliers y variables irrelevantes
- Permite obtener importancia de features

### Regresión Logística
- Modelo lineal interpretable
- Útil para entender el impacto individual de cada variable
- Rápido de entrenar y fácil de explicar a negocio

---

## 📈 Métricas de evaluación

Dado el desbalance de clases, las métricas principales son:

- **F1-Score** (clase minoritaria)
- **ROC-AUC**
- **Precision / Recall**
- **Matriz de confusión**

---

## ⚙️ Configuración

Todos los parámetros del pipeline se encuentran en `config.yaml`, incluyendo rutas de datos, hiperparámetros de los modelos y configuración del preprocesamiento. Modifica este archivo para ajustar el comportamiento sin tocar el código.


---

## 👤 Autor

**Esteban Sempere** — [@estebanSem](https://github.com/estebanSem)
