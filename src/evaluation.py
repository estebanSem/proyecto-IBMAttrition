#Funciones para metricas y graficas
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.pipeline          import Pipeline

import plotly.express   as px
import pandas           as pd



def show_box(x: str, y: str ,df: pd.DataFrame) -> px.box :
    """Genera un grafico box

    Args:
        x (str): varaible x
        y (str): variable y
        df (pd.DataFrame): dataframe

    Returns:
        px.box: grafico
    """

    gr = px.box(df, x=x, y=y)

    return gr

def show_heatmap(cols: list[str], df: pd.DataFrame) -> px.imshow:
    """Genera un mapa de calor

    Args:
        cols (list[str]): columnas a tratar
        df (pd.DataFrame): data

    Returns:
        px.density_heatmap: grafico
    """
    # Matriz completa de correlación
    corr_matrix = df[cols].corr(method='pearson')

    gr = px.imshow(
        corr_matrix,
        text_auto=".2f",      
        color_continuous_scale='RdBu_r',  
        aspect="auto",
        title="Matriz de Correlación"
    )

    return gr

def show_bar(df: pd.DataFrame, x: str, y: str) -> px.bar:
    

    gr = px.bar(
        df, 
        x=x, 
        y=y, 
        color=x
    )
    
    return gr

def show_metricas(y_test, pred) -> str | dict:

    # Obtener Accuracy final
    accuracy_final = accuracy_score(y_test, pred)

    # Obtener el reporte de clasificación (incluye Precision y Recall)
    reporte_final = classification_report(y_test, pred)

    return reporte_final, accuracy_final

def show_confusion_matrix(y_test, pred, model_name: str) -> px.imshow :

    cm = confusion_matrix(y_test, pred)

    # Definimos las etiquetas
    labels = ['Se queda', 'Se va']

    # Creamos el mapa de calor
    fig = px.imshow(
        cm, 
        text_auto=True,               # Muestra los números dentro de los cuadros
        labels=dict(x="Predicción", y="Real", color="Cantidad"),
        x=labels, 
        y=labels,
        color_continuous_scale='Blues'
    )

    # Ajustes estéticos adicionales
    fig.update_layout(
        title=f'Matriz de Confusión: Modelo {model_name}',
        title_x=0.5,                  # Centra el título
        width=600, 
        height=500
    )

    return fig

def get_coeficientes(pipeline: Pipeline, X_train) -> pd.DataFrame:

    coeficientes = pipeline.named_steps['lr'].coef_[0]
    cols = X_train.columns

    importancia_df = pd.DataFrame({
        'Variable': cols,
        'Coeficiente': coeficientes,
        'Abs_Coeficiente': abs(coeficientes) # Usamos el valor absoluto para la importancia total
    })

    importancia_df = importancia_df.sort_values(by='Coeficiente', ascending=False)

    return importancia_df

def show_coeficientes(df: pd.DataFrame) -> px.bar:

    fig = px.bar(
        df.head(25),
        x='Abs_Coeficiente',
        y='Variable',
        orientation='h',                             # Gráfico horizontal
        title='Coeficientes',
        labels={'Abs_Coeficiente': 'Coeficiente', 'Variable': 'Característica'},
        color='Abs_Coeficiente',                          # Color basado en el valor
        color_continuous_scale='Viridis',
        width=1200,
        height=800
    )

    return fig

def show_feature_importances(df: pd.DataFrame ) -> px.bar:
    

    fig = px.bar(
        df.head(25),
        x='Importance',
        y='Feature',
        orientation='h',                             # Gráfico horizontal
        title='25 variables con más peso',
        labels={'Importance': 'Importancia', 'Feature': 'Característica'},
        color='Importance',                          # Color basado en el valor
        color_continuous_scale='Viridis',
        width=1200,
        height=800
    )

    return fig

def get_feature_importances(model: RandomForestClassifier, X_train) -> pd.DataFrame:
    """Genera un dataframe con el peso de las variables en el modelo

    Args:
        model (RandomForestClassifier): modelo
        X_train (_type_): Datos de entrenamiento X

    Returns:
        pd.DataFrame: dataframe con las columnas y los pesos
    """

    feature_importances = model.feature_importances_
    feature_names       = X_train.columns

    # 3. Crear un DataFrame para visualizar las importancias
    importance_df = pd.DataFrame({
        'Feature'   :   feature_names,
        'Importance':   feature_importances
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return importance_df

def show_especificidad(y_test, pred) -> int | float :

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    especificidad = tn / ( tn + fp )

    return especificidad

def show_auc_roc(model, X_test, y_test) -> px.line :

    y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, umbrales = roc_curve(y_test, y_probs)

    auc_score = roc_auc_score(y_test, y_probs)

    df_roc = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr
    })

    fig = px.line(
        df_roc, 
        x='FPR', 
        y='TPR',
        title=f'Curva ROC (AUC = {auc_score:.2f})',
        labels={'FPR': 'Tasa de Falsos Positivos', 'TPR': 'Tasa de Verdaderos Positivos'},
        width=700, 
        height=700
    )

    fig.add_shape(
        type='line', line=dict(dash='dash', color='black'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=[0, 1], constrain='domain')

    return fig