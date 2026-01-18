#Funciones para metricas y graficas
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import accuracy_score, classification_report, confusion_matrix

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

def show_feature_importances(model: RandomForestClassifier) -> px.bar:

    feature_importances = model.feature_importances_
    feature_names       = model.columns

    # 3. Crear un DataFrame para visualizar las importancias
    importance_df = pd.DataFrame({
        'Feature'   :   feature_names,
        'Importance':   feature_importances
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig = px.bar(
        importance_df.head(25),
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