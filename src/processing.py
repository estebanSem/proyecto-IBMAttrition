#Transformaciones, escalados, encoding
import pandas as pd
import numpy as np

def get_one_value_col(df: pd.DataFrame) -> list[str] :
    """Devuelve una lista de las columnas con un unico valor

    Args:
        df (pd.Dataframe): data

    Returns:
        list[str]: nombres de las columnas
    """

    cols = [col for col in df.columns if df[col].nunique() == 1]

    return cols

def get_binary_cols(df: pd.DataFrame) -> list[str] :
    """Devuelve una lista de las columnas object con dos valores

    Args:
        df (pd.Dataframe): data

    Returns:
        list[str]: nombres de las columnas
    """

    cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() == 2]

    return cols

def get_more_two_values_cols(df: pd.DataFrame) -> list[str] :
    """Devuelve una lista de las columnas object con mas de dos valores

    Args:
        df (pd.Dataframe): data

    Returns:
        list[str]: nombres de las columnas
    """

    cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 2]

    return cols

def transform_label_encoding(cols: list[str], df: pd.DataFrame) -> pd.DataFrame:

    """Cambia los valores de las columnas con dos valores a 0 y 1

    Args:
        df (pd.Dataframe): data
        cols (list[str]): columnas a procesar

    Returns:
        pd.DataFrame: data
    """

    map_gender = {'Male':0,'Female':1}
    map_yn = {'No':0,'Yes':1}

    for col in cols:

            values = set(df[col].unique())

            if values == {'Yes','No'}:
                
                df[col] = df[col].replace(map_yn)
            
            if values =={'Male','Female'}:

                df[col] = df[col].replace(map_gender)

    return df

def transform_dummies(cols: list[str] ,df: pd.DataFrame) -> pd.DataFrame :
    """Transforma columnas a dummies y despues a enteros

    Args:
        cols (list[str]): columnas
        df (pd.DataFrame): data

    Returns:
        pd.DataFrame: data
    """
    #dummies
    df_clean = pd.get_dummies(df, columns=cols,drop_first=True)

    #convertir las nuevas columnas a int.
    bool_cols = df_clean.select_dtypes(include='bool').columns
    df_clean[bool_cols] = df_clean[bool_cols].astype(int)

    return df_clean


def transform_log(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Aplica una transformación logarítmica (log1p) a una columna específica de un DataFrame.
    
    Args:
        df (pd.DataFrame): El DataFrame a procesar.
        column_name (str): El nombre de la columna a transformar.
        
    Returns:
        pd.DataFrame: El DataFrame con la columna transformada.
    """
    # Verificamos si la columna existe para evitar errores
    if col in df.columns:
        df[col] = np.log1p(df[col])
    else:
        print(f"Error: La columna '{col}' no se encuentra en el DataFrame.")
        
    return df

def drop_columns(cols: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas del dataframe, las que recibe por parametros, mas 4 extras

    Args:
        cols (list[str]): columnas
        df (pd.DataFrame): data

    Returns:
        pd.DataFrame: data
    """

    df_clean = df.drop(columns=cols)

    return df_clean

def drop_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas del dataframe

    Args:
        df (pd.DataFrame): data

    Returns:
        pd.DataFrame: data
    """
    extra_cols = ['DailyRate','MonthlyRate','HourlyRate','EmployeeNumber']

    df_clean = df.drop(columns=extra_cols)

    return df_clean


     