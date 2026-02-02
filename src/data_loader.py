#Lectura y limpieza inicial
from src.parser                 import YamlParser
from src.extractors             import CsvExtractor
from pathlib                    import Path
from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression

import pandas as pd
import joblib

YAML_FILE = Path(__file__).resolve().parent.parent / "config.yaml"
yaml_parser = YamlParser()
config = yaml_parser.load_yaml(YAML_FILE)
extractor = CsvExtractor()

def create_df(path: str) -> pd.DataFrame :
    """Crea un df del csv que se indica

    Args:
        path (str): ruta del csv

    Returns:
        pd.DataFrame: data
    """
    df = extractor.extract(config['paths']['data'][path])

    return df

def serialize_models(model: LogisticRegression | RandomForestClassifier, name: str) -> str:
    """Serializa el modelo, aplicandole el nombre que se le pasa por parametros

    Args:
        model (LogisticRegression | RandomForestClassifier): Modelo(Pipeline)
        name (str): nombre del modelo

    Returns:
        str: Mensaje de exito o error
    """

    try:
        joblib.dump(model, f'{config["paths"]["models"]["path"]}/{name}.pkl')
        return f"Modelo {name}.pkl guardado con éxito"
    except Exception as err:
        return f"Error al guardar el modelo:\n {err}"