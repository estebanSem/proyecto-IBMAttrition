#Lectura y limpieza inicial
from src.parser import YamlParser
from src.extractors import CsvExtractor
from pathlib import Path

import pandas as pd


YAML_FILE = Path(__file__).resolve().parent.parent / "config.yaml"

def create_df(path: str) -> pd.DataFrame :
    """Crea un df del csv que se indica

    Args:
        path (str): ruta del csv

    Returns:
        pd.DataFrame: data
    """

    yaml_parser = YamlParser()
    config = yaml_parser.load_yaml(YAML_FILE)

    extractor = CsvExtractor()
    df = extractor.extract(config['paths'][path])

    return df



