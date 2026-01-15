#Lectura y limpieza inicial
from src.parser import YamlParser
from src.extractors import CsvExtractor
from pathlib import Path

import pandas as pd


YAML_FILE = Path(__file__).resolve().parent.parent / "config.yml"

def create_df() -> pd.DataFrame :
    """Crea un df 

    Returns:
        pd.DataFrame: dataframe
    """

    yaml_parser = YamlParser()
    config = yaml_parser.load_yaml(YAML_FILE)

    extractor = CsvExtractor()
    df = extractor.extract(config['paths']['raw_data'])

    return df



