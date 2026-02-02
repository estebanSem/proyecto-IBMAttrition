from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    @abstractmethod
    def load(self):
        pass


class CsvLoader(BaseLoader):
    
    """ carga datos en un csv """

    def load(self, df: pd.DataFrame, file_path: str | Path):
        """Carga los datos que recibe en un csv, en la ruta file_path

        Args:
            df (pd.DataFrame): datos a cargar
            file_path (str | Path): ruta donde se cargan
        """
        df.to_csv(file_path, index=False)