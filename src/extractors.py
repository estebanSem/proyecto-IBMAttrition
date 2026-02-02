from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, data):
        pass


class CsvExtractor(BaseExtractor):
    """Extracts data from CSV files."""

    def extract(self, file_path: str | Path) -> pd.DataFrame:
        """Extrae datos de un filepath

        Args:
            file_path (str | Path): ruta al archivo

        Returns:
            pd.DataFrame: dataframe
        """
        return pd.read_csv(file_path)