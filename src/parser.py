from pathlib import Path

import yaml


class YamlParser:

    @staticmethod
    def load_yaml(file_path: str | Path) -> dict:
        """Load YML file

        Args:
            file_path (str | Path): Path to the YAML file

        Returns:
            dict: YAML content
        """
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)