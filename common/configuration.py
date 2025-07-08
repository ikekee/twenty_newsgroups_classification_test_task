"""This module contains classes for define configurations."""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Union

import yaml


def open_yaml(path_to_yaml_file: Path) -> Dict[str, Any]:
    """Parses YAML configuration file.

    Args:
        path_to_yaml_file: A path to YAML file.

    Returns:
        A dictionary containing parsed YAML data
    """
    # Prepare yaml loader
    loader = yaml.FullLoader

    # Load config
    with open(path_to_yaml_file) as yaml_file:
        return yaml.load(yaml_file, Loader=loader)


class PDFParserConfiguration:
    def __init__(self, config_data: Dict[str, Any]):
        """Creates an instance of the class.

        Args:
            config_data: A dictionary containing configuration parameters.
        """
        self.max_pages = config_data["max_pages"]


class ModelInferenceConfiguration:
    """Encapsulates model inference configuration parameters."""

    def __init__(self, config_data: Dict[str, Any]):
        """Creates an instance of the class.

        Args:
            config_data: A dictionary containing configuration parameters.
        """
        self.path_to_model_weights = Path(config_data["path_to_model_weights"])
        self.path_to_tf_idf_vectorizer = Path(config_data["path_to_tf_idf_vectorizer"])


class Configuration:
    """Encapsulates root configuration parameters."""

    def __init__(self, config_file_path: Union[str, Path]):
        """Creates an instance of the class.

        Args:
            config_file_path: A path to YAML config file.
        """
        config_data = open_yaml(config_file_path)
        self.pdf_parser_configuration = PDFParserConfiguration(config_data["pdf_parser"])
        self.model_inference_configuration = ModelInferenceConfiguration(
            config_data["model_inference"]
        )
