import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split


class FileHandler:
    """
    Handles file operations including saving and loading various types of data (JSON, binary, model states)
    to and from a specified project directory.
    """

    def __init__(self, base_folder: Path) -> None:
        """Initialize with the path to the project folder."""
        self.base_folder = Path(base_folder)

    @property
    def file_path(self):
        return self._create_filename("")

    def _create_filename(self, file_name: str) -> Path:
        """Return the full path for a given filename within the project folder."""
        return self.base_folder / file_name

    def save_json(self, data: Any, filename: str) -> None:
        """Save data to a JSON file."""
        file_path = self._create_filename(filename)
        try:
            with open(file_path, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile)
        except Exception as e:
            logging.error("Failed to save JSON to %s: %s", file_path, e)

    def load_json(self, filename: str) -> Optional[Any]:
        """Load data from a JSON file."""
        file_path = self._create_filename(filename)
        try:
            with open(file_path, "r", encoding="utf-8") as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            logging.error("JSON file not found: %s", file_path)
        except Exception as e:
            logging.error("Error decoding JSON from file: %s: %s", file_path, e)
        return None

    def save_pickle(self, obj: Any, filename: str) -> None:
        """Save an object to a binary file using pickle."""
        file_path = self._create_filename(filename)
        try:
            with open(file_path, "wb") as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logging.error("Pickling error while saving object to %s: %s", file_path, e)

    def load_pickle(self, filename: str) -> Optional[Any]:
        """Load a pickled object from a binary file."""
        file_path = self._create_filename(filename)
        try:
            with open(file_path, "rb") as inp:
                return pickle.load(inp)
        except FileNotFoundError:
            logging.error("Object file not found: %s", file_path)
        except Exception as e:
            logging.error("Error unpickling object from file: %s: %s", file_path, e)
        return None

    def save_model_state(self, model: Any, filename: str) -> None:
        """Save the state dictionary of a PyTorch model."""
        file_path = self._create_filename(filename)
        try:
            torch.save(model.state_dict(), file_path)
        except Exception as e:
            logging.error("Failed to save model state to %s: %s", file_path, e)

    def load_model_state(self, model: Any, filename: str) -> Optional[Any]:
        """Load a model's state dictionary into a PyTorch model."""
        file_path = self._create_filename(filename)
        try:
            model.load_state_dict(torch.load(file_path))
            return model
        except FileNotFoundError:
            logging.error("Model state file not found: %s", file_path)
        except Exception as e:
            logging.error("Failed to load model state from %s: %s", file_path, e)
        return None

    def save_model(self, model: Any, filename: str) -> None:
        """Save a complete PyTorch model."""
        file_path = self._create_filename(filename)
        try:
            torch.save(model, file_path)
        except Exception as e:
            logging.error("Failed to save model to %s: %s", file_path, e)

    def load_model(self, filename: str) -> Optional[Any]:
        """Load a complete PyTorch model."""
        file_path = self._create_filename(filename)
        try:
            model = torch.load(file_path)
            model.eval()
            return model
        except FileNotFoundError:
            logging.error("Model file not found: %s", file_path)
        except Exception as e:
            logging.error("Failed to load model from %s: %s", file_path, e)
        return None

    def to_csv(self, filename: str, data: pd.DataFrame, index: bool = False) -> None:
        """Save DataFrame to a CSV file."""
        file_path = self._create_filename(filename)
        try:
            data.to_csv(file_path, index=index)
        except Exception as e:
            logging.error("Failed to save CSV to %s: %s", file_path, e)

    def read_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a CSV file into a DataFrame."""
        file_path = self._create_filename(filename)
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            logging.error("CSV file not found: %s", file_path)
        except Exception as e:
            logging.error("Error reading CSV from file: %s: %s", file_path, e)
        return None

    def to_json(self, data: pd.DataFrame, filename: str,) -> None:
        """Save DataFrame to a Json file."""
        file_path = self._create_filename(filename)
        try:
            data.to_json(file_path, orient="records", lines=True)
        except Exception as e:
            logging.error("Failed to save Json to %s: %s", file_path, e)

    def read_json(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a Json file into a DataFrame."""
        file_path = self._create_filename(filename)
        try:
            return pd.read_json(file_path, lines=True)
        except FileNotFoundError:
            logging.error("Json file not found: %s", file_path)
        except Exception as e:
            logging.error("Error reading Json from file: %s: %s", file_path, e)
        return None

    def write_plotly(self, data: go.Figure, filename: str) -> None:
        """Save a Plotly figure to a JSON file."""
        file_path = self._create_filename(filename)
        try:
            data.write_json(file_path)
        except Exception as e:
            logging.error("Failed to save Json to %s: %s", file_path, e)

    def read_plotly(self, filename: str) -> Optional[go.Figure]:
        """Load a Plotly figure from a JSON file."""
        file_path = self._create_filename(filename)
        try:
            with open(file_path, "r") as file:
                json_data = file.read()
            return pio.from_json(json_data)
        except FileNotFoundError:
            logging.error("Json file not found: %s", file_path)
        except Exception as e:
            logging.error("Error reading Json from file: %s: %s", file_path, e)
        return None

    def save_yaml(self, data, file_name: str):
        """Save data to a YAML file."""
        file_path = self._create_filename(file_name)
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)
            return True
        except IOError as e:
            logging.error("Failed to write to file: %s - %s", file_path, e)
        except yaml.YAMLError as e:
            logging.error(
                "Error while converting data to YAML format: %s - %s", file_path, e
            )
        return False

    def load_yaml(self, file_name: str):
        """Load data from a YAML file."""
        file_path = self._create_filename(file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.error("YAML file not found: %s", file_path)
        except yaml.YAMLError as e:
            logging.error("Error parsing YAML file: %s - %s", file_path, e)
        return None
    
    def save_numpy(self, array: np.ndarray, filename: str) -> None:
        """Save a NumPy array to a binary .npy file."""
        file_path = self._create_filename(filename)
        try:
            np.save(file_path, array)
        except Exception as e:
            logging.error("Failed to save NumPy array to %s: %s", file_path, e)

    def load_numpy(self, filename: str) -> Optional[np.ndarray]:
        """Load a NumPy array from a binary .npy file."""
        file_path = self._create_filename(filename)
        try:
            return np.load(file_path)
        except FileNotFoundError:
            logging.error("NumPy file not found: %s", file_path)
        except Exception as e:
            logging.error("Error loading NumPy array from file: %s: %s", file_path, e)
        return None


class DatasetStrategy(ABC):
    """Abstract base class for dataset strategies."""

    @abstractmethod
    def load_data(self, split: str) -> Any:
        pass

    @abstractmethod
    def process_data(self) -> Dict[str, Any]:
        pass


class ANERCorpStrategy(DatasetStrategy):
    ner_map = {
        "O": 0,
        "B-PERS": 1,
        "I-PERS": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }

    def __init__(self, file_handler: FileHandler, config: Dict[str, Any]) -> None:
        """
        Initializes the ANERCorp data handling strategy with a file handler and configuration.

        :param file_handler: Instance of FileHandler to manage file paths and access.
        :param config: Configuration dictionary that may include path and other options.
        """
        self.file_handler = file_handler
        self.config = config

    def load_data(self, split: str) -> List[Tuple[str, str]]:
        """
        Loads data for a specified split by reading a text file processed into words and tags.

        :param split: The dataset split to load ('train', 'test').
        :return: A list of tuples, each with words and their corresponding tags.
        """
        words, tags, sentences = [], [], []
        logging.info("Generating %s Split", split)
        file_path = self.file_handler.file_path / f"{self.config['path']}_{split}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        if len(parts) < 2:
                            logging.warning(
                                "Malformed line in %s: %s", file_path, line.strip()
                            )
                            continue  # Skip malformed lines
                        words.append(parts[0])
                        tags.append(parts[1])
                        if parts[0] == ".":
                            sentences.append((words, tags))
                            words, tags = [], []
        except FileNotFoundError:
            logging.error("File not found: %s", file_path)
        except IOError as e:
            logging.error("IO error when reading file %s: %s", file_path, e)

        if words:  # Add the last sentence if the file does not end with a period
            sentences.append((words, tags))
        return sentences

    def process_data(self) -> Dict[str, Any]:
        """
        Processes the loaded data and organizes it into a structured format.

        If a validation size is specified in the configuration, it splits the training data into
        training and validation sets.

        :return: A dictionary containing splits data, label mappings, and inverse label mappings.
        """
        validation_size = self.config.get("validation_size", None)
        splits = self.config.get("splits", [])
        # ner_inv_map = {v: k for k, v in self.ner_map.items()}
        splits_data = {}

        # Load each split except validation
        for split in splits:
            splits_data[split] = [
                {"id": sentence_id, "words": sentence[0], "tags": sentence[1]}
                for sentence_id, sentence in enumerate(self.load_data(split))
            ]

        # If validation_size is specified, split the training data
        if "train" in splits_data and validation_size is not None:
            train_data = splits_data["train"]
            train_data, val_data = train_test_split(
                train_data, test_size=validation_size, random_state=1
            )
            splits_data["train"] = train_data
            splits_data["validation"] = val_data

        data = {
            "splits": splits_data,
            "labels": list(self.ner_map.keys()),
            "labels_map": self.ner_map,
            # "inv_map": ner_inv_map,
        }
        return data


class Conll2003Strategy(DatasetStrategy):
    """
    Strategy for handling the CoNLL-2003 dataset.

    This class implements methods to load and process the CoNLL-2003 dataset,
    converting it into a structured format suitable for named entity recognition tasks.
    """

    ner_map = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the Conll2003Strategy with the given configuration.

        :param config: Configuration dictionary that includes dataset name and other options.
        """
        self.config = config
        self.conll2003_dataset = load_dataset(
            self.config.get("dataset_name"), trust_remote_code=True
        )

    def load_data(self, split: str) -> List[Dict[str, Any]]:
        """
        Loads data for a specified split from the CoNLL-2003 dataset.

        :param split: The dataset split to load ('train', 'validation', 'test').
        :return: A list of dictionaries, each containing sentence data with tokens and NER tags.
        """
        logging.info("Generating %s Split", split)
        return self.conll2003_dataset.get(split, [])

    def process_data(self) -> Dict[str, Any]:
        """
        Processes the loaded data and organizes it into a structured format.

        :return: A dictionary containing splits data, label mappings, and inverse label mappings.
        """
        splits = self.config.get("splits", [])
        ner_inv_map = {v: k for k, v in self.ner_map.items()}
        splits_data = {}

        for split in splits:
            splits_data[split] = [
                {
                    "id": sentence.get("id"),
                    "words": sentence.get("tokens"),
                    "tags": [
                        ner_inv_map.get(tid, "O") for tid in sentence.get("ner_tags")
                    ],
                }
                for sentence in self.load_data(split)
            ]

        data = {
            "splits": splits_data,
            "labels": list(self.ner_map.keys()),
            "labels_map": self.ner_map,
            # "inv_map": ner_inv_map,
        }
        return data


class CorporaManager:
    def __init__(self, config_path: Path) -> None:
        """
        Initialize the DataManager with a path to the configuration file.

        :param config_path: Path to the configuration YAML file.
        """
        self.file_handler = FileHandler(config_path.parent)
        self.config = self.file_handler.load_yaml(config_path.name)
        self.validate_configurations(self.config)

    def validate_configurations(self, config: Dict[str, Any]) -> None:
        """Validates the dataset configurations loaded from the YAML file."""
        required_keys = {"strategy", "splits"}
        for dataset_name, settings in config.items():
            if settings is None:
                raise ValueError(
                    f"Configuration for dataset {dataset_name} is missing."
                )

            missing_keys = required_keys - settings.keys()
            if missing_keys:
                raise ValueError(
                    f"Missing required keys {missing_keys} in dataset {dataset_name} configuration."
                )

            if not isinstance(settings.get("splits"), list) or not settings["splits"]:
                raise ValueError(f"No splits defined for dataset {dataset_name}.")

            if settings["strategy"] == "file" and "path" not in settings:
                raise ValueError(
                    f"No file path defined for dataset {dataset_name} with file strategy."
                )

            if settings["strategy"] == "dataset" and "dataset_name" not in settings:
                raise ValueError(
                    f"No dataset name defined for dataset {dataset_name} with dataset strategy."
                )

            if "validation_size" in settings:
                if not isinstance(settings["validation_size"], (float, int)):
                    raise ValueError(
                        f"Invalid type for validation_size in dataset {dataset_name}. It must be a float or integer."
                    )
                if not (0 < settings["validation_size"] < 1):
                    raise ValueError(
                        f"Validation size must be between 0 and 1 for dataset {dataset_name}."
                    )

    def get_strategies(self) -> Dict[str, DatasetStrategy]:
        """
        Get dataset strategies based on the configuration file.

        :return: A dictionary mapping dataset names to their respective strategies.
        """
        dataset_strategies = {}
        for dataset_name, dataset_config in self.config.items():
            if not dataset_config:
                raise ValueError(
                    f"Dataset {dataset_name} not found in the configuration."
                )
            strategy_type = dataset_config["strategy"]
            if strategy_type == "file":
                strategy = ANERCorpStrategy(
                    FileHandler(self.file_handler.file_path), dataset_config
                )
            elif strategy_type == "dataset":
                strategy = Conll2003Strategy(dataset_config)
            else:
                raise ValueError(f"Unknown dataset strategy: {strategy_type}")
            dataset_strategies[dataset_name] = strategy
        return dataset_strategies

    def generate_corpora(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate corpora for all datasets defined in the configuration.

        :return: A dictionary containing processed data for all datasets.
        """
        corpora = {}
        strategies = self.get_strategies()
        for dataset_name, strategy in strategies.items():
            logging.info("Processing %s", dataset_name)
            corpora[dataset_name] = strategy.process_data()
        return corpora
