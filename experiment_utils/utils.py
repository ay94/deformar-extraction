import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


class FileHandler:
    """
    Handles file operations including saving and loading various types of data (JSON, binary, model states)
    to and from a specified project directory.
    """

    def __init__(self, project_folder: Path) -> None:
        """Initialize with the path to the project folder."""
        self.project_folder: Path = Path(project_folder)

    def create_filename(self, file_name: str) -> Path:
        """Return the full path for a given filename within the project folder."""
        return self.project_folder / file_name

    def load_corpora(self, dataset_name: str, path: str) -> Dict[str, Any]:
        """Instantiate and return the corpora from GenerateData using specified dataset and path."""
        corpora = GenerateData(self, dataset_name, path)
        return corpora.corpora

    def save_json(self, path: str, data: Any) -> None:
        """Save data to a JSON file."""
        try:
            with open(self.create_filename(path), "w", encoding="utf-8") as outfile:
                json.dump(data, outfile)
        except json.JSONDecodeError as e:
            logging.error("Failed to encode JSON %s: %s", path, e)

    def load_json(self, path: str) -> Optional[Any]:
        """Load data from a JSON file."""
        try:
            with open(self.create_filename(path), "r", encoding="utf-8") as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            logging.error("JSON file not found: %s", path)
        except json.JSONDecodeError:
            logging.error("Error decoding JSON from file: %s", path)
        return None

    def save_object(self, obj: Any, obj_name: str) -> None:
        """Save an object to a binary file using pickle."""
        try:
            with open(self.create_filename(obj_name), "wb") as output:
                pkl.dump(obj, output, pkl.HIGHEST_PROTOCOL)
        except pkl.PicklingError as e:
            logging.error("Pickling error while saving object to %s: %s", obj_name, e)

    def load_object(self, obj_name: str) -> Optional[Any]:
        """Load a pickled object from a binary file."""
        try:
            with open(self.create_filename(obj_name), "rb") as inp:
                return pkl.load(inp)
        except FileNotFoundError:
            logging.error("Object file not found: %s", obj_name)
        except pkl.UnpicklingError:
            logging.error("Error unpickling object from file: %s", obj_name)
        return None

    def save_model_state(self, model: Any, model_name: str) -> None:
        """Save the state dictionary of a PyTorch model."""
        try:
            torch.save(model.state_dict(), self.create_filename(model_name))
        except Exception as e:
            logging.error("Failed to save model state to %s: %s", model_name, e)

    def load_model_state(self, model: Any, model_name: str) -> Optional[Any]:
        """Load a model's state dictionary into a PyTorch model."""
        try:
            model.load_state_dict(torch.load(self.create_filename(model_name)))
            return model
        except FileNotFoundError:
            logging.error("Model state file not found: %s", model_name)
        except Exception as e:
            logging.error("Failed to load model state from %s: %s", model_name, e)
            return None

    def save_model(self, model: Any, model_name: str) -> None:
        """Save a complete PyTorch model."""
        try:
            torch.save(model, self.create_filename(model_name))
        except Exception as e:
            logging.error("Failed to save model to %s: %s", model_name, e)

    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a complete PyTorch model."""
        try:
            model = torch.load(self.create_filename(model_name))
            model.eval()
            return model
        except FileNotFoundError:
            logging.error("Model state file not found: %s", model_name)
        except Exception as e:
            logging.error("Failed to load model from %s: %s", model_name, e)
            return None


class GenerateData:
    """
    Class for generating and managing data sets from text files and structured data.
    It supports operations to read, split, and generate data sets like ANERCorp_CamelLab and conll2003.
    """

    def __init__(self, fh: FileHandler, dataset_name: str, path: str) -> None:
        """
        Initialize with a file handler, dataset, and path to manage data extraction and processing.

        Parameters:
        fh (FileHandler): A file handler object for managing file paths.
        dataset (Dict[str, Any]): A dictionary containing dataset configurations or references.
        path (str): Base path for dataset file operations.
        use_validation (bool): Flag to determine whether to include a validation split in the datasets.
        """
        self.fh = fh
        self.dataset_name = dataset_name
        self.path = path
        self.corpora = self.initialize_datasets()

    def initialize_datasets(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Initialize datasets based on the configuration provided at class instantiation.

        Returns:
        Dict[str, Optional[Dict[str, Any]]]: A dictionary of initialized datasets.
        """
        logging.info("Initializing datasets")
        corpora = {}

        # Initialize ANERCorp_CamelLab without validation split
        try:
            corpora["ANERCorp_CamelLab"] = self.generate_anercorp(split_data=False)
            logging.info("ANERCorp_CamelLab dataset successfully initialized.")
        except Exception as e:
            logging.error("Failed to initialize ANERCorp_CamelLab: %s", e)
            corpora["ANERCorp_CamelLab"] = None

        # Initialize conll2003 dataset, assuming it always has the same structure
        try:
            corpora["conll2003"] = self.generate_conll2003()
            logging.info("conll2003 dataset successfully initialized.")
        except Exception as e:
            logging.error("Failed to initialize conll2003: %s", e)
            corpora["conll2003"] = None

        # Initialize ANERCorp_CamelLab with validation split
        try:
            corpora["ANERCorp_CamelLab-validation"] = self.generate_anercorp(
                split_data=True
            )
            logging.info(
                "ANERCorp_CamelLab-validation dataset successfully initialized."
            )
        except Exception as e:
            logging.error("Failed to initialize ANERCorp_CamelLab-validation: %s", e)
            corpora["ANERCorp_CamelLab-validation"] = None

        return corpora

    def read_split(self, split):
        """
        Reads and processes a text file split into sentences based on punctuation,
        returning structured data of words and their corresponding tags. Assumes that each line in the file
        contains a word followed by its tag, separated by whitespace, and that sentences are delimited by
        lines containing only a period.

        Parameters:
        split (str): The name of the split, typically 'train', 'test', or 'val', used to generate the filename.

        Returns:
        List[Tuple[List[str], List[str]]]: A list of tuples, each containing a list of words and a list of corresponding tags.
        """
        words, tags, sentences = [], [], []
        logging.info("Generating %s Split", split)
        file_path = self.fh.create_filename(f"{self.path}_{split}.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # import pdb; pdb.set_trace()
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

    def generate_anercorp(
        self, split_data: bool = False, validation_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Processes text files for the ANERCorp_CamelLab dataset, optionally splitting into training,
        validation, and testing datasets, and mapping each with labels and indices.

        Parameters:
        split_data (bool): Whether to split the training data into training and validation sets.
        validation_size (float): The proportion of the training data to be used as validation data.

        Returns:
        Dict[str, Any]: A dictionary containing training, validation (optional), and testing datasets,
                        along with metadata about labels.
        """
        logging.info("Generating ANERCorp_CamelLab")
        
        ner_inv_map = {v: k for k, v in ner_map.items()}

        # Reading the training and test splits
        tr_dt = self.read_split("train")
        test_dt = self.read_split("test")

        # Optionally split the training data into training and validation
        if split_data:
            train_dt, val_dt = train_test_split(
                tr_dt, test_size=validation_size, random_state=1
            )
        else:
            train_dt, val_dt = tr_dt, []

        # Prepare the output datasets
        train = [
            (sentence_id, sentence[0], sentence[1])
            for sentence_id, sentence in enumerate(train_dt)
        ]
        validation = [
            (sentence_id, sentence[0], sentence[1])
            for sentence_id, sentence in enumerate(val_dt)
        ]
        test = [
            (sentence_id, sentence[0], sentence[1])
            for sentence_id, sentence in enumerate(test_dt)
        ]

        # Construct the result dictionary
        result = {
            "train": train,
            "test": test,
            "labels": list(ner_map.keys()),
            "labels_map": ner_map,
            "inv_labels": ner_inv_map,
        }
        if validation:  # Include validation set only if it has been created
            result = {
                "train": train,
                "validation": validation,
                "test": test,
                "labels": list(ner_map.keys()),
                "labels_map": ner_map,
                "inv_labels": ner_inv_map,
            }
        return result

    def generate_split_data(
        self, dataset: Dict[str, Any], split: str, ner_inv_map: Dict[int, str]
    ) -> List[Tuple[int, List[str], List[str]]]:
        """
        Generates data for a specific split from the dataset. It retrieves necessary
        information such as ID, tokens, and NER tags for each example in the split,
        and constructs a list of tuples where each tuple represents an example with its
        corresponding ID, tokens, and NER tags.

        Parameters:
        - dataset (Dict[str, Any]): The dataset containing the splits.
        - split (str): The name of the split, e.g., 'train', 'test'.
        - ner_inv_map (Dict[int, str]): A mapping from numerical NER tags to their string representation.

        Returns:
        - List[Tuple[int, List[str], List[str]]]: A list of tuples, each containing the ID, tokens, and NER tags.
        """
        sentences = []
        data_split = dataset.get(split, [])
        if not data_split:
            logging.error("No data found for split: %s", split)
            return sentences

        logging.info("Generating %s Split", split)
        try:
            for i in tqdm(range(len(data_split)), desc=f"Processing {split}"):
                entry = data_split[i]
                entry_id = int(
                    entry.get("id", -1)
                )  # Default to -1 if 'id' is not found
                tokens = entry.get("tokens", [])
                tags = [ner_inv_map.get(tid, "O") for tid in entry.get("ner_tags", [])]
                sentences.append((entry_id, tokens, tags))
        except Exception as e:
            logging.error("Error processing %s data: %s", split, e)

        return sentences

    def generate_conll2003(self) -> Dict[str, Any]:
        """
        Generates and organizes the conll2003 dataset into structured data for training, validation,
        and testing purposes. It maps NER tags from their ID representations to string labels.

        Returns:
        Dict[str, Any]: A dictionary containing structured datasets for training, validation, and testing,
                        along with labels and their mappings.
        """
        logging.info("Generating conll2003 dataset")

        dataset = load_dataset(self.dataset_name, trust_remote_code=True)
        # Define NER tag mapping
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
        ner_inv_map = {v: k for k, v in ner_map.items()}

        # Generate data for each split using helper function
        try:
            tr = self.generate_split_data(dataset, "train", ner_inv_map)
            vl = self.generate_split_data(dataset, "validation", ner_inv_map)
            te = self.generate_split_data(dataset, "test", ner_inv_map)
            datasets = {"train": tr, "val": vl, "test": te}
            logging.info("Successfully generated all splits for conll2003.")
        except Exception as e:
            logging.error("Failed to generate datasets for conll2003: %s", e)
            return {}

        # Return the datasets along with the NER labels and their mappings
        return {
            "train": datasets["train"],
            "validation": datasets["val"],
            "test": datasets["test"],
            "labels": list(ner_map.keys()),
            "labels_map": ner_map,
            "inv_labels": ner_inv_map,
        }


import logging
import sys


def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    if not logger.handlers:  # To ensure no duplicate handlers are added
        # Create handler that logs to sys.stdout (standard output)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)  # Adjust the logging level as needed

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger

# Then use it in your script or notebook

def main():
    from datasets import load_dataset

    from experiment_utils import colab, utils

    logging.basicConfig(level=logging.INFO)
    local_drive_dir = colab.init("My Drive")
    data_folder = (
        local_drive_dir
        / "Final Year Experiments/Class Imbalance/0_generateExperimentData"
    )
    fh = utils.FileHandler(data_folder)
    conll2003_dataset = load_dataset("conll2003", trust_remote_code=True)
    corpora = fh.load_corpora(
        conll2003_dataset, "ANERcorp-CamelLabSplits/ANERCorp_CamelLab"
    )




if __name__ == "__main__":
    main()
