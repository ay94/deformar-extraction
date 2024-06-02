import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from experiment_utils.utils import FileHandler, GenerateData


@pytest.fixture
def mock_dataset():
    return {
        "train": [{"id": "1", "tokens": ["hello", "world"], "ner_tags": [0, 1]}],
        "test": [{"id": "2", "tokens": ["goodbye", "world"], "ner_tags": [0, 1]}],
    }


@pytest.fixture
def mock_data():
    return {
        "train": [{"id": "3", "tokens": ["foo", "bar"], "ner_tags": [0, 1]}],
        "test": [{"id": "4", "tokens": ["baz", "qux"], "ner_tags": [0, 1]}],
    }


def write_data_with_mock(
    file_path: Path, data: List[Dict[str, Any]], mock_data: List[Dict[str, Any]]
) -> None:
    """
    Write both actual and mock data to a file.

    Parameters:
    - file_path (Path): The path to the file where data should be written.
    - data (List[Dict[str, Any]]): The actual data to be written.
    - mock_data (List[Dict[str, Any]]): The mock data to be written.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for dataset in [data, mock_data]:
            for item in dataset:
                tokens = item["tokens"]
                tags = item["ner_tags"]
                for token, tag in zip(tokens, tags):
                    f.write(f"{token} {tag}\n")
                f.write(". O\n")  # Sentence delimiter


def test_integration(tmp_path, mock_dataset, mock_data):
    # Setup project folder using tmp_path fixture
    project_folder = tmp_path / "project"
    os.makedirs(project_folder, exist_ok=True)

    # Create a FileHandler instance
    fh = FileHandler(project_folder=project_folder)

    # Create temporary files and write mock data
    train_file = fh.create_filename("dummy_path_train.txt")
    test_file = fh.create_filename("dummy_path_test.txt")
    write_data_with_mock(train_file, mock_dataset["train"], mock_data["train"])
    write_data_with_mock(test_file, mock_dataset["test"], mock_data["test"])

    # Initialize GenerateData with the file handler and dataset
    gd = GenerateData(fh=fh, dataset=mock_dataset, path="dummy_path")

    # Generate the ANERCorp data
    anercorp_data = gd.generate_anercorp(split_data=True, validation_size=0.2)
    assert (
        "train" in anercorp_data
        and "validation" in anercorp_data
        and "test" in anercorp_data
    ), "ANERCorp data generation failed."
    print("ANERCorp Data:", anercorp_data)

    # Generate the conll2003 data
    conll_data = gd.generate_conll2003()
    assert (
        "train" in conll_data and "val" in conll_data and "test" in conll_data
    ), "CoNLL2003 data generation failed."
    print("CoNLL2003 Data:", conll_data)
