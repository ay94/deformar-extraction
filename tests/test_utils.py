import pytest
from experiment_utils.utils import FileHandler, GenerateData  # Import classes from your utility file

# Setup a fixture for file handler
@pytest.fixture
def file_handler(tmp_path):
    fh = FileHandler(project_folder=tmp_path)
    return fh

# Test the FileHandler's ability to create filenames correctly
def test_create_filename(file_handler):
    filename = "test.txt"
    expected_path = file_handler.project_folder / filename
    assert file_handler.create_filename(filename) == expected_path, "Filename creation did not construct the correct path."

# Test loading and saving JSON
def test_json_save_load(file_handler):
    test_data = {"key": "value"}
    filename = "test.json"
    file_handler.save_json(filename, test_data)
    
    loaded_data = file_handler.load_json(filename)
    assert loaded_data == test_data, "JSON saved and loaded data do not match."

# Test saving and loading objects with pickle
def test_pickle_save_load(file_handler):
    test_data = {"pickle": "test"}
    filename = "pickle.pkl"
    file_handler.save_object(test_data, filename)

    loaded_data = file_handler.load_object(filename)
    assert loaded_data == test_data, "Pickle saved and loaded data do not match."

# Example test for GenerateData
def test_generate_data_initialization(file_handler):
    dataset = {"train": [{"id": "1", "tokens": ["hello", "world"], "ner_tags": [0, 1]}]}
    gd = GenerateData(fh=file_handler, dataset=dataset, path="dummy_path")
    assert "ANERCorp_CamelLab" in gd.corpora, "Failed to initialize datasets correctly."


