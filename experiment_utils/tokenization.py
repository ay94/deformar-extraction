import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml
from arabert.preprocess import ArabertPreprocessor
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer


@dataclass
class TokenizedOutput:
    sentence_index: int
    core_tokens: List[str] = field(default_factory=list)
    word_pieces: List[List[str]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)

    core_tokens_df: List[str] = field(default_factory=list)
    sentence_index_df: List[int] = field(default_factory=list)
    tokens_df: List[int] = field(default_factory=list)
    word_pieces_df: List[List[str]] = field(default_factory=list)
    words_df: List[str] = field(default_factory=list)
    labels_df: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict):
        """Create an instance from a dictionary."""
        return TokenizedOutput(**data)


class TokenStrategy(ABC):
    IGNORED_TOKEN_LABEL = "IGNORED"

    @abstractmethod
    def handle_tokens(
        self, tokens: List[str], label: str, tokens_data: Dict[str, List[str]]
    ):
        """Process tokens and update tokens_data directly."""
        pass


class CoreTokenStrategy(TokenStrategy):
    def __init__(self, index=0):
        self.index = index

    def handle_tokens(
        self, tokens: List[str], label: str, tokens_data: Dict[str, List[str]] = None
    ):
        if tokens:
            selected_index = min(self.index, len(tokens) - 1)
            selected_token = tokens[selected_index]
            if tokens_data:
                tokens_data["core_tokens"].append(selected_token)
                tokens_data["labels"].append(label)
                tokens_data["core_tokens_df"].extend(
                    [
                        token if token == selected_token else self.IGNORED_TOKEN_LABEL
                        for token in tokens
                    ]
                )
                tokens_data["labels_df"].extend(
                    [
                        label if i == self.index else self.IGNORED_TOKEN_LABEL
                        for i in range(len(tokens))
                    ]
                )
            else:
                core_tokens, labels, processed_tokens, processed_labels = [], [], [], []
                core_tokens.append(selected_token)
                labels.append(label)
                processed_tokens.extend(tokens)
                processed_labels.extend(
                    [
                        label if i == self.index else self.IGNORED_TOKEN_LABEL
                        for i in range(len(tokens))
                    ]
                )
                return core_tokens, labels, processed_tokens, processed_labels
        else:
            logging.warning("No tokens provided for tokenization.")


class AllTokensStrategy(TokenStrategy):
    def __init__(self, schema="BIO"):
        self.schema = schema

    def handle_tokens(
        self, tokens: List[str], label: str, tokens_data: Dict[str, List[str]] = None
    ):
        if tokens:
            all_labels = self.get_labels(label, len(tokens), self.schema)
            if tokens_data:
                tokens_data["core_tokens"].extend(tokens)
                tokens_data["labels"].extend(all_labels)
                tokens_data["core_tokens_df"].extend(tokens)
                tokens_data["labels_df"].extend(all_labels)
            else:
                core_tokens, labels, processed_tokens, processed_labels = [], [], [], []
                core_tokens.extend(tokens)
                labels.extend(all_labels)
                processed_tokens.extend(tokens)
                processed_labels.extend(all_labels)
                return core_tokens, labels, processed_tokens, processed_labels
        else:
            logging.warning("No tokens provided for tokenization.")

    def get_labels(
        self, initial_label: str, token_count: int, schema: str
    ) -> List[str]:
        match schema:
            case "BIO":
                if initial_label.startswith("B-"):
                    return [initial_label] + [
                        f"I-{initial_label[2:]}" for _ in range(1, token_count)
                    ]
                return [initial_label] * token_count
            case _:
                # Default case if the schema is not recognized
                return [initial_label] * token_count


class TokenStrategyFactory:
    def __init__(self, config):
        self.strategies = {"core": CoreTokenStrategy, "all": AllTokensStrategy}
        self.config = config

    def get_strategy(self):

        strategy_type, kwargs = self.get_params()
        strategy_cls = self.strategies.get(strategy_type)

        if not strategy_cls:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return strategy_cls(**kwargs)

    def get_params(self):
        """Validate the configuration and retrieve the appropriate tokenization strategy."""
        strategy = self.config.strategy
        strategy_type = strategy.type
        index = strategy.index
        schema = strategy.schema

        # Validate the configuration
        valid_types = ["core", "all"]  # Define valid strategy types
        if strategy_type not in valid_types:
            logging.error("Invalid or missing strategy type in configuration.")
            raise ValueError("Invalid or missing strategy type in configuration.")

        # Prepare parameters based on strategy type
        params = {}
        if strategy_type == "core":
            if index is None:
                logging.error("Missing 'index' for 'core' strategy in configuration.")
                raise ValueError(
                    "Missing 'index' for 'core' strategy in configuration."
                )
            params["index"] = index
        elif strategy_type == "all":
            if schema is None:
                logging.error("Missing 'schema' for 'all' strategy in configuration.")
                raise ValueError(
                    "Missing 'schema' for 'all' strategy in configuration."
                )
            params["schema"] = schema
        # Use the factory to get the appropriate strategy
        return strategy_type, params


class TokenizedTextProcessor:

    def __init__(
        self, texts, tags, max_seq_len, tokenizer, strategy, preprocessor=None
    ):
        self.texts = texts
        self.tags = tags
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        """Get tokenized output for a single text entry based on its index."""
        words = self.texts[index]
        tags = self.tags[index]
        tokens_data = self._tokenize_and_prepare(words, tags, index)
        tokens_data = self._truncate_and_add_special_tokens(tokens_data)
        return TokenizedOutput.from_dict(tokens_data)

    def _tokenize_and_prepare(self, words, tags, index):
        """Prepare token data, including preprocessing, tokenizing, and updating token data."""
        tokens_data = {
            "sentence_index": 0,
            "core_tokens": [],
            "word_pieces": [],
            "labels": [],
            "words": [],
            "core_tokens_df": [],
            "sentence_index_df": [],
            "tokens_df": [],
            "word_pieces_df": [],
            "words_df": [],
            "labels_df": [],
        }
        for word_id, (word, label) in enumerate(zip(words, tags)):
            tokens = self._preprocess_and_tokenize(word)
            if tokens:
                tokens_data = self._update_tokens_data(
                    tokens_data, tokens, index, word, word_id, label
                )
        return tokens_data

    def _preprocess_and_tokenize(self, word):
        """Apply preprocessing if available, then tokenize the word."""
        if self.preprocessor:
            word = self.preprocessor.preprocess(word)
        return self.tokenizer.tokenize(word)

    def _update_tokens_data(self, tokens_data, tokens, index, word, word_id, label):
        """Update tokens data dictionary with new tokens and associated data."""
        if len(tokens) > 0:
            self.strategy.handle_tokens(tokens, label, tokens_data)
            tokens_data["sentence_index"] = index
            tokens_data["word_pieces"].append(tokens)
            tokens_data["words"].append(word)

            tokens_data["sentence_index_df"].extend([index] * len(tokens))
            tokens_data["tokens_df"].extend(tokens)
            tokens_data["word_pieces_df"].extend([tokens] * len(tokens))
            tokens_data["words_df"].extend([word] * len(tokens))
        return tokens_data

    def _truncate_and_add_special_tokens(self, tokens_data):
        """Truncate tokens data to max sequence length and optionally add special tokens."""
        max_length = self.max_seq_len - self.tokenizer.num_special_tokens_to_add()
        for key in tokens_data:
            if key.endswith("df"):
                tokens_data[key] = tokens_data[key][:max_length]
        self._add_special_tokens(tokens_data)
        return tokens_data

    def _add_special_tokens(self, tokens_data):
        """Add special tokens such as CLS and SEP to the beginning and end of sequences, respectively."""
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        for key in tokens_data:
            if key.endswith("df"):
                if isinstance(tokens_data[key], list):
                    if key != "sentence_index_df":
                        tokens_data[key] = [cls_token] + tokens_data[key] + [sep_token]
                    else:
                        tokens_data[key] = (
                            [tokens_data["sentence_index"]]
                            + tokens_data[key]
                            + [tokens_data["sentence_index"]]
                        )


class TokenizationConfigManager:
    """Manage loading and configuration of tokenizers and preprocessors."""

    def __init__(self, config):
        self.config = config
        self.tokenizer_path = self.config.tokenizer_path
        self.preprocessor_path = self.config.preprocessor_path

    def load_tokenizer(self):
        logging.info("Loading Tokenizer %s", self.tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, do_lower_case=False
        )
        preprocessor = None
        if self.preprocessor_path:
            logging.info("Loading Preprocessor %s", self.preprocessor_path)
            preprocessor = ArabertPreprocessor(self.preprocessor_path)
        return tokenizer, preprocessor


class DataSplitManager:
    """Handles the processing of data across different splits."""

    def __init__(self, data, tokenizer, max_seq_len, strategy, preprocessor=None):
        self.data = data["splits"]
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.preprocessor = preprocessor

    def process_splits(self, specific_split: str = None):
        splits_to_process = [specific_split] if specific_split else self.data.keys()
        split_outputs = {}

        for split in splits_to_process:
            logging.info("Processing %s split", split)
            tokenized_sentences = [
                tokenized_sentence_dataset
                for tokenized_sentence_dataset in tqdm(
                    self.create_split_processor(split)
                )
            ]
            if split == "train":
                subword_index = self.get_split_subwords(split, tokenized_sentences)
            else:
                subword_index = {}
            split_outputs[split] = {
                "tokenized_text": tokenized_sentences,
                "subword_index": subword_index,
            }

        return split_outputs

    def create_split_processor(self, split):
        return TokenizedTextProcessor(
            texts=[x["words"] for x in self.data[split]],
            tags=[x["tags"] for x in self.data[split]],
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
            strategy=self.strategy,
            preprocessor=self.preprocessor,
        )

    def get_split_subwords(self, split, tokenized_data):
        logging.info("Extracting %s subwords", split)
        subwords = defaultdict(list)
        for tokenized_sentence in tqdm(tokenized_data):
            for token, label in zip(
                tokenized_sentence.core_tokens, tokenized_sentence.labels
            ):
                subwords[token].append(
                    {"tag": label, "sentence": tokenized_sentence.sentence_index}
                )
        return subwords


class TokenizationWorkflowManager:
    def __init__(self, data, config) -> None:
        self.data = data
        self.config = config
        self.processed_data = {}
        self.setup()
        self.generate_tokenized_data()

    def setup(self):
        """Set up the tokenization process by loading configurations and preparing managers."""
        self.config_manager = TokenizationConfigManager(self.config)
        try:
            self.tokenizer, self.preprocessor = self.config_manager.load_tokenizer()
            self.max_seq_len = self.config_manager.config.max_seq_len
            self.strategy = TokenStrategyFactory(
                self.config_manager.config
            ).get_strategy()
            self.data_manager = DataSplitManager(
                self.data,
                self.tokenizer,
                self.max_seq_len,
                self.strategy,
                self.preprocessor,
            )
        except Exception as e:
            logging.error("An error occurred during setup: %s", e)
            raise

    def get_split_data(self, split_name):
        split_data = self.processed_data.get(split_name, None)
        if split_data is None or (
            split_name == "train" and not split_data.get("subword_index")
        ):
            logging.warning("%s data is not available.", split_name)
            return {}  # Return an empty dictionary or a default structure
        return split_data

    def generate_tokenized_data(self):
        """Process and store data for all configured splits."""
        if hasattr(self, "data_manager"):
            self.processed_data = self.data_manager.process_splits()
        else:
            logging.error(
                "Data Manager is not properly configured due to setup failure."
            )

    @property
    def train(self):
        return self.get_split_data("train").get("tokenized_text")

    @property
    def test(self):
        return self.get_split_data("test").get("tokenized_text")

    @property
    def validation(self):
        return self.get_split_data("validation").get("tokenized_text")

    @property
    def train_subwords(self):
        return self.get_split_data("train").get("subword_index")

    def get_split(self, split):
        match split:
            case "train":
                return self.train
            case "test":
                return self.test
            case "validation":
                return self.validation
