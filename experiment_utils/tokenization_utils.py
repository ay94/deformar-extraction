import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from abc import ABC, abstractmethod
import logging

import logging
from pathlib import Path
import yaml
from tqdm.autonotebook import tqdm
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoModel, AutoTokenizer
from collections import Counter, defaultdict

import logging
from tqdm.autonotebook import tqdm
from collections import defaultdict
from utils import setup_logging

logger  = setup_logging()



@dataclass
class TokenizedOutput:
    sentence_index: int
    core_tokens: List[str] = field(default_factory=list)
    word_pieces: List[List[str]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)

    core_tokens_df: List[str] = field(default_factory=list)
    sentence_index_df: List[int] = field(default_factory=list)
    word_pieces_df: List[List[str]] = field(default_factory=list)
    words_df: List[str] = field(default_factory=list)
    word_ids_df: List[int] = field(default_factory=list)
    labels_df: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict):
        """Create an instance from a dictionary."""
        return TokenizedOutput(**data)




class TokenStrategy(ABC):
    IGNORED_TOKEN_LABEL = "IGNORED"
    @abstractmethod
    def handle_tokens(self, tokens: List[str], label: str, tokens_data: Dict[str, List[str]]):
        """Process tokens and update tokens_data directly."""
        pass


class CoreTokenStrategy(TokenStrategy):
    def __init__(self, index=0):
        self.index = index

    def handle_tokens(self, tokens: List[str], label: str, tokens_data: Dict[str, List[str]]):
        if tokens:
            selected_index = min(self.index, len(tokens) - 1)
            selected_token = tokens[selected_index]
            tokens_data['core_tokens'].append(selected_token)
            tokens_data['labels'].append(label)
            tokens_data['core_tokens_df'].extend([token if token == selected_token else self.IGNORED_TOKEN_LABEL for token in tokens])
            tokens_data['labels_df'].extend([label if i == self.index else self.IGNORED_TOKEN_LABEL for i in range(len(tokens))])
        else:
            logger.warning("No tokens provided for tokenization.")

class AllTokensStrategy(TokenStrategy):
    def __init__(self, schema='BIO'):
        self.schema = schema

    def handle_tokens(self, tokens: List[str], label: str, tokens_data: Dict[str, List[str]]):
        if tokens:
          labels = self.get_labels(label, len(tokens), self.schema)
          tokens_data['core_tokens'].extend(tokens)
          tokens_data['labels'].extend(labels)
          tokens_data['core_tokens_df'].extend(tokens)
          tokens_data['labels_df'].extend(labels)
        else:
            logger.warning("No tokens provided for tokenization.")

    def get_labels(self, initial_label: str, token_count: int, schema: str) -> List[str]:
        match schema:
            case 'BIO':
                if initial_label.startswith('B-'):
                    return [initial_label] + [f'I-{initial_label[2:]}' for _ in range(1, token_count)]
                return [initial_label] * token_count
            case _:
                # Default case if the schema is not recognized
                return [initial_label] * token_count


class TokenizedTextProcessor:

    def __init__(self, texts, tags, max_seq_len, tokenizer, strategy, preprocessor=None):
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
        text_list = self.texts[index]
        tags = self.tags[index]
        tokens_data = self._tokenize_and_prepare(text_list, tags, index)
        tokens_data = self._truncate_and_add_special_tokens(tokens_data)
        return TokenizedOutput.from_dict(tokens_data)

    def _tokenize_and_prepare(self, text_list, tags, index):
        """Prepare token data, including preprocessing, tokenizing, and updating token data."""
        tokens_data = {
            'sentence_index': 0,
            'core_tokens': [],
            'word_pieces': [],
            'labels': [],
            'words': [],
            'core_tokens_df': [],
            'sentence_index_df': [],
            'word_pieces_df': [],
            'words_df': [],
            'word_ids_df': [],
            'labels_df': [],
        }
        for word_id, (word, label) in enumerate(zip(text_list, tags)):
            tokens = self._preprocess_and_tokenize(word)
            if tokens:
                tokens_data = self._update_tokens_data(tokens_data, tokens, index, word, word_id, label)
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
            # tokens_data['core_tokens'].append(tokens[0])
            tokens_data['sentence_index'] = index
            tokens_data['word_pieces'].append(tokens)
            tokens_data['words'].append(word)
            # tokens_data['labels'].append(label)

            # tokens_data['core_tokens_df'].extend([tokens[0]] + ['IGNORED'] * (len(tokens) - 1))
            tokens_data['sentence_index_df'].extend([index] * len(tokens))
            tokens_data['word_pieces_df'].extend([tokens] * len(tokens))
            tokens_data['words_df'].extend([word] * len(tokens))
            tokens_data['word_ids_df'].extend([word_id] * len(tokens))
            # tokens_data['labels_df'].extend([label if i == 0 else 'IGNORED' for i in range(len(tokens))])
        return tokens_data

    def _truncate_and_add_special_tokens(self, tokens_data):
        """Truncate tokens data to max sequence length and optionally add special tokens."""
        max_length = self.max_seq_len - self.tokenizer.num_special_tokens_to_add()
        for key in tokens_data:
          if key.endswith('df'):
            tokens_data[key] = tokens_data[key][:max_length]
        self._add_special_tokens(tokens_data)
        return tokens_data

    def _add_special_tokens(self, tokens_data):
        """Add special tokens such as CLS and SEP to the beginning and end of sequences, respectively."""
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        for key in tokens_data:
          if key.endswith('df'):
            if isinstance(tokens_data[key], list):
                tokens_data[key] = [cls_token] + tokens_data[key] + [sep_token]


class TokenizationConfigManager:
    """Manage loading and configuration of tokenizers and preprocessors."""
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.tokenizer_path = self.config['tokenizer_path']
        self.preprocessor_path = self.config['preprocessor_path']
        
    def load_tokenizer(self):
        logger.info(f"Loading Tokenizer {self.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, do_lower_case=False)
        preprocessor = None
        if self.preprocessor_path:
            logger.info(f"Loading Preprocessor {self.preprocessor_path}")
            preprocessor = ArabertPreprocessor(self.preprocessor_path)
        return tokenizer, preprocessor
      

    def load_config(self):
        with open(self.config_path, 'r') as file:
          # Load the YAML content from the file
          data = yaml.safe_load(file)
          return data
      
      


class DataSplitManager:
    """Handles the processing of data across different splits."""
    
    def __init__(self, data, tokenizer, max_seq_len, strategy, preprocessor=None):
        self.data = data['splits']
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.preprocessor = preprocessor

    def process_splits(self, specific_split: str = None):
        splits_to_process = [specific_split] if specific_split else self.data.keys()
        split_outputs = {}

        for split in splits_to_process:
            logger.info(f'Processing {split} split')
            tokenized_sentences = [tokenized_sentence_dataset for tokenized_sentence_dataset in tqdm(self.create_split_processor(split))]
            subword_index = self.get_split_subwords(split, tokenized_sentences)
            split_outputs[split] = {'tokenized_text': tokenized_sentences, 'subword_index': subword_index}

        return split_outputs

    def create_split_processor(self, split):
        return TokenizedTextProcessor(
            texts=[x[1] for x in self.data[split]],
            tags=[x[2] for x in self.data[split]],
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
            strategy=self.strategy,
            preprocessor=self.preprocessor,
        )
        

    def get_split_subwords(self, split, tokenized_data):
        logger.info(f'Extracting {split} subwords')
        subwords = defaultdict(list)
        for tokenized_sentence in tqdm(tokenized_data):
            for token, label in zip(tokenized_sentence.core_tokens, tokenized_sentence.labels):
                subwords[token].append({"tag": label, "sentence": tokenized_sentence.sentence_index})
        return subwords
    
    



class TokenizationWorkflowManager: 
    def __init__(self, data, config_path) -> None:
        self.data = data
        self.config_path = Path(config_path)
        self.processed_data = {}
        self.setup()
        self.generate_tokenized_data()
        
    def setup(self):
        """Set up the tokenization process by loading configurations and preparing managers."""
        if not self.config_path.exists():
            logging.error(f"Configuration file at {self.config_path} does not exist.")
            return

        self.config_manager = TokenizationConfigManager(self.config_path)
        self.tokenizer, self.preprocessor = self.config_manager.load_tokenizer()
        self.max_seq_len = self.config_manager.config.get('max_seq_len', 512)  # Default to a sensible value if not set
        self.strategy = self.get_strategy(self.config_manager.config)
        self.data_manager = DataSplitManager(self.data, self.tokenizer, self.max_seq_len, self.strategy, self.preprocessor)
        
    def get_split_data(self, split_name):
        split_data = self.processed_data.get(split_name)
        if split_data is None:
            logging.warning(f"{split_name} data is not available.")
            return {}  # Return an empty dictionary or a default structure
        return split_data
    
    def generate_tokenized_data(self):
        """Process and store data for all configured splits."""
        if hasattr(self, 'data_manager'):
            self.processed_data = self.data_manager.process_splits()
        else:
            logging.error("Data Manager is not properly configured due to setup failure.")

    def get_strategy(self, config):
        """Determine and return the appropriate tokenization strategy based on the configuration."""
        if isinstance(config.get('strategy'), int):
            return CoreTokenStrategy(config['strategy'])
        elif config.get('schema') is not None:
            return AllTokensStrategy(config['schema'])
        else:
            logging.error("Configuration for tokenization strategy is invalid.")
            raise ValueError("Invalid configuration for tokenization strategy.")

    @property
    def train(self):
        return self.get_split_data('train')

    @property
    def test(self):
        return self.get_split_data('test')

    def val(self):
        return self.get_split_data('val')
    
    