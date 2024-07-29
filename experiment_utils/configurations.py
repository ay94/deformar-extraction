import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict

@dataclass
class TrainingConfig:
    train_batch_size: int
    test_batch_size: int
    shuffle: bool
    epochs: int
    splits: int
    learning_rate: float
    warmup_ratio: float
    max_grad_norm: float
    accumulation_steps: int
    logging_step: int

    def __post_init__(self):
        self.validate_config()

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        try:
            config_dict['learning_rate'] = float(config_dict['learning_rate'])
        except ValueError as e:
            logging.error("Invalid format for learning_rate, needs to be convertible to float")
            raise ValueError("Invalid format for learning_rate") from e
        return TrainingConfig(**config_dict)

    def validate_config(self):
        if not (0 < self.learning_rate < 1):
            logging.error("Invalid learning rate: %s", self.learning_rate)
            raise ValueError("Learning rate must be between 0 and 1")
        if not (0 < self.warmup_ratio < 1):
            logging.error("Invalid warmup ratio: %s", self.warmup_ratio)
            raise ValueError("Warmup ratio must be between 0 and 1")
        if self.epochs <= 0:
            logging.error("Invalid number of epochs: %s", self.epochs)
            raise ValueError("Epochs must be greater than 0")
        if self.train_batch_size <= 0 or self.test_batch_size <= 0:
            logging.error("Invalid batch sizes: Train %s, Valid %s", self.train_batch_size, self.test_batch_size)
            raise ValueError("Batch sizes must be greater than 0")
        if self.accumulation_steps < 1:
            logging.error("Invalid accumulation steps: %s", self.accumulation_steps)
            raise ValueError("Accumulation steps must be at least 1")
        if self.logging_step < 1:
            logging.error("Invalid logging steps: %s", self.logging_step)
            raise ValueError("Accumulation steps must be at least 1")
        logging.info("Configuration validated successfully")



@dataclass
class TokenizationStrategy:
    type: str
    index: int
    schema: Optional[str] = None

@dataclass
class TokenizationConfig:
    tokenizer_path: str
    preprocessor_path: str
    max_seq_len: int
    strategy: TokenizationStrategy

    def __post_init__(self):
        self.validate_config()

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        # This assumes the 'strategy' sub-dictionary is properly formatted
        strategy_config = TokenizationStrategy(**config_dict['strategy'])
        return TokenizationConfig(
            tokenizer_path=config_dict['tokenizer_path'],
            preprocessor_path=config_dict['preprocessor_path'],
            max_seq_len=config_dict['max_seq_len'],
            strategy=strategy_config
        )

    def validate_config(self):
        if not isinstance(self.max_seq_len, int) or self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path cannot be empty")
        if not self.preprocessor_path:
            raise ValueError("preprocessor_path cannot be empty")
        if not isinstance(self.strategy, TokenizationStrategy):
            raise ValueError("strategy must be an instance of TokenizationStrategy")
        if self.strategy.type not in ['core', 'all']:  # Example check
            raise ValueError("Invalid strategy type specified")
        if not isinstance(self.strategy.index, int) or self.strategy.index < 0:
            raise ValueError("Strategy index must be a non-negative integer")



@dataclass
class ModelConfig:
    model_path: str
    dropout_rate: float
    enable_attentions: bool
    enable_hidden_states: bool
    initialize_output_layer: bool

    def __post_init__(self):
        self.validate_config()

    @staticmethod
    def from_dict(config_dict):
        return ModelConfig(**config_dict)

    def validate_config(self):
        if not self.model_path:
            raise ValueError("Model path cannot be empty")
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError("Dropout rate must be between 0 and 1")
        if not isinstance(self.enable_attentions, bool):
            raise ValueError("enable_attentions must be a boolean value")
        if not isinstance(self.enable_hidden_states, bool):
            raise ValueError("enable_hidden_states must be a boolean value")
        if not isinstance(self.initialize_output_layer, bool):
            raise ValueError("initialize_output_layer must be a boolean value")



@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 1
    verbose: bool = True
    normalize_embeddings: bool = False

    def set_params(self,
                   n_neighbors: Optional[int] = None,
                   min_dist: Optional[float] = None,
                   metric: Optional[str] = None,
                   normalize_embeddings: Optional[bool] = None):
        """Optionally update UMAP parameters."""
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        if min_dist is not None:
            self.min_dist = min_dist
        if metric is not None:
            self.metric = metric
        if normalize_embeddings is not None:
            self.normalize_embeddings = normalize_embeddings

    @staticmethod
    def from_dict(config_dict):
        """Create UMAPConfig from a dictionary."""
        return UMAPConfig(**config_dict)

    def __post_init__(self):
        """Validate UMAP configuration to ensure valid settings."""
        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        if not isinstance(self.min_dist, float) or self.min_dist < 0:
            raise ValueError("min_dist must be a non-negative float.")
        if not isinstance(self.metric, str):
            raise ValueError("metric must be a string.")
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a boolean.")
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError("normalize_embeddings must be a boolean.")

