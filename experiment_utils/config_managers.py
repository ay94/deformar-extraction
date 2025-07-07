import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_utils.utils import FileHandler


@dataclass
class TrainingConfig:
    train_batch_size: int
    test_batch_size: int
    shuffle: bool
    num_workers: int
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
            config_dict["learning_rate"] = float(config_dict["learning_rate"])
        except ValueError as e:
            logging.error(
                "Invalid format for learning_rate, needs to be convertible to float"
            )
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
            logging.error(
                "Invalid batch sizes: Train %s, Valid %s",
                self.train_batch_size,
                self.test_batch_size,
            )
            raise ValueError("Batch sizes must be greater than 0")
        if self.accumulation_steps < 1:
            logging.error("Invalid accumulation steps: %s", self.accumulation_steps)
            raise ValueError("Accumulation steps must be at least 1")
        if self.logging_step < 1:
            logging.error("Invalid logging steps: %s", self.logging_step)
            raise ValueError("Accumulation steps must be at least 1")
        if self.num_workers < 1 and self.num_workers > 4:
            logging.error("Invalid num_workers: %s", self.num_workers)
            raise ValueError("Accumulation steps must be at least 1")
        logging.info("Training Config validated successfully")


@dataclass
class TokenizationStrategy:
    type: str
    index: int
    scheme: Optional[str] = None


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
        strategy_config = TokenizationStrategy(**config_dict["strategy"])
        return TokenizationConfig(
            tokenizer_path=config_dict["tokenizer_path"],
            preprocessor_path=config_dict["preprocessor_path"],
            max_seq_len=config_dict["max_seq_len"],
            strategy=strategy_config,
        )

    def validate_config(self):
        if not isinstance(self.max_seq_len, int) or self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path cannot be empty")
        # if not self.preprocessor_path:
        #     raise ValueError("preprocessor_path cannot be empty")
        if not isinstance(self.strategy, TokenizationStrategy):
            raise ValueError("strategy must be an instance of TokenizationStrategy")
        if self.strategy.type not in ["core", "all"]:  # Example check
            raise ValueError("Invalid strategy type specified")
        if not isinstance(self.strategy.index, int) or self.strategy.index < 0:
            raise ValueError("Strategy index must be a non-negative integer")
        logging.info("Tokenization Config validated successfully")


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
        logging.info("Model Config validated successfully")


@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 1
    verbose: bool = True
    normalize_embeddings: bool = False

    def set_params(
        self,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None,
        metric: Optional[str] = None,
        random_state: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None,
    ):
        """Optionally update UMAP parameters."""
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        if min_dist is not None:
            self.min_dist = min_dist
        if metric is not None:
            self.metric = metric
        if random_state is not None:
            self.random_state = random_state
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
        logging.info("UMAP Config validated successfully")


@dataclass
class ClusteringConfig:
    init_method: str = "k-means++"
    n_init: int = 10
    random_state: int = 1
    n_clusters: List = field(default_factory=lambda: [3, 4, 9])
    n_clusters_map: List[Dict] = field(
        default_factory=lambda: {3: "boundary", 4: "entity", 9: "token"}
    )
    silhouette_metric: str = "cosine"
    norm: str = "l2"

    def set_params(
        self,
        init_method: Optional[str] = None,
        n_init: Optional[int] = None,
        random_state: Optional[int] = None,
        n_clusters: Optional[List[Dict]] = None,
        silhouette_metric: Optional[str] = None,
        norm: Optional[str] = None,
    ):
        """Optionally update clustering parameters."""
        if init_method is not None:
            self.init_method = init_method
        if n_init is not None:
            self.n_init = n_init
        if random_state is not None:
            self.random_state = random_state
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if silhouette_metric is not None:
            self.silhouette_metric = silhouette_metric
        if norm is not None:
            self.norm = norm

    @staticmethod
    def from_dict(config_dict):
        """Create ClusteringConfig from a dictionary."""

        return ClusteringConfig(**config_dict)

    def __post_init__(self):
        """Validate clustering configuration to ensure valid settings."""
        valid_init_methods = ["k-means++", "random"]
        if self.init_method not in valid_init_methods:
            raise ValueError(f"init_method must be one of {valid_init_methods}.")
        if not isinstance(self.n_init, int) or self.n_init <= 0:
            raise ValueError("n_init must be a positive integer.")
        if not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer.")
        if not isinstance(self.n_clusters, list):
            raise ValueError("n_clusters must be a list int.")
        if not isinstance(self.n_clusters_map, Dict):
            raise ValueError("n_clusters must be a list of dictionary.")
        valid_silhouette_metrics = ["euclidean", "cosine"]
        if self.silhouette_metric not in valid_silhouette_metrics:
            raise ValueError(
                f"silhouette_metric must be one of {valid_silhouette_metrics}."
            )
        valid_norms = ["l1", "l2"]
        if self.norm not in valid_norms:
            raise ValueError(f"norm must be one of {valid_norms}.")
        logging.info("Clustering Config validated successfully")


@dataclass
class EvaluationConfig:
    scheme: Optional[str] = None
    # mode: Optional[str] = None

    def __post_init__(self):
        self.validate_config()

    @staticmethod
    def from_dict(config_dict):
        return EvaluationConfig(**config_dict)

    def validate_config(self):
        allowed_schemes = ["IOB2", "IOE2", "IOBES", "BILOU"]
        # allowed_modes = [None, "strict"]

        if self.scheme is not None and self.scheme not in allowed_schemes:
            raise ValueError(f"Scheme must be one of {allowed_schemes} or None")
        # if self.mode not in allowed_modes:
        #     raise ValueError(f"Mode must be one of {allowed_modes}")
        logging.info("Evaluation Config validated successfully")


@dataclass
class ExperimentConfig:
    experiment_dir: Path
    corpora_dir: Path
    variant_dir: Path
    extraction_dir: Path
    results_dir: Path
    fine_tuning_dir: Path
    dataset_name: str
    model_name: str
    model_path: str

    @staticmethod
    def from_dict(base_folder, experiment_name, variant) -> "ExperimentConfig":
        """Initialize from a dictionary"""
        experiment_configs_dir = base_folder / experiment_name / variant / "configs"
        if experiment_configs_dir.exists():
            config_fh = FileHandler(experiment_configs_dir)
            config_dict = config_fh.load_yaml("experiment_config.yaml")

            experiment_dir = base_folder / config_dict["experiment_dir"]
            corpora_dir = base_folder / config_dict["corpora_dir"]
            variant_dir = base_folder / experiment_name / config_dict["variant_dir"]
            extraction_dir = experiment_configs_dir / config_dict["extraction_dir"]
            results_dir = experiment_configs_dir / config_dict["results_dir"]
            fine_tuning_dir = experiment_configs_dir / config_dict["fine_tuning_dir"]
        else:
            raise ValueError("Experiment Config doesn't exist please review the path")
        return ExperimentConfig(
            experiment_dir=experiment_dir,
            corpora_dir=corpora_dir,
            variant_dir=variant_dir,
            extraction_dir=extraction_dir,
            results_dir=results_dir,
            fine_tuning_dir=fine_tuning_dir,
            dataset_name=config_dict["dataset_name"],
            model_name=config_dict["model_name"],
            model_path=config_dict["model_path"],
        )




@dataclass
class ExtractionConfigManager:
    def __init__(self, config_path: Path):
        config_fh = FileHandler(config_path.parent)
        self.config = config_fh.load_yaml(config_path.name)

    @property
    def dataset_name(self) -> str:
        return self.config.get("dataset_name", None)

    @property
    def model_path(self) -> str:
        return self.config.get("model_path", None)

    @property
    def training_config(self) -> TrainingConfig:
        return TrainingConfig.from_dict(
            self.config.get("fine_tuning", {}).get("args", {})
        )

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig.from_dict(
            self.config.get("fine_tuning", {}).get("model", {})
        )

    @property
    def evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig.from_dict(
            self.config.get("fine_tuning", {}).get("evaluation", {})
        )

    @property
    def tokenization_config(self) -> TokenizationConfig:
        return TokenizationConfig.from_dict(
            self.config.get("extraction", {}).get("tokenization", {})
        )

    @property
    def umap_config(self) -> UMAPConfig:
        return UMAPConfig.from_dict(self.config.get("extraction", {}).get("umap", {}))

    @property
    def clustering_config(self) -> ClusteringConfig:
        return ClusteringConfig.from_dict(
            self.config.get("extraction", {}).get("clustering", {})
        )


class ResultsConfigManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        config_fh = FileHandler(config_path.parent)
        self.config = config_fh.load_yaml(config_path.name)

    @property
    def results_dir(self) -> Path:
        results_dir = self.config_path.parents[1] / self.config["results_dir"]
        return results_dir

    @property
    def analysis_data(self) -> Dict[str, Any]:
        return self.config.get("analysis_data", {})

    @property
    def train_data(self) -> Dict[str, Any]:
        return self.config.get("train_data", {})

    @property
    def entity_report(self) -> Dict[str, Any]:
        return self.config.get("entity_report", {})
    
    @property
    def token_report(self) -> Dict[str, Any]:
        return self.config.get("token_report", {})
    
    @property
    def kmeans_results(self) -> Dict[str, Any]:
        return self.config.get("kMeans_results", {})
    
    @property
    def entity_non_strict_confusion_data(self) -> Dict[str, Any]:
        return self.config.get("entity_non_strict_confusion_data", {})
    
    @property
    def entity_strict_confusion_data(self) -> Dict[str, Any]:
        return self.config.get("entity_strict_confusion_data", {})
    
    @property
    def attention_weights_similarity_matrix(self) -> Dict[str, Any]:
        return self.config.get("attention_weights_similarity_matrix", {})
    
    @property
    def attention_weights_similarity_heatmap(self) -> Dict[str, Any]:
        return self.config.get("attention_weights_similarity_heatmap", {})
    
    @property
    def attention_similarity_matrix(self) -> Dict[str, Any]:
        return self.config.get("attention_similarity_matrix", {})
    
    @property
    def attention_similarity_heatmap(self) -> Dict[str, Any]:
        return self.config.get("attention_similarity_heatmap", {})
    
    @property
    def centroids_avg_similarity_matrix(self) -> Dict[str, Any]:
        return self.config.get("centroids_avg_similarity_matrix", {})

    


class FineTuningConfigManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        config_fh = FileHandler(config_path.parent)
        self.config = config_fh.load_yaml(config_path.name)

    @property
    def save_dir(self) -> Path:
        save_dir = self.config_path.parents[1] / self.config["save_dir"]
        return save_dir

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.config.get("metrics", {})

    @property
    def state_dict(self) -> Dict[str, Any]:
        return self.config.get("model", {}).get("state_dict", {})

    @property
    def binary(self) -> Dict[str, Any]:
        return self.config.get("model", {}).get("binary", {})
