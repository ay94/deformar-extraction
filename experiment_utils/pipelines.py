import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from transformers import AutoModel

from experiment_utils.analysis import (AnalysisWorkflowManager, Entity,
                                       TrainingImpact)
from experiment_utils.config_managers import (ExperimentConfig,
                                              ExtractionConfigManager,
                                              FineTuningConfigManager,
                                              ResultsConfigManager)
from experiment_utils.evaluation import Metrics
from experiment_utils.model_outputs import (
    ModelOutputWorkflowManager, PretrainedModelOutputWorkflowManager)
from experiment_utils.tokenization import TokenizationWorkflowManager
from experiment_utils.train import DatasetManager, ModelManager, Trainer
from experiment_utils.utils import FileHandler


class OutputGenerationPipeline:
    def __init__(self, model, data_manager, extraction_manager, pretrained_model_path: str):
        """
        Initialize the OutputGenerationPipeline with the necessary components.

        Args:
            model: The model to generate outputs from.
            data_manager: Manager for handling dataset operations.
            config_manager: Manager for accessing configuration settings.
            pretrained_model_path (str): Path to the pretrained model.
        """
        self.model = model
        self.data_manager = data_manager
        self.extraction_manager = extraction_manager
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_model = None
        self.outputs = None

    def load_pretrained_model(self, pretrained_model_path):
        """
        Load the pretrained model from the given path.

        Returns:
            AutoModel: The loaded pretrained model.
        """
        try:
            logging.info(f"Loading pretrained model from: {pretrained_model_path}")
            pretrained_model = AutoModel.from_pretrained(
                pretrained_model_path, output_attentions=True, output_hidden_states=True
            )
            return pretrained_model
        except Exception as e:
            logging.error(f"Error loading pretrained model: {e}")
            raise

    def initialize(self):
        """
        Initialize the OutputGenerationPipeline.
        """
        if self.pretrained_model is None:
            self.pretrained_model = self.load_pretrained_model(
                self.pretrained_model_path
            )

    def run(self, split: str) -> Dict[str, object]:
        """
        Run the pipeline to generate model and tokenization outputs.

        Args:
            split (str): The data split to process (e.g., 'train', 'test', 'validation').

        Returns:
            Dict[str, object]: A dictionary containing model and tokenization outputs.
        """
        try:

            self.initialize()

            # Generate model outputs
            logging.info(f"Generating model outputs")
            model_outputs_manager = ModelOutputWorkflowManager(
                self.model,
                self.data_manager,
                self.extraction_manager.training_config,
            )

            # Generate pretrained model outputs
            logging.info(f"Generating pretrained model outputs for split: {split}")
            pretrained_model_outputs_manager = PretrainedModelOutputWorkflowManager(
                self.pretrained_model,
                self.data_manager,
                self.extraction_manager.training_config,
                split,
            )

            # Generate tokenization outputs
            logging.info(f"Generating tokenization outputs")
            tokenization_outputs_manager = TokenizationWorkflowManager(
                self.data_manager.corpus, self.extraction_manager.tokenization_config
            )

            self.outputs = {
                "model_outputs": model_outputs_manager,
                "pretrained_model_outputs": pretrained_model_outputs_manager,
                "tokenization_outputs": tokenization_outputs_manager,
            }
        except Exception as e:
            logging.error(f"Error during output generation: {e}")
            raise

    @property
    def model_outputs(self):
        return self.outputs.get("model_outputs")

    @property
    def pretrained_model_outputs(self):
        return self.outputs.get("pretrained_model_outputs")

    @property
    def tokenization_outputs(self):
        return self.outputs.get("tokenization_outputs")


class AnalysisExtractionPipeline:
    def __init__(
        self,
        output_pipeline: OutputGenerationPipeline,
        evaluation_results,
        extraction_manager,
        results_manager,
        split: str,
    ):
        """
        Initialize the AnalysisExtractionPipeline with the necessary components.

        Args:
            output_pipeline (Dict[str, object]): Outputs from the previous pipeline (model and tokenization outputs).
            evaluation_results: Metrics from the model evaluation.
            extraction_manager: Manager for accessing configuration settings.
        """

        self.output_pipeline = output_pipeline
        self.evaluation_results = evaluation_results
        self.extraction_manager = extraction_manager
        self.results_manager = results_manager
        self.split = split
        self.outputs = None
        self.analysis_manager = None
        self.entity_confusion = None
        self.training_impact = None
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            if not self.output_pipeline.outputs:
                logging.warning(
                    "Output pipeline is empty. Please run the Output Generation Pipeline first."
                )
                raise ValueError(
                    "Output pipeline outputs are required but not available."
                )

            try:

                self.setup_analysis_components()
                self.initialized = True
                logging.info("Analysis extraction pipeline initialized successfully.")
            except Exception as e:
                logging.error("Error initializing AnalysisExtractionPipeline: %s", e)
                raise

    def setup_analysis_components(self):
        """Setup all components required for analysis."""
        self.analysis_manager = AnalysisWorkflowManager(
            self.extraction_manager,
            self.evaluation_results,
            self.output_pipeline.tokenization_outputs,
            self.output_pipeline.model_outputs,
            self.output_pipeline.pretrained_model_outputs,
            self.output_pipeline.data_manager,
            self.split,
        )
        self.entity_confusion = Entity(self.evaluation_results.entity_outputs)
        self.training_impact = TrainingImpact(
            self.output_pipeline.data_manager.data[self.split],
            self.output_pipeline.tokenization_outputs,
            self.output_pipeline.pretrained_model,
            self.output_pipeline.model.bert,
        )

    def run(self) -> Dict[str, object]:
        """
        Run the analysis extraction pipeline.

        Returns:
            Dict[str, object]: A dictionary containing analysis data, cluster analysis, training impact, and entity evaluation.
        """
        try:
            self.initialize()
            # analysis_data, average_silhouette_score, kmeans_metrics =  self.analysis_manager.run()
            (
                analysis_data,
                average_silhouette_score,
                kmeans_results,
                centroids_avg_similarity_matrix,
            ) = self.analysis_manager.run()
            attention_similarity_heatmap = (
                self.training_impact.compute_attention_similarities()
            )
            attention_weights_similarity_heatmap = self.training_impact.compare_weights()
            entity_confusion_data = (
                self.entity_confusion.generate_entity_confusion_data()
            )

            self.outputs = {
                "analysis_data": analysis_data,
                "entity_report": self.evaluation_results.entity_report,
                "token_report": self.evaluation_results.token_report,
                "results": AnalysisExtractionPipeline.combine_results(
                    pd.DataFrame(self.evaluation_results.entity_results),
                    pd.DataFrame(self.evaluation_results.token_results),
                    average_silhouette_score,
                ),
                "kmeans_results": AnalysisExtractionPipeline.combine_kmeans_results(
                    kmeans_results
                ),
                "centroids_avg_similarity_matrix": centroids_avg_similarity_matrix,
                "attention_similarity_heatmap": attention_similarity_heatmap,
                "attention_similarity_matrix": self.training_impact.similarity_matrix,
                "attention_weights_similarity_heatmap": attention_weights_similarity_heatmap,
                "attention_weights_similarity_matrix": self.training_impact.weight_diff_matrix,
                "entity_confusion_data": entity_confusion_data,
            }
        except Exception as e:
            logging.error("Error running AnalysisExtractionPipeline: %s", e)
            raise
    def generate_training_data(self):
        train_data = self.analysis_manager.generate_train_df()
        return train_data

    @staticmethod
    def combine_results(entity_results, token_results, average_silhouette_scores):

        entity_results["Type"] = "Entity"
        token_results["Type"] = "Token"
        df_combined = pd.concat([entity_results, token_results]).reset_index(drop=True)
        # Add scores as new columns
        for key, value in average_silhouette_scores.items():
            df_combined[key] = value
        return df_combined

    @staticmethod
    def combine_kmeans_results(data):
        return pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={'index': 'n_clusters'})
    @property
    def analysis_data(self):
        return self.outputs.get("analysis_data")
    @property
    def entity_report(self):
        return self.outputs.get("entity_report")
    @property
    def token_report(self):
        return self.outputs.get("token_report")
    @property
    def results(self):
        return self.outputs.get("results")
    @property
    def kmeans_results(self):
        return self.outputs.get("kmeans_results")
    @property
    def centroids_avg_similarity_matrix(self):
        return self.outputs.get("centroids_avg_similarity_matrix")
    @property
    def attention_similarity_matrix(self):
        return self.outputs.get("attention_similarity_matrix")
    @property
    def attention_similarity_heatmap(self):
        return self.outputs.get("attention_similarity_heatmap")
    @property
    def attention_weights_similarity(self):
        return self.outputs.get("attention_weights_similarity")
    @property
    def attention_weights_similarity_heatmap(self):
        return self.outputs.get("attention_weights_similarity_heatmap")
    @property
    def entity_confusion_data(self):
        return self.outputs.get("entity_confusion_data")

    @property
    def train_df(self):
        return self.outputs.get("train_df", None)


class ExperimentInitializer:
    def __init__(
        self,
        base_folder,
        experiment_config,
        extraction_config,
        results_config,
        fine_tuning_config,
    ):
        self.base_folder = base_folder
        self.experiment_config = experiment_config
        self.extraction_config = extraction_config
        self.results_config = results_config
        self.fine_tuning_config = fine_tuning_config

    def setup_experiment(self):
        # Create the main experiment directory
        experiment_dir = self.base_folder / self.experiment_config["experiment_name"]
        corpora_dir = self.base_folder / self.experiment_config["corpora_path"]
        variant_dir = experiment_dir / self.experiment_config["variant"]
        experiment_dir.mkdir(parents=True, exist_ok=True)

        configs_dir = variant_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        extraction_dir = configs_dir / self.experiment_config["extraction_config"]
        results_dir = configs_dir / self.experiment_config["results_config"]
        fine_tuning_dir = configs_dir / self.experiment_config["fine_tuning_config"]

        experiment_config = {
            "experiment_dir": str(experiment_dir.name),
            "corpora_dir": str(corpora_dir.name),
            "variant_dir": str(variant_dir.name),
            "dataset_name": self.experiment_config["dataset_name"],
            "model_name": self.experiment_config["model_name"],
            "extraction_dir": str(extraction_dir.name),
            "results_dir": str(results_dir.name),
            "fine_tuning_dir": str(fine_tuning_dir.name),
            "model_path": self.experiment_config["model_path"],
        }
        self.write_config(experiment_config, configs_dir, "experiment_config.yaml")
        self.write_config(
            self.extraction_config,
            configs_dir,
            self.experiment_config["extraction_config"],
        )
        self.write_config(
            self.results_config, configs_dir, self.experiment_config["results_config"]
        )
        self.write_config(
            self.fine_tuning_config,
            configs_dir,
            self.experiment_config["fine_tuning_config"],
        )

    def write_config(self, config, path, file_name):
        config_fh = FileHandler(path)
        config_fh.save_yaml(config, file_name)


class ResultsSaver:
    def __init__(self, results_manager):
        self.results_manager = results_manager
        self.results_fh = FileHandler(self.results_manager.results_dir)

    def save(self, data, config):
        file_path = self.results_manager.results_dir / config["folder"] / config["filename"]
        file_path.mkdir(exist_ok=True, parents=True)
        fmt = config["format"]
        if fmt == "np":
            self.results_fh.save_numpy(file_path.with_suffix(".np"), data)
        if fmt == "json":
            if isinstance(data, pd.DataFrame):
                self.results_fh.to_json(file_path.with_suffix(".json"), data)
            elif isinstance(data, (go.Figure)):
                data.write_json(file_path.with_suffix(".json"))
        else:
            raise ValueError(f"Unsupported data type or format: {fmt}")

    def save_all(self, results):
        results_dir = self.results_manager.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Saving Extraction Results in %s", results_dir)
        for key, data in results.items():
            if key in self.results_manager.config:
                self.save(data, self.results_manager.config[key])


class FineTuningSaver:
    def __init__(self, fine_tuning_manager):
        self.fine_tuning_manager = fine_tuning_manager
        self.fine_tuning_fh = FileHandler(self.fine_tuning_manager.save_dir)

    def save(self, data, config):
        file_path = self.fine_tuning_manager.save_dir / config["filename"]
        fmt = config["format"]
        if fmt == "json":
            self.fine_tuning_fh.save_json(file_path.with_suffix(".json"), data)
        elif fmt == "pth":
            self.fine_tuning_fh.save_model_state(data, file_path.with_suffix(".pth"))
        elif fmt == "bin":
            self.fine_tuning_fh.save_model(data, file_path.with_suffix(".bin"))
        else:
            raise ValueError(f"Unsupported data type or format: {fmt}")

    def save_model(self, model):
        model_state_dict = self.fine_tuning_manager.state_dict
        if model_state_dict:
            self.save(model, model_state_dict)

        model_bin = self.fine_tuning_manager.binary
        if model_bin:
            self.save(model, model_bin)

    def save_metrics(self, metrics):
        metrics_config = self.fine_tuning_manager.metrics
        if metrics_config:
            self.save(metrics, metrics_config)

    def save_all(self, model, eval_metrics):
        save_dir = self.fine_tuning_manager.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Saving Fine Tuning Results in %s", save_dir)
        self.save_model(model)
        self.save_metrics(eval_metrics)


class DataExtractionPhase:
    def __init__(
        self, experiment_base_folder: Path, experiment_name: str, variant: str
    ):
        self.experiment_manager = None
        self.extraction_manager = None
        self.results_manager = None
        self.fine_tuning_manager = None
        self.data_manager = None
        self.num_tags = None
        self.trainer = None
        self.model_manager = None
        self.evaluation_results = None
        self.output_generation_pipeline = None
        self.analysis_extraction_pipeline = None
        self.setup_managers(experiment_base_folder, experiment_name, variant)


    def setup_managers(self, experiment_base_folder, experiment_name, variant):
        try:
            self.experiment_manager = ExperimentConfig.from_dict(
                experiment_base_folder, experiment_name, variant
            )
            logging.info("Experiment manager set up successfully.")
        except Exception as e:
            raise Exception("Experiment manager setup failed.") from e
            
        try:
            self.extraction_manager = ExtractionConfigManager(
                self.experiment_manager.extraction_dir
            )
            logging.info("Extraction manager set up successfully.")
        except Exception as e:
            raise Exception("Extraction manager setup failed.") from e
        try:
            self.results_manager = ResultsConfigManager(
                self.experiment_manager.results_dir
            )
            logging.info("Results manager set up successfully.")
        except Exception as e:
            raise Exception("Results manager setup failed.") from e
        try:
            self.fine_tuning_manager = FineTuningConfigManager(
                self.experiment_manager.fine_tuning_dir
            )
            logging.info("Fine Tuning manager set up successfully.")
        except Exception as e:
            raise Exception("Fine Tuning manager setup failed.") from e
        try:
            self.data_manager = DatasetManager(
                self.experiment_manager.corpora_dir,
                self.experiment_manager.dataset_name,
                self.extraction_manager.tokenization_config,
            )
            logging.info("Dataset manager set up successfully.")
        except Exception as e:
            raise Exception("Dataset manager setup failed.") from e

    def setup_trainer(self):
        """Setup necessary configurations for the phase."""
        try:
            self.num_tags = len(self.data_manager.labels)
            self.model_manager = ModelManager(
                self.num_tags, self.extraction_manager.model_config
            )
            self.trainer = Trainer(
                self.data_manager,
                self.model_manager,
                self.extraction_manager.training_config,
                self.extraction_manager.evaluation_config,
            )
            logging.info("Trainer set up successfully.")
        except Exception as e:
            logging.error(f"Error setting up trainer: {e}")
            raise

    def fine_tune(self):
        """Perform the fine-tuning phase."""
        try:
            if not self.trainer:
                self.setup_trainer()
            self.trainer.training_loop()
            logging.info("Fine-tuning completed.")
        except Exception as e:
            logging.error(f"Error during fine-tuning: {e}")
            raise

    def setup_extractors(self):
        """Load fine-tuned model and generate output."""
        try:
            fine_tuning_fh = FileHandler(self.fine_tuning_manager.save_dir)
            results_dict = fine_tuning_fh.load_json("evaluation_metrics.json")
            model = fine_tuning_fh.load_model("model_binary.bin")
            model.enable_attentions()
            self.evaluation_results = Metrics.from_dict(results_dict)
            self.output_generation_pipeline = OutputGenerationPipeline(
                model,
                self.data_manager,
                self.extraction_manager,
                self.experiment_manager.model_path,
            )
            self.analysis_extraction_pipeline = AnalysisExtractionPipeline(
                output_pipeline=self.output_generation_pipeline,
                evaluation_results=self.evaluation_results,
                extraction_manager=self.extraction_manager,
                results_manager=self.results_manager,
                split="test",
            )

            logging.info("Extractors setup.")
        except Exception as e:
            logging.error(f"Error during output generation: {e}")
            raise

    def extract_data(self):
        """Run the analysis extraction pipeline."""
        try:
            if (
                not self.output_generation_pipeline
                and not self.analysis_extraction_pipeline
            ):
                self.setup_extractors()
            self.output_generation_pipeline.run("test")
            self.analysis_extraction_pipeline.run()
            logging.info("Data extraction completed successfully.")
        except Exception as e:
            logging.error(f"Error during data extraction: {e}")
            raise

    def save_fine_tuning(self):
        """Save all fine tuninig results."""
        try:
            saver = FineTuningSaver(self.fine_tuning_manager)
            saver.save_all(self.trainer.model, self.trainer.eval_metrics.to_dict())
            logging.info("Fine-tuning results saved successfully.")
        except Exception as e:
            logging.error(f"Error saving fine-tuning results: {e}")
            raise

    def save_results(self):
        """Save all extracted results."""
        try:
            saver = ResultsSaver(self.results_manager)
            saver.save_all(self.analysis_extraction_pipeline.outputs)
            logging.info("Extracted results saved successfully.")
        except Exception as e:
            logging.error(f"Error saving extracted results: {e}")
            raise

    def run_fine_tuning_phase(self):
        """Run the fine-tuning phase."""
        try:
            self.fine_tune()
            self.save_fine_tuning()
            logging.info("Fine-tuning phase completed successfully.")
        except Exception as e:
            logging.error(f"Error during fine-tuning phase: {e}")
            raise

    def run_extraction_phase(self):
        """Run the extraction phase."""
        try:
            self.setup_extractors()
            self.extract_data()
            self.save_results()
            logging.info("Extraction phase completed successfully.")
        except Exception as e:
            logging.error(f"Error during extraction phase: {e}")
            raise
        
    def generate_train_df(self):
        return self.analysis_extraction_pipeline.generate_training_data()
