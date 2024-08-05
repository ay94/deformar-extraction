from typing import Dict
import logging
from experiment_utils.analysis import AnalysisWorkflowManager, TrainingImpact, Entity
from experiment_utils.tokenization import TokenizationWorkflowManager
from experiment_utils.model_outputs import ModelOutputWorkflowManager, PretrainedModelOutputWorkflowManager
from transformers import AutoModel



class OutputGenerationPipeline:
    def __init__(self, model, data_manager, config_manager, pretrained_model_path: str):
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
        self.config_manager = config_manager
        self.pretrained_model = self.load_pretrained_model(pretrained_model_path)

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

    def run(self, split: str) -> Dict[str, object]:
        """
        Run the pipeline to generate model and tokenization outputs.

        Args:
            split (str): The data split to process (e.g., 'train', 'test', 'validation').

        Returns:
            Dict[str, object]: A dictionary containing model and tokenization outputs.
        """
        try:
            
            

            # Generate model outputs
            logging.info(f"Generating model outputs for split: {split}")
            model_outputs_manager = ModelOutputWorkflowManager(
                self.model, self.data_manager, self.config_manager.training_config, split
            )
            
            # Generate pretrained model outputs
            logging.info(f"Generating pretrained model outputs for split: {split}")
            pertrained_model_outputs_manager = PretrainedModelOutputWorkflowManager(
                self.pretrained_model, self.data_manager, self.config_manager.training_config, split
            )
            
            # Generate tokenization outputs
            logging.info(f"Generating tokenization outputs")
            tokenization_outputs_manager = TokenizationWorkflowManager(
                self.data_manager.corpus, self.config_manager.tokenization_config
            )

            return {
                "pretrained_model_outputs": pertrained_model_outputs_manager,
                "model_outputs": model_outputs_manager,
                "tokenization_outputs": tokenization_outputs_manager
            }
        except Exception as e:
            logging.error(f"Error during output generation: {e}")
            raise
# class OutputGenerationPipeline:
#     def __init__(self, model, data_manager, config_manager):
#         """
#         Initialize the OutputGenerationPipeline with the necessary components.
        
#         Args:
#             model: The model to generate outputs from.
#             data_manager: Manager for handling dataset operations.
#             config_manager: Manager for accessing configuration settings.
#         """
#         self.model = model
#         self.data_manager = data_manager
#         self.config_manager = config_manager

#     def run(self, split: str) -> Dict[str, object]:
#         """
#         Run the pipeline to generate model and tokenization outputs.

#         Args:
#             split (str): The data split to process (e.g., 'train', 'test', 'validation').

#         Returns:
#             Dict[str, object]: A dictionary containing model and tokenization outputs.
#         """
#         try:
#             logging.info("Generating model outputs for split: %s", split)
#             model_outputs_manager = ModelOutputWorkflowManager(
#                 self.model, self.data_manager, self.config_manager.training_config, split
#             )
            
#             logging.info(f"Generating tokenization outputs")
#             tokenization_outputs_manager = TokenizationWorkflowManager(
#                 self.data_manager.corpus, self.config_manager.tokenization_config
#             )

#             return {
#                 "model_outputs": model_outputs_manager,
#                 "tokenization_outputs": tokenization_outputs_manager
#             }
#         except Exception as e:
#             logging.error("Error during output generation: %s", e)
#             raise



class AnalysisExtractionPipeline:
    def __init__(self, output_pipeline: Dict[str, object], results, config_manager, split: str):
        """
        Initialize the AnalysisExtractionPipeline with the necessary components.
        
        Args:
            output_pipeline (Dict[str, object]): Outputs from the previous pipeline (model and tokenization outputs).
            metrics: Metrics from the model evaluation.
            config_manager: Manager for accessing configuration settings.
        """
        try:
            self.analysis_manager = AnalysisWorkflowManager(
                config_manager, results, output_pipeline.get('tokenization_outputs'), output_pipeline.get('model_outputs'), 
                output_pipeline.get('pretrained_model_outputs'), output_pipeline.data_manager, split
            )
            
            self.entity_evaluation = Entity(
                results.entity_outputs
            )
            
            self.training_impact = TrainingImpact(
                output_pipeline.data_manager.data[split], output_pipeline['tokenization_outputs'], output_pipeline.pretrained_model, output_pipeline.model.bert
            )

        except Exception as e:
            logging.error("Error initializing AnalysisExtractionPipeline: %s", e)
            raise

    def run(self) -> Dict[str, object]:
        """
        Run the analysis extraction pipeline.
        
        Returns:
            Dict[str, object]: A dictionary containing analysis data, cluster analysis, training impact, and entity evaluation.
        """
        try:
            analysis_data, average_silhouette_score, kmeans_metrics =  self.analysis_manager.run()
            
            
            return {
                  "analysis_data": analysis_data,
                  "average_silhouette_score": average_silhouette_score,
                  "kmeans_metrics": kmeans_metrics,
            }
        except Exception as e:
            logging.error("Error running AnalysisExtractionPipeline: %s", e)
            raise

