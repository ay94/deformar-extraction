from typing import Dict
import logging
from experiment_utils.analysis import AnalysisWrokflowManager, TrainingImpact, Entity




class OutputGenerationPipeline:
    def __init__(self, model, data_manager, config_manager):
        """
        Initialize the OutputGenerationPipeline with the necessary components.
        
        Args:
            model: The model to generate outputs from.
            data_manager: Manager for handling dataset operations.
            config_manager: Manager for accessing configuration settings.
        """
        self.model = model
        self.data_manager = data_manager
        self.config_manager = config_manager

    def run(self, split: str) -> Dict[str, object]:
        """
        Run the pipeline to generate model and tokenization outputs.

        Args:
            split (str): The data split to process (e.g., 'train', 'test', 'validation').

        Returns:
            Dict[str, object]: A dictionary containing model and tokenization outputs.
        """
        try:
            logging.info(f"Generating model outputs for split: {split}")
            model_outputs_manager = ModelOutputWorkflowManager(
                self.model, self.data_manager, self.config_manager.training_config, split
            )
            
            logging.info(f"Generating tokenization outputs")
            tokenization_outputs_manager = TokenizationWorkflowManager(
                self.data_manager.corpus, self.config_manager.tokenization_config
            )

            return {
                "model_outputs": model_outputs_manager,
                "tokenization_outputs": tokenization_outputs_manager
            }
        except Exception as e:
            logging.error(f"Error during output generation: {e}")
            raise



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
            self.analysis_manager = AnalysisWrokflowManager(
                config_manager, results, output_pipeline.get('tokenization_outputs'), output_pipeline.get('model_outputs'), output_pipeline.get('data_manager'), split
            )
            
            # self.entity_evaluation = Entity(
            #     metrics.entity_outputs
            # )
            
            # # Training impact workflow
            # self.training_impact = TrainingImpact(
            #     output_pipeline['model_outputs'].data['test'], output_pipeline['tokenization_outputs'], 'aubmindlab/bert-base-arabertv02', output_pipeline['model'].bert
            # )

        except Exception as e:
            logging.error(f"Error initializing AnalysisExtractionPipeline: {e}")
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
            logging.error(f"Error running AnalysisExtractionPipeline: {e}")
            raise

