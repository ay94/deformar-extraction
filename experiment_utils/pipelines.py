from typing import Dict
import logging
from experiment_utils.analysis import AnalysisWorkflowManager, TrainingImpact, Entity
from experiment_utils.tokenization import TokenizationWorkflowManager
from experiment_utils.model_outputs import ModelOutputWorkflowManager, PretrainedModelOutputWorkflowManager
from transformers import AutoModel



from typing import Dict
from transformers import AutoModel
import logging
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
        self.pretrained_model = self.load_pretrained_model(self.pretrained_model_path) 

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
            logging.info(f"Generating model outputs for split: {split}")
            model_outputs_manager = ModelOutputWorkflowManager(
                self.model, self.data_manager, self.config_manager.training_config, split
            )
            
            # Generate pretrained model outputs
            logging.info(f"Generating pretrained model outputs for split: {split}")
            pretrained_model_outputs_manager = PretrainedModelOutputWorkflowManager(
                self.pretrained_model, self.data_manager, self.config_manager.training_config, split
            )
            
            # Generate tokenization outputs
            logging.info(f"Generating tokenization outputs")
            tokenization_outputs_manager = TokenizationWorkflowManager(
                self.data_manager.corpus, self.config_manager.tokenization_config
            )

            self.outputs =  {
                "model_outputs": model_outputs_manager,
                "pretrained_model_outputs": pretrained_model_outputs_manager,
                "tokenization_outputs": tokenization_outputs_manager
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
# test
class AnalysisExtractionPipeline:
    def __init__(self, output_pipeline: OutputGenerationPipeline, evaluation_results, config_manager, split: str):
        """
        Initialize the AnalysisExtractionPipeline with the necessary components.
        
        Args:
            output_pipeline (Dict[str, object]): Outputs from the previous pipeline (model and tokenization outputs).
            evaluation_results: Metrics from the model evaluation.
            config_manager: Manager for accessing configuration settings.
        """
        
        self.output_pipeline = output_pipeline
        self.evaluation_results = evaluation_results
        self.config_manager = config_manager
        self.split = split
        self.outputs = None
        self.analysis_manager = None
        self.entity_confusion = None
        self.training_impact = None
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            if not self.output_pipeline.outputs:
                logging.warning("Output pipeline is empty. Please run the Output Generation Pipeline first.")
                raise ValueError("Output pipeline outputs are required but not available.")
            
            try:
                self.analysis_manager = AnalysisWorkflowManager(
                    self.config_manager, self.evaluation_results, 
                    self.output_pipeline.tokenization_outputs, self.output_pipeline.model_outputs, 
                    self.output_pipeline.pretrained_model_outputs, self.output_pipeline.data_manager, self.split
                )
                self.entity_confusion = Entity(self.evaluation_results.entity_outputs)
                self.training_impact = TrainingImpact(
                    self.output_pipeline.data_manager.data[self.split], self.output_pipeline.tokenization_outputs, 
                    self.output_pipeline.pretrained_model, self.output_pipeline.model.bert
                )
                self.initialized = True
                logging.info("Analysis extraction pipeline initialized successfully.")
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
            self.initialize()
            analysis_data, average_silhouette_score, kmeans_metrics =  self.analysis_manager.run()
            attention_similarity_matrix = self.training_impact.compute_attention_similarities()
            attention_weights_similarity = self.training_impact.compare_weights()
            entity_confusion_data = self.entity_confusion.generate_entity_confusion_data()
            
            
            self.outputs =  {
                  "analysis_data": analysis_data,
                  "average_silhouette_score": average_silhouette_score,
                  "kmeans_metrics": kmeans_metrics,
                  "attention_similarity_matrix": attention_similarity_matrix,
                  "attention_weights_similarity": attention_weights_similarity,
                  "entity_confusion_data": entity_confusion_data,
            }
        except Exception as e:
            logging.error("Error running AnalysisExtractionPipeline: %s", e)
            raise
        
      
    @property
    def analysis_data(self):
        return self.outputs.get("analysis_data")
    
    @property
    def average_silhouette_score(self):
        return self.outputs.get("average_silhouette_score")
    
    @property
    def kmeans_metrics(self):
        return self.outputs.get("kmeans_metrics")
      
    @property
    def attention_similarity_matrix(self):
        return self.outputs.get("attention_similarity_matrix")
    
    @property
    def attention_weights_similarity(self):
        return self.outputs.get("attention_weights_similarity")
    
    @property
    def entity_confusion_data(self):
        return self.outputs.get("entity_confusion_data")
