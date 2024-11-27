import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from seqeval.metrics import classification_report as seq_classification
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score as seq_precision
from seqeval.metrics import recall_score as seq_recall
from sklearn.metrics import classification_report as skl_classification
from sklearn.metrics import f1_score as skl_f1
from sklearn.metrics import precision_score as skl_precision
from sklearn.metrics import recall_score as skl_recall
from torch import nn
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.scheme import IOB1, IOB2, IOE1, IOE2, IOBES, BILOU, auto_detect, Entities

VALID_SCHEMES = {
    'IOB1':IOB1,
    'IOB2':IOB2,
    'IOE1':IOE1,
    'IOE2':IOE2,
    'IOBES':IOBES,
    'BILOU':BILOU
    }


# class EvaluationStrategy(ABC):
#     def __init__(self, inv_map):
#         self.inv_map = inv_map
#         self.ignore_index = nn.CrossEntropyLoss().ignore_index

    # def align_predictions(self, preds, truth):
    #     preds = np.argmax(preds, axis=2)
    #     batch_size, seq_len = preds.shape

    #     truth_list = [[] for _ in range(batch_size)]
    #     preds_list = [[] for _ in range(batch_size)]

    #     for i in range(batch_size):
    #         for j in range(seq_len):
    #             if truth[i, j] != self.ignore_index:
    #                 truth_list[i].append(self.inv_map[truth[i][j]])
    #                 preds_list[i].append(self.inv_map[preds[i][j]])
    #     return truth_list, preds_list

    # def create_classification_report(self, results):
    #     lines = []
    #     for line in results.strip().split("\n")[1:]:
    #         if line.strip():
    #             tokens = line.split()
    #             # Remove intermediate aggregation if exists (multi-class)
    #             if len(tokens) > 5:
    #                 del tokens[1]
    #             lines.append(tokens)
    #     report = pd.DataFrame(
    #         lines, columns=["Tag", "Precision", "Recall", "F1", "Support"]
    #     )

    #     return report

    # @abstractmethod
    # def compute_metrics(self, true_labels, predictions):
    #     pass

class EvaluationStrategy(ABC):
    def __init__(self, inv_map):
        self.inv_map = inv_map
        self.ignore_index = nn.CrossEntropyLoss().ignore_index

    def align_predictions(self, predictions, truth):
        predictions = np.argmax(predictions, axis=2)
        batch_size, seq_len = predictions.shape

        truth_list = [[] for _ in range(batch_size)]
        pred_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if truth[i, j] != self.ignore_index:
                    truth_list[i].append(self.inv_map[truth[i][j]])
                    pred_list[i].append(self.inv_map[predictions[i][j]])
                    
        if len(truth_list) != len(pred_list):
            raise ValueError("Aligned predictions and truth have mismatched lengths.")
        return truth_list, pred_list

    def create_classification_report(self, results):
        lines = []
        for line in results.strip().split("\n")[1:]:
            if line.strip():
                tokens = line.split()
                # Remove intermediate aggregation if exists (multi-class)
                if len(tokens) > 5:
                    del tokens[1]
                lines.append(tokens)
        report = pd.DataFrame(
            lines, columns=["Tag", "Precision", "Recall", "F1", "Support"]
        )
        return report
    
    @abstractmethod
    def compute_metrics(self, true_labels, predictions):
        pass

# class TokenEvaluationStrategy(EvaluationStrategy):
#     def compute_metrics(self, true_labels, predictions):
#         truth_list, pred_list = true_labels, predictions
#         flat_truth = [item for sublist in truth_list for item in sublist]
#         flat_preds = [item for sublist in pred_list for item in sublist]
#         report = skl_classification(y_true=flat_truth, y_pred=flat_preds, digits=4)
#         report = self.create_classification_report(report)
#         cleaned_report = self.clean_report(report)
#         return {
#             "Precision": skl_precision(
#                 y_true=flat_truth, y_pred=flat_preds, average="macro"
#             ),
#             "Recall": skl_recall(
#                 y_true=flat_truth, y_pred=flat_preds, average="macro"
#                 ),
#             "F1": skl_f1(
#                 y_true=flat_truth, y_pred=flat_preds, average="macro"
#                 ),
#             "classification": cleaned_report,
#             "output": {"y_true": flat_truth, "y_pred": flat_preds},
#         }

#     def clean_report(self, report):
#         report = report.copy()
#         mask = report["Tag"] == "accuracy"
#         accuracy_row = report[mask]
#         if not accuracy_row.empty:
#             # Get the accuracy value
#             accuracy_value = accuracy_row["Precision"].values[
#                 0
#             ]  # Assuming accuracy is stored in the 'Precision' column
#             accuracy_support = accuracy_row["Recall"].values[
#                 0
#             ]  # Assuming accuracy is stored in the 'Precision' column

#             # Set the precision, recall, and F1-score to the accuracy value
#             report.loc[mask, "Precision"] = accuracy_value
#             report.loc[mask, "Recall"] = accuracy_value
#             report.loc[mask, "F1"] = accuracy_value
#             report.loc[mask, "Support"] = accuracy_support

#             # Rename the tag from 'accuracy' to 'accuracy/micro' for clarity
#             report.loc[report["Tag"] == "accuracy", "Tag"] = "accuracy/micro"
#         return report

class TokenEvaluationStrategy(EvaluationStrategy):
    def compute_metrics(self, true_labels, predictions):
        try:
            truth_list, pred_list = self.align_predictions(predictions, true_labels)
        except:
            logging.info('The labels already aligned, proceed with evaluation')
            truth_list, pred_list = true_labels, predictions
        
        flat_truth = [item for sublist in truth_list for item in sublist]
        flat_preds = [item for sublist in pred_list for item in sublist]
        report = skl_classification(y_true=flat_truth, y_pred=flat_preds, digits=4)
        report = self.create_classification_report(report)
        cleaned_report = self.clean_report(report)
        return {
            "Precision": skl_precision(
                y_true=flat_truth, y_pred=flat_preds, average="macro"
            ),
            "Recall": skl_recall(
                y_true=flat_truth, y_pred=flat_preds, average="macro"
                ),
            "F1": skl_f1(
                y_true=flat_truth, y_pred=flat_preds, average="macro"
                ),
            "classification": cleaned_report,
            "output": {"y_true": flat_truth, "y_pred": flat_preds},
        }

    def clean_report(self, report):
        report = report.copy()
        mask = report["Tag"] == "accuracy"
        accuracy_row = report[mask]
        if not accuracy_row.empty:
            # Get the accuracy value
            accuracy_value = accuracy_row["Precision"].values[
                0
            ]  # Assuming accuracy is stored in the 'Precision' column
            accuracy_support = accuracy_row["Recall"].values[
                0
            ]  # Assuming accuracy is stored in the 'Precision' column

            # Set the precision, recall, and F1-score to the accuracy value
            report.loc[mask, "Precision"] = accuracy_value
            report.loc[mask, "Recall"] = accuracy_value
            report.loc[mask, "F1"] = accuracy_value
            report.loc[mask, "Support"] = accuracy_support

            # Rename the tag from 'accuracy' to 'accuracy/micro' for clarity
            report.loc[report["Tag"] == "accuracy", "Tag"] = "accuracy/micro"
        return report
    


# class EntityEvaluationStrategy(EvaluationStrategy):
#     def compute_metrics(self, true_labels, predictions, average_loss, entity_config):
#         truth_list, preds_list = self.align_predictions(predictions, true_labels)
#         '''
#             TODO: There are some changes needed in this class:
#             1 -  allow the function to return both strict and none strict mode
#             2 - create a dictionary that holds all the different schemes supported by seqeval and map it to a string so we don't have to hard code it
#         '''
#         if entity_config.mode:
#             report = seq_classification(
#                 y_true=truth_list,
#                 y_pred=preds_list,
#                 digits=7,
#                 mode=entity_config.mode,
#                 scheme=entity_config.scheme,
#             )
#         else:
#             report = seq_classification(y_true=truth_list, y_pred=preds_list, digits=7)
#         return {
#             "Precision": seq_precision(
#                 y_true=truth_list, y_pred=preds_list, average="micro"
#             ),
#             "Recall": seq_recall(y_true=truth_list, y_pred=preds_list, average="micro"),
#             "F1": seq_f1(y_true=truth_list, y_pred=preds_list, average="micro"),
#             "Loss": average_loss,
#             "classification": self.create_classification_report(report),
#             "output": {"y_true": truth_list, "y_pred": preds_list},
#         }

class EntityEvaluationStrategy(EvaluationStrategy):
    def compute_metrics(self, true_labels, predictions, entity_config):
        scheme = entity_config.scheme  # Default to 'none' if not specified

        # Check if the scheme is valid and not 'none'
        try:
            truth_list, pred_list = self.align_predictions(predictions, true_labels)
        except:
            logging.info('The labels already aligned, proceed with evaluation')
            truth_list, pred_list = true_labels, predictions
            
        strict_outputs = self._evaluate_strict(truth_list, pred_list, scheme)
        non_strict_outputs = self._evaluate_non_strict(truth_list, pred_list)
        
        return {
            "strict": strict_outputs,
            "non_strict": non_strict_outputs,
            "output": {"y_true": truth_list, "y_pred": pred_list}
        }
        
        
    
    def _evaluate_strict(self, truth_list, pred_list, scheme):
        
        if scheme is not None and scheme in VALID_SCHEMES:
            scheme_class = VALID_SCHEMES[scheme]
            report = seq_classification(
                    y_true=truth_list,
                    y_pred=pred_list,
                    digits=4,
                    mode='strict',
                    scheme=scheme_class,
                )
            precision = seq_precision(
                    y_true=truth_list, y_pred=pred_list, average="micro", mode='strict', scheme = scheme_class
                )
            recall = seq_recall(
                    y_true=truth_list, y_pred=pred_list, average="micro", mode='strict', scheme = scheme_class
                )
            f1 = seq_f1(
                    y_true=truth_list, y_pred=pred_list, average="micro", mode='strict', scheme = scheme_class
                )
            
        else:
            logging.info("The scheme is unspecified; seqeval will auto-detect the scheme.")
            report = seq_classification(
                    y_true=truth_list,
                    y_pred=pred_list,
                    digits=4,
                    mode='strict',
                )
            scheme_class = auto_detect(pred_list, False)
            precision = seq_precision(
                    y_true=truth_list, y_pred=pred_list, average="micro", mode='strict', scheme = scheme_class
                )
            recall = seq_recall(
                    y_true=truth_list, y_pred=pred_list, average="micro", mode='strict', scheme = scheme_class
                )
            f1 = seq_f1(
                    y_true=truth_list, y_pred=pred_list, average="micro", mode='strict', scheme = scheme_class
                )
        
        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "classification": self.create_classification_report(report),
                
            }
    
        
    def _evaluate_non_strict(self, truth_list, pred_list,):
        
        report = seq_classification(
                    y_true=truth_list,
                    y_pred=pred_list,
                    digits=4,
                )
        precision = seq_precision(
                y_true=truth_list, y_pred=pred_list, average="micro"
            )
        recall = seq_recall(
                    y_true=truth_list, y_pred=pred_list, average="micro"
                )
        f1 = seq_f1(
                y_true=truth_list, y_pred=pred_list, average="micro"
            )
        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "classification": self.create_classification_report(report),
                
            }

class Evaluation:
    def __init__(self, inv_map, y_true, y_pred, evaluation_config):
        self.truths = y_true
        self.predictions = y_pred
        self.evaluation_config = evaluation_config
        self.token_strategy = TokenEvaluationStrategy(inv_map)
        self.entity_strategy = EntityEvaluationStrategy(inv_map)

    

    def evaluate(self):
        token_metrics = self.token_strategy.compute_metrics(
            self.truths, self.predictions
        )
        entity_metrics = self.entity_strategy.compute_metrics(
            self.truths, self.predictions, self.evaluation_config
        )

        # Combine or store results as needed
        return {"Token_Level": token_metrics, "Entity_Level": entity_metrics}

    def _prepare_results(self, metrics):
        results = pd.DataFrame.from_dict(self._round_and_slice(metrics))
        report = metrics["classification"]
        output = metrics["output"]
        return results, report, output
    
    def _prepare_entity_results(self, metrics):
        strict = metrics["strict"]
        non_strict = metrics["non_strict"]
        entity_strict_results = pd.DataFrame.from_dict(self._round_and_slice(strict))
        entity_non_strict_results = pd.DataFrame.from_dict(self._round_and_slice(non_strict))
        entity_strict_report = strict['classification']
        entity_non_strict_report = non_strict['classification']
        output = metrics["output"]
        return {
            'entity_strict_results': entity_strict_results,
            'entity_non_strict_results': entity_non_strict_results,
            'entity_strict_report': entity_strict_report,
            'entity_non_strict_report': entity_non_strict_report,
            'output': output,
        }

    def _round_and_slice(self, dictionary):
        # Slicing and rounding results for cleaner presentation
        keys_for_slicing = ["Precision", "Recall", "F1"]
        sliced_dict = {key: [round(dictionary[key], 4)] for key in keys_for_slicing}
        return sliced_dict
    
    def generate_results(self):
        metrics = self.evaluate()
        token_results, token_report, token_outputs = self._prepare_results(
            metrics["Token_Level"]
        )
        entity_level_outputs = self._prepare_entity_results(
            metrics["Entity_Level"]
        )
        
        

        return {
            "token_results": token_results,
            "token_report": token_report,
            "token_outputs": token_outputs,
            "entity_strict_results": entity_level_outputs['entity_strict_results'],
            "entity_non_strict_results": entity_level_outputs['entity_non_strict_results'],
            "entity_strict_report": entity_level_outputs['entity_strict_report'],
            "entity_non_strict_report": entity_level_outputs['entity_non_strict_report'],
            "entity_outputs": entity_level_outputs['output'],
        }


@dataclass
class Metrics:
    token_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    token_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    token_outputs: dict = field(default_factory=dict)
    entity_strict_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    entity_non_strict_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    entity_strict_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    entity_non_strict_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    entity_outputs: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict):
        """Create an instance from a dictionary."""
        required_keys = [
        "token_results", "token_report", "token_outputs",
        "entity_strict_results", "entity_non_strict_results",
        "entity_strict_report", "entity_non_strict_report", "entity_outputs"
        ]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys in data: {missing_keys}")
        return Metrics(**data)

    def to_dict(self):
        return {
            "token_results": self.token_results.to_dict(orient="records"),
            "token_report": self.token_report.to_dict(orient="records"),
            "token_outputs": self.token_outputs,
            "entity_strict_results": self.entity_strict_results.to_dict(orient="records"),     
            "entity_non_strict_results": self.entity_non_strict_results.to_dict(orient="records"),
            "entity_strict_report": self.entity_strict_report.to_dict(orient="records"), 
            "entity_non_strict_report": self.entity_non_strict_report.to_dict(orient="records"), 
            "entity_outputs": self.entity_outputs,
        }

# class Evaluation:
#     def __init__(self, inv_map, truths, predictions, average_loss, evaluation_config):
#         self.predictions = predictions
#         self.truths = truths
#         self.loss = average_loss
#         self.evaluation_config = evaluation_config
#         self.token_strategy = TokenEvaluationStrategy(inv_map)
#         self.entity_strategy = EntityEvaluationStrategy(inv_map)

#     def generate_results(self):
#         metrics = self.evaluate()
#         token_results, token_report, token_outputs = self.prepare_results(
#             metrics["Token_Level"]
#         )
#         entity_results, entity_report, entity_outputs = self.prepare_results(
#             metrics["Entity_Level"]
#         )

#         return {
#             "token_results": token_results,
#             "token_report": token_report,
#             "token_outputs": token_outputs,
#             "entity_results": entity_results,
#             "entity_report": entity_report,
#             "entity_outputs": entity_outputs,
#         }

#     def evaluate(self):
#         token_metrics = self.token_strategy.compute_metrics(
#             self.truths, self.predictions, self.loss
#         )
#         entity_metrics = self.entity_strategy.compute_metrics(
#             self.truths, self.predictions, self.loss, self.evaluation_config
#         )

#         # Combine or store results as needed
#         return {"Token_Level": token_metrics, "Entity_Level": entity_metrics}

#     def prepare_results(self, metrics):
#         results = pd.DataFrame.from_dict(self.round_and_slice(metrics))
#         report = metrics["classification"]
#         output = metrics["output"]
#         return results, report, output

#     def round_and_slice(self, dictionary):
#         # Slicing and rounding results for cleaner presentation
#         keys_for_slicing = ["Precision", "Recall", "F1", "Loss"]
#         sliced_dict = {key: [round(dictionary[key], 4)] for key in keys_for_slicing}
#         return sliced_dict


# @dataclass
# class Metrics:
#     token_results: pd.DataFrame = field(default_factory=pd.DataFrame)
#     token_report: pd.DataFrame = field(default_factory=pd.DataFrame)
#     token_outputs: dict = field(default_factory=dict)
#     entity_results: pd.DataFrame = field(default_factory=pd.DataFrame)
#     entity_report: pd.DataFrame = field(default_factory=pd.DataFrame)
#     entity_outputs: dict = field(default_factory=dict)

#     @staticmethod
#     def from_dict(data: dict):
#         """Create an instance from a dictionary."""
#         return Metrics(**data)

#     def to_dict(self):
#         return {
#             "token_results": self.token_results.to_dict(orient="records"),
#             "token_report": self.token_report.to_dict(orient="records"),
#             "token_outputs": self.token_outputs,
#             "entity_results": self.entity_results.to_dict(orient="records"),
#             "entity_report": self.entity_report.to_dict(orient="records"),
#             "entity_outputs": self.entity_outputs,
#         }



class StrictEntityConfusion:
    def __init__(self, evaluation_results):
        """
        Initialize the StrictEntityConfusion class.

        Args:
            y_true (list): The ground truth entities.
            y_pred (list): The predicted entities.
        """
        self.y_true = evaluation_results.entity_outputs['y_true']
        self.y_pred = evaluation_results.entity_outputs['y_pred']
        
    
    def compute(self):
        """
        Compute confusion matrix, false positives, and false negatives for all entities.

        Returns:
            dict: A dictionary containing:
                - 'confusion_matrix': The confusion matrix for all entity types.
                - 'false_negatives': A dictionary with false negative counts categorized by type and subcategory.
                - 'false_positives': A dictionary with false positive counts categorized by type and subcategory.
        """
        # Prepare entities (this initializes and formats entities based on the input scheme)
        self.prepare_entities()

        # Compute the confusion matrix for all entities
        confusion_matrix = self.compute_confusion_matrix()

        # Initialize dictionaries for false negatives and false positives
        false_negatives = defaultdict(Counter)
        false_positives = defaultdict(Counter)

        # Get all unique entity types from the data
        entity_types = set(
            ent[1] for ent in self.true_entities
        ).union(set(ent[1] for ent in self.pred_entities))

        # Iterate over all entity types to calculate false negatives and positives
        for entity_type in entity_types:
            # Compute false negatives for this type
            fn_counts = self.compute_false_negatives(entity_type)

            # Compute false positives for this type
            fp_counts = self.compute_false_positives(entity_type)

            # Merge the results into the global dictionaries
            for t_type, counts in fn_counts.items():
                for subtype, count in counts.items():
                    false_negatives[t_type][subtype] += count

            for t_type, counts in fp_counts.items():
                for subtype, count in counts.items():
                    false_positives[t_type][subtype] += count

        # Return the aggregated results
        return {
            'confusion_matrix': confusion_matrix,
            'false_negatives': dict(false_negatives),  # Convert to standard dict for output clarity
            'false_positives': dict(false_positives),  # Convert to standard dict for output clarity
        }

    
    
    def prepare_entities(self):
        # Initialize true and predicted entities
        self.scheme = auto_detect(self.y_true, False)
        entities_true = self.extract_entities(self.y_true)
        entities_pred = self.extract_entities(self.y_pred)
        self.true_entities = self.flatten_strict_entities(entities_true)
        self.pred_entities = self.flatten_strict_entities(entities_pred)

    def extract_entities(self, y_data):
        # Replace with the Entities() logic if provided
        return Entities(y_data, self.scheme, False)

    @staticmethod
    def flatten_strict_entities(entities):
        """Flatten entities extracted in strict mode into tuples."""
        return [e.to_tuple() for sen in entities.entities for e in sen]

    def compute_confusion_matrix(self):
        """Compute confusion matrix across all entity types."""
        types = set([ent[1] for ent in self.true_entities]).union(
            [ent[1] for ent in self.pred_entities]
        )

        confusion_matrix = {typ: {'TP': 0, 'FP': 0, 'FN': 0} for typ in types}

        for entity_type in types:
            TP, FP, FN = self.extract_strict_entity_confusion(entity_type)
            confusion_matrix[entity_type]['TP'] = TP
            confusion_matrix[entity_type]['FP'] = FP
            confusion_matrix[entity_type]['FN'] = FN

        return confusion_matrix

    def extract_strict_entity_confusion(self, entity):
        """Extract TP, FP, and FN for a given entity type."""
        fns = set([e for e in self.true_entities if e[1] == entity]) - set(
            [e for e in self.pred_entities if e[1] == entity]
        )
        fps = set([e for e in self.pred_entities if e[1] == entity]) - set(
            [e for e in self.true_entities if e[1] == entity]
        )
        tps = set([e for e in self.pred_entities if e[1] == entity]).intersection(
            set([e for e in self.true_entities if e[1] == entity])
        )
        return len(tps), len(fps), len(fns)

    def compute_false_positives(self, entity_type):
        """Analyze false positives for a specific entity type."""
        false_positives = set(
            [e for e in self.pred_entities if e[1] == entity_type]
        ) - set([e for e in self.true_entities if e[1] == entity_type])

        return self.analyze_errors(false_positives, self.true_entities, "FP")

    def compute_false_negatives(self, entity_type):
        """Analyze false negatives for a specific entity type."""
        false_negatives = set(
            [e for e in self.true_entities if e[1] == entity_type]
        ) - set([e for e in self.pred_entities if e[1] == entity_type])

        return self.analyze_errors(false_negatives, self.pred_entities, "FN")

    def analyze_errors(self, target_entities, comparison_entities, error_type):
        """Analyze entity-level errors (FP or FN)."""
        counts = defaultdict(Counter)
        non_o_errors = set()
        indexed_entities = defaultdict(list)

        # Index comparison entities by sentence
        for entity in comparison_entities:
            sen, entity_type, start, end = entity
            indexed_entities[sen].append(entity)
        
        # Track processed pairs to avoid duplicates in counting
        processed_pairs = set()
        # First pass: entity errors
        for target_entity in target_entities:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_type, c_start, c_end = comp_entity[1:]

                # Check for entity type mismatch with exact boundary match
                if (
                t_start == c_start
                and t_end == c_end
                and t_type != c_type
                and target_entity not in non_o_errors
            ):
                    counts[t_type][c_type] += 1
                    non_o_errors.add(target_entity)

        # Second pass: boundary errors
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_type, c_start, c_end = comp_entity[1:]

                # Check for boundary issues with the same entity type
                if (
                t_type == c_type
                and (t_start <= c_start <= t_end or t_start <= c_end <= t_end)
                and target_entity not in non_o_errors
            ):
                    counts[t_type]['Boundary'] += 1
                    non_o_errors.add(target_entity)

        # Third pass: combined entity and boundary errors
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_type, c_start, c_end = comp_entity[1:]

                # Check for combined entity and boundary issues with different types
                if (
                c_type != t_type
                and (t_start <= c_start <= t_end or t_start <= c_end <= t_end)
                and target_entity not in non_o_errors
            ):
                    counts[t_type]['Entity and Boundary'] += 1
                    non_o_errors.add(target_entity)

        # Remaining errors are "O" errors (completely unmatched)
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity
            counts[t_type]['O'] += 1

        return dict(counts)




class EntityConfusion:
    def __init__(self, evaluation_results):
        """
        Initialize the EntityConfusionMatrix class.

        Args:
            y_true (list): The ground truth entities.
            y_pred (list): The predicted entities.
        """
        self.y_true = evaluation_results.entity_outputs['y_true']
        self.y_pred = evaluation_results.entity_outputs['y_pred']

    def prepare_entities(self):
        """
        Prepare entities for confusion matrix calculation.
        - In non-strict mode, entities are extracted using `get_entities`.
        """    
        # Use non-strict extraction
        self.true_entities = get_entities(self.y_true)
        self.pred_entities = get_entities(self.y_pred)

    @staticmethod
    def extract_entity_confusion(entity, true_entities, pred_entities):
        fns = set([e for e in true_entities if e[0] == entity]) - set([e for e in pred_entities if e[0] == entity])
        fps = set([e for e in pred_entities if e[0] == entity]) - set([e for e in true_entities if e[0] == entity])
        tps = set([e for e in pred_entities if e[0] == entity]).intersection(set([e for e in true_entities if e[0] == entity]))
        return len(tps), len(fps), len(fns)


    def compute(self):
        """
        Compute the confusion matrix, false negatives, and false positives.

        Returns:
            dict: A dictionary containing:
                  - 'confusion_matrix': The confusion matrix for entity recognition.
                  - 'false_negatives': Detailed false negatives.
                  - 'false_positives': Detailed false positives.
        """
        self.prepare_entities()
        return {
            'confusion_matrix': self.compute_confusion_matrix(),
            'false_negatives': self.compute_false_negatives_with_boundary(),
            'false_positives': self.compute_false_positives_with_boundary()
        }
    

    def compute_confusion_matrix(self):
        """
        Compute a confusion matrix for Named Entity Recognition (NER) predictions.

        Returns:
            dict: A confusion matrix structured as:
                  {entity_type: {'TP': count, 'FP': count, 'FN': count}}
        """
        # Extract all unique entity types from true and predicted entities
        types = set([ent[0] for ent in self.true_entities]).union([ent[0] for ent in self.pred_entities])

        # Initialize the confusion matrix
        confusion_matrix = {typ: {'TP': 0, 'FP': 0, 'FN': 0} for typ in types}

        
        # Populate the confusion matrix for each entity type
        for entity_type in types:
            TP, FP, FN = self.extract_entity_confusion(entity_type, self.true_entities, self.pred_entities)
            confusion_matrix[entity_type]['TP'] = TP
            confusion_matrix[entity_type]['FP'] = FP
            confusion_matrix[entity_type]['FN'] = FN

        return confusion_matrix
    

    def compute_false_negatives_with_boundary(self):
        """
        Compute false negatives with detailed categorization:
        - 'Boundary': Incorrect boundaries for the same entity type.
        - 'Missed': Predicted as O or no match at all.

        Returns:
            dict: False negatives categorized by entity type.
        """
        fn_counts = defaultdict(Counter)
        true_indexed = {(t[1], t[2]): t[0] for t in self.true_entities}  # Index true entities by boundaries
        pred_indexed = {(p[1], p[2]): p[0] for p in self.pred_entities}  # Index predicted entities by boundaries
        # Iterate through true entities to classify false negatives
        for (t_start, t_end), t_type in true_indexed.items():
            if (t_start, t_end) in pred_indexed:
                if pred_indexed[(t_start, t_end)] != t_type:
                    # Type mismatch at the exact position
                    matched_type = pred_indexed.get((t_start, t_end))
                    fn_counts[t_type][matched_type] += 1
            else:
                # No exact match found, check for other errors
                boundary_error = False
                entity_error = False
                for (p_start, p_end), p_type in pred_indexed.items():
                    if t_type == p_type:
                        if (p_start <= t_start <= p_end) or (p_start <= t_end <= p_end):
                            # Boundary error for the same type
                            fn_counts[t_type]['Boundary'] += 1
                            boundary_error = True
                            break
                    else:
                        if (p_start <= t_start <= p_end) or (p_start <= t_end <= p_end):
                            # Boundary error with a different entity type
                            fn_counts[t_type]['Entity and Boundary'] += 1
                            entity_error = True
                            break
                if not boundary_error and not entity_error:
                    # Missed entity entirely
                    fn_counts[t_type]['O'] += 1

        return dict(fn_counts)


    def compute_false_positives_with_boundary(self):
        """
        Compute false positives with boundary categorization:
        - 'Boundary': Incorrect boundaries for the same entity type.
        - 'Missed': Predicted as O or no match at all.

        Returns:
            dict: False positives categorized by entity type.
        """
        fp_counts = defaultdict(Counter)
        true_indexed = {(t[1], t[2]): t[0] for t in self.true_entities}  # Index true entities by boundaries
        pred_indexed = {(p[1], p[2]): p[0] for p in self.pred_entities}  # Index predicted entities by boundaries
        # Iterate through predicted entities to find false positives
        for (p_start, p_end), p_type in pred_indexed.items():
            entity_error = False
            if (p_start, p_end) not in true_indexed or true_indexed[(p_start, p_end)] != p_type:
                # No matching true entity or type mismatch at the same position
                matched_type = true_indexed.get((p_start, p_end))
                if matched_type:
                    fp_counts[p_type][matched_type] += 1
                    entity_error = True
                boundary_error = False
                entity_boundary_error = False
                for (t_start, t_end), t_type in true_indexed.items():
                    if t_type == p_type and not (p_start == t_start and p_end == t_end):
                        if (p_start <= t_start <= p_end) or (p_start <= t_end <= p_end):
                            # Detected boundary error for the same entity type
                            fp_counts[p_type]['Boundary'] += 1
                            boundary_error = True
                            break
                    elif t_type != p_type and not (p_start == t_start and p_end == t_end):
                        if (p_start <= t_start <= p_end) or (p_start <= t_end <= p_end):
                            fp_counts[p_type]['Entity and Boundary'] += 1
                            entity_boundary_error = True
                            break
                if not boundary_error and not entity_error and not entity_boundary_error:
                    # Missed entity entirely (e.g., predicted as O)
                    fp_counts[p_type]['O'] += 1

        return dict(fp_counts)
    
