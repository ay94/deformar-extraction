from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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


class EvaluationStrategy(ABC):
    def __init__(self, inv_map):
        self.inv_map = inv_map
        self.ignore_index = nn.CrossEntropyLoss().ignore_index

    def align_predictions(self, preds, truth):
        preds = np.argmax(preds, axis=2)
        batch_size, seq_len = preds.shape

        truth_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if truth[i, j] != self.ignore_index:
                    truth_list[i].append(self.inv_map[truth[i][j]])
                    preds_list[i].append(self.inv_map[preds[i][j]])
        return truth_list, preds_list

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


class TokenEvaluationStrategy(EvaluationStrategy):
    def compute_metrics(self, true_labels, predictions, average_loss):
        truth_list, preds_list = self.align_predictions(predictions, true_labels)
        flat_truth = [item for sublist in truth_list for item in sublist]
        flat_preds = [item for sublist in preds_list for item in sublist]
        report = skl_classification(y_true=flat_truth, y_pred=flat_preds, digits=7)
        report = self.create_classification_report(report)
        cleaned_report = self.clean_report(report)
        return {
            "Precision": skl_precision(
                y_true=flat_truth, y_pred=flat_preds, average="macro"
            ),
            "Recall": skl_recall(y_true=flat_truth, y_pred=flat_preds, average="macro"),
            "F1": skl_f1(y_true=flat_truth, y_pred=flat_preds, average="macro"),
            "Loss": average_loss,
            "classification": cleaned_report,
            "output": {"y_true": flat_truth, "y_pred": flat_preds},
        }

    def clean_report(self, report):
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


class EntityEvaluationStrategy(EvaluationStrategy):
    def compute_metrics(self, true_labels, predictions, average_loss, entity_config):
        truth_list, preds_list = self.align_predictions(predictions, true_labels)

        if entity_config.mode:
            report = seq_classification(
                y_true=truth_list,
                y_pred=preds_list,
                digits=7,
                mode=entity_config.mode,
                scheme=entity_config.scheme,
            )
        else:
            report = seq_classification(y_true=truth_list, y_pred=preds_list, digits=7)
        return {
            "Precision": seq_precision(
                y_true=truth_list, y_pred=preds_list, average="micro"
            ),
            "Recall": seq_recall(y_true=truth_list, y_pred=preds_list, average="micro"),
            "F1": seq_f1(y_true=truth_list, y_pred=preds_list, average="micro"),
            "Loss": average_loss,
            "classification": self.create_classification_report(report),
            "output": {"y_true": truth_list, "y_pred": preds_list},
        }


class Evaluation:
    def __init__(self, inv_map, truths, predictions, average_loss, evaluation_config):
        self.predictions = predictions
        self.truths = truths
        self.loss = average_loss
        self.evaluation_config = evaluation_config
        self.token_strategy = TokenEvaluationStrategy(inv_map)
        self.entity_strategy = EntityEvaluationStrategy(inv_map)

    def generate_results(self):
        metrics = self.evaluate()
        token_results, token_report, token_outputs = self.prepare_results(
            metrics["Token_Level"]
        )
        entity_results, entity_report, entity_outputs = self.prepare_results(
            metrics["Entity_Level"]
        )

        return {
            "token_results": token_results,
            "token_report": token_report,
            "token_outputs": token_outputs,
            "entity_results": entity_results,
            "entity_report": entity_report,
            "entity_outputs": entity_outputs,
        }

    def evaluate(self):
        token_metrics = self.token_strategy.compute_metrics(
            self.truths, self.predictions, self.loss
        )
        entity_metrics = self.entity_strategy.compute_metrics(
            self.truths, self.predictions, self.loss, self.evaluation_config
        )

        # Combine or store results as needed
        return {"Token_Level": token_metrics, "Entity_Level": entity_metrics}

    def prepare_results(self, metrics):
        results = pd.DataFrame.from_dict(self.round_and_slice(metrics))
        report = metrics["classification"]
        output = metrics["output"]
        return results, report, output

    def round_and_slice(self, dictionary):
        # Slicing and rounding results for cleaner presentation
        keys_for_slicing = ["Precision", "Recall", "F1", "Loss"]
        sliced_dict = {key: [round(dictionary[key], 4)] for key in keys_for_slicing}
        return sliced_dict


@dataclass
class Metrics:
    token_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    token_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    token_outputs: dict = field(default_factory=dict)
    entity_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    entity_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    entity_outputs: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict):
        """Create an instance from a dictionary."""
        return Metrics(**data)

    def to_dict(self):
        return {
            "token_results": self.token_results.to_dict(orient="records"),
            "token_report": self.token_report.to_dict(orient="records"),
            "token_outputs": self.token_outputs,
            "entity_results": self.entity_results.to_dict(orient="records"),
            "entity_report": self.entity_report.to_dict(orient="records"),
            "entity_outputs": self.entity_outputs,
        }
