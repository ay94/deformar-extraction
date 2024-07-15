import torch
from torch import nn
import logging
from torch.utils.data import Dataset
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer
from experiment_utils.tokenization import TokenStrategyFactory
from dataclasses import dataclass, field
import pandas as pd
from tqdm.autonotebook import tqdm

import torch
import logging
from torch import nn
from transformers import AutoModel



from abc import ABC, abstractmethod
import numpy as np
from seqeval.metrics import classification_report as seq_classification
from seqeval.metrics import precision_score as seq_precision
from seqeval.metrics import recall_score as seq_recall
from seqeval.metrics import f1_score as seq_f1
from sklearn.metrics import classification_report as skl_classification
from sklearn.metrics import precision_score as skl_precision
from sklearn.metrics import recall_score as skl_recall
from sklearn.metrics import f1_score as skl_f1

from torch import nn

@dataclass
class FineTuneConfig:
    train_batch_size: int = 16
    valid_batch_size: int = 8
    epochs: int = 4
    splits: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1

    @staticmethod
    def from_dict(config_dict):
        """
        Create a FineTuneConfig instance from a dictionary of configuration values.
        This method assumes that the keys in the dictionary match the parameters of the data class exactly.
        """
        return FineTuneConfig(**config_dict)

class TCDataset:
    def __init__(self, texts, tags, label_map, config):
        self.texts = texts
        self.tags = tags
        self.label_map = label_map
        self.max_seq_len = config.get('max_seq_len', 256)
        self.tokenizer, self.preprocessor = self.initialize_tokenizer_and_preprocessor(config)
        # Use cross entropy ignore_index as padding label id so that only real label ids contribute to the loss later.
        self.pad_label_id = nn.CrossEntropyLoss().ignore_index
        self.strategy = TokenStrategyFactory(config).get_strategy()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        words = self.texts[item]
        tags = self.tags[item]
        tokens_data = self.process_tokens(words, tags)
        tokens_data = self.truncate_and_add_special_tokens(tokens_data)
        tokens_data = self.convert_to_tensors(tokens_data)
        return tokens_data

        
    def initialize_tokenizer_and_preprocessor(self, config):
        tokenizer_path = config.get('tokenizer_path')
        preprocessor_path = config.get('preprocessor_path')
        tokenizer = preprocessor = None
        
        if preprocessor_path:
            logging.info("Loading Preprocessor: %s", preprocessor_path)
            preprocessor = ArabertPreprocessor(preprocessor_path)
            
        if tokenizer_path:
            logging.info("Loading Tokenizer: %s", tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=(tokenizer_path != "bert-base-multilingual-cased"))
        
        if not tokenizer:
            raise ValueError("Tokenizer path is not valid or missing.")
            
        return tokenizer, preprocessor   
         
    def process_tokens(self, words, tags):
        tokens_data = {
            'input_ids': [],
            'labels': []
        }
        
        for word, label in zip(words, tags):
            tokenized_word = self._preprocess_and_tokenize(word)
            if tokenized_word:
                tokens_data = self._update_tokens_data(tokenized_word, label, tokens_data)
                
        return tokens_data
    
    def _preprocess_and_tokenize(self, word):
        """Apply preprocessing if available, then tokenize the word."""
        if self.preprocessor:
            word = self.preprocessor.preprocess(word)
        return self.tokenizer.tokenize(word)

    def _update_tokens_data(self, tokens, label, tokens_data):
        """Update tokens data dictionary with new tokens and associated data."""
        if len(tokens) > 0:
            _, _, processed_tokens, processed_labels = self.strategy.handle_tokens(tokens, label)
            tokens_data['input_ids'].extend(processed_tokens)
            tokens_data['labels'].extend(processed_labels)
            return tokens_data
    
    def truncate_and_add_special_tokens(self, tokens_data):
        """Truncate tokens data to max sequence length and optionally add special tokens."""
        max_length = self.max_seq_len - self.tokenizer.num_special_tokens_to_add()
        for key in tokens_data:
            tokens_data[key] = tokens_data[key][:max_length]
        self._add_special_tokens_and_convert_input_ids(tokens_data)
        return tokens_data

    def _add_special_tokens_and_convert_input_ids(self, tokens_data):
        """Add special tokens such as CLS and SEP to the beginning and end of sequences, respectively."""
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        for key in tokens_data:
            if isinstance(tokens_data[key], list):
                tokens_data[key] = [cls_token] + tokens_data[key] + [sep_token]
            if key == 'input_ids':
                input_ids = self._convert_to_input_ids(tokens_data[key])
                tokens_data[key] = input_ids
                token_type_ids = [0] * len(input_ids)
                attention_mask = [1] * len(input_ids)
        tokens_data['token_type_ids'] = token_type_ids
        tokens_data['attention_mask'] = attention_mask
        return self.add_padding(tokens_data)
                
    def _convert_to_input_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
        
    
    def add_padding(self, tokens_data):
        input_ids = tokens_data.get('input_ids')
        padding_length = self.max_seq_len - len(input_ids)
        for key in tokens_data:
            if isinstance(tokens_data[key], list):
                if key == 'input_ids':
                    tokens_data[key] += [self.tokenizer.pad_token_id] * padding_length
                elif key == 'labels':
                    
                    tokens_data[key] = [
                    self.label_map[label] if label not in  [
                        self.tokenizer.cls_token, 
                        self.tokenizer.sep_token,
                        self.strategy.IGNORED_TOKEN_LABEL] else self.pad_label_id
                    
                    for label in tokens_data[key]
                    ]
                    tokens_data[key] += [self.pad_label_id] * padding_length
                else:
                    tokens_data[key] += [0] * padding_length
        self.validate_padding(tokens_data)
        return tokens_data
    
    def validate_padding(self, tokens_data):
        assert len(tokens_data['input_ids']) == self.max_seq_len, "Padding validation failed for input_ids"
        assert len(tokens_data['attention_mask']) == self.max_seq_len, "Padding validation failed for attention_mask"
        assert len(tokens_data['token_type_ids']) == self.max_seq_len, "Padding validation failed for token_type_ids"
        assert len(tokens_data['labels']) == self.max_seq_len, "Padding validation failed for labels"
    
    def convert_to_tensors(self, tokens_data):
        return {key: torch.tensor(value, dtype=torch.long) for key, value in tokens_data.items()}




# TODO: config to setup the different options in the model
class TCModel(nn.Module):
    def __init__(self, num_tag, path, dropout_rate=0.3, output_hidden_states=False, output_attentions=False, initialize=False):
        super(TCModel, self).__init__()
        self.num_tag = num_tag
        logging.info("Loading BERT Model from: %s", path)
        self.bert = AutoModel.from_pretrained(
            path, output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.bert_drop = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.bert.config.hidden_size, num_tag)
        if initialize:
            self.init_weights()
        self.avg_loss = nn.CrossEntropyLoss()
        self.loss_per_item = nn.CrossEntropyLoss(reduction="none")

    def init_weights(self):
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def loss_fn(self, output, target, mask):
        active_loss = mask.view(-1) == 1
        active_logits = output.view(-1, self.num_tag)
        active_labels = torch.where(
            active_loss, target.view(-1), torch.tensor(self.avg_loss.ignore_index).type_as(target)
        )
        average_loss = self.avg_loss(active_logits, active_labels)
        items_loss = self.loss_per_item(active_logits, active_labels)
        return average_loss, items_loss
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_out = self.bert_drop(output["last_hidden_state"])
        logits = self.output_layer(bert_out)
        average_loss, items_loss = self.loss_fn(logits, labels, attention_mask)
        
        outputs = {
            "average_loss": average_loss,
            "losses": items_loss,
            "logits": logits,
            "last_hidden_state": output["last_hidden_state"],
        }
        
        if self.output_hidden_states:
            outputs["hidden_states"] = output["hidden_states"]
        
        if self.output_attentions:
            outputs["attentions"] = output["attentions"]
        
        return outputs
    
    def enable_hidden_states(self, enable=True):
        logging.info("Setting output_hidden_states to %s.", 'enabled' if enable else 'disabled')
        self.output_hidden_states = enable
        self.bert.config.output_hidden_states = enable
    
    def enable_attentions(self, enable=True):
        logging.info("Setting output_attentions to %s.", 'enabled' if enable else 'disabled')
        self.output_attentions = enable
        self.bert.config.output_attentions = enable



class EvaluationStrategy(ABC):
    def __init__(self, inv_label_map):
        self.inv_label_map = inv_label_map
        self.ignore_index = nn.CrossEntropyLoss().ignore_index

    def align_predictions(self, preds, truth):
        preds = np.argmax(preds, axis=2)
        batch_size, seq_len = preds.shape

        truth_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if truth[i, j] != self.ignore_index:
                    truth_list[i].append(self.inv_label_map[truth[i][j]])
                    preds_list[i].append(self.inv_label_map[preds[i][j]])
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
        report =  pd.DataFrame(lines, columns=["Tag", "Precision", "Recall", "F1", "Support"])
       
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
                "Recall": skl_recall(
                    y_true=flat_truth, y_pred=flat_preds, average="macro"
                ),
                "F1": skl_f1(
                    y_true=flat_truth, y_pred=flat_preds, average="macro"
                ),
                "Loss": average_loss,
                "classification": cleaned_report,
                "output": {"y_true": flat_truth, "y_pred": flat_preds},
            }
    def clean_report(self, report):
        mask = report['Tag'] == 'accuracy'
        accuracy_row = report[mask]
        if not accuracy_row.empty:
            # Get the accuracy value
            accuracy_value = accuracy_row['Precision'].values[0]  # Assuming accuracy is stored in the 'Precision' column
            accuracy_support = accuracy_row['Recall'].values[0]  # Assuming accuracy is stored in the 'Precision' column

            # Set the precision, recall, and F1-score to the accuracy value
            report.loc[mask, 'Precision'] = accuracy_value
            report.loc[mask, 'Recall'] = accuracy_value
            report.loc[mask, 'F1'] = accuracy_value
            report.loc[mask, 'Support'] = accuracy_support	

            # Rename the tag from 'accuracy' to 'accuracy/micro' for clarity
            report.loc[report['Tag'] == 'accuracy', 'Tag'] = 'accuracy/micro'
        return report   


    
class EntityEvaluationStrategy(EvaluationStrategy):
    def compute_metrics(self, true_labels, predictions, average_loss):
        truth_list, preds_list = self.align_predictions(predictions, true_labels)
        report = seq_classification(y_true=truth_list, y_pred=preds_list, digits=7)
        return {
                "Precision": seq_precision(
                    y_true=truth_list, y_pred=preds_list, average="micro"
                ),
                "Recall": seq_recall(
                    y_true=truth_list, y_pred=preds_list, average="micro"
                ),
                "F1": seq_f1(y_true=truth_list, y_pred=preds_list, average="micro"),
                "Loss": average_loss,
                "classification": self.create_classification_report(report),
                "output": {"y_true": truth_list, "y_pred": preds_list},
            }
        
    
class Evaluation:
    def __init__(self, inv_map, truths, predictions, average_loss):
        self.predictions = predictions
        self.truths = truths
        self.loss = average_loss
        self.token_strategy = TokenEvaluationStrategy(inv_map)
        self.entity_strategy = EntityEvaluationStrategy(inv_map)

    
    def generate_results(self):
        metrics = self.evaluate()
        token_results, token_report, token_outputs = self.prepare_results(metrics['Token_Level'])
        entity_results, entity_report, entity_outputs = self.prepare_results(metrics['Entity_Level'])
        
        return {
            "token_results": token_results,
            "token_report": token_report,
            "token_outputs": token_outputs,
            "entity_results": entity_results,
            "entity_report": entity_report,
            "entity_outputs": entity_outputs
        }
        
    def evaluate(self):
        token_metrics = self.token_strategy.compute_metrics(self.truths, self.predictions, self.loss)
        entity_metrics = self.entity_strategy.compute_metrics(self.truths, self.predictions, self.loss)
        
        
        # Combine or store results as needed
        return {
            'Token_Level': token_metrics, 
            'Entity_Level': entity_metrics
        }
    
    
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
            "token_results": self.token_results.to_dict(orient='records'),
            "token_report": self.token_report.to_dict(orient='records'),
            "token_outputs": self.token_outputs,
            "entity_results": self.entity_results.to_dict(orient='records'),
            "entity_report": self.entity_report.to_dict(orient='records'),
            "entity_outputs": self.entity_outputs,
        }


class FineTuneUtils:
    def train_fn(self, data_loader, model, optimizer, device, scheduler, config):
        model.train()
        final_loss = 0
        for i, data in enumerate(tqdm(data_loader, total=len(data_loader))):
            for k, v in data.items():
                data[k] = v.to(device)
            outputs = model(**data)
            loss = outputs["average_loss"]
            # calculate the gradient and we can say loss.grad where the gradient is stored
            loss.backward()
            # item returns the scalar only not the whole tensor
            final_loss += loss.item()
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                # Gradient accumulation is a technique used when the effective batch size is larger than the memory capacity of the GPU. Instead of updating the model parameters after every mini-batch, gradients are accumulated over multiple mini-batches and then used to update the parameters.
                # Gradient clipping is a technique used to prevent the exploding gradient problem, where gradients become excessively large during training, leading to unstable updates and convergence issues.
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        return final_loss / len(data_loader)

    def eval_fn(self, data_loader, model, device, inv_labels):
        model.eval()
        with torch.no_grad():
            final_loss = 0
            preds = None
            labels = None
            for data in tqdm(data_loader, total=len(data_loader)):
                for k, v in data.items():
                    data[k] = v.to(device)
                outputs = model(**data)
                loss = outputs["average_loss"]
                logits = outputs["logits"]
                final_loss += loss.item()
                if logits is not None:
                    preds = (
                        logits if preds is None else torch.cat((preds, logits), dim=0)
                    )
                if data["labels"] is not None:
                    labels = (
                        data["labels"]
                        if labels is None
                        else torch.cat((labels, data["labels"]), dim=0)
                    )
            preds = preds.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            # TODO: connecting the dots once the workflow is complete we can figure out how ot handle this
            # evaluator = Evaluation(model_outputs.data['inv_labels'], labels, preds, final_loss)
            # results = evaluator.generate_results()
            # metrics = Metrics.from_dict(results)
            # evaluation = Evaluation(labels, preds, inv_labels, final_loss)
            # metrics = evaluation.compute_metrics()
        # return metrics, final_loss