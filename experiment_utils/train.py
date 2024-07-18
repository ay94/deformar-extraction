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


class TCDataset:
    def __init__(self, texts, tags, label_map, config):
        self.texts = texts
        self.tags = tags
        self.label_map = label_map
        self.max_seq_len = config.max_seq_len
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
        tokenizer_path = config.tokenizer_path
        preprocessor_path = config.preprocessor_path
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





class TCModel(nn.Module):
    def __init__(self, num_tags: int, config: dict):
        super(TCModel, self).__init__()
        self.num_tags = num_tags
        self.configure_model(config)
        
    def configure_model(self, config):
        model_path = config.model_path
        if not model_path:
            raise ValueError("Model path must be specified in the configuration.")

        enable_attentions = config.enable_attentions
        enable_hidden_states = config.enable_hidden_states
        initialize_output_layer = config.initialize_output_layer
        dropout_rate = config.dropout_rate
        
        if not 0 <= dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1.")

        logging.info("Loading BERT Model from: %s", model_path)
        self.bert = AutoModel.from_pretrained(
            model_path, output_attentions=enable_attentions, output_hidden_states=enable_hidden_states
        )

        self.output_attentions = enable_attentions
        self.output_hidden_states = enable_hidden_states
        self.bert_drop = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.bert.config.hidden_size, self.num_tags)

        if initialize_output_layer:
            self.init_weights()

        self.avg_loss = nn.CrossEntropyLoss()
        self.loss_per_item = nn.CrossEntropyLoss(reduction="none")

    def init_weights(self):
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def loss_fn(self, output, target, mask):
        active_loss = mask.view(-1) == 1
        active_logits = output.view(-1, self.num_tags)
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
    @staticmethod
    # def train_fn(data_loader, model, optimizer, device, scheduler, args):
    #     model.train()
    #     final_loss = 0
    #     for batch_idx, data in enumerate(tqdm(data_loader, total=len(data_loader))):
    #         for k, v in data.items():
    #             data[k] = v.to(device)
    #         outputs = model(**data)
    #         loss = outputs["average_loss"]
    #         # calculate the gradient and we can say loss.grad where the gradient is stored
    #         loss.backward()
    #         # item returns the scalar only not the whole tensor
    #         final_loss += loss.item()
    #         if (batch_idx + 1) % args.accumulation_steps == 0:                
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #             optimizer.step()
    #             scheduler.step()
    #             optimizer.zero_grad()
    #         logging.info("Batch %s: Loss = %s", batch_idx+1 / len(data_loader), loss.item())
    #     average_loss = final_loss / len(data_loader)
    #     logging.info("Epoch completed. Average Loss per Batch: %s", average_loss)
    #     return average_loss
    @staticmethod
    def train_fn(data_loader, model, optimizer, device, scheduler, args):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")

        for batch_idx, data in progress_bar:
            try:
                # Move data to the appropriate device
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                # Forward pass
                outputs = model(**data)
                loss = outputs["average_loss"]

                # Backward and optimize
                loss.backward()
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Logging and progress update
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

                # Detailed logging
                if (batch_idx + 1) % args.logging_step == 0:
                    logging.info("Batch %d/%d - Loss: %.4f", batch_idx + 1, len(data_loader), loss.item())

            except Exception as e:
                logging.error("Error during training at batch %d: %s", batch_idx, str(e))
                continue

        average_loss = total_loss / len(data_loader)
        logging.info("Training completed. Average Loss: %.4f", average_loss)
        return average_loss
    
    @staticmethod
    def eval_fn(data_loader, model, device, inv_map, args):
        model.eval()
        total_loss = 0
        preds = None
        labels = None
        progress_bar = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
        
        with torch.no_grad():
            for batch_idx, data in enumerate(progress_bar):
                
                # Move data to the appropriate device
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                
                outputs = model(**data)
                loss = outputs["average_loss"]
                logits = outputs["logits"]
                total_loss += loss.item()

                if logits is not None:
                    preds = logits if preds is None else torch.cat((preds, logits), dim=0)
                if data["labels"] is not None:
                    labels = data["labels"] if labels is None else torch.cat((labels, data["labels"]), dim=0)

                # Set loss for the current batch in the progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

                # Detailed logging for each logging step
                if (batch_idx + 1) % args.logging_step == 0:
                    logging.info("Batch %d/%d - Loss: %.4f", batch_idx + 1, len(data_loader), loss.item())

            preds = preds.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            average_loss = total_loss / len(data_loader)
            logging.info("Evaluation completed. Average Loss per Batch: %.4f", average_loss)

            evaluator = Evaluation(inv_map, labels, preds, average_loss)
            results = evaluator.generate_results()
            metrics = Metrics.from_dict(results)

        return metrics, average_loss

    # def eval_fn(data_loader, model, device, inv_map):
    #     model.eval()
    #     with torch.no_grad():
    #         final_loss = 0
    #         preds = None
    #         labels = None
    #         for data in tqdm(data_loader, total=len(data_loader)):
    #             for k, v in data.items():
    #                 data[k] = v.to(device)
    #             outputs = model(**data)
    #             loss = outputs["average_loss"]
    #             logits = outputs["logits"]
    #             final_loss += loss.item()
    #             if logits is not None:
    #                 preds = (
    #                     logits if preds is None else torch.cat((preds, logits), dim=0)
    #                 )
    #             if data["labels"] is not None:
    #                 labels = (
    #                     data["labels"]
    #                     if labels is None
    #                     else torch.cat((labels, data["labels"]), dim=0)
    #                 )
    #         preds = preds.detach().cpu().numpy()
    #         labels = labels.cpu().numpy()
    #         average_loss = final_loss / len(data_loader)
    #         logging.info("Evaluation completed. Average Loss per Batch: %s", average_loss)

    #         evaluator = Evaluation(inv_map, labels, preds, final_loss)
    #         results = evaluator.generate_results()
    #         metrics = Metrics.from_dict(results)
    #     return metrics, average_loss
        
        
        