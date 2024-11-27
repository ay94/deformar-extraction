import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

import torch
from arabert.preprocess import ArabertPreprocessor
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from experiment_utils.config_managers import TokenizationConfig
from experiment_utils.evaluation import Evaluation, Metrics
from experiment_utils.tokenization import TokenStrategyFactory
from experiment_utils.utils import FileHandler


class TCDataset:
    def __init__(self, texts, tags, label_map, config):
        self.texts = texts
        self.tags = tags
        self.label_map = label_map
        self.max_seq_len = config.max_seq_len
        self.tokenizer, self.preprocessor = self.initialize_tokenizer_and_preprocessor(
            config
        )
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
            # lower_case = tokenizer_path == "bert-base-multilingual-cased"
            lower_case = False
            logging.info(
                "Loading Tokenizer: %s, lower_case: %s", tokenizer_path, lower_case
            )

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, do_lower_case=lower_case
            )

        if not tokenizer:
            raise ValueError("Tokenizer path is not valid or missing.")

        return tokenizer, preprocessor

    def process_tokens(self, words, tags):
        tokens_data = {"input_ids": [], "labels": []}

        for word, label in zip(words, tags):
            tokenized_word = self._preprocess_and_tokenize(word)
            if tokenized_word:
                tokens_data = self._update_tokens_data(
                    tokenized_word, label, tokens_data
                )

        return tokens_data

    def _preprocess_and_tokenize(self, word):
        """Apply preprocessing if available, then tokenize the word."""
        if self.preprocessor:
            word = self.preprocessor.preprocess(word)
        return self.tokenizer.tokenize(word)

    def _update_tokens_data(self, tokens, label, tokens_data):
        """Update tokens data dictionary with new tokens and associated data."""
        if len(tokens) > 0:
            _, _, processed_tokens, processed_labels = self.strategy.handle_tokens(
                tokens, label
            )
            tokens_data["input_ids"].extend(processed_tokens)
            tokens_data["labels"].extend(processed_labels)
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
            if key == "input_ids":
                input_ids = self._convert_to_input_ids(tokens_data[key])
                tokens_data[key] = input_ids
                token_type_ids = [0] * len(input_ids)
                attention_mask = [1] * len(input_ids)
        tokens_data["token_type_ids"] = token_type_ids
        tokens_data["attention_mask"] = attention_mask
        return self.add_padding(tokens_data)

    def _convert_to_input_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def add_padding(self, tokens_data):
        input_ids = tokens_data.get("input_ids")
        padding_length = self.max_seq_len - len(input_ids)
        for key in tokens_data:
            if isinstance(tokens_data[key], list):
                if key == "input_ids":
                    tokens_data[key] += [self.tokenizer.pad_token_id] * padding_length
                elif key == "labels":

                    tokens_data[key] = [
                        (
                            self.label_map[label]
                            if label
                            not in [
                                self.tokenizer.cls_token,
                                self.tokenizer.sep_token,
                                self.strategy.IGNORED_TOKEN_LABEL,
                            ]
                            else self.pad_label_id
                        )
                        for label in tokens_data[key]
                    ]
                    tokens_data[key] += [self.pad_label_id] * padding_length
                else:
                    tokens_data[key] += [0] * padding_length
        self.validate_padding(tokens_data)
        return tokens_data

    def validate_padding(self, tokens_data):
        assert (
            len(tokens_data["input_ids"]) == self.max_seq_len
        ), "Padding validation failed for input_ids"
        assert (
            len(tokens_data["attention_mask"]) == self.max_seq_len
        ), "Padding validation failed for attention_mask"
        assert (
            len(tokens_data["token_type_ids"]) == self.max_seq_len
        ), "Padding validation failed for token_type_ids"
        assert (
            len(tokens_data["labels"]) == self.max_seq_len
        ), "Padding validation failed for labels"

    def convert_to_tensors(self, tokens_data):
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in tokens_data.items()
        }


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
            model_path,
            output_attentions=enable_attentions,
            output_hidden_states=enable_hidden_states,
        )

        self.output_attentions = enable_attentions
        self.output_hidden_states = enable_hidden_states
        self.bert_drop = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.bert.config.hidden_size, self.num_tags)

        if initialize_output_layer:
            self.init_weights()

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.loss_per_item = nn.CrossEntropyLoss(reduction="none")

    def init_weights(self):
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def loss_fn(self, output, target, mask):
        active_loss = mask.view(-1) == 1
        active_logits = output.view(-1, self.num_tags)
        active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(self.cross_entropy_loss.ignore_index).type_as(target),
        )
        average_loss = self.cross_entropy_loss(active_logits, active_labels)
        items_loss = self.loss_per_item(active_logits, active_labels)
        return average_loss, items_loss

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
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
        logging.info(
            "Setting output_hidden_states to %s.", "enabled" if enable else "disabled"
        )
        self.output_hidden_states = enable
        self.bert.config.output_hidden_states = enable

    def enable_attentions(self, enable=True):
        logging.info(
            "Setting output_attentions to %s.", "enabled" if enable else "disabled"
        )
        self.output_attentions = enable
        self.bert.config.output_attentions = enable


class DatasetManager:
    def __init__(
        self,
        corpora_path: Path,
        dataset_name: str,
        config: TokenizationConfig,
        sample: bool,
        corpora_file_name: str = "corpora.json",
    ):
        """
        Initialize the DatasetManager with the path to corpora, dataset name, and configuration.

        Args:
            corpora_path (Path): Path to the directory containing corpora.
            dataset_name (str): Name of the dataset to use.
            config (Dict[str, Any]): Configuration dictionary for the dataset.
            corpora_file_name (str, optional): Name of the corpora JSON file. Defaults to 'corpora.json'.
        """
        corpora_fh = FileHandler(corpora_path)
        self.config = config
        self.corpora = corpora_fh.load_json(corpora_file_name)
        self.corpus = self.get_corpus(dataset_name, sample)
        self.data = self.corpus["splits"]
        self.labels = self.corpus["labels"]
        self.labels_map = self.corpus["labels_map"]
        self.inv_labels_map = {v: k for k, v in self.labels_map.items()}

    def get_corpus(self, data_name: str, sample=False) -> Dict[str, Any]:
        """
        Retrieve the corpus information for the specified dataset name.

        Args:
            data_name (str): Name of the dataset.
            corpora (Dict[str, Any]): Dictionary of available corpora.

        Returns:
            Dict[str, Any]: The corpus information for the specified dataset.

        Raises:
            ValueError: If the dataset name is not found in the corpora.
        """
        if data_name not in self.corpora:
            raise ValueError(f"Data name {data_name} not found in corpora.")
        data = self.corpora[data_name].copy()  # Make a copy of the data to avoid modifying the original

        if sample:
            # Only modify the 'splits' key with a random sample of 100 from each dataset split if available
            splits_data = data.get('splits', {})
            sampled_splits = {}
            for key, items in splits_data.items():
                if len(items) > 100:
                    import random
                    sampled_splits[key] = random.sample(items, 100)
                else:
                    sampled_splits[key] = items
            data['splits'] = sampled_splits  # Replace the original splits data with sampled data
        
        return data
        # return self.corpora[data_name]

    def get_dataset(self, split: str) -> TCDataset:
        """
        Retrieve the dataset for a specific split.

        Args:
            split (str): The split of the dataset ('train', 'test', 'validation').

        Returns:
            TCDataset: The dataset for the specified split.
        """
        return self.create_dataset(split)

    def create_dataset(self, split: str) -> TCDataset:
        """
        Create a TCDataset instance for the specified split.

        Args:
            split (str): The split of the dataset ('train', 'test', 'validation').

        Returns:
            TCDataset: The dataset for the specified split.
        """
        # data = self.data[split][:1000] if self.testing else self.data[split]
        data = self.data[split]
        
        return TCDataset(
            texts=[x["words"] for x in data],
            tags=[x["tags"] for x in data],
            label_map=self.labels_map,
            config=self.config,
        )

    def get_dataloader(
        self, split: str, batch_size: int, num_workers: int, shuffle: bool = False
    ) -> Union[DataLoader, None]:
        """
        Get a DataLoader for the specified split.

        Args:
            split (str): The split of the dataset ('train', 'test', 'validation').
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.

        Returns:
            Union[DataLoader, None]: The DataLoader for the specified split or None if the split doesn't exist.
        """
        try:
            return DataLoader(
                dataset=self.get_dataset(split),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
        except KeyError:
            logging.error("The %s Split Doesn't Exist", split)
        return None


class ModelManager:
    def __init__(self, num_tags, config):
        self.num_tags = num_tags
        self.config = config

    def configure_model(self):
        device = self.get_device()
        model = TCModel(self.num_tags, self.config)
        model.to(device)
        self.original_state = model.state_dict()
        logging.info("Model %s loaded and sent to %s", self.config.model_path, device)
        return model

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model_parameters(self):
        model = self.configure_model()
        return model.named_parameters()


class FineTuneUtils:

    @staticmethod
    def train_fn(data_loader, model, optimizer, device, scheduler, args):
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(
            tqdm(data_loader, total=len(data_loader), desc="Training")
        ):
            data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }

            outputs = model(**data)
            loss = outputs["average_loss"]
            loss.backward()

            total_loss += loss.item()

            if (batch_idx + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        return total_loss / len(data_loader)

    @staticmethod
    def eval_fn(data_loader, model, device, inv_labels_map, evaluation_config):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            preds = None
            labels = None
            for data in tqdm(data_loader, total=len(data_loader), desc="Evaluation"):
                data = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                }

                outputs = model(**data)
                loss = outputs["average_loss"]
                logits = outputs["logits"]
                total_loss += loss.item()

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

            
            y_true = labels.cpu().numpy()
            y_pred = preds.detach().cpu().numpy()
            average_loss = total_loss / len(data_loader)

            evaluator = Evaluation(
                inv_labels_map, y_true, y_pred, evaluation_config
            )
            results = evaluator.generate_results()
            metrics = Metrics.from_dict(results)

        return metrics, average_loss


class Trainer:
    def __init__(
        self,
        data_manager,
        model_manager,
        args,
        evaluation_config,
        use_cross_validation=False,
    ) -> None:
        self.model = None
        self.eval_metrics = None
        self.device = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.validation_dataloader = None
        self.args = args
        self.evaluation_config = evaluation_config
        self.data_manager = data_manager
        self.use_cross_validation = use_cross_validation

        self.setup_trainer(model_manager)

    def setup_trainer(self, model_manager):
        self.model = model_manager.configure_model()
        self.device = model_manager.get_device()

        self.train_dataloader = self.data_manager.get_dataloader(
            "train", self.args.train_batch_size, self.args.num_workers
        )
        self.test_dataloader = self.data_manager.get_dataloader(
            "test", self.args.test_batch_size, self.args.num_workers
        )
        self.validation_dataloader = self.data_manager.get_dataloader(
            "validation", self.args.test_batch_size, self.args.num_workers
        )
        # Initialize optimizer and scheduler
        self.setup_optimizer_scheduler(self.model, self.args)

    def setup_optimizer_scheduler(self, model, args):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        num_train_steps = int(
            (
                len(self.train_dataloader.dataset)
                / (args.train_batch_size * args.accumulation_steps)
            )
            * args.epochs
        )
        logging.info("num_train_steps: %s", num_train_steps)

        self.optimizer = AdamW(optimizer_parameters, lr=args.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(args.warmup_ratio * num_train_steps),
            num_training_steps=num_train_steps,
        )

    def train(self):
        training_loss = FineTuneUtils.train_fn(
            self.train_dataloader,
            self.model,
            self.optimizer,
            self.device,
            self.scheduler,
            self.args,
        )
        return training_loss

    def evaluate(self, dataloader):
        eval_metrics, eval_loss = FineTuneUtils.eval_fn(
            dataloader,
            self.model,
            self.device,
            self.data_manager.inv_labels_map,
            self.evaluation_config,
        )
        return eval_metrics, eval_loss

    def standard_training_loop(self):
        for epoch in range(self.args.epochs):
            logging.info("Start Training Epoch: %s", epoch + 1)
            start_time = time.time()
            training_loss = self.train()

            logging.info("Start Evaluation")
            if self.validation_dataloader:
                logging.info("Validation Split is Available")
                evaluation_dataloader = self.validation_dataloader
            else:
                logging.info("Test Split is Used for Evaluation")
                evaluation_dataloader = self.test_dataloader

            eval_metrics, eval_loss = self.evaluate(evaluation_dataloader)

            logging.info("Training Loss: %s | Eval Loss: %s", training_loss, eval_loss)
            elapsed_time = time.time() - start_time
            logging.info("Epoch completed in %s s", elapsed_time)
            logging.info("\nToken-Level Evaluation Metrics:")
            logging.info(
                "\n" 
                + 
                eval_metrics.token_results.to_markdown(
                    index=False, tablefmt="fancy_grid"
                )
            )
            
            logging.info("\nEntity-Level Non Strict Evaluation Metrics:")
            logging.info(
                "\n" 
                + 
                eval_metrics.entity_non_strict_results.to_markdown(
                    index=False, tablefmt="fancy_grid"
                )
            )

            logging.info("\nEntity-Level Strict Evaluation Metrics:")
            logging.info(
                "\n" 
                + 
                eval_metrics.entity_strict_results.to_markdown(
                    index=False, tablefmt="fancy_grid"
                )
            )

        # Final evaluation after training loop is finished
        logging.info("Final Evaluation on Test Set")
        final_eval_metrics, final_eval_loss = self.evaluate(self.test_dataloader)

        logging.info("Final Test Loss: %s", final_eval_loss)
        logging.info("\nFinal Token-Level Evaluation Metrics:")
        logging.info(
            "\n"
            + 
            final_eval_metrics.token_results.to_markdown(
                index=False, tablefmt="fancy_grid"
            )
        )

        logging.info("\nFinal Entity-Level Non Strict Evaluation Metrics:")
        logging.info(
            "\n" 
            + 
            eval_metrics.entity_non_strict_results.to_markdown(
                index=False, tablefmt="fancy_grid"
            )
        )

        logging.info("\nFinal Entity-Level Strict Evaluation Metrics:")
        logging.info(
            "\n" 
            + 
            eval_metrics.entity_strict_results.to_markdown(
                index=False, tablefmt="fancy_grid"
            )
        )
        
        # Store the final evaluation metrics for the test set
        self.eval_metrics = final_eval_metrics

    def training_loop(self):
        if self.use_cross_validation:
            logging.info("Cross Validation")
            self.cross_validation_loop()
        else:
            logging.info("Standard")
            self.standard_training_loop()

    def cross_validation_loop(self):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.args.splits)
        for fold, (train_index, val_index) in enumerate(
            kf.split(self.data_manager.get_dataset("train"))
        ):
            print(f"\nStarting Fold {fold+1}/{self.args.splits}")
            # Create dataloaders for the current fold
            train_subset = torch.utils.data.Subset(
                self.data_manager.get_dataset("train"), train_index
            )
            val_subset = torch.utils.data.Subset(
                self.data_manager.get_dataset("train"), val_index
            )
            self.train_dataloader = DataLoader(
                train_subset, batch_size=self.args.train_batch_size
            )
            self.validation_dataloader = DataLoader(
                val_subset, batch_size=self.args.test_batch_size
            )
            self.standard_training_loop()  # or a separate function if the training loop varies by fold


class EarlyStopping:
    def __init__(
        self, patience=5, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
