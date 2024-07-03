import ast
import copy
import math
import os
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import arabert
from arabert.preprocess import ArabertPreprocessor
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

from experiment_utils.finetune_utils import TCModel


class WordPieceDataset:
    def __init__(self, texts, tags, config, tokenizer, preprocessor=None):
        self.texts = texts
        self.tags = tags
        self.config = config
        self.PREPROCESSOR = preprocessor
        self.TOKENIZER = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text_list = self.texts[item]
        tags = self.tags[item]
        self.first_tokens = []
        self.sentence_ind = []
        self.wordpieces = []
        self.words = []
        self.word_ids = []
        self.labels = []
        self.first_tokens_df = []
        self.sentence_ind_df = []
        self.wordpieces_df = []
        self.words_df = []
        self.word_ids_df = []
        self.labels_df = []
        self.tokens = []
        self.sentence_len = 0
        self.wordpieces_len = 0
        self.removed_words = []
        for word_id, (word, label) in enumerate(zip(text_list, tags)):
            if self.PREPROCESSOR is not None:
                clean_word = self.PREPROCESSOR.preprocess(word)
                word_tokens = self.TOKENIZER.tokenize(clean_word)
            else:
                word_tokens = self.TOKENIZER.tokenize(word)
            if len(word_tokens) > 0:
                self.first_tokens.append(word_tokens[0])
                self.sentence_ind.append(item)
                self.wordpieces.append(word_tokens)
                self.words.append(word)
                self.word_ids.append(word_id)
                self.labels.append(label)
                self.first_tokens_df.extend(
                    [
                        word_tokens[i] if i == 0 else "IGNORED"
                        for i, w in enumerate(word_tokens)
                    ]
                )
                self.sentence_ind_df.extend([item for i in range(len(word_tokens))])
                self.tokens.extend(word_tokens)
                self.wordpieces_df.extend(
                    [word_tokens for i in range(len(word_tokens))]
                )
                self.words_df.extend([word for i in range(len(word_tokens))])
                self.word_ids_df.extend([word_id for i in range(len(word_tokens))])
                self.labels_df.extend(
                    [label if i == 0 else "IGNORED" for i, w in enumerate(word_tokens)]
                )
            else:
                self.removed_words.append((item, word))
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.TOKENIZER.num_special_tokens_to_add()
        if len(self.tokens) > self.config.MAX_SEQ_LEN - special_tokens_count:
            self.tokens = self.tokens[
                : (self.config.MAX_SEQ_LEN - special_tokens_count)
            ]
            self.first_tokens_df = self.first_tokens_df[
                : (self.config.MAX_SEQ_LEN - special_tokens_count)
            ]
            self.sentence_ind_df = self.sentence_ind_df[
                : (self.config.MAX_SEQ_LEN - special_tokens_count)
            ]
            self.wordpieces_df = self.wordpieces_df[
                : (self.config.MAX_SEQ_LEN - special_tokens_count)
            ]
            self.words_df = self.words_df[
                : (self.config.MAX_SEQ_LEN - special_tokens_count)
            ]
            self.word_ids_df = self.word_ids_df[
                : (self.config.MAX_SEQ_LEN - special_tokens_count)
            ]
            self.labels_df = self.labels_df[
                : (self.config.MAX_SEQ_LEN - special_tokens_count)
            ]

        # Add special tokens
        self._add_special_tokens()

        # Length information
        self.sentence_len = len(self.words)
        self.wordpieces_len = len(self.tokens)

    def _add_special_tokens(self):
        """
        Add special tokens [CLS] and [SEP] to the tokenized data.
        """
        self.first_tokens_df = (
            [self.TOKENIZER.cls_token]
            + self.first_tokens_df
            + [self.TOKENIZER.sep_token]
        )
        self.sentence_ind_df = (
            [self.TOKENIZER.cls_token]
            + self.sentence_ind_df
            + [self.TOKENIZER.sep_token]
        )
        self.wordpieces_df = (
            [self.TOKENIZER.cls_token] + self.wordpieces_df + [self.TOKENIZER.sep_token]
        )
        self.words_df = (
            [self.TOKENIZER.cls_token] + self.words_df + [self.TOKENIZER.sep_token]
        )
        self.word_ids_df = (
            [self.TOKENIZER.cls_token] + self.word_ids_df + [self.TOKENIZER.sep_token]
        )
        self.tokens = (
            [self.TOKENIZER.cls_token] + self.tokens + [self.TOKENIZER.sep_token]
        )
        self.labels_df = (
            [self.TOKENIZER.cls_token] + self.labels_df + [self.TOKENIZER.sep_token]
        )


class GenerateSplitOutputs:
    def __init__(self, batches: list, labels: list) -> None:
        """
        Initialize the GenerateSplitOutputs class.

        :param batches: List of data batches.
        :param labels: List of data labels.
        """
        self.data_labels = labels
        self.label_map = {label: i for i, label in enumerate(self.data_labels)}
        self.scores = []  # silhouette score for each sentence
        self.errors = []  # sentences failed to be scored
        self.aligned_losses = []  # loss for each instance
        self.label_score = defaultdict(list)  # score for each label
        self.sentence_samples = defaultdict(
            list
        )  # silhouette score for each word in the sentence

        self.generate_split_outputs(batches)

    # def compute_silhouette(self, batches: list) -> None:
    #     """
    #     Compute silhouette scores for each batch.
    #
    #     :param batches: List of data batches.
    #     """
    #     #  loop through each batch
    #     for batch_num, batch in tqdm(enumerate(batches), total=len(batches)):
    #         # for each batch give me the sentence
    #         sentence_score = []
    #         for labels, sentence_nums, outputs, input_ids in zip(batch['labels'], batch['sentence_num'],
    #                                                              batch['last_hidden_state'], batch['input_ids']):
    #             # input ids identify the tokens included in the sentence it is used to compute sentence length 0 means padding nonzero means token/subtoken
    #             sentence_len = input_ids.nonzero().shape[0]
    #             # get the unique values to extract sentence_number
    #             sentence_num = torch.unique(sentence_nums[sentence_nums != -100]).tolist()[0]
    #             # get labels that belong to the sentence get the indices of labels that are not ignored convert them to list then get the unique labels to idenity the number of unique values in tensor which gives the numebr of labels in the sentence
    #             num_of_labels = len(torch.unique(labels[labels != -100]))
    #             # mask indices that are ignored
    #             label_mask = labels[:sentence_len] != -100
    #             # apply the mask to keep the actual labels only and remove the ignored ones
    #             considered_labels = labels[:sentence_len][label_mask]
    #             try:
    #                 # compute the average silhouette score for all tokens
    #                 sentence_score.append(silhouette_score(outputs[:sentence_len][label_mask].detach().cpu().numpy(),
    #                                                        considered_labels.detach().cpu().numpy()))
    #                 # compute sample silhouette score for each token
    #                 silhouette_sample = silhouette_samples(outputs[:sentence_len][label_mask].detach().cpu().numpy(),
    #                                                        considered_labels.detach().cpu().numpy())
    #                 self.compute_label_score(considered_labels, silhouette_sample)
    #                 self.sentence_samples[sentence_num].extend(silhouette_sample)
    #
    #             except Exception as e:
    #                 sentence_score.append(-100)
    #                 silhouette_sample = np.array([-100] * len(considered_labels))
    #                 self.compute_label_score(considered_labels, silhouette_sample)
    #                 self.errors.append((batch_num, sentence_num, num_of_labels))
    #                 self.sentence_samples[sentence_num] = [-100] * len(considered_labels)
    #         self.scores.extend(sentence_score)

    # def compute_label_score(self, considered_labels, silhouette_sample):
    #     """
    #     Compute the label score.
    #
    #     :param considered_labels: Tensor of considered labels.
    #     :param silhouette_sample: Array of silhouette scores.
    #     """
    #     for lb in self.data_labels:
    #         # identify the indices of the samples that has silhouette score
    #         label_indices = considered_labels.detach().cpu().numpy() == self.label_map[lb]
    #         # for each label assign the samples score that belong to that label
    #         self.label_score[lb].extend(silhouette_sample[label_indices])

    def generate_split_outputs(self, batches):
        """
        Generate outputs for each split.
        :param batches: List of data batches.
        """
        # self.compute_silhouette(batches)  # Uncomment if silhouette computation is needed
        self.align_loss_input_ids(batches)

    def align_loss_input_ids(self, batches):
        """
        Align loss with input IDs.

        :param batches: List of data batches.
        """
        # for each batch take the unique indices and get the losses
        for batch in batches:
            # return tensor of unique values and tensor of indices the tensor of indices contains the location of the unique element in the unique list this location in itself is not necessary but we use it to mask the right loss boundaries
            unique_values, indices = torch.unique(
                batch["input_ids"], return_inverse=True
            )
            # mask the losses with the indices because 0 index here is only refering to the first element of the unique index which is zero
            self.aligned_losses.append(batch["losses"][indices.view(-1) != 0])


class GenerateSplitBatches:
    def __init__(self, results, model, data_loader) -> None:
        self.model = model
        self.data_loader = data_loader
        self.device = self.load_device()
        self.batches = self.detach_batches(
            self.eval_fn(self.data_loader, self.model, self.device)
        )
        self.outputs = None  # Initialize the attribute
        self.compute_outputs(results)

    # @staticmethod
    # def detach_batches(batches):
    #     for i in range(len(batches)):
    #         for k, v in batches[i].items():
    #             batches[i][k] = v.detach().cpu()
    #     return batches
    @staticmethod
    def detach_batches(batches):
        detached_batches = []
        for batch in batches:
            if isinstance(batch, dict):
                # If the batch is a dictionary, detach each tensor in the dictionary.
                detached_batch = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
            elif isinstance(batch, torch.Tensor):
                # If the batch is a single tensor, detach it directly.
                detached_batch = batch.detach().cpu()
            elif isinstance(batch, tuple) and all(torch.is_tensor(x) for x in batch):
                # If the batch is a tuple of tensors (like hidden states), process each tensor.
                detached_batch = tuple(x.detach().cpu() for x in batch)
            else:
                raise TypeError("Unsupported batch type: {}".format(type(batch)))
            detached_batches.append(detached_batch)
        return detached_batches


    @staticmethod
    def load_device():
        use_cuda = torch.cuda.is_available()
        return torch.device("cuda:0" if use_cuda else "cpu")

    @staticmethod
    def eval_fn(data_loader, model, device):
        model.eval()
        with torch.no_grad():
            batches = []
            for data in tqdm(data_loader, total=len(data_loader)):
                for k, v in data.items():
                    data[k] = v.to(device)

                outputs = model(**data)
                batches.append(
                    {
                        "labels": data["labels"],
                        "words_ids": data["words_ids"],
                        "sentence_num": data["sentence_num"],
                        "attention_mask": data["attention_mask"],
                        "input_ids": data["input_ids"],
                        "losses": outputs["losses"],
                        "logits": outputs["logits"],
                        "last_hidden_state": outputs["last_hidden_state"],
                        # "hidden_states": outputs["hidden_states"],
                    }
                )
        return batches

    def compute_outputs(self, results):
        print("Compute Outputs")
        self.outputs = GenerateSplitOutputs(self.batches, results.data["labels"])


class GenerateSplitTokenizationOutputs:
    def __init__(self, wordpiece_data) -> None:
        self.first_tokens_df = []
        self.sentence_ind_df = []
        self.wordpieces_df = []
        self.words_df = []
        self.word_ids_df = []
        self.labels_df = []
        self.sentence_len_df = []
        self.wordpieces_len_df = []

        self.first_tokens = []
        self.sentence_ind = []
        self.tokens = []
        self.wordpieces = []
        self.words = []
        self.word_ids = []
        self.labels = []
        self.sentence_len = []
        self.wordpieces_len = []
        self.get_wordpiece_data(wordpiece_data)

    def get_wordpiece_data(self, wordpiece_data):
        for i in tqdm(range(wordpiece_data.__len__())):
            wordpiece_data.__getitem__(i)
            self.first_tokens_df.append(wordpiece_data.first_tokens_df)
            self.sentence_ind_df.append(wordpiece_data.sentence_ind_df)
            self.tokens.append(wordpiece_data.tokens)
            self.wordpieces_df.append(wordpiece_data.wordpieces_df)
            self.words_df.append(wordpiece_data.words_df)
            self.word_ids_df.append(wordpiece_data.word_ids_df)
            self.labels_df.append(wordpiece_data.labels_df)
            self.sentence_len_df.append(wordpiece_data.sentence_len)
            self.wordpieces_len_df.append(wordpiece_data.wordpieces_len)

            self.first_tokens.append(wordpiece_data.first_tokens)
            self.sentence_ind.append(wordpiece_data.sentence_ind)
            self.wordpieces.append(wordpiece_data.wordpieces)
            self.words.append(wordpiece_data.words)
            self.word_ids.append(wordpiece_data.word_ids)
            self.labels.append(wordpiece_data.labels)


class BatchOutputs:
    def __init__(self, outputs, model) -> None:
        """
        Initialize the BatchOutputs class and generate batches for training, validation, and test data.

        Args:
            outputs: The outputs object containing data loaders and other relevant information.
            model: The model to be used for generating batches.
        """
        self.train_batches = self.generate_batches(outputs, model, "train")
        self.val_batches = self.generate_batches(outputs, model, "val")
        self.test_batches = self.generate_batches(outputs, model, "test")

    @staticmethod
    def generate_batches(outputs, model, mode):
        """
        Generate batches for a given mode (train, val, test).

        Args:
            outputs: The outputs object containing data loaders and other relevant information.
            model: The model to be used for generating batches.
            mode: The mode for which to generate batches (train, val, test).

        Returns:
            The generated batches for the specified mode.
        """
        data_loader = getattr(outputs, f"{mode}_dataloader")
        print(f"Generate {mode.capitalize()} Batches")
        return GenerateSplitBatches(outputs, model, data_loader)


class ModelOutputs:
    def __init__(self, batches) -> None:
        self.train_outputs = batches.train_batches.outputs
        self.val_outputs = batches.val_batches.outputs
        self.test_outputs = batches.test_batches.outputs


class TokenizationOutputs:
    def __init__(self, outputs, tokenizer_path, preprocessor_path=None) -> None:
        """
        Initialize the TokenizationOutputs class and generate word pieces and subword locations.

        Args:
            outputs: The outputs object containing data loaders and other relevant information.
            tokenizer_path: Path to the tokenizer.
            preprocessor_path: Path to the preprocessor (if any).
        """

        self.tokenizer_path = tokenizer_path
        self.preprocessor_path = preprocessor_path
        TOKENIZER, PREPROCESSOR = self.load_tokenizer()
        # subword sentence locations and wh at tag they had in each sentence
        self.generate_wordpieces(outputs, TOKENIZER, PREPROCESSOR)
        self.train_subwords = None
        self.val_subwords = None
        self.test_subwords = None
        self.train_tokenization_output = None
        self.val_tokenization_output = None
        self.test_tokenization_output = None

    def load_tokenizer(self):
        """
        Load the tokenizer and preprocessor.

        Returns:
            TOKENIZER: The loaded tokenizer.
            PREPROCESSOR: The loaded preprocessor (or None if not provided).
        """
        if self.preprocessor_path is not None:
            print(f"Loading Preprocessor {self.preprocessor_path}")
            PREPROCESSOR = ArabertPreprocessor(self.preprocessor_path)
        else:
            PREPROCESSOR = None
        print(f"Loading Tokenizer {self.tokenizer_path}")
        TOKENIZER = AutoTokenizer.from_pretrained(
            self.tokenizer_path, do_lower_case=False
        )
        return TOKENIZER, PREPROCESSOR

    @staticmethod
    def load_wordpieces(outputs, mode, tokenizer, preprocessor):
        """
        Load the word pieces for a given mode.

        Args:
            outputs: The outputs object containing data loaders and other relevant information.
            mode: The mode for which to load word pieces (train, val, test).
            tokenizer: The tokenizer to be used.
            preprocessor: The preprocessor to be used (if any).

        Returns:
            wordpieces: The loaded word pieces dataset.
        """
        wordpieces = WordPieceDataset(
            texts=[x[1] for x in outputs.data[mode]],
            tags=[x[2] for x in outputs.data[mode]],
            config=outputs.config,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
        )
        return wordpieces

    @staticmethod
    def get_subwords(wordpieces):
        """
        Get subword locations and their tags.

        Args:
            wordpieces: The word pieces dataset.

        Returns:
            subwords: A dictionary of subwords and their corresponding tags and sentence indices.
        """
        subwords = defaultdict(list)
        for i in tqdm(range(wordpieces.__len__())):
            wordpieces.__getitem__(i)
            for w, t in zip(wordpieces.first_tokens, wordpieces.labels):
                subwords[w].append({"tag": t, "sentence": i})
        return subwords

    def generate_wordpieces(self, outputs, tokenizer, preprocessor):
        """
        Generate word pieces for training, validation, and test data.

        Args:
            outputs: The outputs object containing data loaders and other relevant information.
            tokenizer: The tokenizer to be used.
            preprocessor: The preprocessor to be used (if any).
        """
        train_wordpieces = self.load_wordpieces(
            outputs, "train", tokenizer, preprocessor
        )
        val_wordpieces = self.load_wordpieces(outputs, "val", tokenizer, preprocessor)
        test_wordpieces = self.load_wordpieces(outputs, "test", tokenizer, preprocessor)

        self.generate_tokenization_output(
            train_wordpieces, val_wordpieces, test_wordpieces
        )
        self.get_subword_locations(train_wordpieces, val_wordpieces, test_wordpieces)

    def get_subword_locations(self, train_wordpieces, val_wordpieces, test_wordpieces):
        """
        Get subword locations for training, validation, and test data.

        Args:
            train_wordpieces: The word pieces dataset for training data.
            val_wordpieces: The word pieces dataset for validation data.
            test_wordpieces: The word pieces dataset for test data.
        """
        print("Generate Training Subwords Locations")
        self.train_subwords = self.get_subwords(train_wordpieces)
        print("Generate Validation Subwords Locations")
        self.val_subwords = self.get_subwords(val_wordpieces)
        print("Generate Test Subwords Locations")
        self.test_subwords = self.get_subwords(test_wordpieces)

    def generate_tokenization_output(
        self, train_wordpieces, val_wordpieces, test_wordpieces
    ):
        """
        Generate tokenization outputs for training, validation, and test data.

        Args:
            train_wordpieces: The word pieces dataset for training data.
            val_wordpieces: The word pieces dataset for validation data.
            test_wordpieces: The word pieces dataset for test data.
        """
        print("Generate Training Tokenization Outputs")
        self.train_tokenization_output = GenerateSplitTokenizationOutputs(
            train_wordpieces
        )
        print("Generate Validation Tokenization Outputs")
        self.val_tokenization_output = GenerateSplitTokenizationOutputs(val_wordpieces)
        print("Generate Test Tokenization Outputs")
        self.test_tokenization_output = GenerateSplitTokenizationOutputs(
            test_wordpieces
        )


class ModelResults:
    def __init__(self, outputs) -> None:
        """
        Initialize the ModelResults class and generate results for training, validation, and test data.

        Args:
            outputs: The outputs object containing metrics for training, validation, and test.
        """
        self.train_metrics = outputs.train_metrics
        self.val_metrics = outputs.val_metrics
        self.test_metrics = outputs.test_metrics


class SaveModelOutputs:
    def __init__(self, fh, data_name, model_name, outputs, tokenization, results):
        """
        Initialize the SaveModelOutputs class and save the outputs, tokenization, and results.

        Args:
           fh: File handler object for saving files.
           data_name: Name of the dataset.
           model_name: Name of the model.
           outputs: Outputs object containing model outputs.
           tokenization: Tokenization object containing tokenization outputs.
           results: Results object containing evaluation metrics.
        """
        self.outputs = outputs
        self.tokenization = tokenization
        self.results = results

        self.save_outputs(fh, data_name, model_name)

    def save_outputs(self, fh, data_name, model_name):
        """
        Save the outputs, tokenization, and results to files.

        Args:
            fh: File handler object for saving files.
            data_name: Name of the dataset.
            model_name: Name of the model.
        """
        fh.save_object(
            self.outputs, f"modelOutputs/{data_name}_{model_name}_model_outputs.pkl"
        )
        fh.save_object(
            self.tokenization,
            f"modelOutputs/{data_name}_{model_name}_tokenization_outputs.pkl",
        )
        fh.save_object(
            self.results, f"modelOutputs/{data_name}_{model_name}_model_results.pkl"
        )


# Data Utility
class UtilityFunctions:
    @staticmethod
    def entropy(probabilities):
        return -sum(p * math.log2(p) for p in probabilities.values())

    @staticmethod
    def label_probabilities(dataset):
        label_counts = defaultdict(Counter)
        for token, label in dataset:
            label_counts[token][label] += 1
        probabilities = {
            token: {
                label: count / sum(counts.values()) for label, count in counts.items()
            }
            for token, counts in label_counts.items()
        }
        return probabilities


# Data Preparation
class LabelAligner:
    def __init__(self, batches, tokenization_outputs):
        """
        Initialize the DataPreparation class.

        Args:
            batches: The batches object containing batch data.
            tokenization_outputs: The tokenization outputs object.
        """
        self.batches = batches
        self.tokenization = tokenization_outputs

    def label_alignment(self):
        """
        Create a map for label alignment based on tokenization outputs.

        Returns:
            alignment_map: A dictionary mapping sentence IDs to token indices and tokens.
        """
        alignment_map = defaultdict(list)
        for sen_id, sen in enumerate(self.tokenization.labels_df):
            for tok_id, tok in enumerate(sen):
                if tok in ["[CLS]", "[SEP]", "IGNORED"]:
                    alignment_map[sen_id].append((tok_id, tok))
        return alignment_map

    @staticmethod
    def change_preds(preds, pred_map):
        """
        Modify predictions based on the alignment map.

        Args:
            preds: A list of predictions.
            pred_map: A dictionary mapping sentence IDs to token indices and tokens.

        Returns:
            modified_preds: A list of modified predictions.
        """
        modified_preds = []
        for sen_id, sen in enumerate(preds):
            sentence = sen.copy()
            for idx, tok in pred_map[sen_id]:
                sentence.insert(idx, tok)
            modified_preds.append(sentence)
        return modified_preds


# Analysis Computations
class AmbiguityComputer:
    @staticmethod
    def compute_consistency(analysis_df, subwords_locations):
        # Creating a DataFrame from subwords_locations for easier manipulation
        subwords_df = pd.DataFrame(
            [
                {"first_token": token, "tag": info["tag"], "count": 1}
                for token, tags in subwords_locations.items()
                for info in tags
            ]
        )

        # Summing up counts by first_token and tag
        tag_counts = subwords_df.groupby(["first_token", "tag"]).sum().reset_index()

        # Merging with analysis_df
        merged_df = analysis_df.merge(
            tag_counts, how="left", left_on="first_tokens", right_on="first_token"
        )

        # Determine consistency
        merged_df["is_consistent"] = merged_df["tag"] == merged_df["truth"]
        consistency_counts = merged_df.pivot_table(
            index="index", columns="is_consistent", values="count", fill_value=0
        )

        # Adding to original DataFrame
        analysis_df["first_tokens_consistency"] = consistency_counts[True]
        analysis_df["first_tokens_inconsistency"] = consistency_counts[False]

        return analysis_df

    @staticmethod
    def token_ambiguity(analysis_df, subwords_counter):
        # Convert counter to DataFrame
        subwords_df = pd.DataFrame(subwords_counter, columns=["token", "tag"])
        probabilities = (
            subwords_df.groupby("token")["tag"]
            .value_counts(normalize=True)
            .rename("probability")
            .reset_index()
        )

        # Calculate entropy
        probabilities["entropy_contribution"] = -probabilities["probability"] * np.log2(
            probabilities["probability"]
        )
        entropy_df = (
            probabilities.groupby("token")["entropy_contribution"].sum().reset_index()
        )

        # Merge back to analysis_df
        analysis_df = analysis_df.merge(
            entropy_df, how="left", left_on="first_tokens", right_on="token"
        )
        analysis_df["token_entropy"] = analysis_df["entropy_contribution"].fillna(-1)
        analysis_df.drop(["token", "entropy_contribution"], axis=1, inplace=True)

        return analysis_df["token_entropy"].values

    @staticmethod
    def word_ambiguity(analysis_df, wordsDict):
        # Flatten the dictionary into a DataFrame
        words_df = pd.DataFrame(
            [
                {"word": word, "tag": tag["tag"]}
                for word, tags in wordsDict.items()
                for tag in tags
            ]
        )
        probabilities = (
            words_df.groupby("word")["tag"]
            .value_counts(normalize=True)
            .rename("probability")
            .reset_index()
        )

        # Calculate entropy
        probabilities["entropy_contribution"] = -probabilities["probability"] * np.log2(
            probabilities["probability"]
        )
        entropy_df = (
            probabilities.groupby("word")["entropy_contribution"].sum().reset_index()
        )

        # Merge back to analysis_df
        analysis_df = analysis_df.merge(
            entropy_df, how="left", left_on="words", right_on="word"
        )
        analysis_df["word_entropy"] = analysis_df["entropy_contribution"].fillna(-1)
        analysis_df.drop(["word", "entropy_contribution"], axis=1, inplace=True)

        return analysis_df["word_entropy"].values

    # def compute_consistency(analysis_df, subwords_locations):
    #     """
    #     Compute consistency and inconsistency scores for tokens in the analysis dataframe.
    #
    #     Args:
    #         analysis_df: The dataframe containing analysis data.
    #         subwords_locations: A dictionary of subwords and their corresponding tags and sentence indices.
    #
    #     Returns:
    #         analysis_df: The updated dataframe with consistency and inconsistency scores.
    #     """
    #     consistent = []
    #     inconsistent = []
    #     for i in tqdm(range(len(analysis_df))):
    #         consistent_count = []
    #         inconsistent_count = []
    #         for t, count in Counter(
    #                 [tok['tag'] for tok in subwords_locations[analysis_df.iloc[i]['first_tokens']]]
    #         ).items():
    #             if t == analysis_df.iloc[i]['truth']:
    #                 consistent_count.append(count)
    #             else:
    #                 inconsistent_count.append(count)
    #         consistent.append(sum(consistent_count))
    #         inconsistent.append(sum(inconsistent_count))
    #     analysis_df['first_tokens_consistency'] = consistent
    #     analysis_df['first_tokens_inconsistency'] = inconsistent
    #     return analysis_df

    # def token_ambiguity(self, analysis_df):
    #     """
    #     Calculate token ambiguity for the analysis dataframe.
    #
    #     Args:
    #         analysis_df: The dataframe containing analysis data.
    #
    #     Returns:
    #         token_entropy: A list of token entropy values.
    #     """
    #     subwords_counter = [(subword, tag['tag']) for subword, tag_dis in self.subwords.items() for tag in tag_dis]
    #     probabilities = UtilityFunctions.label_probabilities(subwords_counter)
    #     # token_entropies = {token: abs(UtilityFunctions.entropy(probs)) for token, probs in probabilities.items()}
    #     token_entropies = {token: UtilityFunctions.entropy(probs) for token, probs in probabilities.items()}
    #     computed_token_entropy = pd.DataFrame(token_entropies.items(), columns=['first_tokens', 'entropy'])
    #
    #     token_entropy = []
    #     for tk in tqdm(analysis_df['first_tokens']):
    #         token_data = computed_token_entropy[computed_token_entropy['first_tokens'] == tk]
    #         if not token_data.empty:
    #             token_entropy.append(token_data['entropy'].values[0])
    #         else:
    #             token_entropy.append(-1)
    #     return token_entropy

    # def word_ambiguity(self, analysis_df):
    #     """
    #     Calculate word ambiguity for the analysis dataframe.
    #
    #     Args:
    #         analysis_df: The dataframe containing analysis data.
    #
    #     Returns:
    #         word_entropy: A list of word entropy values.
    #     """
    #     wordsDict = defaultdict(list)
    #     for i, sen in enumerate(self.dataset_outputs.data['train']):
    #         for w, t in zip(sen[1], sen[2]):
    #             wordsDict[w].append({'tag': t, 'sentence': i})
    #
    #     words_counter = [(word, tag['tag']) for word, tag_dis in wordsDict.items() for tag in tag_dis]
    #     probabilities = UtilityFunctions.label_probabilities(words_counter)
    #     # word_entropies = {token: abs(UtilityFunctions.entropy(probs)) for token, probs in probabilities.items()}
    #     word_entropies = {token: UtilityFunctions.entropy(probs) for token, probs in probabilities.items()}
    #     computed_word_entropy = pd.DataFrame(word_entropies.items(), columns=['words', 'entropy'])
    #
    #     word_entropy = []
    #     for tk in tqdm(analysis_df['words']):
    #         token_data = computed_word_entropy[computed_word_entropy['words'] == tk]
    #         if not token_data.empty:
    #             word_entropy.append(token_data['entropy'].values[0])
    #         else:
    #             word_entropy.append(-1)
    #     return word_entropy


class DataExtractor:
    def __init__(self, preparation, outputs, results):
        """
        Initialize the FlatDataExtractor class.

        Args:
            preparation: The DataPreparation object.
            outputs: The outputs object containing batch outputs.
            results: The results object containing model results.
        """
        self.preparation = preparation
        self.outputs = outputs
        self.results = results

    def extract_flat_data(self):
        """
        Extract and flatten the data from the preparation, outputs, and results.

        Returns:
            A tuple of flattened data elements.
        """
        flat_last_hidden_state = torch.cat(
            [
                hidden_state[ids != 0]
                for batch in self.preparation.batches
                for ids, hidden_state in zip(
                    batch["input_ids"], batch["last_hidden_state"]
                )
            ]
        )
        flat_labels = torch.cat(
            [
                labels[ids != 0]
                for batch in self.preparation.batches
                for ids, labels in zip(batch["input_ids"], batch["labels"])
            ]
        )
        flat_losses = torch.cat([losses for losses in self.outputs.aligned_losses])
        flat_words = [
            tok for sen in self.preparation.tokenization.words_df for tok in sen
        ]
        flat_tokens = [
            tok for sen in self.preparation.tokenization.tokens for tok in sen
        ]
        flat_wordpieces = [
            str(tok)
            for sen in self.preparation.tokenization.wordpieces_df
            for tok in sen
        ]
        flat_first_tokens = [
            tok for sen in self.preparation.tokenization.first_tokens_df for tok in sen
        ]
        token_id_strings = [
            f"{tok}@#{tok_id}@#{i}"
            for sen, sen_id in zip(
                self.preparation.tokenization.first_tokens_df,
                self.preparation.tokenization.sentence_ind_df,
            )
            for i, (tok, tok_id) in
            # for handling special token, get all sentence ids except the first and last then concatenate the first and last
            enumerate(
                zip(
                    sen,
                    list(set(sen_id[1:-1])) + sen_id[1:-1] + list(set(sen_id[1:-1])),
                )
            )
        ]
        flat_true_labels = [
            tok for sen in self.preparation.tokenization.labels_df for tok in sen
        ]

        prediction_map = self.preparation.label_alignment()
        modified_predictions = self.preparation.change_preds(
            self.results.seq_output["y_pred"].copy(), prediction_map
        )
        flat_predictions = [tok for sen in modified_predictions for tok in sen]
        flat_sentence_ids = [
            tok
            for sen in self.preparation.tokenization.sentence_ind_df
            for tok in list(set(sen[1:-1])) + sen[1:-1] + list(set(sen[1:-1]))
        ]
        flat_agreements = np.array(flat_true_labels) == np.array(flat_predictions)

        token_ids = [
            int(w)
            for batch in self.preparation.batches
            for ids in batch["input_ids"]
            for w in ids[ids != 0]
        ]
        word_ids = [
            w_id
            for batch in self.preparation.batches
            for ids in batch["input_ids"]
            for w_id in range(len(ids[ids != 0]))
        ]

        return (
            flat_last_hidden_state,
            flat_labels,
            flat_losses,
            flat_words,
            flat_tokens,
            flat_wordpieces,
            flat_first_tokens,
            token_id_strings,
            flat_true_labels,
            flat_predictions,
            flat_sentence_ids,
            flat_agreements,
            token_ids,
            word_ids,
        )


class Config:
    def __init__(self):
        self.umap_config = UMAPConfig()
        # Add other configuration sections as needed


class UMAPConfig:
    def __init__(self):
        # UMAP parameters
        self.n_neighbors = 15
        self.min_dist = 0.1
        self.metric = "cosine"
        self.random_state = 1
        self.verbose = True
        self.normalize_embeddings = False

    def set_params(
        self, n_neighbors=None, min_dist=None, metric=None, normalize_embeddings=None
    ):
        """Set parameters for UMAP if provided."""
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        if min_dist is not None:
            self.min_dist = min_dist
        if metric is not None:
            self.metric = metric
        if normalize_embeddings is not None:
            self.normalize_embeddings = normalize_embeddings


class DataTransformer:
    def __init__(self, umap_config):
        """
        Initialize the DataTransformer class with UMAP parameters and normalization option.

        Args:
            umap_config: configuration of umap paramaters
        """

        self.umap_model = UMAP(
            n_neighbors=umap_config.n_neighbors,
            min_dist=umap_config.min_dist,
            metric=umap_config.metric,
            random_state=umap_config.random_state,
            verbose=umap_config.verbose,
        )
        self.normalize_embeddings = umap_config.normalize_embeddings

    def apply_umap(self, flat_states):
        """
        Apply UMAP dimensionality reduction to the given flat states.

        Args:
            flat_states: The high-dimensional data to reduce.

        Returns:
            The reduced data.
        """
        if self.normalize_embeddings:
            flat_states = normalize(flat_states, axis=1)
        return self.umap_model.fit_transform(flat_states).transpose()

    @staticmethod
    def transform_to_dataframe(flat_data, layer_reduced):
        (
            flat_last_hidden_state,
            flat_labels,
            flat_losses,
            flat_words,
            flat_tokens,
            flat_wordpieces,
            flat_first_tokens,
            token_id_strings,
            flat_true_labels,
            flat_predictions,
            flat_sentence_ids,
            flat_agreements,
            token_ids,
            word_ids,
        ) = flat_data

        analysis_df = pd.DataFrame(
            {
                "token_id": token_ids,
                "word_id": word_ids,
                "sentence_id": flat_sentence_ids,
                "token_id_string": token_id_strings,
                "label_ids": flat_labels.tolist(),
                "words": flat_words,
                "wordpieces": flat_wordpieces,
                "tokens": flat_tokens,
                "first_tokens": flat_first_tokens,
                "truth": flat_true_labels,
                "pred": flat_predictions,
                "agreement": flat_agreements,
                "losses": flat_losses.tolist(),
                "x": layer_reduced[0],
                "y": layer_reduced[1],
            }
        )

        return analysis_df


class DataAnnotator:
    def __init__(self, subwords, dataset_outputs):
        """
        Initialize the AnalysisComputations class.

        Args:
            subwords: A dictionary containing subword information.
            dataset_outputs: The dataset outputs object containing data for analysis.
        """
        self.subwords = subwords
        self.dataset_outputs = dataset_outputs

    @staticmethod
    def annotate_tokenization_rate(analysis_df):
        """
        Annotate the tokenization rate (fertility) for each word in the dataframe.

        Args:
            analysis_df (pd.DataFrame): DataFrame containing analysis data.

        Returns:
            pd.DataFrame: Updated DataFrame with 'tokenization_rate' column.
        """
        num_tokens = []
        for wps in analysis_df["wordpieces"]:
            try:
                # Evaluate the string representation of the word pieces to a list
                word_pieces = ast.literal_eval(wps)
                num_tokens.append(len(word_pieces))
            except ValueError:
                num_tokens.append(1)
        analysis_df["tokenization_rate"] = num_tokens
        return analysis_df

    @staticmethod
    def annotate_first_token_frequencies(analysis_df, subword_locations):
        """
        Add a column 'first_tokens_freq' to the DataFrame with the frequency of each first token.

        Args:
            analysis_df (pd.DataFrame): DataFrame containing analysis data.
            subword_locations (dict): Dictionary containing subword locations.

        Returns:
            pd.DataFrame: Updated DataFrame with 'first_tokens_freq' column.
        """
        if "first_tokens" in analysis_df.columns:
            subword_freq_series = pd.Series(
                {k: len(v) for k, v in subword_locations.items()}
            )
            analysis_df["first_tokens_freq"] = (
                analysis_df["first_tokens"]
                .map(subword_freq_series)
                .fillna(0)
                .astype(int)
            )
        else:
            raise KeyError("The DataFrame does not contain a 'first_tokens' column.")
        return analysis_df

    @staticmethod
    def error_type(row):
        """
        Determine the type of error for a given row in the analysis dataframe.

        Args:
            row: A row from the analysis dataframe.

        Returns:
            str: The type of error ('Correct', 'Entity', or 'Chunk').
        """
        true, pred = row["truth"], row["pred"]
        if true == pred:
            return "Correct"
        elif true[1:] != pred[1:]:
            return "Entity"
        else:
            return "Chunk"

    def annotate_additional_info(self, analysis_df):
        print("Compute Consistency")
        analysis_df = AmbiguityComputer.compute_consistency(
            analysis_df, copy.deepcopy(self.subwords)
        )
        print("Compute Token Ambiguity")
        analysis_df["token_entropy"] = AmbiguityComputer.token_ambiguity(
            analysis_df.copy(), copy.deepcopy(self.subwords)
        )
        print("Compute Word Ambiguity")
        analysis_df["word_entropy"] = AmbiguityComputer.word_ambiguity(
            analysis_df.copy(), copy.deepcopy(self.subwords)
        )

        analysis_df["tr_entity"] = analysis_df["truth"].apply(
            lambda x: x if x in ["[CLS]", "IGNORED"] else x.split("-")[-1]
        )
        analysis_df["pr_entity"] = analysis_df["pred"].apply(
            lambda x: x if x in ["[CLS]", "IGNORED"] else x.split("-")[-1]
        )

        analysis_df["error_type"] = analysis_df[["truth", "pred"]].apply(
            DataAnnotator.error_type, axis=1
        )
        return analysis_df


class AnalysisBuilder:
    def __init__(self, preparation, ambiguity_computer, outputs, results, config):
        """
        Initialize the DataFrameCreator class.

        Args:
            preparation: The DataPreparation object.
            ambiguity_computer: The AmbiguityComputer object.
            outputs: The outputs object containing batch outputs.
            results: The results object containing model results.
            config: General configuration file for different configurations
        """
        self.preparation = preparation
        self.ambiguity_computer = ambiguity_computer
        self.outputs = outputs
        self.results = results
        self.config = config

    def construct_analysis_df(self):
        """
        Create the analysis DataFrame by extracting, transforming, and annotating data.

        Returns:
            pd.DataFrame: The final annotated DataFrame.
        """
        # Extract flat data
        extractor = DataExtractor(self.preparation, self.outputs, self.results)
        flat_data = extractor.extract_flat_data()

        # Apply UMAP transformation
        transformer = DataTransformer(self.config.umap_config)
        layer_reduced = transformer.apply_umap(flat_data[0])
        analysis_df = transformer.transform_to_dataframe(flat_data, layer_reduced)

        # Annotate the DataFrame
        analysis_df = DataAnnotator.annotate_tokenization_rate(analysis_df)
        analysis_df = DataAnnotator.annotate_first_token_frequencies(
            analysis_df, copy.deepcopy(self.computations.subwords)
        )
        analysis_df = DataAnnotator.annotate_additional_info(analysis_df)

        return analysis_df


# Main Dataset Characteristics Class
class AnalysisManager:
    def __init__(
        self,
        dataset_outputs,
        batch_outputs,
        tokenization_outputs,
        subword_outputs,
        model_outputs,
        results,
        config,
    ):
        self.label_aligner = LabelAligner(batch_outputs.batches, tokenization_outputs)
        self.ambiguity_computer = AmbiguityComputer(subword_outputs, dataset_outputs)
        self.analysis_builder = AnalysisBuilder(
            self.label_aligner, self.ambiguity_computer, model_outputs, results, config
        )
        self.analysis_df = self.analysis_builder.construct_analysis_df()


# class DatasetCharacteristics:
#     def __init__(self, dataset_outputs, batch_outputs, tokenization_outputs, subword_outputs, model_outputs, results):
#         """
#         Initialize the DatasetCharacteristics class and create the analysis DataFrame.
#
#         Args:
#             dataset_outputs: Dataset outputs object.
#             batch_outputs: Batch outputs object containing batches.
#             tokenization_outputs: Tokenization outputs object.
#             subword_outputs: Subword outputs object.
#             model_outputs: Model outputs object.
#             results: Results object containing evaluation metrics.
#         """
#         self.dataset_outputs = dataset_outputs
#         self.batches = batch_outputs.batches
#         self.tokenization = tokenization_outputs
#         self.subwords = subword_outputs
#         self.outputs = model_outputs
#
#         self.results = results
#         self.analysis_df = self.create_analysis_df()
#
#     # def extract_token_scores(self, sentence_samples, tokenization_output):
#     #     sentence_scores = []
#     #     for token_scores, labels in zip(sentence_samples.values(), self.tokenization.labels_df):
#     #         token_score = []
#     #         i = 0
#     #         for lb in labels:
#     #             if lb in ['[CLS]', 'IGNORED', '[SEP]']:
#     #                 token_score.append(-100)
#     #             else:
#     #                 if i < len(token_scores):
#     #                     token_score.append(token_scores[i])
#     #                     i += 1
#     #         sentence_scores.extend(token_score)
#     #     return sentence_scores
#
#     def label_alignment(self):
#         label_map = defaultdict(list)
#         for sen_id, sen in enumerate(self.tokenization.labels_df):
#             for tok_id, tok in enumerate(sen):
#                 if tok in ['[CLS]', '[SEP]', 'IGNORED']:
#                     label_map[sen_id].append((tok_id, tok))
#         return label_map
#
#     @staticmethod
#     def change_preds(pred, pred_map):
#         modified_pred = []
#         for sen_id, sen in enumerate(pred):
#             sentence = sen.copy()
#             for idx, tok in pred_map[sen_id]:
#                 sentence.insert(idx, tok)
#             modified_pred.append(sentence)
#         return modified_pred
#
#     def create_analysis_df(self):
#         flat_states = torch.cat([hidden_state[ids != 0] for batch in self.batches for ids, hidden_state in
#                                  zip(batch['input_ids'], batch['last_hidden_state'])])
#         flat_labels = torch.cat([labels[ids != 0] for batch in self.batches for ids, labels in
#                                  zip(batch['input_ids'], batch['labels'])])
#         flat_losses = torch.cat([losses for losses in
#                                  self.outputs.aligned_losses])
#         # flat_scores = self.extract_token_scores(self.outputs.sentence_samples,
#         #                                         self.tokenization)
#         flat_words = [tok for sen in self.tokenization.words_df for tok in sen]
#         flat_tokens = [tok for sen in self.tokenization.tokens for tok in sen]
#         flat_wordpieces = [str(tok) for sen in self.tokenization.wordpieces_df for tok in sen]
#         flat_first_tokens = [tok for sen in self.tokenization.first_tokens_df for tok in sen]
#         flat_token_ids = [f'{tok}@#{id}@#{i}' for sen, sen_id in
#                           zip(self.tokenization.first_tokens_df, self.tokenization.sentence_ind_df) for i, (tok, id) in
#                           enumerate(zip(sen, list(set(sen_id[1:-1])) + sen_id[1:-1] + list(set(sen_id[1:-1]))))]
#
#         flat_trues = [tok for sen in self.tokenization.labels_df for tok in sen]
#
#         pred_map = self.label_alignment()
#
#         modified_preds = self.change_preds(self.results.seq_output['y_pred'].copy(), pred_map)
#         flat_preds = [tok for sen in modified_preds for tok in sen]
#         flat_sen_ids = [tok for sen in self.tokenization.sentence_ind_df for tok in
#                         list(set(sen[1:-1])) + sen[1:-1] + list(set(sen[1:-1]))]
#         flat_agreement = np.array(flat_trues) == np.array(flat_preds)
#
#         t_ids = [int(w) for batch_id, batch in enumerate(self.batches) for sen_id, ids in
#                  enumerate(batch['input_ids']) for w_id, w in enumerate(ids[ids != 0])]
#
#         w_ids = [w_id for batch_id, batch in enumerate(self.batches) for sen_id, ids in
#                  enumerate(batch['input_ids']) for w_id, w in enumerate(ids[ids != 0])]
#
#         layer_reduced = UMAP(verbose=True, random_state=1).fit_transform(flat_states).transpose()
#
#         analysis_df = pd.DataFrame(
#             {'token_id': t_ids, 'word_id': w_ids, 'sen_id': flat_sen_ids, 'token_ids': flat_token_ids,
#              'label_ids': flat_labels.tolist(),
#              'words': flat_words, 'wordpieces': flat_wordpieces, 'tokens': flat_tokens,
#              'first_tokens': flat_first_tokens,
#              'truth': flat_trues, 'pred': flat_preds, 'agreement': flat_agreement,
#              'losses': flat_losses.tolist(), 'x': layer_reduced[0],
#              'y': layer_reduced[1]})
#
#         analysis_df = self.annotate_tokenization_rate(analysis_df.copy())
#         analysis_df = self.get_first_tokens(analysis_df, copy.deepcopy(self.subwords))
#         print('Compute Consistency')
#         analysis_df = self.compute_consistency(analysis_df, copy.deepcopy(self.subwords))
#         print('Compute Token Ambiguity')
#         analysis_df['token_entropy'] = self.token_ambiguity(analysis_df.copy())
#         print('Compute Word Ambiguity')
#         analysis_df['word_entropy'] = self.word_ambiguity(analysis_df.copy())
#
#         analysis_df['tr_entity'] = analysis_df['truth'].apply(
#             lambda x: x if x == '[CLS]' or x == 'IGNORED' else x.split('-')[-1])
#         analysis_df['pr_entity'] = analysis_df['pred'].apply(
#             lambda x: x if x == '[CLS]' or x == 'IGNORED' else x.split('-')[-1])
#
#         analysis_df['error_type'] = analysis_df[['truth', 'pred']].apply(self.error_type, axis=1)
#
#         return analysis_df
#
#     @staticmethod
#     def annotate_tokenization_rate(analysis_df):
#         num_tokens = []
#         for wps in analysis_df['wordpieces']:
#             try:
#                 num_tokens.append(len(ast.literal_eval(wps)))
#             except:
#                 num_tokens.append(1)
#         analysis_df['tokenization_rate'] = num_tokens
#         return analysis_df
#
#     @staticmethod
#     def get_first_tokens(analysis, subword_locations):
#         fr_tk = []
#         try:
#             analysis.insert(5, 'first_tokens_freq', analysis['first_tokens'].apply(lambda x: len(subword_locations[x])))
#         except:
#             print('')
#         return analysis
#
#     @staticmethod
#     def compute_consistency(analysis, subwords_locations):
#         consistent = []
#         inconsistent = []
#         for i in tqdm(range(len(analysis))):
#             con_count = []
#             incon_count = []
#             for t, count in Counter(
#                     [tok['tag'] for tok in subwords_locations[analysis.iloc[i]['first_tokens']]]).items():
#                 if t == analysis.iloc[i]['truth']:
#                     con_count.append(count)
#                 else:
#                     incon_count.append(count)
#             consistent.append(sum(con_count))
#             inconsistent.append(sum(incon_count))
#             try:
#                 analysis.insert(6, 'first_tokens_consistency', consistent)
#                 analysis.insert(7, 'first_tokens_inconsistency', inconsistent)
#             except:
#                 continue
#         return analysis
#
#     # def entropy(self, probabilities):
#     #     return -sum(p * math.log2(p) for p in probabilities.values())
#
#     # def label_probabilities(self, dataset):
#     #     # Count the frequencies of each label for each token
#     #     label_counts = {}
#     #     for token, label in dataset:
#     #         if token not in label_counts:
#     #             label_counts[token] = Counter()
#     #         label_counts[token][label] += 1
#     #
#     #     # Calculate the probabilities of each label for each token
#     #     probabilities = {}
#     #     for token, counts in label_counts.items():
#     #         total = sum(counts.values())
#     #         probabilities[token] = {label: count / total for label, count in counts.items()}
#     #     return probabilities
#
#     def token_ambiguity(self, analysis_df):
#         subwords_counter = []
#         for subword, tag_dis in tqdm(self.subwords.items()):
#             for tag in tag_dis:
#                 subwords_counter.append((subword, tag['tag']))
#         probabilities = self.label_probabilities(subwords_counter)
#         # Calculate the entropy for each token
#         token_entropies = {token: abs(self.entropy(probs)) for token, probs in probabilities.items()}
#         computed_token_entropy = pd.DataFrame(token_entropies.items(), columns=['first_tokens', 'entropy'])
#
#         token_entropy = []
#         for tk in tqdm(analysis_df['first_tokens']):
#             token_data = computed_token_entropy[computed_token_entropy['first_tokens'] == tk]
#             if len(token_data) > 0:
#                 token_entropy.append(token_data['entropy'].values[0])
#             else:
#                 token_entropy.append(-1)
#         return token_entropy
#
#     def word_ambiguity(self, analysis_df):
#         wordsDict = defaultdict(list)
#         for i, sen in enumerate(self.dataset_outputs.data['train']):
#             for w, t in zip(sen[1], sen[2]):
#                 wordsDict[w].append({'tag': t, 'sentence': i})
#
#         words_counter = []
#         for word, tag_dis in tqdm(wordsDict.items()):
#             for tag in tag_dis:
#                 words_counter.append((word, tag['tag']))
#
#         probabilities = self.label_probabilities(words_counter)
#         word_entropies = {token: abs(self.entropy(probs)) for token, probs in probabilities.items()}
#         computed_word_entropy = pd.DataFrame(word_entropies.items(), columns=['words', 'entropy'])
#
#         word_entropy = []
#         for tk in tqdm(analysis_df['words']):
#             token_data = computed_word_entropy[computed_word_entropy['words'] == tk]
#             if len(token_data) > 0:
#                 word_entropy.append(token_data['entropy'].values[0])
#             else:
#                 word_entropy.append(-1)
#         return word_entropy
#
#     def error_type(self, row):
#         true, pred = row['truth'], row['pred']
#
#         # Check if both entity type and boundaries are correct
#         if true == pred:
#             return 'Correct'
#
#         # Check if the entity type is incorrect but the boundaries are correct
#         elif true[1:] != pred[1:]:
#             return 'Entity'
#
#         # If neither of the above conditions are met, the error must be in the boundaries
#         else:
#             return 'Chunk'


class Entity:
    def __init__(self, outputs):
        self.y_true = outputs["y_true"]
        self.y_pred = outputs["y_pred"]
        true = self.get_entities(self.y_true)
        pred = self.get_entities(self.y_pred)
        self.seq_true, self.seq_pred = self.compute_entity_location(true, pred)
        self.entity_prediction = self.extract_entity_predictions(true, pred)

    def end_of_chunk(self, prev_tag, tag, prev_type, type_):
        """Checks if a chunk ended between the previous and current word.

        Args:
            prev_tag: previous chunk tag.
            tag: current chunk tag.
            prev_type: previous type.
            type_: current type.

        Returns:
            chunk_end: boolean.
        """
        chunk_end = False

        if prev_tag == "E":
            chunk_end = True
        if prev_tag == "S":
            chunk_end = True

        if prev_tag == "B" and tag == "B":
            chunk_end = True
        if prev_tag == "B" and tag == "S":
            chunk_end = True
        if prev_tag == "B" and tag == "O":
            chunk_end = True
        if prev_tag == "I" and tag == "B":
            chunk_end = True
        if prev_tag == "I" and tag == "S":
            chunk_end = True
        if prev_tag == "I" and tag == "O":
            chunk_end = True

        if prev_tag != "O" and prev_tag != "." and prev_type != type_:
            chunk_end = True

        return chunk_end

    def start_of_chunk(self, prev_tag, tag, prev_type, type_):
        """Checks if a chunk started between the previous and current word.

        Args:
            prev_tag: previous chunk tag.
            tag: current chunk tag.
            prev_type: previous type.
            type_: current type.

        Returns:
            chunk_start: boolean.
        """
        chunk_start = False

        if tag == "B":
            chunk_start = True
        if tag == "S":
            chunk_start = True

        if prev_tag == "E" and tag == "E":
            chunk_start = True
        if prev_tag == "E" and tag == "I":
            chunk_start = True
        if prev_tag == "S" and tag == "E":
            chunk_start = True
        if prev_tag == "S" and tag == "I":
            chunk_start = True
        if prev_tag == "O" and tag == "E":
            chunk_start = True
        if prev_tag == "O" and tag == "I":
            chunk_start = True

        if tag != "O" and tag != "." and prev_type != type_:
            chunk_start = True

        return chunk_start

    def get_entities(self, seq, suffix=False):
        """Gets entities from sequence.

        Args:
            seq (list): sequence of labels.

        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).

        Example:
            >>> from seqeval.metrics.sequence_labeling import get_entities
            >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            >>> get_entities(seq)
            [('PER', 0, 1), ('LOC', 3, 3)]
        """

        def _validate_chunk(chunk, suffix):
            if chunk in ["O", "B", "I", "E", "S"]:
                return

            if suffix:
                if not chunk.endswith(("-B", "-I", "-E", "-S")):
                    warnings.warn("{} seems not to be NE tag.".format(chunk))

            else:
                if not chunk.startswith(("B-", "I-", "E-", "S-")):
                    warnings.warn("{} seems not to be NE tag.".format(chunk))

        # for nested list
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ["O"]]

        prev_tag = "O"
        prev_type = ""
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ["O"]):
            _validate_chunk(chunk, suffix)

            if suffix:
                tag = chunk[-1]
                type_ = chunk[:-1].rsplit("-", maxsplit=1)[0] or "_"
            else:
                tag = chunk[0]
                type_ = chunk[1:].split("-", maxsplit=1)[-1] or "_"

            if self.end_of_chunk(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i - 1))
            if self.start_of_chunk(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_

        return chunks

    def compute_entity_location(self, true, pred):

        set1 = set(true)
        set2 = set(pred)

        # Find elements present in list1 but not in list2
        only_in_set1 = set1 - set2

        # Find elements present in list2 but not in list1
        only_in_set2 = set2 - set1

        # Add the missing elements to the corresponding lists
        true.extend([("O",) + t[1:] for t in only_in_set2])
        pred.extend([("O",) + t[1:] for t in only_in_set1])

        seq_true = [t[0] for t in sorted(true, key=lambda x: x[1:])]
        seq_pred = [t[0] for t in sorted(pred, key=lambda x: x[1:])]
        return seq_true, seq_pred

    def extract_tag(self, id, lst):
        i = 0
        for j, sen in enumerate(lst):
            for t in sen:
                if i + j == id:
                    return t, j
                i = i + 1

    def extract_entity_predictions(self, true, pred):
        aligned_tags = defaultdict(list)
        used_idices = []
        for t in true:
            used_idices.append(t[1:])
            aligned_tags[t[0]].append(t[1:])

        for t in pred:
            if t[1:] not in used_idices:
                aligned_tags[t[0]].append(t[1:])

        entities = []
        for tag, idxs in aligned_tags.items():
            for idx in sorted(idxs):
                for i in range(idx[0], idx[1] + 1):
                    entities.append(
                        (
                            tag,
                            self.extract_tag(i, self.y_true)[0],
                            self.extract_tag(i, self.y_pred)[0],
                            i,
                            self.extract_tag(i, self.y_true)[1],
                        )
                    )
        entity_prediction = pd.DataFrame(
            entities,
            columns=["entity", "true_token", "pred_token", "token_id", "sen_id"],
        )
        entity_prediction["agreement"] = (
            entity_prediction["true_token"] == entity_prediction["pred_token"]
        )
        return entity_prediction


class DecisionBoundary:
    def __init__(self, batches, analysis_df, outputs):

        self.batches = batches.batches
        self.analysis_df = analysis_df
        self.entropy_df = self.extract_prediction_entropy(analysis_df, outputs)

    def softmax(self, logits):
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def calculate_entropy(self, probabilities):
        return -np.sum(probabilities * np.log2(probabilities), axis=1)

    def extract_prediction_entropy(self, analysis_df, outputs):
        token_logits = []
        for batch in self.batches:
            unique_values, indices = torch.unique(
                batch["input_ids"], return_inverse=True
            )
            for token in batch["logits"][indices != 0]:
                token_logits.append(token.tolist())

        logits_matrix = np.array(token_logits)
        probabilities_matrix = self.softmax(logits_matrix)

        # Calculate entropy for each token
        prediction_entropy = self.calculate_entropy(probabilities_matrix)

        prediction_confidence = [
            max(prob_scores) for prob_scores in probabilities_matrix
        ]
        prediction_variability = [
            np.std(prob_scores) for prob_scores in probabilities_matrix
        ]

        prediction_probabilities = pd.DataFrame(probabilities_matrix).rename(
            columns=outputs.data["inv_labels"]
        )
        prediction_probabilities = prediction_probabilities.reset_index()
        prediction_probabilities = prediction_probabilities.rename(
            columns={"index": "global_id"}
        )

        analysis_df = analysis_df.reset_index()
        analysis_df = analysis_df.rename(columns={"index": "global_id"})
        analysis_df["prediction_entropy"] = prediction_entropy
        analysis_df["confidences"] = prediction_confidence
        analysis_df["variability"] = prediction_variability
        entropy_df = analysis_df.merge(prediction_probabilities, on="global_id")
        return entropy_df

    def cluster_data(self, k, states):
        # Define the number of clusters
        n_clusters = k

        # Create an instance of the KMeans algorithm
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1)

        # Fit the algorithm to the data
        kmeans.fit(states)

        # Get the cluster assignments for each data point
        labels = [f"cluster-{lb}" for lb in kmeans.labels_]

        # Get the centroid locations
        centroids = kmeans.cluster_centers_
        return centroids, labels

    def annotate_clusters(self, k):
        flat_states = torch.cat(
            [
                hidden_state[ids != 0]
                for batch in self.batches
                for ids, hidden_state in zip(
                    batch["input_ids"], batch["last_hidden_state"]
                )
            ]
        )

        flat_labels = torch.cat(
            [
                labels[ids != 0]
                for batch in self.batches
                for ids, labels in zip(batch["input_ids"], batch["labels"])
            ]
        )
        mask = np.array(flat_labels != -100)
        states = flat_states[mask]

        centroids, labels = self.cluster_data(k, states)
        self.entropy_df[f"{k}_clusters"] = "IGNORED"
        self.entropy_df.loc[mask, f"{k}_clusters"] = labels
        self.centroid_df = self.generate_centroid_data(centroids, k)
        return self.entropy_df, self.centroid_df

    def generate_centroid_data(self, centroids, k):
        flat_states = torch.cat(
            [
                hidden_state[ids != 0]
                for batch in self.batches
                for ids, hidden_state in zip(
                    batch["input_ids"], batch["last_hidden_state"]
                )
            ]
        )
        flat_labels = torch.cat(
            [
                labels[ids != 0]
                for batch in self.batches
                for ids, labels in zip(batch["input_ids"], batch["labels"])
            ]
        )
        mask = np.array(flat_labels != -100)
        states = flat_states[mask]
        c_df = self.entropy_df[mask].copy()
        centroid_df = pd.DataFrame()
        centroid_df["token_ids"] = list(c_df["token_ids"].values) + ["C"] * k
        centroid_df["truth"] = list(c_df["truth"].values) + ["C"] * k
        centroid_df["pred"] = list(c_df["pred"].values) + ["C"] * k
        centroid_df["agreement"] = list(c_df["agreement"].values) + ["C"] * k
        centroid_df["error_type"] = list(c_df["error_type"].values) + ["C"] * k
        centroid_df["centroid"] = f"Centroid-{k}"
        centroid_df["clusters"] = list(c_df[f"{k}_clusters"].values) + ["C"] * k

        centroid_data = torch.cat([states, torch.from_numpy(centroids)])

        centroid_reduced = (
            UMAP(verbose=True, random_state=1).fit_transform(centroid_data).transpose()
        )
        centroid_df["x"] = centroid_reduced[0]
        centroid_df["y"] = centroid_reduced[1]

        return centroid_df

    def generate_token_score(self):
        flat_states = torch.cat(
            [
                hidden_state[ids != 0]
                for batch in self.batches
                for ids, hidden_state in zip(
                    batch["input_ids"], batch["last_hidden_state"]
                )
            ]
        )
        flat_labels = torch.cat(
            [
                labels[ids != 0]
                for batch in self.batches
                for ids, labels in zip(batch["input_ids"], batch["labels"])
            ]
        )

        flat_mask = ~self.analysis_df["pred"].isin(["IGNORED", "[SEP]", "[CLS]"])
        flat_pred = self.analysis_df["pred"]

        self.overall_score = silhouette_score(
            flat_states[flat_labels != -100], flat_labels[flat_labels != -100]
        )
        silhouette_sample = silhouette_samples(
            flat_states[flat_labels != -100], flat_labels[flat_labels != -100]
        )
        pred_silhouette_sample = silhouette_samples(
            flat_states[flat_mask], flat_pred[flat_mask]
        )
        without_ignore = self.entropy_df[self.entropy_df["label_ids"] != -100].copy()
        without_ignore["truth_token_score"] = silhouette_sample
        without_ignore["pred_token_score"] = pred_silhouette_sample
        return without_ignore


class AttentionSimilarity:
    def __init__(self, device, model1, model2, tokeniser, preprocessor):
        self.device = device
        self.model1 = model1
        self.model2 = model2
        self.tokenizer = tokeniser
        self.preprocessor = preprocessor

    def compute_similarity(self, example):
        scores = []

        sentence_a = " ".join(example)

        if self.preprocessor == None:
            inputs = self.tokenizer.encode_plus(
                sentence_a,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=True,
            )
        else:
            inputs = self.tokenizer.encode_plus(
                self.preprocessor.preprocess(sentence_a),
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]

        with torch.no_grad():

            outputs = self.model1(**inputs)
            model1_att = outputs.attentions

        with torch.no_grad():
            outputs = self.model2(**inputs)
            model2_att = outputs.attentions

        model1_mat = np.array([atten[0].cpu().numpy() for atten in model1_att])
        model2_mat = np.array([atten[0].cpu().numpy() for atten in model2_att])

        layer = []
        head = []

        for i in range(12):
            for j in range(12):
                head.append(
                    1
                    - distance.cosine(
                        model1_mat[i][j].flatten(), model2_mat[i][j].flatten()
                    )
                )
            layer.append(head)
            head = []
        scores.append(layer)
        return scores[0]


class TrainingImpact:
    def __init__(self, mode, outputs, model_path, model):
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.data = outputs.data[mode]
        tokenizer = outputs.test_dataloader.dataset.TOKENIZER
        preprocessor = outputs.test_dataloader.dataset.PREPROCESSOR
        self.pretrained_model = AutoModel.from_pretrained(
            model_path, output_attentions=True, output_hidden_states=True
        ).to(self.device)
        self.fine_tuned_model = model.to(self.device)
        self.attention_impact = AttentionSimilarity(
            self.device,
            self.pretrained_model,
            self.fine_tuned_model,
            tokenizer,
            preprocessor,
        )

    def compute_attention_similarities(self):
        similarities = [
            self.attention_impact.compute_similarity(example[1])
            for example in tqdm(self.data[:500])
        ]
        change_fig = px.imshow(
            np.array(similarities).mean(0),
            labels=dict(x="Heads", y="Layers", color="Similarity Score"),
        )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        return change_fig

    def compute_example_similarities(self, id):
        scores = self.attention_impact.compute_similarity(self.data[id][1])
        change_fig = px.imshow(
            scores,
            labels=dict(x="Heads", y="Layers", color="Similarity Score"),
        )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        change_fig.show()

    def extract_weights(self, layer):
        self_attention_weights = torch.cat(
            [
                layer.attention.self.query.weight,
                layer.attention.self.key.weight,
                layer.attention.self.value.weight,
            ],
            dim=0,
        )

        return self_attention_weights

    def compare_weights(self):
        num_layers = len(self.pretrained_model.encoder.layer)
        num_heads = self.pretrained_model.config.num_attention_heads
        weight_diff_matrix = np.zeros((num_layers, num_heads))

        for layer in range(num_layers):
            for head in range(num_heads):
                pretrained_weight = (
                    self.extract_weights(self.pretrained_model.encoder.layer[layer])[
                        :, head::num_heads
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                fine_tuned_weight = (
                    self.extract_weights(self.fine_tuned_model.encoder.layer[layer])[
                        :, head::num_heads
                    ]
                    .cpu()
                    .detach()
                    .cpu()
                    .numpy()
                )
                weight_diff = 1 - distance.cosine(
                    pretrained_weight.flatten(), fine_tuned_weight.flatten()
                )
                weight_diff_matrix[layer, head] = weight_diff

        return self.weight_difference(weight_diff_matrix)

    def weight_difference(self, weight_diff_matrix):
        change_fig = px.imshow(
            weight_diff_matrix,
            labels=dict(x="Heads", y="Layers", color="Similarity Score"),
        )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        return change_fig


class ErrorAnalysis:
    def __init__(
        self,
        dataset_outputs,
        batches,
        tokenization_outputs,
        model_outputs,
        results,
        model,
    ):
        self.dataset_outputs = dataset_outputs
        self.batches = batches
        self.tokenization_outputs = tokenization_outputs
        self.model_outputs = model_outputs
        self.results = results
        self.model = model

    def compute_analysis_data(self, mode, model_path):
        if mode == "train":
            batches = self.batches.train_batches
            toks = self.tokenization_outputs.train_tokenization_output
            subwords = self.tokenization_outputs.train_subwords
            md_out = self.model_outputs.train_outputs
            res = self.results.train_metrics
            self.seq_report = res.seq_report
            self.skl_report = res.skl_report
            self.ent = Entity(res.seq_output)
        elif mode == "val":
            batches = self.batches.val_batches
            toks = self.tokenization_outputs.val_tokenization_output
            subwords = self.tokenization_outputs.train_subwords
            md_out = self.model_outputs.val_outputs
            res = self.results.val_metrics
            self.seq_report = res.seq_report
            self.skl_report = res.skl_report
            self.ent = Entity(res.seq_output)
        else:
            batches = self.batches.test_batches
            toks = self.tokenization_outputs.test_tokenization_output
            subwords = self.tokenization_outputs.train_subwords
            md_out = self.model_outputs.test_outputs
            res = self.results.test_metrics
            self.seq_report = res.seq_report
            self.skl_report = res.skl_report
            self.ent = Entity(res.seq_output)
        self.dc = DatasetCharacteristics(
            self.dataset_outputs, batches, toks, subwords, md_out, res
        )
        self.db = DecisionBoundary(batches, self.dc.analysis_df, self.dataset_outputs)
        self.tr_im = TrainingImpact(
            mode, self.dataset_outputs, model_path, self.model.bert
        )


class SaveAnalysis:
    def __init__(self, out_fh, error_analysis, mode, model_path):
        self.ea = error_analysis
        self.out_fh = out_fh
        self.mode = mode
        self.ea.compute_analysis_data(mode, model_path)
        self.generate_split_outputs()

    def generate_confusion(
        self,
    ):
        confusion_data = pd.DataFrame()
        confusion_data["truth"] = self.ea.ent.seq_true
        confusion_data["pred"] = self.ea.ent.seq_pred
        entity_prediction = self.ea.ent.entity_prediction
        return confusion_data, entity_prediction

    def generate_clustering(
        self,
    ):
        centroid_data = []
        cols = [3, 4, 9]
        for col in cols:
            cluster_df, centroid = self.ea.db.annotate_clusters(col)
            centroid_data.append(centroid)
        centroid_df = pd.concat(centroid_data)
        return cluster_df, centroid_df

    def generate_split_outputs(self):
        print("Generate Analysis Data")
        self.cluster_df, self.centroid_df = self.generate_clustering()
        print("Generate Prediction Data")
        self.confusion_data, self.entity_prediction = self.generate_confusion()
        self.seq_report = self.ea.seq_report
        self.skl_report = self.ea.skl_report
        print("Generate Scores Data")
        self.token_score_df = self.ea.db.generate_token_score()
        print("Generate Training Impact Data")
        self.activations = self.ea.tr_im.compute_attention_similarities()
        self.weights = self.ea.tr_im.compare_weights()

    def save(self):
        # because all the clustering fuction affecting the same df that is why we used one of theme because everytime we call the function the annotation is added
        self.cluster_df.to_json(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_analysis_df.jsonl.gz"),
            lines=True,
            orient="records",
        )

        self.centroid_df.to_json(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_centroid_df.jsonl.gz"),
            lines=True,
            orient="records",
        )

        # this is adding token silhouette score because it is ignoring the IGNORED tokens and only considering entities
        self.token_score_df.to_json(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_token_score_df.jsonl.gz"),
            lines=True,
            orient="records",
        )

        self.confusion_data.to_json(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_confusion_data.jsonl.gz"),
            lines=True,
            orient="records",
        )

        self.entity_prediction.to_json(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_entity_prediction.jsonl.gz"),
            lines=True,
            orient="records",
        )

        self.seq_report.to_csv(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_seq_report.csv"), index=False
        )
        self.skl_report.to_csv(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_skl_report.csv"), index=False
        )

        self.activations.write_json(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_activations.json")
        )
        self.weights.write_json(
            self.out_fh.cr_fn(f"{self.mode}/{self.mode}_weights.json")
        )


class AnalysisOutputs:
    def __init__(
        self, fh, out_fh, data_name, model_name, model_path, preprocessor=None
    ):
        self.out_fh = out_fh
        self.model_path = model_path
        self.outputs = fh.load_object(
            f"evalOutputs/{model_name}_{data_name}_regular_outputs.pkl"
        )
        load_model_path = fh.cr_fn(f"trainOutputs/{model_name}_{data_name}_regular.bin")
        self.model = torch.load(load_model_path)
        self.batch_outputs = BatchOutputs(self.outputs, self.model)
        self.model_outputs = ModelOutputs(self.batch_outputs)
        self.results = ModelResults(self.outputs)
        if preprocessor is not None:
            self.tokenization_outputs = TokenizationOutputs(
                self.outputs, model_path, preprocessor
            )
        else:
            self.tokenization_outputs = TokenizationOutputs(self.outputs, model_path)
        print("Finished Tokenisation Outputs")
        self.ea = ErrorAnalysis(
            self.outputs,
            self.batch_outputs,
            self.tokenization_outputs,
            self.model_outputs,
            self.results,
            self.model,
        )
        self.create_folder(out_fh)
        print("Save Tokenizatoin Subwords Output")
        self.out_fh.save_json(
            "train_subwords.json", self.tokenization_outputs.train_subwords
        )
        save_model_path = self.out_fh.cr_fn("initialization")
        torch.save(
            self.model, f"{save_model_path}/{model_name}_{data_name}_regular.bin"
        )

    def create_folder(self, out_fh):
        os.makedirs(out_fh.cr_fn("train"), exist_ok=True)
        os.makedirs(out_fh.cr_fn("val"), exist_ok=True)
        os.makedirs(out_fh.cr_fn("test"), exist_ok=True)
        os.makedirs(out_fh.cr_fn("initialization"), exist_ok=True)

    def save_analysis(self, mode):
        self.analysis = SaveAnalysis(self.out_fh, self.ea, mode, self.model_path)
        self.analysis.save()


class AuxilariyOutputs:
    def __init__(self, model_name, data_name, model_path, fh, out_fh):
        device = torch.device("cuda")
        outputs = fh.load_object(
            f"evalOutputs/{model_name}_{data_name}_regular_outputs.pkl"
        )
        pretrained_model = TCModel(len(outputs.data["labels"]), model_path)
        self.out_fh = out_fh
        self.batch_outputs = BatchOutputs(outputs, pretrained_model.to(device))
        self.light_train_df = pd.read_json(
            out_fh.cr_fn("train/train_analysis_df.jsonl.gz"), lines=True
        )[["token_ids", "words", "agreement", "truth", "pred", "x", "y"]]

        self.light_train_df.to_json(
            out_fh.cr_fn(f"light_train_df.jsonl.gz"), lines=True, orient="records"
        )

    def create_df(self, batches):
        flat_states = torch.cat(
            [
                hidden_state[ids != 0]
                for batch in batches
                for ids, hidden_state in zip(
                    batch["input_ids"], batch["last_hidden_state"]
                )
            ]
        )

        layer_reduced = (
            UMAP(verbose=True, random_state=1).fit_transform(flat_states).transpose()
        )

        analysis_df = pd.DataFrame(
            {"pre_x": layer_reduced[0], "pre_y": layer_reduced[1]}
        )
        analysis_df = analysis_df.reset_index()
        return analysis_df.rename(columns={"index": "global_id"})

    def save(self, mode):
        if mode == "train":
            pre_df = self.create_df(self.batch_outputs.train_batches.batches)
        elif mode == "val":
            pre_df = self.create_df(self.batch_outputs.val_batches.batches)
        else:
            pre_df = self.create_df(self.batch_outputs.test_batches.batches)

        pre_df.to_json(
            self.out_fh.cr_fn(f"{mode}/{mode}_pre_df.jsonl.gz"),
            lines=True,
            orient="records",
        )
