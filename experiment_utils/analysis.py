import logging
import math
import random
import time
import warnings
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure, silhouette_samples
from sklearn.preprocessing import normalize
from tqdm.autonotebook import tqdm
from transformers import AutoModel
from umap import UMAP

from experiment_utils.tokenization import TokenizationWorkflowManager


class LabelAligner:
    def __init__(self, predictions, tokenized_sentences):
        """
        Initialize the DataPreparation class.

        Args:
            batches: The batches object containing batch data.
            tokenization_outputs: The tokenization outputs object.
        """
        self.tokenized_sentences = tokenized_sentences
        self.predictions = predictions

    def get_alignment_locations(self):
        """
        Create a map for label alignment based on tokenization outputs.

        Returns:
            alignment_map: A dictionary mapping sentence IDs to token indices and tokens.
        """
        alignment_locations = defaultdict(list)
        for sentence_id, sentence in enumerate(self.tokenized_sentences):
            for token_label_id, token_label in enumerate(sentence.labels_df):
                if token_label in ["[CLS]", "[SEP]", "IGNORED"]:
                    alignment_locations[sentence_id].append(
                        (token_label_id, token_label)
                    )
        return alignment_locations

    def align_labels(self):
        """
        Modify predictions based on the alignment map.

        Args:
            preds: A list of predictions.
            pred_map: A dictionary mapping sentence IDs to token indices and tokens.

        Returns:
            modified_preds: A list of modified predictions.
        """
        alignment_locations = self.get_alignment_locations()
        modified_predictions = []
        for sentence_id, original_sentence in enumerate(self.predictions):
            sentence = original_sentence[:]  # Create a shallow copy of the list
            for index, token in alignment_locations[sentence_id]:
                # no need to shift as the indices were calculated according to the tokenized version in the truth labels
                sentence.insert(index, token)
            modified_predictions.append(sentence)
        return modified_predictions


class UMAPTransformer:
    def __init__(self, umap_config):
        """
        Initialize the UMAPTransformer with configuration for UMAP.
        """
        self.umap_config = umap_config

    def apply_umap(self, data):
        """
        Apply UMAP dimensionality reduction to the given data.
        """
        if self.umap_config.normalize_embeddings:
            data = normalize(data, axis=1)
        umap_model = UMAP(
            n_neighbors=self.umap_config.n_neighbors,
            min_dist=self.umap_config.min_dist,
            metric=self.umap_config.metric,
            random_state=self.umap_config.random_state,
            verbose=self.umap_config.verbose,
        )
        return umap_model.fit_transform(data).transpose()


@dataclass
class DeprecatedDataExtractor:
    tokenization_outputs: Optional[list] = field(default_factory=list)
    model_outputs: Optional[list] = field(default_factory=list)
    aligner: Optional["LabelAligner"] = field(default=None, repr=False)
    transformer: Optional["UMAPTransformer"] = field(default=None, repr=False)
    last_hidden_states: Optional[torch.Tensor] = field(init=False, default=None)
    labels: Optional[torch.Tensor] = field(init=False, default=None)
    losses: Optional[torch.Tensor] = field(init=False, default=None)
    token_ids: Optional[torch.Tensor] = field(init=False, default=None)
    words: Optional[list] = field(init=False, default_factory=list)
    tokens: Optional[list] = field(init=False, default_factory=list)
    word_pieces: Optional[list] = field(init=False, default_factory=list)
    core_tokens: Optional[list] = field(init=False, default_factory=list)
    true_labels: Optional[list] = field(init=False, default_factory=list)
    pred_labels: Optional[list] = field(init=False, default_factory=list)
    sentence_ids: Optional[list] = field(init=False, default_factory=list)
    token_positions: Optional[list] = field(init=False, default_factory=list)
    token_selector_id: Optional[list] = field(init=False, default_factory=list)
    agreements: Optional[list] = field(init=False, default_factory=list)
    x: Optional[list] = field(init=False, default_factory=list)
    y: Optional[list] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.model_outputs and self.tokenization_outputs:
            self.extract_features()
        if self.model_outputs:
            self.extract_model_features()
        if self.tokenization_outputs:
            self.extract_tokenization_features()
        if self.aligner:
            self.align_labels()
        if self.transformer:
            self.apply_umap()

    def extract_features(self):
        self.extract_model_features()
        self.extract_tokenization_features()
        return self

    def extract_model_features(self):
        """Extract features from the model outputs."""
        self.last_hidden_states = torch.cat(
            [s.last_hidden_states for s in self.model_outputs]
        )
        if hasattr(self.model_outputs[0], "labels"):
            self.labels = torch.cat([s.labels for s in self.model_outputs])
        if hasattr(self.model_outputs[0], "losses"):
            self.losses = torch.cat([s.losses for s in self.model_outputs])
        if hasattr(self.model_outputs[0], "input_ids"):
            self.token_ids = torch.cat([s.input_ids for s in self.model_outputs])

    def extract_tokenization_features(self):
        self.words = [
            word for sentence in self.tokenization_outputs for word in sentence.words_df
        ]
        self.tokens = [
            token
            for sentence in self.tokenization_outputs
            for token in sentence.tokens_df
        ]
        self.word_pieces = [
            wp
            for sentence in self.tokenization_outputs
            for wp in sentence.word_pieces_df
        ]
        self.core_tokens = [
            ct
            for sentence in self.tokenization_outputs
            for ct in sentence.core_tokens_df
        ]
        self.true_labels = [
            label
            for sentence in self.tokenization_outputs
            for label in sentence.labels_df
        ]
        self.sentence_ids = [
            index
            for sentence in self.tokenization_outputs
            for index in sentence.sentence_index_df
        ]
        self.token_positions = [
            position
            for sentence in self.tokenization_outputs
            for position in range(len(sentence.tokens_df))
        ]
        self.token_selector_id = [
            f"{core_token}@#{token_position}@#{sentence_index}"
            for core_token, token_position, sentence_index in zip(
                self.core_tokens, self.token_positions, self.sentence_ids
            )
        ]

    def align_labels(self):
        """Align labels according to aligner's method."""
        aligned_labels = self.aligner.align_labels()
        self.pred_labels = [label for sentence in aligned_labels for label in sentence]
        self.agreements = np.array(self.true_labels) == np.array(self.pred_labels)
        return self

    def apply_umap(self):
        """Apply dimension reduction using UMAP."""
        coordinates = self.transformer.apply_umap(self.last_hidden_states.cpu().numpy())
        self.x, self.y = coordinates
        return self

    def to_dict(self):
        """Convert extracted data to a dictionary."""
        exclude_fields = {
            "tokenization_outputs",
            "model_outputs",
            "aligner",
            "transformer",
            "last_hidden_states",
        }
        data_dict = {}
        for field_name in self.__dataclass_fields__:
            if field_name not in exclude_fields:
                value = getattr(self, field_name)
                if value is not None and (
                    not isinstance(value, (list, np.ndarray, torch.Tensor))
                    or len(value) > 0
                ):
                    if isinstance(value, torch.Tensor):
                        data_dict[field_name] = value.cpu().tolist()
                    elif isinstance(value, np.ndarray):
                        data_dict[field_name] = value.tolist()
                    else:
                        data_dict[field_name] = value
        return data_dict

    def to_df(self, is_pretrained: bool = False):
        """Convert data to pandas DataFrame and compute global ID."""
        data_dict = self.to_dict()
        if is_pretrained:
            df = pd.DataFrame(data_dict)
            df["global_id"] = UtilityFunctions.global_ids_from_df(df)
            df = df.rename(columns={"x": "pre_x", "y": "pre_y"})
            df = df[["global_id", "pre_x", "pre_y"]]
        else:
            df = pd.DataFrame(data_dict)
            df["global_id"] = UtilityFunctions.global_ids_from_df(df)
        return df


class ModelFeatureExtractor:
    def __init__(self, model_outputs):
        self.model_outputs = model_outputs

    def extract_features(self):
        # Initialize lists to store features
        last_hidden_states = []
        labels = []
        losses = []
        token_ids = []
        sentence_ids = []
        token_positions = []

        # Iterate over each model output, which represents a sentence
        for idx, output in enumerate(self.model_outputs):
            if hasattr(output, "input_ids"):
                # Convert tensor of input_ids to a list of integers
                input_ids = (
                    output.input_ids.tolist()
                    if torch.is_tensor(output.input_ids)
                    else output.input_ids
                )
                token_ids.extend(input_ids)
                sentence_ids.extend([idx] * len(input_ids))
                token_positions.extend(list(range(len(input_ids))))

            if hasattr(output, "last_hidden_states"):
                last_hidden_states.append(output.last_hidden_states)

            if hasattr(output, "labels"):
                labels.extend(
                    output.labels
                    if isinstance(output.labels, list)
                    else output.labels.tolist()
                )

            if hasattr(output, "losses"):
                losses.extend(
                    output.losses
                    if isinstance(output.losses, list)
                    else output.losses.tolist()
                )

        # Concatenate tensors where applicable and convert to list
        features = {
            "last_hidden_states": (
                torch.cat(last_hidden_states) if last_hidden_states else []
            ),
            "labels": labels,
            "losses": losses,
            "token_ids": token_ids,
            "sentence_ids": sentence_ids,
            "token_positions": token_positions,
        }
        return features


class TokenizationFeatureExtractor:
    def __init__(self, tokenization_outputs):
        self.tokenization_outputs = tokenization_outputs

    def extract_features(self):
        attributes = ["words", "tokens", "word_pieces", "core_tokens"]
        features = {
            attr: [
                item
                for output in self.tokenization_outputs
                for item in getattr(output, f"{attr}_df")
            ]
            for attr in attributes
        }
        features["sentence_ids"] = [
            index
            for sentence in self.tokenization_outputs
            for index in sentence.sentence_index_df
        ]
        features["true_labels"] = [
            label
            for sentence in self.tokenization_outputs
            for label in sentence.labels_df
        ]
        features["token_positions"] = [
            pos
            for output in self.tokenization_outputs
            for pos in range(len(output.tokens_df))
        ]
        features["token_selector_id"] = [
            f"{ct}@#{tp}@#{si}"
            for ct, tp, si in zip(
                features["core_tokens"],
                features["token_positions"],
                features["sentence_ids"],
            )
        ]
        return features


@dataclass
class DataExtractor:
    model_outputs: Optional[List] = field(default_factory=list)
    tokenization_outputs: Optional[List] = field(default_factory=list)
    aligner: Optional["LabelAligner"] = None
    transformer: Optional["UMAPTransformer"] = None

    # Attributes from model features
    last_hidden_states: Optional[torch.Tensor] = field(init=False, default=None)
    labels: Optional[List] = field(init=False, default=list)
    losses: Optional[List] = field(init=False, default=list)
    token_ids: Optional[List[int]] = field(init=False, default=list)
    sentence_ids: Optional[List[int]] = field(init=False, default=list)
    token_positions: Optional[List[int]] = field(init=False, default=list)

    # Attributes from tokenization features
    words: Optional[list] = field(init=False, default_factory=list)
    tokens: Optional[list] = field(init=False, default_factory=list)
    word_pieces: Optional[list] = field(init=False, default_factory=list)
    core_tokens: Optional[list] = field(init=False, default_factory=list)
    true_labels: Optional[list] = field(init=False, default_factory=list)
    sentence_ids: Optional[list] = field(init=False, default_factory=list)
    token_positions: Optional[list] = field(init=False, default_factory=list)
    token_selector_id: Optional[list] = field(init=False, default_factory=list)

    # Attributes for aligned data or transformed data
    pred_labels: Optional[list] = field(init=False, default_factory=list)
    agreements: Optional[list] = field(init=False, default_factory=list)
    x: Optional[list] = field(init=False, default_factory=list)
    y: Optional[list] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.extract_features()

    def extract_features(self):
        if self.model_outputs:
            logging.info("Extracting model features...")
            model_features = ModelFeatureExtractor(
                self.model_outputs
            ).extract_features()
            for key, value in model_features.items():
                setattr(self, key, value)

        if self.tokenization_outputs:
            logging.info("Extracting tokenization features...")
            token_features = TokenizationFeatureExtractor(
                self.tokenization_outputs
            ).extract_features()
            for key, value in token_features.items():
                setattr(self, key, value)

        # Example of handling alignment and transformation
        if self.aligner:
            logging.info("Aligning labels...")
            self.align_labels()
        if self.transformer:
            self.apply_umap()

        return self

    def align_labels(self):
        aligned_labels = self.aligner.align_labels()
        self.pred_labels = [label for sentence in aligned_labels for label in sentence]
        self.agreements = np.array(self.true_labels) == np.array(self.pred_labels)

    def apply_umap(self):
        if self.last_hidden_states is not None:
            logging.info("Applying UMAP...")
            if isinstance(self.last_hidden_states, torch.Tensor):
                # Ensuring the tensor is on CPU and converting to NumPy array
                coordinates = self.transformer.apply_umap(
                    self.last_hidden_states.cpu().numpy()
                )
                self.x, self.y = coordinates
            else:
                logging.error(
                    "Expected last_hidden_states to be a torch.Tensor but got another type."
                )
        else:
            logging.warning("last_hidden_states is None, skipping UMAP application.")

    def validate_attribute_lengths(self, attributes):
        """Validate that all relevant attributes in the dictionary have the same length."""
        length = None
        attribute_name = None
        try:
            for attribute, value in attributes.items():
                # Check if the value is an instance and not empty
                if isinstance(value, (list, np.ndarray)) and (len(value) > 0):
                    if length is None:
                        length = len(value)
                        attribute_name = attribute
                    elif length != len(value):
                        raise ValueError(
                            f"Length mismatch: '{attribute_name}' has length {length}, but '{attribute}' has length {len(value)}"
                        )
        except Exception as e:
            raise ValueError(f"Error during length validation: {str(e)}")

    def to_dict(self):
        # Exclude fields with None values and return as dictionary
        exclude_fields = {
            "model_outputs",
            "tokenization_outputs",
            "aligner",
            "transformer",
            "last_hidden_states",
        }
        return {
            k: v
            for k, v in vars(self).items()
            if k not in exclude_fields and v is not None and len(v) > 0
        }

    def to_df(self, columns_map=None, required_columns=None):
        # Convert to DataFrame, handling cases where lists are involved
        data_dict = self.to_dict()
        self.validate_attribute_lengths(data_dict)
        df = pd.DataFrame(data_dict)

        # Apply global IDs computation if the necessary columns exist
        if (
            "token_ids" in df.columns
            and "sentence_ids" in df.columns
            and "token_positions" in df.columns
            and "labels" in df.columns
        ):
            df["global_id"] = UtilityFunctions.global_ids_from_df(df)

        # Rename columns based on a passed dictionary
        if columns_map:
            df.rename(columns=columns_map, inplace=True)

        # Select only the required columns if specified
        if required_columns:
            df = df[required_columns]
        return df


@dataclass
class DynamicDataExtractor:

    model_outputs: InitVar[Optional[List]] = None
    tokenization_outputs: InitVar[Optional[List]] = None
    aligner: InitVar[Optional["LabelAligner"]] = None
    transformer: InitVar[Optional["UMAPTransformer"]] = None

    def __post_init__(self, model_outputs, tokenization_outputs, aligner, transformer):
        # Automatically process the data if not already processed
        self.extract_features(model_outputs, tokenization_outputs, aligner, transformer)

    def extract_features(
        self, model_outputs, tokenization_outputs, aligner, transformer
    ):
        if model_outputs:
            model_features = ModelFeatureExtractor(model_outputs).extract_features()
            self.__dict__.update(model_features)

        if tokenization_outputs:
            token_features = TokenizationFeatureExtractor(
                tokenization_outputs
            ).extract_features()
            self.__dict__.update(token_features)

        if aligner:
            self.align_labels(aligner, true_labels)

        if transformer and "last_hidden_states" in self.__dict__:
            self.apply_umap(transformer, last_hidden_states)
        return self

    def align_labels(self, aligner, true_labels):
        aligned_labels = aligner.align_labels()
        self.pred_labels = [label for sentence in aligned_labels for label in sentence]
        self.agreements = np.array(true_labels) == np.array(self.pred_labels)
        self.__dict__.update(
            {"pred_labels": self.pred_labels, "agreements": self.agreements}
        )

    def apply_umap(self, transformer, last_hidden_states):
        coordinates = transformer.apply_umap(last_hidden_states.cpu().numpy())
        self.x, self.y = coordinates
        self.__dict__.update({"x": self.x, "y": self.y})

    def to_dict(self):
        exclude_fields = {"last_hidden_states"}
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in exclude_fields and v is not None and not callable(v)
        }

    def to_df(self, columns_map=None, required_columns=None):
        data_dict = self.to_dict()
        df = pd.DataFrame(data_dict)

        # Apply global IDs computation if the necessary columns exist
        if (
            "token_ids" in df.columns
            and "sentence_ids" in df.columns
            and "token_positions" in df.columns
            and "labels" in df.columns
        ):
            df["global_id"] = UtilityFunctions.global_ids_from_df(df)

        # Rename columns based on a passed dictionary
        if columns_map:
            df.rename(columns=columns_map, inplace=True)

        # Select only the required columns if specified
        if required_columns:
            df = df[required_columns]
        return df


class UtilityFunctions:
    @staticmethod
    def per_token_entropy(p):
        """
        Calculate the Shannon entropy of a distribution of probabilities.

        Args:
            p (list of float): A list of probabilities associated with different classes.

        Returns:
            float: The calculated entropy value.
        """
        return -p * np.log2(p) if p > 0 else 0

    @staticmethod
    def entropy(probabilities):
        """
        Calculate the Shannon entropy of a distribution of probabilities.

        Args:
            probabilities (list of float): A list of probabilities associated with different classes.

        Returns:
            float: The calculated entropy value.
        """
        if not probabilities.size:
            return 0
        return -np.sum(probabilities * np.log2(probabilities), axis=1)

    @staticmethod
    def max_entropy(num_classes):
        """
        Calculate the maximum entropy for a given number of distinct classes.
        Maximum entropy occurs when all classes are equally probable.

        Args:
            num_classes (int): The number of distinct classes.

        Returns:
            float: The maximum entropy in bits. Returns 0 if num_classes is 1 or less.
        """
        if num_classes > 1:
            p = 1 / num_classes
            return -num_classes * p * math.log2(p)
        return 0

    @staticmethod
    def error_type(row):
        """
        Determine the type of error for a given row based on the 'truth' and 'pred' columns.
        """

        true, pred = row["true_labels"], row["pred_labels"]

        # Check if both labels are exactly the same, including 'O'
        if true == pred:
            return "Correct"

        # Check for 'O' to avoid IndexError when accessing parts of the string
        elif "O" in [true, pred]:
            return "Chunk"  # 'Chunk' error since one is 'O' and the other isn't

        # Extract parts after the dash and compare
        elif true.split("-")[1] != pred.split("-")[1]:
            return "Entity"

        # If not correct and no entity type mismatch, it must be a chunk error
        else:
            return "Chunk"

    @staticmethod
    def softmax(logits):
        """Apply softmax to logits to get probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    @staticmethod
    def generate_global_ids(model_outputs):
        global_ids = []
        for sentence_id, sentence in enumerate(model_outputs):
            for token_position, (token_id, label) in enumerate(
                zip(sentence.input_ids.tolist(), sentence.labels.tolist())
            ):
                global_ids.append(
                    f"{str(token_id)}_{sentence_id}_{token_position}_{str(label)}"
                )
        return global_ids

    @staticmethod
    def global_ids_from_df(df):
        return (
            df["token_ids"].astype(str)
            + "_"
            + df["sentence_ids"].astype(str)
            + "_"
            + df["token_positions"].astype(str)
            + "_"
            + df["labels"].astype(str)
        ).values


class TokenConsistencyCalculator:
    @staticmethod
    def calculate(subword_index, analysis_df):
        """Calculate the token consistency from subword tagging data.

        Args:
            subword_index (dict): A dictionary mapping tokens to their corresponding tags and other metadata.
            analysis_df (DataFrame): A DataFrame with analysis data including 'core_tokens' and 'true_labels'.

        Returns:
            DataFrame: A DataFrame with new columns for consistency count, inconsistency count, and total occurrences.
        """
        # Create frequency dictionary for each token based on tags
        train_label_freq = defaultdict(lambda: defaultdict(int))
        for token, tags in tqdm(subword_index.items(), desc="Extract Subwords"):
            for tag in tags:
                train_label_freq[token][tag["tag"]] += 1

        # Initialize list to hold consistency information
        consistency_info = {
            "consistency_count": [],
            "inconsistency_count": [],
            "total_train_occurrences": [],
        }

        # Calculate consistency information for each token in analysis dataframe
        for row in tqdm(
            analysis_df.itertuples(index=False),
            total=len(analysis_df),
            desc="Calculate Consistency",
        ):
            token = getattr(row, "core_tokens")
            test_label = getattr(row, "true_labels")
            label_dict = train_label_freq[token]
            consistency_count = label_dict.get(test_label, 0)
            total_train_occurrences = sum(label_dict.values())
            inconsistency_count = total_train_occurrences - consistency_count

            consistency_info["consistency_count"].append(consistency_count)
            consistency_info["inconsistency_count"].append(inconsistency_count)
            consistency_info["total_train_occurrences"].append(total_train_occurrences)

        # Return the results as a DataFrame
        return pd.DataFrame(consistency_info)


class TokenEntropyCalculator:
    @staticmethod
    def create_subwords_dataframe(subword_index):
        """
        Transforms subword index into a DataFrame of tokens and their corresponding tags.

        Args:
            subword_index (dict): A dictionary where keys are tokens and values are lists of tags.

        Returns:
            DataFrame: A pandas DataFrame with columns 'token' and 'tag'.
        """
        if not subword_index:
            return pd.DataFrame(columns=["train_token", "tag"])
        subwords_counter = [
            (subword, tag["tag"])
            for subword, tags in subword_index.items()
            for tag in tags
        ]
        return pd.DataFrame(subwords_counter, columns=["train_token", "tag"])

    @staticmethod
    def calculate_tag_counts(subwords_df):
        """
        Calculates the count of each tag for tokens and computes initial probability contributions.

        Args:
            subwords_df (DataFrame): DataFrame containing token and tag information.

        Returns:
            DataFrame: Updated DataFrame with probability and initial entropy contribution for each tag.
        """
        token_counts = (
            subwords_df.groupby(["train_token", "tag"]).size().reset_index(name="count")
        )
        token_total = (
            token_counts.groupby(["train_token"])["count"]
            .sum()
            .reset_index(name="total")
        )
        token_specific_prob = pd.merge(token_counts, token_total, on="train_token")
        token_specific_prob["probability"] = (
            token_specific_prob["count"] / token_specific_prob["total"]
        )
        token_specific_prob["entropy_contribution"] = token_specific_prob[
            "probability"
        ].apply(UtilityFunctions.per_token_entropy)
        return token_specific_prob

    @staticmethod
    def calculate_entropy(subwords_df, token_specific_prob):
        """
        Calculates local and dataset-wide entropy for tokens.

        Args:
            subwords_df (DataFrame): DataFrame containing token and tag information.
            token_specific_prob (DataFrame): DataFrame containing calculated probabilities for each tag.

        Returns:
            DataFrame: DataFrame with token entropy values including maximum entropy.
        """
        num_tags_per_token = (
            subwords_df.groupby("train_token")["tag"]
            .nunique()
            .reset_index()
            .rename(columns={"tag": "num_tags"})
        )
        num_tags_per_token["token_max_entropy"] = num_tags_per_token["num_tags"].apply(
            UtilityFunctions.max_entropy
        )

        token_entropy = (
            token_specific_prob.groupby("train_token")["entropy_contribution"]
            .sum()
            .reset_index()
            .rename(columns={"entropy_contribution": "local_token_entropy"})
        )
        token_entropy = pd.merge(
            token_entropy,
            num_tags_per_token[["train_token", "token_max_entropy"]],
            on="train_token",
        )

        dataset_total = len(subwords_df)
        dataset_prob = (
            subwords_df.groupby(["train_token", "tag"])
            .size()
            .div(dataset_total)
            .reset_index(name="dataset_probability")
        )
        dataset_prob["dataset_entropy_contribution"] = dataset_prob[
            "dataset_probability"
        ].apply(UtilityFunctions.per_token_entropy)
        dataset_entropy = (
            dataset_prob.groupby("train_token")["dataset_entropy_contribution"]
            .sum()
            .reset_index()
            .rename(columns={"dataset_entropy_contribution": "dataset_token_entropy"})
        )

        entropy_df = pd.merge(token_entropy, dataset_entropy, on="train_token")
        return entropy_df

    @staticmethod
    def calculate(subword_index):
        """
        Main method to calculate token entropy from subword index.

        Args:
            subword_index (dict): Dictionary with token keys and lists of tags as values.

        Returns:
            DataFrame: Final DataFrame with all entropy calculations for tokens.
        """
        subwords_df = TokenEntropyCalculator.create_subwords_dataframe(subword_index)
        token_specific_prob = TokenEntropyCalculator.calculate_tag_counts(subwords_df)
        entropy_df = TokenEntropyCalculator.calculate_entropy(
            subwords_df, token_specific_prob
        )
        return entropy_df


class WordEntropyCalculator:
    @staticmethod
    def create_words_dataframe(data):
        """
        Transforms a nested data structure into a DataFrame of words and their associated tags.

        Args:
            data (list of dicts): A list where each element is a dict representing a sentence
                                  with 'words' and 'tags' as keys.

        Returns:
            DataFrame: A pandas DataFrame with columns 'word', 'tag', and 'sentence_index'.
        """
        wordsDict = defaultdict(list)
        for i, s in enumerate(data):
            for w, t in zip(s["words"], s["tags"]):
                wordsDict[w].append({"tag": t, "sentence": i})

        return pd.DataFrame(
            [
                {
                    "train_word": word,
                    "tag": tag["tag"],
                    "sentence_index": tag["sentence"],
                }
                for word, tags in wordsDict.items()
                for tag in tags
            ]
        )

    @staticmethod
    def calculate_entropy_components(words_df):
        """
        Calculates local and dataset-wide entropy components for each word.

        Args:
            words_df (DataFrame): DataFrame containing 'word', 'tag', and 'sentence_index'.

        Returns:
            DataFrame: DataFrame enriched with local and dataset entropy components.
        """
        # Calculate number of distinct tags per word and maximum entropy
        num_tags_per_word = (
            words_df.groupby("train_word")["tag"]
            .nunique()
            .reset_index()
            .rename(columns={"tag": "num_tags"})
        )
        num_tags_per_word["word_max_entropy"] = num_tags_per_word["num_tags"].apply(
            UtilityFunctions.max_entropy
        )

        # Local entropy: normalize by the number of occurrences of each word
        local_probabilities = (
            words_df.groupby(["train_word", "tag"])
            .size()
            .div(words_df.groupby("train_word").size(), axis=0)
            .reset_index(name="local_probability")
        )
        local_probabilities["local_entropy_contribution"] = local_probabilities[
            "local_probability"
        ].apply(UtilityFunctions.per_token_entropy)
        local_entropy_df = (
            local_probabilities.groupby("train_word")["local_entropy_contribution"]
            .sum()
            .reset_index()
            .rename(columns={"local_entropy_contribution": "local_word_entropy"})
        )
        local_entropy_df = pd.merge(
            local_entropy_df,
            num_tags_per_word[["train_word", "word_max_entropy"]],
            on="train_word",
        )

        # Dataset-wide entropy: normalize by the total number of tags in the dataset
        dataset_probabilities = (
            words_df.groupby(["train_word", "tag"])
            .size()
            .div(len(words_df))
            .reset_index(name="dataset_probability")
        )
        dataset_probabilities["dataset_entropy_contribution"] = dataset_probabilities[
            "dataset_probability"
        ].apply(UtilityFunctions.per_token_entropy)
        dataset_entropy_df = (
            dataset_probabilities.groupby("train_word")["dataset_entropy_contribution"]
            .sum()
            .reset_index()
            .rename(columns={"dataset_entropy_contribution": "dataset_word_entropy"})
        )

        # Merge local and dataset entropy dataframes
        entropy_df = pd.merge(local_entropy_df, dataset_entropy_df, on="train_word")

        return entropy_df

    @staticmethod
    def calculate(data):
        """
        Main method to calculate word entropy from structured input data.

        Args:
            data (list of dicts): Input data with each dict containing 'words' and 'tags' keys.

        Returns:
            DataFrame: Final DataFrame with all word entropy calculations.
        """
        words_df = WordEntropyCalculator.create_words_dataframe(data)
        entropy_df = WordEntropyCalculator.calculate_entropy_components(words_df)
        return entropy_df


class PredictionEntropyCalculator:

    @staticmethod
    def calculate(model_outputs, labels_map):
        """Extract prediction entropy from logits."""
        token_logits = []
        for sentence in model_outputs:
            for token in sentence.logits:
                token_logits.append(token.tolist())

        logits_matrix = np.array(token_logits)
        probabilities_matrix = UtilityFunctions.softmax(logits_matrix)
        prediction_entropy = UtilityFunctions.entropy(probabilities_matrix)
        prediction_confidence = [
            max(prob_scores) for prob_scores in probabilities_matrix
        ]
        prediction_variability = [
            np.std(prob_scores) for prob_scores in probabilities_matrix
        ]
        prediction_analysis = pd.DataFrame(probabilities_matrix, columns=labels_map)

        prediction_analysis["prediction_entropy"] = prediction_entropy
        prediction_analysis["prediction_max_entropy"] = UtilityFunctions.max_entropy(
            len(labels_map)
        )
        prediction_analysis["confidence"] = prediction_confidence
        prediction_analysis["variability"] = prediction_variability
        prediction_analysis["global_id"] = UtilityFunctions.generate_global_ids(
            model_outputs
        )
        return prediction_analysis


class ClusterAnalysis:
    def __init__(self, flat_data, analysis_df, config):
        self.flat_data = flat_data
        self.labels_mask = np.array(self.flat_data.labels) != -100
        self.states = self.flat_data.last_hidden_states[self.labels_mask]
        self.true_labels = analysis_df["true_labels"][self.labels_mask]
        self.pred_labels = analysis_df["pred_labels"][self.labels_mask]
        self.df = pd.DataFrame()
        self.df["global_id"] = UtilityFunctions.global_ids_from_df(
            analysis_df[self.labels_mask]
        ).copy()
        self.config = config

    def normalize_states(self):
        return normalize(self.states, norm=self.config.norm, axis=1)

    def calculate_silhouette_scores(self):
        truth_token_score = silhouette_samples(
            self.states, self.true_labels, metric=self.config.silhouette_metric
        )
        pred_token_score = silhouette_samples(
            self.states, self.pred_labels, metric=self.config.silhouette_metric
        )

        self.df["true_token_score"] = truth_token_score
        self.df["pred_token_score"] = pred_token_score
        average_silhouette_score = {
            "true_score": self.df["true_token_score"].mean(),
            "pred_score": self.df["pred_token_score"].mean(),
        }
        return average_silhouette_score

    def apply_kmeans(self, k):
        kmeans_model = KMeans(
            n_clusters=k,
            init=self.config.init_method,
            n_init=self.config.n_init,
            random_state=self.config.random_state,
        )
        normalized_states = self.normalize_states()
        kmeans_model.fit(normalized_states)
        cluster_labels = [f"cluster-{lb}" for lb in kmeans_model.labels_]
        return kmeans_model.cluster_centers_, cluster_labels, kmeans_model.labels_

    def generate_clustering_outputs(self, k):
        centroids, cluster_labels, kmeans_labels = self.apply_kmeans(k)

        self.df[f"k={k}"] = cluster_labels
        aligned_labels = self.align_labels(self.true_labels, k)
        self.df[self.config.n_clusters_map[k]] = aligned_labels

        metrics = homogeneity_completeness_v_measure(aligned_labels, kmeans_labels)
        clustering_metrics = {
            "homogeneity": metrics[0],
            "completeness": metrics[1],
            "v_measure": metrics[2],
            "centroids": centroids,
            "labels": kmeans_labels,
            "aligned_labels": aligned_labels,
        }

        return clustering_metrics

    def align_labels(self, true_labels, k):
        aligned_labels = []
        for label in true_labels:
            match k:
                case 3:
                    aligned_labels.append(label.split("-")[0])
                case 4:
                    aligned_labels.append(label.split("-")[-1])
                case 9:
                    aligned_labels.append(label)
        return aligned_labels

    def calculate_centroid_average_similarity(self, kmeans_metrics):

        ner_labels = np.array(kmeans_metrics["k=9"]["aligned_labels"])
        cluster_labels = kmeans_metrics["k=9"]["labels"]
        centroids = kmeans_metrics["k=9"]["centroids"]
        # Calculate cosine similarities and then convert to a DataFrame
        similarity_matrix = 1 - cdist(self.states, centroids, metric="cosine")
        df = pd.DataFrame(
            similarity_matrix,
            columns=[f"Centroid_{i}" for i in range(centroids.shape[0])],
        )
        df["NER_Labels"] = ner_labels

        # Compute average similarities for each NER label per centroid
        avg_similarities = df.groupby("NER_Labels").mean().reset_index()
        avg_similarities.rename(columns={"NER_Labels": "NER_Label"}, inplace=True)

        return avg_similarities

    def calculate(self):
        logging.info("Calculating Silhouette Score")
        average_silhouette_score = self.calculate_silhouette_scores()
        kmeans_metrics = {}
        for k in self.config.n_clusters:
            logging.info("Processing K=%s", k)
            clustering_results = self.generate_clustering_outputs(k)
            kmeans_metrics[f"k={k}"] = clustering_results
        logging.info("Calculating Centorid Average Similarity Matrix")
        centroids_avg_similarities = self.calculate_centroid_average_similarity(
            kmeans_metrics
        )

        for k in kmeans_metrics:
            kmeans_metrics[k].pop("centroids", None)
            kmeans_metrics[k].pop("aligned_labels", None)
            kmeans_metrics[k].pop("labels", None)

        return (
            average_silhouette_score,
            kmeans_metrics,
            self.df,
            centroids_avg_similarities,
        )


class DataAnnotator:
    def __init__(
        self,
        subwords,
        analysis_data,
        pretrained_coordinates,
        train_data,
        model_outputs,
        labels_map,
    ):
        self.subwords = subwords
        self.original_data = (
            analysis_data.copy()
        )  # Keep an original copy for fresh starts
        self.pretrained_coordinates = pretrained_coordinates
        self.train_data = train_data
        self.model_outputs = model_outputs
        self.labels_map = labels_map
        self.analysis_data = analysis_data.copy()

    def annotate_tokenization_rate(self):
        """Annotate the tokenization rate for each word based on the number of subwords."""
        self.analysis_data["tokenization_rate"] = self.analysis_data[
            "word_pieces"
        ].apply(lambda x: len(x) if isinstance(x, list) else 0)

    def annotate_consistency(self):
        logging.info("Annotating consistency...")
        consistency_results = TokenConsistencyCalculator.calculate(
            self.subwords, self.analysis_data
        )
        self.analysis_data = pd.concat(
            [self.analysis_data, consistency_results], axis=1
        )

    def annotate_token_entropy(self):
        logging.info("Annotating token entropy...")
        token_entropy = TokenEntropyCalculator.calculate(self.subwords)
        self.analysis_data = self.analysis_data.merge(
            token_entropy, left_on="core_tokens", right_on="train_token", how="left"
        )
        self.analysis_data["local_token_entropy"] = self.analysis_data[
            "local_token_entropy"
        ].fillna(-1)
        self.analysis_data["token_max_entropy"] = self.analysis_data[
            "token_max_entropy"
        ].fillna(-1)
        self.analysis_data["dataset_token_entropy"] = self.analysis_data[
            "dataset_token_entropy"
        ].fillna(-1)
        self.analysis_data.drop("train_token", axis=1, inplace=True)

    def annotate_word_entropy(self):
        logging.info("Annotating word entropy...")
        word_entropy = WordEntropyCalculator.calculate(self.train_data)
        self.analysis_data = self.analysis_data.merge(
            word_entropy, left_on="words", right_on="train_word", how="left"
        )
        self.analysis_data["local_word_entropy"] = self.analysis_data[
            "local_word_entropy"
        ].fillna(-1)
        self.analysis_data["word_max_entropy"] = self.analysis_data[
            "word_max_entropy"
        ].fillna(-1)
        self.analysis_data["dataset_word_entropy"] = self.analysis_data[
            "dataset_word_entropy"
        ].fillna(-1)
        self.analysis_data.drop("train_word", axis=1, inplace=True)

    def annotate_error_types(self):
        logging.info("Annotating error types...")
        self.analysis_data["error_type"] = self.analysis_data.apply(
            UtilityFunctions.error_type, axis=1
        )

    def annotate_entity(self):
        logging.info("Annotating entity...")
        self.analysis_data["tr_entity"] = self.analysis_data["true_labels"].apply(
            lambda x: x if x in ["[CLS]", "[SEP]", "IGNORED"] else x.split("-")[-1]
        )
        self.analysis_data["pr_entity"] = self.analysis_data["pred_labels"].apply(
            lambda x: x if x in ["[CLS]", "[SEP]", "IGNORED"] else x.split("-")[-1]
        )

    def annotate_prediction_entropy(self):
        logging.info("Annotating prediction entropy...")
        prediction_entropy_df = PredictionEntropyCalculator.calculate(
            self.model_outputs, self.labels_map
        )
        self.analysis_data = self.analysis_data.merge(
            prediction_entropy_df, on="global_id"
        )

    def annotate_pretrained_coordinates(self):
        logging.info("Annotating pretrained coordinates...")
        if "global_id" not in self.analysis_data.columns:
            logging.error("Column 'global_id' is missing from analysis data.")
            raise KeyError("Column 'global_id' is missing from analysis data.")
        if "global_id" not in self.pretrained_coordinates.columns:
            logging.error("Column 'global_id' is missing from pretrained coordinates.")
            raise KeyError("Column 'global_id' is missing from pretrained coordinates.")
        self.analysis_data = self.analysis_data.merge(
            self.pretrained_coordinates, on="global_id"
        )

    def reset_data(self):
        """Reset analysis data to the original state to avoid cumulative effects."""
        self.analysis_data = self.original_data.copy()

    def annotate_all(self):
        logging.info("Annotating all...")
        self.reset_data()  # Start with a fresh state
        self.annotate_consistency()
        self.annotate_token_entropy()
        self.annotate_word_entropy()
        self.annotate_tokenization_rate()
        self.annotate_entity()
        self.annotate_error_types()
        self.annotate_prediction_entropy()
        self.annotate_pretrained_coordinates()
        return self.analysis_data


class AnalysisWorkflowManager:
    def __init__(
        self,
        config_manager,
        evaluation_results,
        tokenization_outputs,
        model_outputs,
        pretrained_model_outputs,
        data_manager,
        split,
    ):
        self.transformer = UMAPTransformer(config_manager.umap_config)
        self.aligner = LabelAligner(
            evaluation_results.entity_outputs["y_pred"].copy(),
            tokenization_outputs.get_split(split),
        )
        self.config_manager = config_manager
        self.tokenization_outputs = tokenization_outputs
        self.model_outputs = model_outputs
        self.pretrained_model_outputs = pretrained_model_outputs
        self.data_manager = data_manager
        self.split = split

    def extract_analysis_data(self):
        try:
            analysis_flat_data = DataExtractor(
                tokenization_outputs=self.tokenization_outputs.get_split(self.split),
                model_outputs=self.model_outputs.get_split(self.split),
                aligner=self.aligner,
                transformer=self.transformer,
            )
            analysis_df = analysis_flat_data.to_df()
            pretrained_flat_data = DataExtractor(
                tokenization_outputs=self.tokenization_outputs.get_split(self.split),
                model_outputs=self.pretrained_model_outputs.get_split(self.split),
                transformer=self.transformer,
            )
            pretrained_coordinates = pretrained_flat_data.to_df(
                {"x": "pre_x", "y": "pre_y"}, ["global_id", "pre_x", "pre_y"]
            )
            return analysis_df, analysis_flat_data, pretrained_coordinates
        except Exception as e:
            logging.error("Error in data extraction: %s", e)
            raise

    def perform_clustering(self, analysis_df, flat_data):
        try:
            clustering_analyser = ClusterAnalysis(
                flat_data, analysis_df, self.config_manager.clustering_config
            )
            (
                average_silhouette_score,
                kmeans_metrics,
                clustering_df,
                centroids_avg_similarities,
            ) = clustering_analyser.calculate()
            merged_clustering_data = analysis_df.merge(
                clustering_df, on="global_id", how="left"
            )
            return (
                merged_clustering_data,
                average_silhouette_score,
                kmeans_metrics,
                centroids_avg_similarities,
            )
        except Exception as e:
            logging.error("Error in clustering analysis: %s", e)
            raise

    def annotate_data(self, merged_data, pretrained_data):
        try:
            data_annotator = DataAnnotator(
                self.tokenization_outputs.train_subwords,
                merged_data,
                pretrained_data,
                self.data_manager.data.get("train"),
                self.model_outputs.test,
                self.data_manager.labels_map,
            )
            return data_annotator.annotate_all()
        except Exception as e:
            logging.error("Error in data annotation:%s", e)
            raise

    def run(self):
        start_time = time.time()
        analysis_df, flat_data, pretrained_coordinates = self.extract_analysis_data()
        (
            merged_clustering_data,
            average_silhouette_score,
            kmeans_metrics,
            centroids_avg_similarities,
        ) = self.perform_clustering(analysis_df, flat_data)
        analysis_data = self.annotate_data(
            merged_clustering_data, pretrained_coordinates
        )
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info("Analysis workflow execution time: %s seconds", execution_time)
        return (
            analysis_data,
            average_silhouette_score,
            kmeans_metrics,
            centroids_avg_similarities,
        )

    def generate_train_df(self):
        try:
            # Extract data using the DataExtractor
            train_extractor = DataExtractor(
                model_outputs=self.model_outputs.get_split("train"),
                transformer=self.transformer,
            )
            tr_df = train_extractor.to_df()

            # Copy the inverse labels map from the data manager and modify it
            inv_labels_map = self.data_manager.inv_labels_map.copy()
            inv_labels_map[-100] = "IGNORED"

            # Map the labels in the DataFrame
            tr_df["true_labels"] = tr_df["labels"].map(inv_labels_map)

            # Return the modified DataFrame
            return tr_df
        except Exception as e:
            logging.error("Error in generating training data frame: %s", e)
            raise


class Entity:
    def __init__(self, outputs):
        self.y_true = outputs["y_true"]
        self.y_pred = outputs["y_pred"]
        true = self.get_entities(self.y_true)
        pred = self.get_entities(self.y_pred)
        self.seq_true, self.seq_pred = self.compute_entity_location(true, pred)

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

    def generate_entity_confusion_data(
        self,
    ):
        confusion_data = pd.DataFrame()
        confusion_data["true_entity"] = self.seq_true
        confusion_data["pred_entity"] = self.seq_pred
        return confusion_data


class AttentionSimilarity:
    def __init__(
        self,
        device: torch.device,
        model1: AutoModel,
        model2: AutoModel,
        tokenizer,
        preprocessor=None,
    ):
        """
        Initialize AttentionSimilarity with models, tokenizer, and preprocessor.

        Args:
            device (torch.device): The device to run the computations on.
            model1 (AutoModel): The pretrained model.
            model2 (AutoModel): The fine-tuned model.
            tokenizer: Tokenizer to process input data.
            preprocessor: Preprocessor to preprocess input data.
        """
        self.device = device
        self.model1 = model1
        self.model2 = model2
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

    def compute_similarity(self, example: List[str]) -> List[List[float]]:
        """
        Compute the similarity between attention heads of two models for a given example.

        Args:
            example (List[str]): The input example as a list of tokens.

        Returns:
            List[List[float]]: Similarity scores for each attention head across all layers.
        """
        sentence = " ".join(example)
        inputs = self.tokenizer.encode_plus(
            self.preprocessor.preprocess(sentence) if self.preprocessor else sentence,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            model1_att = self.model1(**inputs).attentions
            model2_att = self.model2(**inputs).attentions

        model1_mat = np.array([atten[0].cpu().numpy() for atten in model1_att])
        model2_mat = np.array([atten[0].cpu().numpy() for atten in model2_att])

        scores = [
            [
                1
                - distance.cosine(
                    model1_mat[layer][head].flatten(), model2_mat[layer][head].flatten()
                )
                for head in range(12)
            ]
            for layer in range(12)
        ]
        return scores


class TrainingImpact:
    def __init__(
        self,
        data: List,
        tokenization_outputs: TokenizationWorkflowManager,
        pretrained_model: AutoModel,
        model: AutoModel,
    ):
        """
        Initialize TrainingImpact with mode, outputs, model path, and model.

        Args:
            mode (str): The mode of the data (e.g., 'train', 'test').
            outputs: The outputs containing data and dataloaders.
            model_path (str): Path to the pretrained model.
            model (AutoModel): The fine-tuned model.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.tokenizer = tokenization_outputs.tokenizer
        self.preprocessor = tokenization_outputs.preprocessor
        self.pretrained_model = pretrained_model.to(self.device)
        self.fine_tuned_model = model.to(self.device)
        self.attention_impact = AttentionSimilarity(
            self.device,
            self.pretrained_model,
            self.fine_tuned_model,
            self.tokenizer,
            self.preprocessor,
        )

    def compute_attention_similarities(self, n_examples: int = None) -> px.imshow:
        """
        Compute attention similarities for multiple examples and visualize the average similarity.

        Returns:
            px.imshow: Plotly heatmap figure of the average similarity scores.
        """
        logging.info("Computing attention similarities")
        sampled_data = random.sample(
            self.data, n_examples if n_examples else len(self.data)
        )

        # Compute similarities and display progress
        similarities = [
            self.attention_impact.compute_similarity(example["words"])
            for example in tqdm(sampled_data, desc="Computing attention similarities")
        ]
        self.similarity_matrix = np.array(similarities).mean(0)
        change_fig = px.imshow(
            self.similarity_matrix,
            color_continuous_scale='RdBu_r',  # Set color scale here
            template="plotly_white",
            labels=dict(x="Heads", y="Layers", color="Similarity Score"),
        )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        return change_fig

    def compute_example_similarities(self, id: int) -> None:
        """
        Compute attention similarities for a specific example and visualize the similarity.

        Args:
            id (int): The index of the example to visualize.
        """
        scores = self.attention_impact.compute_similarity(self.data[id]["words"])
        change_fig = px.imshow(
            scores,
            labels=dict(x="Heads", y="Layers", color="Similarity Score"),
        )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        change_fig.show()

    def extract_weights(self, layer: torch.nn.Module) -> torch.Tensor:
        """
        Extract weights from a model layer.

        Args:
            layer (torch.nn.Module): The model layer.

        Returns:
            torch.Tensor: The concatenated query, key, and value weights.
        """
        return torch.cat(
            [
                layer.attention.self.query.weight,
                layer.attention.self.key.weight,
                layer.attention.self.value.weight,
            ],
            dim=0,
        )

    def compare_weights(self) -> px.imshow:
        """
        Compare weights of the pretrained and fine-tuned models and visualize the differences.

        Returns:
            px.imshow: Plotly heatmap figure of the weight differences.
        """
        logging.info("Comparing weights")
        num_layers = len(self.pretrained_model.encoder.layer)
        num_heads = self.pretrained_model.config.num_attention_heads
        self.weight_diff_matrix = np.zeros((num_layers, num_heads))

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
                    .detach()
                    .cpu()
                    .numpy()
                )
                weight_diff = 1 - distance.cosine(
                    pretrained_weight.flatten(), fine_tuned_weight.flatten()
                )
                self.weight_diff_matrix[layer, head] = weight_diff

        return self.visualize_weight_difference(self.weight_diff_matrix)

    def visualize_weight_difference(self, weight_diff_matrix: np.ndarray) -> px.imshow:
        """
        Visualize weight differences across layers and heads.

        Args:
            weight_diff_matrix (np.ndarray): Matrix of weight differences.

        Returns:
            px.imshow: Plotly heatmap figure of the weight differences.
        """
        change_fig = px.imshow(
            weight_diff_matrix,
            color_continuous_scale='RdBu_r',  # Set color scale here
            template="plotly_white",
            labels=dict(x="Heads", y="Layers", color="Similarity Score"),
        )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        return change_fig
