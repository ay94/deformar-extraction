
import pandas as pd
import numpy as np
import math
import logging
from collections import defaultdict
from tqdm.autonotebook import tqdm
import torch
from dataclasses import dataclass, field, asdict

from umap import UMAP
from sklearn.preprocessing import normalize
import numpy as np
from typing import Optional

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
                    alignment_locations[sentence_id].append((token_label_id, token_label))
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




class DataTransformer:
    def __init__(self, umap_config):
        """
        Initialize the DataTransformer with configuration for UMAP.
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
class DataExtractor:
    tokenization_outputs: list
    model_outputs: list
    aligner: LabelAligner
    transformer: DataTransformer
    last_hidden_states: torch.Tensor = field(init=False)
    labels: torch.Tensor = field(init=False)
    losses: torch.Tensor = field(init=False)
    token_ids: torch.Tensor = field(init=False)
    words: list = field(init=False)
    tokens: list = field(init=False)
    word_pieces: list = field(init=False)
    core_tokens: list = field(init=False)
    true_labels: list = field(init=False)
    pred_labels: list = field(init=False, default_factory=list)
    sentence_ids: list = field(init=False)
    token_positions: list = field(init=False)
    token_selector_id: list = field(init=False)
    agreements: list = field(init=False)
    x: list = field(init=False)
    y: list = field(init=False)

    def __post_init__(self):
      self.process_data()

    def process_data(self):
        self.extract_features()
        self.align_labels()
        self.apply_umap()
    def extract_features(self):
        """
        Extract features from the model outputs.
        """
        self.last_hidden_states = torch.concat([s.last_hidden_states for s in self.model_outputs])
        self.labels = torch.concat([s.labels for s in self.model_outputs])
        self.losses = torch.concat([s.losses for s in self.model_outputs])
        self.token_ids = torch.concat([s.input_ids for s in self.model_outputs])
        self.words = [word for sentence in self.tokenization_outputs for word in sentence.words_df]
        self.tokens = [token for sentence in self.tokenization_outputs for token in sentence.tokens_df]
        self.word_pieces = [wp for sentence in self.tokenization_outputs for wp in sentence.word_pieces_df]
        self.core_tokens = [ct for sentence in self.tokenization_outputs for ct in sentence.core_tokens_df]
        self.true_labels = [label for sentence in self.tokenization_outputs for label in sentence.labels_df]
        self.sentence_ids = [index for sentence in self.tokenization_outputs for index in sentence.sentence_index_df]
        self.token_positions = [position for sentence in self.tokenization_outputs for position in range(len(sentence.tokens_df))]
        self.token_selector_id = [
            f"{core_token}@#{token_position}@#{sentence_index}"
            for core_token, token_position, sentence_index in
            zip(self.core_tokens, self.token_positions, self.sentence_ids)
        ]


    def align_labels(self):
        aligned_labels = self.aligner.align_labels()
        self.pred_labels = [label for sentence in aligned_labels for label in sentence]
        self.agreements = np.array(self.true_labels) == np.array(self.pred_labels)

    def apply_umap(self):
        coordinates = self.transformer.apply_umap(self.last_hidden_states)
        self.x, self.y = coordinates

    def to_dict(self):
        analysis_data = {
            "labels": self.labels,
            "losses": self.losses,
            "token_ids": self.token_ids,
            "words": self.words,
            "tokens": self.tokens,
            "word_pieces": self.word_pieces,
            "core_tokens": self.core_tokens,
            "true_labels": self.true_labels,
            "pred_labels": self.pred_labels,
            "sentence_ids": self.sentence_ids,
            "token_positions": self.token_positions,
            "token_selector_id": self.token_selector_id,
            "agreements": self.agreements,
            "x": self.x,
            "y": self.y
        }
        return analysis_data


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
        
        true, pred = row['true_labels'], row['pred_labels']

        # Check if both labels are exactly the same, including 'O'
        if true == pred:
            return 'Correct'

        # Check for 'O' to avoid IndexError when accessing parts of the string
        elif 'O' in [true, pred]:
            return 'Chunk'  # 'Chunk' error since one is 'O' and the other isn't

        # Extract parts after the dash and compare
        elif true.split('-')[1] != pred.split('-')[1]:
            return 'Entity'

        # If not correct and no entity type mismatch, it must be a chunk error
        else:
            return 'Chunk'
        
    @staticmethod
    def softmax(logits):
        """Apply softmax to logits to get probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)






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
                train_label_freq[token][tag['tag']] += 1

        # Initialize list to hold consistency information
        consistency_info = {
            'consistency_count': [],
            'inconsistency_count': [],
            'total_train_occurrences': []
        }

        # Calculate consistency information for each token in analysis dataframe
        for row in tqdm(analysis_df.itertuples(index=False), total=len(analysis_df), desc="Calculate Consistency"):
            token = getattr(row, 'core_tokens')
            test_label = getattr(row, 'true_labels')
            label_dict = train_label_freq[token]
            consistency_count = label_dict.get(test_label, 0)
            total_train_occurrences = sum(label_dict.values())
            inconsistency_count = total_train_occurrences - consistency_count
            
            consistency_info['consistency_count'].append(consistency_count)
            consistency_info['inconsistency_count'].append(inconsistency_count)
            consistency_info['total_train_occurrences'].append(total_train_occurrences)
        
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
            return pd.DataFrame(columns=["token", "tag"])
        subwords_counter = [(subword, tag['tag']) for subword, tags in subword_index.items() for tag in tags]
        return pd.DataFrame(subwords_counter, columns=["token", "tag"])

    @staticmethod
    def calculate_tag_counts(subwords_df):
        """
        Calculates the count of each tag for tokens and computes initial probability contributions.

        Args:
            subwords_df (DataFrame): DataFrame containing token and tag information.

        Returns:
            DataFrame: Updated DataFrame with probability and initial entropy contribution for each tag.
        """
        token_counts = subwords_df.groupby(["token", "tag"]).size().reset_index(name='count')
        token_total = token_counts.groupby(["token"])['count'].sum().reset_index(name='total')
        token_specific_prob = pd.merge(token_counts, token_total, on="token")
        token_specific_prob['probability'] = token_specific_prob['count'] / token_specific_prob['total']
        token_specific_prob['entropy_contribution'] = token_specific_prob['probability'].apply(UtilityFunctions.per_token_entropy)
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
        num_tags_per_token = subwords_df.groupby('token')['tag'].nunique().reset_index().rename(columns={'tag': 'num_tags'})
        num_tags_per_token['token_max_entropy'] = num_tags_per_token['num_tags'].apply(UtilityFunctions.max_entropy)
        
        token_entropy = token_specific_prob.groupby("token")['entropy_contribution'].sum().reset_index().rename(columns={'entropy_contribution': 'local_token_entropy'})
        token_entropy = pd.merge(token_entropy, num_tags_per_token[['token', 'token_max_entropy']], on="token")
        
        dataset_total = len(subwords_df)
        dataset_prob = subwords_df.groupby(["token", "tag"]).size().div(dataset_total).reset_index(name='dataset_probability')
        dataset_prob['dataset_entropy_contribution'] = dataset_prob['dataset_probability'].apply(UtilityFunctions.per_token_entropy)
        dataset_entropy = dataset_prob.groupby("token")['dataset_entropy_contribution'].sum().reset_index().rename(columns={'dataset_entropy_contribution': 'dataset_token_entropy'})

        entropy_df = pd.merge(token_entropy, dataset_entropy, on="token")
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
        entropy_df = TokenEntropyCalculator.calculate_entropy(subwords_df, token_specific_prob)
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
            for w, t in zip(s['words'], s['tags']):
                wordsDict[w].append({'tag': t, 'sentence': i})

        return pd.DataFrame([
            {"word": word, "tag": tag["tag"], "sentence_index": tag["sentence"]}
            for word, tags in wordsDict.items()
            for tag in tags
        ])

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
        num_tags_per_word = words_df.groupby('word')['tag'].nunique().reset_index().rename(columns={'tag': 'num_tags'})
        num_tags_per_word['word_max_entropy'] = num_tags_per_word['num_tags'].apply(UtilityFunctions.max_entropy)

        # Local entropy: normalize by the number of occurrences of each word
        local_probabilities = words_df.groupby(["word", "tag"]).size().div(words_df.groupby("word").size(), axis=0).reset_index(name='local_probability')
        local_probabilities['local_entropy_contribution'] = local_probabilities['local_probability'].apply(UtilityFunctions.per_token_entropy)
        local_entropy_df = local_probabilities.groupby("word")['local_entropy_contribution'].sum().reset_index().rename(columns={'local_entropy_contribution': 'local_word_entropy'})
        local_entropy_df = pd.merge(local_entropy_df, num_tags_per_word[['word', 'word_max_entropy']], on='word')

        # Dataset-wide entropy: normalize by the total number of tags in the dataset
        dataset_probabilities = words_df.groupby(["word", "tag"]).size().div(len(words_df)).reset_index(name='dataset_probability')
        dataset_probabilities['dataset_entropy_contribution'] = dataset_probabilities['dataset_probability'].apply(UtilityFunctions.per_token_entropy)
        dataset_entropy_df = dataset_probabilities.groupby("word")['dataset_entropy_contribution'].sum().reset_index().rename(columns={'dataset_entropy_contribution': 'dataset_word_entropy'})

        # Merge local and dataset entropy dataframes
        entropy_df = pd.merge(local_entropy_df, dataset_entropy_df, on='word')

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
    def __init__(self, model_outputs, data):
        self.model_outputs = model_outputs
        self.data = data

    
    @staticmethod
    def calculate():
        """Extract prediction entropy from logits."""
        token_logits = []
        for sentence in self.model_outputs:
            for token in sentence.logits:
                token_logits.append(token.tolist())

        logits_matrix = np.array(token_logits)
        probabilities_matrix = UtilityFunctions.softmax(logits_matrix)
        prediction_entropy = UtilityFunctions.entropy(probabilities_matrix)
        prediction_confidence = [max(prob_scores) for prob_scores in probabilities_matrix]
        prediction_variability = [np.std(prob_scores) for prob_scores in probabilities_matrix]
        prediction_analysis = pd.DataFrame(probabilities_matrix, columns=self.data["labels_map"])
        
        prediction_analysis["prediction_entropy"] = prediction_entropy
        prediction_analysis["prediction_max_entropy"] = UtilityFunctions.max_entropy(len(self.data["labels_map"]))
        prediction_analysis["confidence"] = prediction_confidence
        prediction_analysis["variability"] = prediction_variability
        return prediction_analysis



class DataAnnotator:
    def __init__(self, subwords, analysis_data, train_data):
        self.subwords = subwords
        self.train_data = train_data
        self.analysis_data = analysis_data
        self.analysis_df = None

    def annotate_tokenization_rate(self):
        """Annotate the tokenization rate for each word based on the number of subwords."""
        self.analysis_df['tokenization_rate'] = self.analysis_df['word_pieces'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    def annotate_consistency(self):
        logging.info("Annotating consistency...")
        consistency_results = TokenConsistencyCalculator.calculate(self.subwords, self.analysis_df)
        self.analysis_df = pd.concat([self.analysis_df, consistency_results], axis=1)
    
    def annotate_token_entropy(self):
        logging.info("Annotating token entropy...")
        token_entropy = TokenEntropyCalculator.calculate(self.subwords)
        self.analysis_df = self.analysis_df.merge(token_entropy, left_on='core_tokens', right_on='token', how='left')
        self.analysis_df['local_token_entropy'] = self.analysis_df['local_token_entropy'].fillna(-1)
        self.analysis_df['token_max_entropy'] = self.analysis_df['token_max_entropy'].fillna(-1)
        self.analysis_df['dataset_token_entropy'] = self.analysis_df['dataset_token_entropy'].fillna(-1)
        self.analysis_df.drop('token', axis=1, inplace=True)
    
    def annotate_word_entropy(self):
        logging.info("Annotating word entropy...")
        word_entropy = WordEntropyCalculator.calculate(self.train_data)
        self.analysis_df = self.analysis_df.merge(word_entropy, left_on='words', right_on='word', how='left')
        self.analysis_df['local_word_entropy'] = self.analysis_df['local_word_entropy'].fillna(-1)
        self.analysis_df['word_max_entropy'] = self.analysis_df['word_max_entropy'].fillna(-1)
        self.analysis_df['dataset_word_entropy'] = self.analysis_df['dataset_word_entropy'].fillna(-1)
        self.analysis_df.drop('word', axis=1, inplace=True)
    
    def annotate_error_types(self):
        logging.info("Annotating error types...")
        self.analysis_df['error_type'] = self.analysis_df.apply(UtilityFunctions.error_type, axis=1)
    
    def annotate_entity(self):
        logging.info("Annotating entity...")
        self.analysis_df["tr_entity"] = self.analysis_df["true_labels"].apply(
            lambda x: x if x in ["[CLS]", "[SEP]", "IGNORED"] else x.split("-")[-1]
        )
        self.analysis_df["pr_entity"] = self.analysis_df["pred_labels"].apply(
            lambda x: x if x in ["[CLS]", "[SEP]", "IGNORED"] else x.split("-")[-1]
        )
    def annotate_prediction_entropy(self):
        prediction_entropy_df = PredictionEntropyCalculator.calculate()
        self.analysis_df.merge(prediction_entropy_df, left_index=True, right_index=True, how='left')

    def annotate_all(self):
        self.analysis_df = self.analysis_data.copy()
        self.annotate_consistency()
        self.annotate_token_entropy()
        self.annotate_word_entropy()
        self.annotate_tokenization_rate()
        self.annotate_entity()
        self.annotate_error_types()
        self.annotate_prediction_entropy()
        return self.analysis_df
    
    

# from collections import Counter, defaultdict
# import math
# import torch
# from sklearn.preprocessing import normalize
# import numpy as np
# from umap import UMAP
# import ast
# import copy

# import pandas as pd



# class LabelAligner:
#     def __init__(self, results, tokenized_sentences):
#         """
#         Initialize the DataPreparation class.

#         Args:
#             batches: The batches object containing batch data.
#             tokenization_outputs: The tokenization outputs object.
#         """
#         self.tokenized_sentences = tokenized_sentences
#         self.predictions = results.entity_outputs['y_pred'].copy()

#     def label_alignment(self):
#         """
#         Create a map for label alignment based on tokenization outputs.

#         Returns:
#             alignment_map: A dictionary mapping sentence IDs to token indices and tokens.
#         """
#         alignment_map = defaultdict(list)
#         for sentence_id, sentence in enumerate(self.tokenized_sentences):
#             for token_label_id, token_label in enumerate(sentence.labels_df):
#                 if token_label in ["[CLS]", "[SEP]", "IGNORED"]:
#                     alignment_map[sentence_id].append((token_label_id, token_label))
#         return alignment_map

#     def change_preds(self):
#         """
#         Modify predictions based on the alignment map.

#         Args:
#             preds: A list of predictions.
#             pred_map: A dictionary mapping sentence IDs to token indices and tokens.

#         Returns:
#             modified_preds: A list of modified predictions.
#         """
#         self.alignment_map = self.label_alignment()
#         modified_predictions = []
#         for sentence_id, original_sentence in enumerate(self.predictions):
#             sentence = original_sentence[:]  # Create a shallow copy of the list
#             for index, token in self.alignment_map[sentence_id]:
#                 # no need to shift as the indices were calculated according to the tokenized version in the truth labels
#                 sentence.insert(index, token)
#             modified_predictions.append(sentence)
#         return modified_predictions
# # Data Utility
# class UtilityFunctions:
#     @staticmethod
#     def entropy(probabilities):
#         return -sum(p * math.log2(p) for p in probabilities.values())

#     @staticmethod
#     def label_probabilities(dataset):
#         label_counts = defaultdict(Counter)
#         for token, label in dataset:
#             label_counts[token][label] += 1
#         probabilities = {
#             token: {
#                 label: count / sum(counts.values()) for label, count in counts.items()
#             }
#             for token, counts in label_counts.items()
#         }
#         return probabilities



# # Analysis Computations
# class AmbiguityComputer:
#     @staticmethod
#     def compute_consistency(analysis_df, subwords_locations):
#         # Creating a DataFrame from subwords_locations for easier manipulation
#         subwords_df = pd.DataFrame(
#             [
#                 {"first_token": token, "tag": info["tag"], "count": 1}
#                 for token, tags in subwords_locations.items()
#                 for info in tags
#             ]
#         )

#         # Summing up counts by first_token and tag
#         tag_counts = subwords_df.groupby(["first_token", "tag"]).sum().reset_index()

#         # Merging with analysis_df
#         merged_df = analysis_df.merge(
#             tag_counts, how="left", left_on="first_tokens", right_on="first_token"
#         )

#         # Determine consistency
#         merged_df["is_consistent"] = merged_df["tag"] == merged_df["truth"]
#         consistency_counts = merged_df.pivot_table(
#             index="index", columns="is_consistent", values="count", fill_value=0
#         )

#         # Adding to original DataFrame
#         analysis_df["first_tokens_consistency"] = consistency_counts[True]
#         analysis_df["first_tokens_inconsistency"] = consistency_counts[False]

#         return analysis_df

#     @staticmethod
#     def token_ambiguity(analysis_df, subwords_counter):
#         # Convert counter to DataFrame
#         subwords_df = pd.DataFrame(subwords_counter, columns=["token", "tag"])
#         probabilities = (
#             subwords_df.groupby("token")["tag"]
#             .value_counts(normalize=True)
#             .rename("probability")
#             .reset_index()
#         )

#         # Calculate entropy
#         probabilities["entropy_contribution"] = -probabilities["probability"] * np.log2(
#             probabilities["probability"]
#         )
#         entropy_df = (
#             probabilities.groupby("token")["entropy_contribution"].sum().reset_index()
#         )

#         # Merge back to analysis_df
#         analysis_df = analysis_df.merge(
#             entropy_df, how="left", left_on="first_tokens", right_on="token"
#         )
#         analysis_df["token_entropy"] = analysis_df["entropy_contribution"].fillna(-1)
#         analysis_df.drop(["token", "entropy_contribution"], axis=1, inplace=True)

#         return analysis_df["token_entropy"].values

#     @staticmethod
#     def word_ambiguity(analysis_df, wordsDict):
#         # Flatten the dictionary into a DataFrame
#         words_df = pd.DataFrame(
#             [
#                 {"word": word, "tag": tag["tag"]}
#                 for word, tags in wordsDict.items()
#                 for tag in tags
#             ]
#         )
#         probabilities = (
#             words_df.groupby("word")["tag"]
#             .value_counts(normalize=True)
#             .rename("probability")
#             .reset_index()
#         )

#         # Calculate entropy
#         probabilities["entropy_contribution"] = -probabilities["probability"] * np.log2(
#             probabilities["probability"]
#         )
#         entropy_df = (
#             probabilities.groupby("word")["entropy_contribution"].sum().reset_index()
#         )

#         # Merge back to analysis_df
#         analysis_df = analysis_df.merge(
#             entropy_df, how="left", left_on="words", right_on="word"
#         )
#         analysis_df["word_entropy"] = analysis_df["entropy_contribution"].fillna(-1)
#         analysis_df.drop(["word", "entropy_contribution"], axis=1, inplace=True)

#         return analysis_df["word_entropy"].values

#     # def compute_consistency(analysis_df, subwords_locations):
#     #     """
#     #     Compute consistency and inconsistency scores for tokens in the analysis dataframe.
#     #
#     #     Args:
#     #         analysis_df: The dataframe containing analysis data.
#     #         subwords_locations: A dictionary of subwords and their corresponding tags and sentence indices.
#     #
#     #     Returns:
#     #         analysis_df: The updated dataframe with consistency and inconsistency scores.
#     #     """
#     #     consistent = []
#     #     inconsistent = []
#     #     for i in tqdm(range(len(analysis_df))):
#     #         consistent_count = []
#     #         inconsistent_count = []
#     #         for t, count in Counter(
#     #                 [tok['tag'] for tok in subwords_locations[analysis_df.iloc[i]['first_tokens']]]
#     #         ).items():
#     #             if t == analysis_df.iloc[i]['truth']:
#     #                 consistent_count.append(count)
#     #             else:
#     #                 inconsistent_count.append(count)
#     #         consistent.append(sum(consistent_count))
#     #         inconsistent.append(sum(inconsistent_count))
#     #     analysis_df['first_tokens_consistency'] = consistent
#     #     analysis_df['first_tokens_inconsistency'] = inconsistent
#     #     return analysis_df

#     # def token_ambiguity(self, analysis_df):
#     #     """
#     #     Calculate token ambiguity for the analysis dataframe.
#     #
#     #     Args:
#     #         analysis_df: The dataframe containing analysis data.
#     #
#     #     Returns:
#     #         token_entropy: A list of token entropy values.
#     #     """
#     #     subwords_counter = [(subword, tag['tag']) for subword, tag_dis in self.subwords.items() for tag in tag_dis]
#     #     probabilities = UtilityFunctions.label_probabilities(subwords_counter)
#     #     # token_entropies = {token: abs(UtilityFunctions.entropy(probs)) for token, probs in probabilities.items()}
#     #     token_entropies = {token: UtilityFunctions.entropy(probs) for token, probs in probabilities.items()}
#     #     computed_token_entropy = pd.DataFrame(token_entropies.items(), columns=['first_tokens', 'entropy'])
#     #
#     #     token_entropy = []
#     #     for tk in tqdm(analysis_df['first_tokens']):
#     #         token_data = computed_token_entropy[computed_token_entropy['first_tokens'] == tk]
#     #         if not token_data.empty:
#     #             token_entropy.append(token_data['entropy'].values[0])
#     #         else:
#     #             token_entropy.append(-1)
#     #     return token_entropy

#     # def word_ambiguity(self, analysis_df):
#     #     """
#     #     Calculate word ambiguity for the analysis dataframe.
#     #
#     #     Args:
#     #         analysis_df: The dataframe containing analysis data.
#     #
#     #     Returns:
#     #         word_entropy: A list of word entropy values.
#     #     """
#     #     wordsDict = defaultdict(list)
#     #     for i, sen in enumerate(self.dataset_outputs.data['train']):
#     #         for w, t in zip(sen[1], sen[2]):
#     #             wordsDict[w].append({'tag': t, 'sentence': i})
#     #
#     #     words_counter = [(word, tag['tag']) for word, tag_dis in wordsDict.items() for tag in tag_dis]
#     #     probabilities = UtilityFunctions.label_probabilities(words_counter)
#     #     # word_entropies = {token: abs(UtilityFunctions.entropy(probs)) for token, probs in probabilities.items()}
#     #     word_entropies = {token: UtilityFunctions.entropy(probs) for token, probs in probabilities.items()}
#     #     computed_word_entropy = pd.DataFrame(word_entropies.items(), columns=['words', 'entropy'])
#     #
#     #     word_entropy = []
#     #     for tk in tqdm(analysis_df['words']):
#     #         token_data = computed_word_entropy[computed_word_entropy['words'] == tk]
#     #         if not token_data.empty:
#     #             word_entropy.append(token_data['entropy'].values[0])
#     #         else:
#     #             word_entropy.append(-1)
#     #     return word_entropy


# class DataExtractor:
#     def __init__(self, preparation, outputs, results):
#         """
#         Initialize the FlatDataExtractor class.

#         Args:
#             preparation: The DataPreparation object.
#             outputs: The outputs object containing batch outputs.
#             results: The results object containing model results.
#         """
#         self.preparation = preparation
#         self.outputs = outputs
#         self.results = results

#     def extract_flat_data(self):
#         """
#         Extract and flatten the data from the preparation, outputs, and results.

#         Returns:
#             A tuple of flattened data elements.
#         """
#         flat_last_hidden_state = torch.cat(
#             [
#                 hidden_state[ids != 0]
#                 for batch in self.preparation.batches
#                 for ids, hidden_state in zip(
#                     batch["input_ids"], batch["last_hidden_state"]
#                 )
#             ]
#         )
#         flat_labels = torch.cat(
#             [
#                 labels[ids != 0]
#                 for batch in self.preparation.batches
#                 for ids, labels in zip(batch["input_ids"], batch["labels"])
#             ]
#         )
#         flat_losses = torch.cat([losses for losses in self.outputs.aligned_losses])
#         flat_words = [
#             tok for sen in self.preparation.tokenization.words_df for tok in sen
#         ]
#         flat_tokens = [
#             tok for sen in self.preparation.tokenization.tokens for tok in sen
#         ]
#         flat_wordpieces = [
#             str(tok)
#             for sen in self.preparation.tokenization.wordpieces_df
#             for tok in sen
#         ]
#         flat_first_tokens = [
#             tok for sen in self.preparation.tokenization.first_tokens_df for tok in sen
#         ]
#         token_id_strings = [
#             f"{tok}@#{tok_id}@#{i}"
#             for sen, sen_id in zip(
#                 self.preparation.tokenization.first_tokens_df,
#                 self.preparation.tokenization.sentence_ind_df,
#             )
#             for i, (tok, tok_id) in
#             # for handling special token, get all sentence ids except the first and last then concatenate the first and last
#             enumerate(
#                 zip(
#                     sen,
#                     list(set(sen_id[1:-1])) + sen_id[1:-1] + list(set(sen_id[1:-1])),
#                 )
#             )
#         ]
#         flat_true_labels = [
#             tok for sen in self.preparation.tokenization.labels_df for tok in sen
#         ]

#         prediction_map = self.preparation.label_alignment()
#         modified_predictions = self.preparation.change_preds(
#             self.results.seq_output["y_pred"].copy(), prediction_map
#         )
#         flat_predictions = [tok for sen in modified_predictions for tok in sen]
#         flat_sentence_ids = [
#             tok
#             for sen in self.preparation.tokenization.sentence_ind_df
#             for tok in list(set(sen[1:-1])) + sen[1:-1] + list(set(sen[1:-1]))
#         ]
#         flat_agreements = np.array(flat_true_labels) == np.array(flat_predictions)

#         token_ids = [
#             int(w)
#             for batch in self.preparation.batches
#             for ids in batch["input_ids"]
#             for w in ids[ids != 0]
#         ]
#         word_ids = [
#             w_id
#             for batch in self.preparation.batches
#             for ids in batch["input_ids"]
#             for w_id in range(len(ids[ids != 0]))
#         ]

#         return (
#             flat_last_hidden_state,
#             flat_labels,
#             flat_losses,
#             flat_words,
#             flat_tokens,
#             flat_wordpieces,
#             flat_first_tokens,
#             token_id_strings,
#             flat_true_labels,
#             flat_predictions,
#             flat_sentence_ids,
#             flat_agreements,
#             token_ids,
#             word_ids,
#         )


# class Config:
#     def __init__(self):
#         self.umap_config = UMAPConfig()
#         # Add other configuration sections as needed


# class UMAPConfig:
#     def __init__(self):
#         # UMAP parameters
#         self.n_neighbors = 15
#         self.min_dist = 0.1
#         self.metric = "cosine"
#         self.random_state = 1
#         self.verbose = True
#         self.normalize_embeddings = False

#     def set_params(
#         self, n_neighbors=None, min_dist=None, metric=None, normalize_embeddings=None
#     ):
#         """Set parameters for UMAP if provided."""
#         if n_neighbors is not None:
#             self.n_neighbors = n_neighbors
#         if min_dist is not None:
#             self.min_dist = min_dist
#         if metric is not None:
#             self.metric = metric
#         if normalize_embeddings is not None:
#             self.normalize_embeddings = normalize_embeddings


# class DataTransformer:
#     def __init__(self, umap_config):
#         """
#         Initialize the DataTransformer class with UMAP parameters and normalization option.

#         Args:
#             umap_config: configuration of umap paramaters
#         """

#         self.umap_model = UMAP(
#             n_neighbors=umap_config.n_neighbors,
#             min_dist=umap_config.min_dist,
#             metric=umap_config.metric,
#             random_state=umap_config.random_state,
#             verbose=umap_config.verbose,
#         )
#         self.normalize_embeddings = umap_config.normalize_embeddings

#     def apply_umap(self, flat_states):
#         """
#         Apply UMAP dimensionality reduction to the given flat states.

#         Args:
#             flat_states: The high-dimensional data to reduce.

#         Returns:
#             The reduced data.
#         """
#         if self.normalize_embeddings:
#             flat_states = normalize(flat_states, axis=1)
#         return self.umap_model.fit_transform(flat_states).transpose()

#     @staticmethod
#     def transform_to_dataframe(flat_data, layer_reduced):
#         (
#             flat_last_hidden_state,
#             flat_labels,
#             flat_losses,
#             flat_words,
#             flat_tokens,
#             flat_wordpieces,
#             flat_first_tokens,
#             token_id_strings,
#             flat_true_labels,
#             flat_predictions,
#             flat_sentence_ids,
#             flat_agreements,
#             token_ids,
#             word_ids,
#         ) = flat_data

#         analysis_df = pd.DataFrame(
#             {
#                 "token_id": token_ids,
#                 "word_id": word_ids,
#                 "sentence_id": flat_sentence_ids,
#                 "token_id_string": token_id_strings,
#                 "label_ids": flat_labels.tolist(),
#                 "words": flat_words,
#                 "wordpieces": flat_wordpieces,
#                 "tokens": flat_tokens,
#                 "first_tokens": flat_first_tokens,
#                 "truth": flat_true_labels,
#                 "pred": flat_predictions,
#                 "agreement": flat_agreements,
#                 "losses": flat_losses.tolist(),
#                 "x": layer_reduced[0],
#                 "y": layer_reduced[1],
#             }
#         )

#         return analysis_df


# class DataAnnotator:
#     def __init__(self, subwords, dataset_outputs):
#         """
#         Initialize the AnalysisComputations class.

#         Args:
#             subwords: A dictionary containing subword information.
#             dataset_outputs: The dataset outputs object containing data for analysis.
#         """
#         self.subwords = subwords
#         self.dataset_outputs = dataset_outputs

#     @staticmethod
#     def annotate_tokenization_rate(analysis_df):
#         """
#         Annotate the tokenization rate (fertility) for each word in the dataframe.

#         Args:
#             analysis_df (pd.DataFrame): DataFrame containing analysis data.

#         Returns:
#             pd.DataFrame: Updated DataFrame with 'tokenization_rate' column.
#         """
#         num_tokens = []
#         for wps in analysis_df["wordpieces"]:
#             try:
#                 # Evaluate the string representation of the word pieces to a list
#                 word_pieces = ast.literal_eval(wps)
#                 num_tokens.append(len(word_pieces))
#             except ValueError:
#                 num_tokens.append(1)
#         analysis_df["tokenization_rate"] = num_tokens
#         return analysis_df

#     @staticmethod
#     def annotate_first_token_frequencies(analysis_df, subword_locations):
#         """
#         Add a column 'first_tokens_freq' to the DataFrame with the frequency of each first token.

#         Args:
#             analysis_df (pd.DataFrame): DataFrame containing analysis data.
#             subword_locations (dict): Dictionary containing subword locations.

#         Returns:
#             pd.DataFrame: Updated DataFrame with 'first_tokens_freq' column.
#         """
#         if "first_tokens" in analysis_df.columns:
#             subword_freq_series = pd.Series(
#                 {k: len(v) for k, v in subword_locations.items()}
#             )
#             analysis_df["first_tokens_freq"] = (
#                 analysis_df["first_tokens"]
#                 .map(subword_freq_series)
#                 .fillna(0)
#                 .astype(int)
#             )
#         else:
#             raise KeyError("The DataFrame does not contain a 'first_tokens' column.")
#         return analysis_df

#     @staticmethod
#     def error_type(row):
#         """
#         Determine the type of error for a given row in the analysis dataframe.

#         Args:
#             row: A row from the analysis dataframe.

#         Returns:
#             str: The type of error ('Correct', 'Entity', or 'Chunk').
#         """
#         true, pred = row["truth"], row["pred"]
#         if true == pred:
#             return "Correct"
#         elif true[1:] != pred[1:]:
#             return "Entity"
#         else:
#             return "Chunk"

#     def annotate_additional_info(self, analysis_df):
#         print("Compute Consistency")
#         analysis_df = AmbiguityComputer.compute_consistency(
#             analysis_df, copy.deepcopy(self.subwords)
#         )
#         print("Compute Token Ambiguity")
#         analysis_df["token_entropy"] = AmbiguityComputer.token_ambiguity(
#             analysis_df.copy(), copy.deepcopy(self.subwords)
#         )
#         print("Compute Word Ambiguity")
#         analysis_df["word_entropy"] = AmbiguityComputer.word_ambiguity(
#             analysis_df.copy(), copy.deepcopy(self.subwords)
#         )

#         analysis_df["tr_entity"] = analysis_df["truth"].apply(
#             lambda x: x if x in ["[CLS]", "IGNORED"] else x.split("-")[-1]
#         )
#         analysis_df["pr_entity"] = analysis_df["pred"].apply(
#             lambda x: x if x in ["[CLS]", "IGNORED"] else x.split("-")[-1]
#         )

#         analysis_df["error_type"] = analysis_df[["truth", "pred"]].apply(
#             DataAnnotator.error_type, axis=1
#         )
#         return analysis_df


# class AnalysisBuilder:
#     def __init__(self, preparation, ambiguity_computer, outputs, results, config):
#         """
#         Initialize the DataFrameCreator class.

#         Args:
#             preparation: The DataPreparation object.
#             ambiguity_computer: The AmbiguityComputer object.
#             outputs: The outputs object containing batch outputs.
#             results: The results object containing model results.
#             config: General configuration file for different configurations
#         """
#         self.preparation = preparation
#         self.ambiguity_computer = ambiguity_computer
#         self.outputs = outputs
#         self.results = results
#         self.config = config

#     def construct_analysis_df(self):
#         """
#         Create the analysis DataFrame by extracting, transforming, and annotating data.

#         Returns:
#             pd.DataFrame: The final annotated DataFrame.
#         """
#         # Extract flat data
#         extractor = DataExtractor(self.preparation, self.outputs, self.results)
#         flat_data = extractor.extract_flat_data()

#         # Apply UMAP transformation
#         transformer = DataTransformer(self.config.umap_config)
#         layer_reduced = transformer.apply_umap(flat_data[0])
#         analysis_df = transformer.transform_to_dataframe(flat_data, layer_reduced)

#         # Annotate the DataFrame
#         analysis_df = DataAnnotator.annotate_tokenization_rate(analysis_df)
#         analysis_df = DataAnnotator.annotate_first_token_frequencies(
#             analysis_df, copy.deepcopy(self.computations.subwords)
#         )
#         analysis_df = DataAnnotator.annotate_additional_info(analysis_df)

#         return analysis_df


# # Main Dataset Characteristics Class
# class AnalysisManager:
#     def __init__(
#         self,
#         dataset_outputs,
#         batch_outputs,
#         tokenization_outputs,
#         subword_outputs,
#         model_outputs,
#         results,
#         config,
#     ):
#         self.label_aligner = LabelAligner(batch_outputs.batches, tokenization_outputs)
#         self.ambiguity_computer = AmbiguityComputer(subword_outputs, dataset_outputs)
#         self.analysis_builder = AnalysisBuilder(
#             self.label_aligner, self.ambiguity_computer, model_outputs, results, config
#         )
#         self.analysis_df = self.analysis_builder.construct_analysis_df()
