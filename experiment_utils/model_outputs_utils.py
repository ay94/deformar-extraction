import os
import ast
import math
import copy
import torch
import warnings
import numpy as np
import pandas as pd
from umap import UMAP
import plotly.express as px
from tqdm.notebook import tqdm
from collections import defaultdict, Counter
from scipy.spatial import distance
from sklearn.cluster import KMeans
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

from sklearn.metrics import silhouette_samples, silhouette_score
from experiment_utils.error_analysis import TokenAmbiguity
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
        textlist = self.texts[item]
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
        for word_id, (word, label) in enumerate(zip(textlist, tags)):
            if self.PREPROCESSOR != None:
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
                    [word_tokens[i] if i == 0 else 'IGNORED' for i, w in enumerate(word_tokens)])
                self.sentence_ind_df.extend([item for i in range(len(word_tokens))])
                self.tokens.extend(word_tokens)
                self.wordpieces_df.extend([word_tokens for i in range(len(word_tokens))])
                self.words_df.extend([word for i in range(len(word_tokens))])
                self.word_ids_df.extend([word_id for i in range(len(word_tokens))])
                self.labels_df.extend([label if i == 0 else 'IGNORED' for i, w in enumerate(word_tokens)])
            else:
                self.removed_words.append((item, word))
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.TOKENIZER.num_special_tokens_to_add()
        if len(self.tokens) > self.config.MAX_SEQ_LEN - special_tokens_count:
            self.tokens = self.tokens[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.first_tokens_df = self.first_tokens_df[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.sentence_ind_df = self.sentence_ind_df[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.wordpieces_df = self.wordpieces_df[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.words_df = self.words_df[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.word_ids_df = self.word_ids_df[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.labels_df = self.labels_df[: (self.config.MAX_SEQ_LEN - special_tokens_count)]

        # Add the [SEP] token
        self.first_tokens_df += [self.TOKENIZER.sep_token]
        self.sentence_ind_df += [self.TOKENIZER.sep_token]
        self.wordpieces_df += [self.TOKENIZER.sep_token]
        self.words_df += [self.TOKENIZER.sep_token]
        self.word_ids_df += [self.TOKENIZER.sep_token]
        self.tokens += [self.TOKENIZER.sep_token]
        self.labels_df += [self.TOKENIZER.sep_token]
        # Add the [CLS] TOKEN
        self.first_tokens_df = [self.TOKENIZER.cls_token] + self.first_tokens_df
        self.sentence_ind_df = [self.TOKENIZER.cls_token] + self.sentence_ind_df
        self.wordpieces_df = [self.TOKENIZER.cls_token] + self.wordpieces_df
        self.words_df = [self.TOKENIZER.cls_token] + self.words_df
        self.word_ids_df = [self.TOKENIZER.cls_token] + self.word_ids_df
        self.tokens = [self.TOKENIZER.cls_token] + self.tokens
        self.labels_df = [self.TOKENIZER.cls_token] + self.labels_df
        # Length information
        self.sentence_len = len(self.words)
        self.wordpieces_len = len(self.tokens)


class GenerateSplitOutputs:
    def __init__(self, batches, labels) -> None:

        self.data_labels = labels
        self.label_map = {label: i for i, label in enumerate(self.data_labels)}
        # silhouette score for each sentence
        self.scores = []
        # sentences failed to be scored
        self.errors = []
        # loss for each instance
        self.aligned_losses = []
        # score for each label
        self.label_score = defaultdict(list)
        # silhouette score for each word in the sentence
        self.sentence_samples = defaultdict(list)
        self.generate_split_outputs(batches)

    def compute_silhouette(self, batcehs):
        #  loop through each batch
        for batch_num, batch in tqdm(enumerate(batcehs)):
            # for each batch give me the sentence
            sentence_score = []
            for labels, sentence_nums, outputs, input_ids in zip(batch['labels'], batch['sentence_num'],
                                                                 batch['last_hidden_state'], batch['input_ids']):
                # input ids identify the tokens included in the sentence it is used to compute sentence length 0 means padding nonzero means token/subtoken
                sentence_len = input_ids.nonzero().shape[0]
                # get the unique values to extract sentence_number
                sentence_num = torch.unique(sentence_nums[sentence_nums != -100]).tolist()[0]
                # get labels that belong to the sentence get the indices of labels that are not ignored convert them to list then get the unique labels to idenity the number of unique values in tensor which gives the numebr of labels in the sentence
                num_of_labels = len(torch.unique(labels[labels != -100]))
                # mask indices that are ignored
                label_mask = labels[:sentence_len] != -100
                # apply the mask to keep the actual labels only and remove the ignored ones
                considered_labels = labels[:sentence_len][label_mask]
                try:
                    # compute the average silhouette score for all tokens
                    sentence_score.append(silhouette_score(outputs[:sentence_len][label_mask].detach().cpu().numpy(),
                                                           considered_labels.detach().cpu().numpy()))
                    # compute sample silhouette score for each token
                    silhouette_sample = silhouette_samples(outputs[:sentence_len][label_mask].detach().cpu().numpy(),
                                                           considered_labels.detach().cpu().numpy())
                    self.compute_label_score(considered_labels, silhouette_sample)
                    self.sentence_samples[sentence_num].extend(silhouette_sample)

                except:
                    sentence_score.append(-100)
                    silhouette_sample = np.array([-100] * len(considered_labels))
                    self.compute_label_score(considered_labels, silhouette_sample)
                    self.errors.append((batch_num, sentence_num, num_of_labels))
                    self.sentence_samples[sentence_num] = [-100] * len(considered_labels)
            self.scores.extend(sentence_score)

    def compute_label_score(self, considered_labels, silhouette_sample):
        for lb in self.data_labels:
            # identify the indices of the samples that has silhouette score
            label_indices = considered_labels.detach().cpu().numpy() == self.label_map[lb]
            # for each label assign the samples score that belong to that label
            self.label_score[lb].extend(silhouette_sample[label_indices])

    def generate_split_outputs(self, batches):
        # self.compute_silhouette(batches)
        self.align_loss_input_ids(batches)

    def align_loss_input_ids(self, batches):
        # for each batch take the unique indices and get the losses
        for batch in batches:
            # return tensor of unique values and tensor of indices the tensor of indices contains the location of the unique element in the unique list this location in itself is not necessary but we use it to mask the right loss boundaries
            unique_values, indices = torch.unique(batch['input_ids'], return_inverse=True)
            # mask the losses with the indices because 0 index here is only refering to the first element of the unique index which is zero
            self.aligned_losses.append(batch['losses'][indices.view(-1) != 0])


class GenerateSplitBathces:
    def __init__(self, results, model, data_loader) -> None:
        self.model = model
        self.data_loader = data_loader
        self.device = self.load_device()
        self.batches = self.detache_batches(self.eval_fn(self.data_loader, self.model, self.device))
        self.compute_outputs(results)

    def detache_batches(self, batches):
        for i in range(len(batches)):
            for k, v in batches[i].items():
                batches[i][k] = v.detach().cpu()
        return batches

    def load_device(self):
        use_cuda = torch.cuda.is_available()
        return torch.device("cuda:0" if use_cuda else "cpu")

    def eval_fn(self, data_loader, model, device):
        model.eval()
        with torch.no_grad():
            batches = []
            for data in tqdm(data_loader, total=len(data_loader)):
                for k, v in data.items():
                    data[k] = v.to(device)

                outputs = model(**data)
                batches.append(({'labels': data['labels'], 'words_ids': data['words_ids'],
                                 'sentence_num': data['sentence_num'], 'attention_mask': data['attention_mask'],
                                 'input_ids': data['input_ids'], 'losses': outputs['losses'],
                                 'logits': outputs['logits'], 'last_hidden_state': outputs['last_hidden_state']}))
        return batches

    def compute_outputs(self, results):
        print('Compute Outputs')
        self.outputs = GenerateSplitOutputs(self.batches, results.data['labels'])


class GenerateSplitTokenizationOutputs:
    def __init__(self, wordpiece_data) -> None:
        self.first_tokens_df = []
        self.sentence_ind_df = []
        self.wordpieces_df = []
        self.words_df = []
        self.word_ids_df = []
        self.labels_df = []
        self.sentence_len = []
        self.wordpieces_len = []

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
            self.sentence_len.append(wordpiece_data.sentence_len)
            self.wordpieces_len.append(wordpiece_data.wordpieces_len)

            self.first_tokens.append(wordpiece_data.first_tokens)
            self.sentence_ind.append(wordpiece_data.sentence_ind)
            self.wordpieces.append(wordpiece_data.wordpieces)
            self.words.append(wordpiece_data.words)
            self.word_ids.append(wordpiece_data.word_ids)
            self.labels.append(wordpiece_data.labels)


class BatchOutputs:
    def __init__(self, outputs, model) -> None:
        self.generate_batches(outputs, model)

    def generate_batches(self, outputs, model):
        print('Generate Training Batches')
        self.train_batches = GenerateSplitBathces(outputs, model, outputs.train_dataloader)
        print('Generate Validation Batches')
        self.val_batches = GenerateSplitBathces(outputs, model, outputs.val_dataloader)
        print('Generate Test Batches')
        self.test_batches = GenerateSplitBathces(outputs, model, outputs.test_dataloader)


class ModelOutputs:
    def __init__(self, batches) -> None:
        self.generate_outputs(batches)

    def generate_outputs(self, batches):
        self.train_outputs = batches.train_batches.outputs
        self.val_outputs = batches.val_batches.outputs
        self.test_outputs = batches.test_batches.outputs


class TokenizationOutputs:
    def __init__(self, outputs, tokenizer_path, preprocessor_path=None) -> None:

        self.tokenizer_path = tokenizer_path
        self.preprocessor_path = preprocessor_path
        TOKENIZER, PREPROCESSOR = self.load_tokenizer()
        # subword sentence locations and wh at tag they had in each sentence
        self.generate_wordpieces(outputs, TOKENIZER, PREPROCESSOR)

    def load_tokenizer(self):
        if self.preprocessor_path != None:
            print(f'Loading Preprocessor {self.preprocessor_path}')
            PREPROCESSOR = ArabertPreprocessor(self.preprocessor_path)
        else:
            PREPROCESSOR = None
        print(f'Loading Tokenizer {self.tokenizer_path}')
        TOKENIZER = AutoTokenizer.from_pretrained(self.tokenizer_path, do_lower_case=False)
        return TOKENIZER, PREPROCESSOR

    def load_wordpieces(self, outputs, mode, tokenizer, preprocessor):
        wordpices = WordPieceDataset(
            texts=[x[1] for x in outputs.data[mode]],
            tags=[x[2] for x in outputs.data[mode]],
            config=outputs.config,
            tokenizer=tokenizer,
            preprocessor=preprocessor)
        return wordpices

    def get_subwords(self, wordpieces):
        subwords = defaultdict(list)
        for i in tqdm(range(wordpieces.__len__())):
            wordpieces.__getitem__(i)
            for w, t in zip(wordpieces.first_tokens, wordpieces.labels):
                subwords[w].append({'tag': t, 'sentence': i})
        return subwords

    def generate_wordpieces(self, outputs, tokenizer, preporcessor):
        train_wordpieces = self.load_wordpieces(outputs, 'train', tokenizer, preporcessor)
        val_wordpieces = self.load_wordpieces(outputs, 'val', tokenizer, preporcessor)
        test_wordpieces = self.load_wordpieces(outputs, 'test', tokenizer, preporcessor)

        self.generate_tokenization_output(train_wordpieces, val_wordpieces, test_wordpieces)
        self.get_subword_locations(train_wordpieces, val_wordpieces, test_wordpieces)

    def get_subword_locations(self, train_wordpieces, val_wordpieces, test_wordpieces):
        print('Generate Training Subwords Locations')
        self.train_subwords = self.get_subwords(train_wordpieces)
        print('Generate Validation Subwords Locations')
        self.val_subwords = self.get_subwords(val_wordpieces)
        print('Generate Test Subwords Locations')
        self.test_subwords = self.get_subwords(test_wordpieces)

    def generate_tokenization_output(self, train_wordpieces, val_wordpieces, test_wordpieces):
        print('Generate Training Tokenization Outputs')
        self.train_tokenizatin_output = GenerateSplitTokenizationOutputs(train_wordpieces)
        print('Generate Validation Tokenization Outputs')
        self.val_tokenizatin_output = GenerateSplitTokenizationOutputs(val_wordpieces)
        print('Generate Test Tokenization Outputs')
        self.test_tokenizatin_output = GenerateSplitTokenizationOutputs(test_wordpieces)


class ModelResults:
    def __init__(self, outputs) -> None:
        self.generate_results(outputs)

    def generate_results(self, outputs):
        self.train_metrics = outputs.train_metrics
        self.val_metrics = outputs.val_metrics
        self.test_metrics = outputs.test_metrics


class SaveModelOutputs:
    def __init__(self, fh, data_name, model_name, outputs, tokenization, results):
        self.outputs = outputs
        self.tokenization = tokenization
        self.results = results
        fh.save_object(outputs, f'modelOutputs/{data_name}_{model_name}_model_outputs.pkl')
        fh.save_object(tokenization, f'modelOutputs/{data_name}_{model_name}_tokenization_outputs.pkl')
        fh.save_object(results, f'modelOutputs/{data_name}_{model_name}_model_results.pkl')


class DatasetCharacteristics:
    def __init__(self, dataset_outputs, batch_outputs, tokenization_outputs, subword_outputs, model_outputs, results):
        self.dataset_outputs = dataset_outputs
        self.batches = batch_outputs.batches
        self.tokenization = tokenization_outputs
        self.subwords = subword_outputs
        self.outputs = model_outputs

        self.results = results
        self.analysis_df = self.create_analysis_df()

    def extract_token_scores(self, sentence_samples, tokenization_output):
        sentence_scores = []
        for token_scores, labels in zip(sentence_samples.values(), self.tokenization.labels_df):
            token_score = []
            i = 0
            for lb in labels:
                if lb in ['[CLS]', 'IGNORED', '[SEP]']:
                    token_score.append(-100)
                else:
                    if i < len(token_scores):
                        token_score.append(token_scores[i])
                        i += 1
            sentence_scores.extend(token_score)
        return sentence_scores

    def label_alignment(self):
        map = defaultdict(list)
        for sen_id, sen in enumerate(self.tokenization.labels_df):
            for tok_id, tok in enumerate(sen):
                if tok in ['[CLS]', '[SEP]', 'IGNORED']:
                    map[sen_id].append((tok_id, tok))
        return map

    def change_preds(self, pred, pred_map):
        modified_pred = []
        for sen_id, sen in enumerate(pred):
            sentnece = sen.copy()
            for idx, tok in pred_map[sen_id]:
                sentnece.insert(idx, tok)
            modified_pred.append(sentnece)
        return modified_pred

    def create_analysis_df(self):
        flat_states = torch.cat([hidden_state[ids != 0] for batch in self.batches for ids, hidden_state in
                                 zip(batch['input_ids'], batch['last_hidden_state'])])
        flat_labels = torch.cat([labels[ids != 0] for batch in self.batches for ids, labels in
                                 zip(batch['input_ids'], batch['labels'])])
        flat_losses = torch.cat([losses for losses in
                                 self.outputs.aligned_losses])
        flat_scores = self.extract_token_scores(self.outputs.sentence_samples,
                                                self.tokenization)
        flat_words = [tok for sen in self.tokenization.words_df for tok in sen]
        flat_tokens = [tok for sen in self.tokenization.tokens for tok in sen]
        flat_wordpieces = [str(tok) for sen in self.tokenization.wordpieces_df for tok in sen]
        flat_first_tokens = [tok for sen in self.tokenization.first_tokens_df for tok in sen]
        flat_token_ids = [f'{tok}@#{id}@#{i}' for sen, sen_id in
                          zip(self.tokenization.first_tokens_df, self.tokenization.sentence_ind_df) for i, (tok, id) in
                          enumerate(zip(sen, list(set(sen_id[1:-1])) + sen_id[1:-1] + list(set(sen_id[1:-1]))))]

        flat_trues = [tok for sen in self.tokenization.labels_df for tok in sen]

        pred_map = self.label_alignment()

        modified_preds = self.change_preds(self.results.seq_output['y_pred'].copy(), pred_map)
        flat_preds = [tok for sen in modified_preds for tok in sen]
        flat_sen_ids = [tok for sen in self.tokenization.sentence_ind_df for tok in
                        list(set(sen[1:-1])) + sen[1:-1] + list(set(sen[1:-1]))]
        flat_agreement = np.array(flat_trues) == np.array(flat_preds)

        t_ids = [int(w) for batch_id, batch in enumerate(self.batches) for sen_id, ids in
                 enumerate(batch['input_ids']) for w_id, w in enumerate(ids[ids != 0])]

        w_ids = [w_id for batch_id, batch in enumerate(self.batches) for sen_id, ids in
                 enumerate(batch['input_ids']) for w_id, w in enumerate(ids[ids != 0])]

        layer_reduced = UMAP(verbose=True, random_state=1).fit_transform(flat_states).transpose()

        analysis_df = pd.DataFrame(
            {'token_id': t_ids, 'word_id': w_ids, 'sen_id': flat_sen_ids, 'token_ids': flat_token_ids,
             'label_ids': flat_labels.tolist(),
             'words': flat_words, 'wordpieces': flat_wordpieces, 'tokens': flat_tokens,
             'first_tokens': flat_first_tokens,
             'truth': flat_trues, 'pred': flat_preds, 'agreement': flat_agreement,
             'losses': flat_losses.tolist(), 'x': layer_reduced[0],
             'y': layer_reduced[1]})

        analysis_df = self.annotate_tokenization_rate(analysis_df.copy())
        analysis_df = self.get_first_tokens(analysis_df, copy.deepcopy(self.subwords))
        print('Compute Consistency')
        analysis_df = self.compute_consistency(analysis_df, copy.deepcopy(self.subwords))
        print('Compute Token Ambiguity')
        analysis_df['token_entropy'] = self.token_ambiguity(analysis_df.copy())
        print('Compute Word Ambiguity')
        analysis_df['word_entropy'] = self.word_ambiguity(analysis_df.copy())

        analysis_df['tr_entity'] = analysis_df['truth'].apply(
            lambda x: x if x == '[CLS]' or x == 'IGNORED' else x.split('-')[-1])
        analysis_df['pr_entity'] = analysis_df['pred'].apply(
            lambda x: x if x == '[CLS]' or x == 'IGNORED' else x.split('-')[-1])

        analysis_df['error_type'] = analysis_df[['truth', 'pred']].apply(self.error_type, axis=1)

        return analysis_df

    def annotate_tokenization_rate(self, analysis_df):
        num_tokens = []
        for wps in analysis_df['wordpieces']:
            try:
                num_tokens.append(len(ast.literal_eval(wps)))
            except:
                num_tokens.append(1)
        analysis_df['tokenization_rate'] = num_tokens
        return analysis_df

    def get_first_tokens(self, analysis, subword_locations):
        fr_tk = []
        try:
            analysis.insert(5, 'first_tokens_freq', analysis['first_tokens'].apply(lambda x: len(subword_locations[x])))
        except:
            print('')
        return analysis

    def compute_consistency(self, analysis, subwords_locations):
        consistent = []
        inconsistent = []
        for i in tqdm(range(len(analysis))):
            con_count = []
            incon_count = []
            for t, count in Counter(
                    [tok['tag'] for tok in subwords_locations[analysis.iloc[i]['first_tokens']]]).items():
                if t == analysis.iloc[i]['truth']:
                    con_count.append(count)
                else:
                    incon_count.append(count)
            consistent.append(sum(con_count))
            inconsistent.append(sum(incon_count))
            try:
                analysis.insert(6, 'first_tokens_consistency', consistent)
                analysis.insert(7, 'first_tokens_inconsistency', inconsistent)
            except:
                continue
        return analysis

    def entropy(self, probabilities):
        return -sum(p * math.log2(p) for p in probabilities.values())

    def label_probabilities(self, dataset):
        # Count the frequencies of each label for each token
        label_counts = {}
        for token, label in dataset:
            if token not in label_counts:
                label_counts[token] = Counter()
            label_counts[token][label] += 1

        # Calculate the probabilities of each label for each token
        probabilities = {}
        for token, counts in label_counts.items():
            total = sum(counts.values())
            probabilities[token] = {label: count / total for label, count in counts.items()}
        return probabilities

    def token_ambiguity(self, analysis_df):
        subwords_counter = []
        for subword, tag_dis in tqdm(self.subwords.items()):
            for tag in tag_dis:
                subwords_counter.append((subword, tag['tag']))
        probabilities = self.label_probabilities(subwords_counter)
        # Calculate the entropy for each token
        token_entropies = {token: abs(self.entropy(probs)) for token, probs in probabilities.items()}
        computed_token_entropy = pd.DataFrame(token_entropies.items(), columns=['first_tokens', 'entropy'])

        token_entropy = []
        for tk in tqdm(analysis_df['first_tokens']):
            token_data = computed_token_entropy[computed_token_entropy['first_tokens'] == tk]
            if len(token_data) > 0:
                token_entropy.append(token_data['entropy'].values[0])
            else:
                token_entropy.append(-1)
        return token_entropy

    def word_ambiguity(self, analysis_df):
        wordsDict = defaultdict(list)
        for i, sen in enumerate(self.dataset_outputs.data['train']):
            for w, t in zip(sen[1], sen[2]):
                wordsDict[w].append({'tag': t, 'sentence': i})

        words_counter = []
        for word, tag_dis in tqdm(wordsDict.items()):
            for tag in tag_dis:
                words_counter.append((word, tag['tag']))

        probabilities = self.label_probabilities(words_counter)
        word_entropies = {token: abs(self.entropy(probs)) for token, probs in probabilities.items()}
        computed_word_entropy = pd.DataFrame(word_entropies.items(), columns=['words', 'entropy'])

        word_entropy = []
        for tk in tqdm(analysis_df['words']):
            token_data = computed_word_entropy[computed_word_entropy['words'] == tk]
            if len(token_data) > 0:
                word_entropy.append(token_data['entropy'].values[0])
            else:
                word_entropy.append(-1)
        return word_entropy

    def error_type(self, row):
        true, pred = row['truth'], row['pred']

        # Check if both entity type and boundaries are correct
        if true == pred:
            return 'Correct'

        # Check if the entity type is incorrect but the boundaries are correct
        elif true[1:] != pred[1:]:
            return 'Entity'

        # If neither of the above conditions are met, the error must be in the boundaries
        else:
            return 'Chunk'


class Entity:
    def __init__(self, outputs):
        self.y_true = outputs['y_true']
        self.y_pred = outputs['y_pred']
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

        if prev_tag == 'E':
            chunk_end = True
        if prev_tag == 'S':
            chunk_end = True

        if prev_tag == 'B' and tag == 'B':
            chunk_end = True
        if prev_tag == 'B' and tag == 'S':
            chunk_end = True
        if prev_tag == 'B' and tag == 'O':
            chunk_end = True
        if prev_tag == 'I' and tag == 'B':
            chunk_end = True
        if prev_tag == 'I' and tag == 'S':
            chunk_end = True
        if prev_tag == 'I' and tag == 'O':
            chunk_end = True

        if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
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

        if tag == 'B':
            chunk_start = True
        if tag == 'S':
            chunk_start = True

        if prev_tag == 'E' and tag == 'E':
            chunk_start = True
        if prev_tag == 'E' and tag == 'I':
            chunk_start = True
        if prev_tag == 'S' and tag == 'E':
            chunk_start = True
        if prev_tag == 'S' and tag == 'I':
            chunk_start = True
        if prev_tag == 'O' and tag == 'E':
            chunk_start = True
        if prev_tag == 'O' and tag == 'I':
            chunk_start = True

        if tag != 'O' and tag != '.' and prev_type != type_:
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
            if chunk in ['O', 'B', 'I', 'E', 'S']:
                return

            if suffix:
                if not chunk.endswith(('-B', '-I', '-E', '-S')):
                    warnings.warn('{} seems not to be NE tag.'.format(chunk))

            else:
                if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                    warnings.warn('{} seems not to be NE tag.'.format(chunk))

        # for nested list
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ['O']]

        prev_tag = 'O'
        prev_type = ''
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ['O']):
            _validate_chunk(chunk, suffix)

            if suffix:
                tag = chunk[-1]
                type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
            else:
                tag = chunk[0]
                type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

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
        true.extend([('O',) + t[1:] for t in only_in_set2])
        pred.extend([('O',) + t[1:] for t in only_in_set1])

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
                    entities.append((tag, self.extract_tag(i, self.y_true)[0], self.extract_tag(i, self.y_pred)[0], i,
                                     self.extract_tag(i, self.y_true)[1]))
        entity_prediction = pd.DataFrame(entities, columns=['entity', 'true_token', 'pred_token', 'token_id', 'sen_id'])
        entity_prediction['agreement'] = entity_prediction['true_token'] == entity_prediction['pred_token']
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
            unique_values, indices = torch.unique(batch['input_ids'], return_inverse=True)
            for token in batch['logits'][indices != 0]:
                token_logits.append(token.tolist())

        logits_matrix = np.array(token_logits)
        probabilities_matrix = self.softmax(logits_matrix)

        # Calculate entropy for each token
        prediction_entropy = self.calculate_entropy(probabilities_matrix)

        prediction_confidence = [max(prob_scores) for prob_scores in probabilities_matrix]
        prediction_variability = [np.std(prob_scores) for prob_scores in probabilities_matrix]

        prediction_probabilities = pd.DataFrame(probabilities_matrix).rename(columns=outputs.data['inv_labels'])
        prediction_probabilities = prediction_probabilities.reset_index()
        prediction_probabilities = prediction_probabilities.rename(columns={'index': 'global_id'})

        analysis_df = analysis_df.reset_index()
        analysis_df = analysis_df.rename(columns={'index': 'global_id'})
        analysis_df['prediction_entropy'] = prediction_entropy
        analysis_df['confidences'] = prediction_confidence
        analysis_df['variability'] = prediction_variability
        entropy_df = analysis_df.merge(prediction_probabilities, on='global_id')
        return entropy_df

    def cluster_data(self, k, states):
        # Define the number of clusters
        n_clusters = k

        # Create an instance of the KMeans algorithm
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1)

        # Fit the algorithm to the data
        kmeans.fit(states)

        # Get the cluster assignments for each data point
        labels = [f'cluster-{lb}' for lb in kmeans.labels_]

        # Get the centroid locations
        centroids = kmeans.cluster_centers_
        return centroids, labels

    def annotate_clusters(self, k):
        flat_states = torch.cat([hidden_state[ids != 0] for batch in self.batches for ids, hidden_state in
                                 zip(batch['input_ids'], batch['last_hidden_state'])])

        flat_labels = torch.cat([labels[ids != 0] for batch in self.batches for ids, labels in
                                 zip(batch['input_ids'], batch['labels'])])
        mask = np.array(flat_labels != -100)
        states = flat_states[mask]

        centroids, labels = self.cluster_data(k, states)
        self.entropy_df[f'{k}_clusters'] = 'IGNORED'
        self.entropy_df.loc[mask, f'{k}_clusters'] = labels
        self.centroid_df = self.generate_centroid_data(centroids, k)
        return self.entropy_df, self.centroid_df

    def generate_centroid_data(self, centroids, k):
        flat_states = torch.cat([hidden_state[ids != 0] for batch in self.batches for ids, hidden_state in
                                 zip(batch['input_ids'], batch['last_hidden_state'])])
        flat_labels = torch.cat([labels[ids != 0] for batch in self.batches for ids, labels in
                                 zip(batch['input_ids'], batch['labels'])])
        mask = np.array(flat_labels != -100)
        states = flat_states[mask]
        c_df = self.entropy_df[mask].copy()
        centroid_df = pd.DataFrame()
        centroid_df['token_ids'] = list(c_df['token_ids'].values) + ['C'] * k
        centroid_df['truth'] = list(c_df['truth'].values) + ['C'] * k
        centroid_df['pred'] = list(c_df['pred'].values) + ['C'] * k
        centroid_df['agreement'] = list(c_df['agreement'].values) + ['C'] * k
        centroid_df['error_type'] = list(c_df['error_type'].values) + ['C'] * k
        centroid_df['centroid'] = f'Centroid-{k}'
        centroid_df['clusters'] = list(c_df[f'{k}_clusters'].values) + ['C'] * k

        centroid_data = torch.cat([states, torch.from_numpy(centroids)])

        centroid_reduced = UMAP(verbose=True, random_state=1).fit_transform(centroid_data).transpose()
        centroid_df['x'] = centroid_reduced[0]
        centroid_df['y'] = centroid_reduced[1]

        return centroid_df

    def generate_token_score(self):
        flat_states = torch.cat([hidden_state[ids != 0] for batch in self.batches for ids, hidden_state in
                                 zip(batch['input_ids'], batch['last_hidden_state'])])
        flat_labels = torch.cat([labels[ids != 0] for batch in self.batches for ids, labels in
                                 zip(batch['input_ids'], batch['labels'])])

        flat_mask = ~self.analysis_df['pred'].isin(['IGNORED', '[SEP]', '[CLS]'])
        flat_pred = self.analysis_df['pred']

        self.overall_score = silhouette_score(flat_states[flat_labels != -100], flat_labels[flat_labels != -100])
        silhouette_sample = silhouette_samples(flat_states[flat_labels != -100], flat_labels[flat_labels != -100])
        pred_silhouette_sample = silhouette_samples(flat_states[flat_mask], flat_pred[flat_mask])
        without_ignore = self.entropy_df[self.entropy_df['label_ids'] != -100].copy()
        without_ignore['truth_token_score'] = silhouette_sample
        without_ignore['pred_token_score'] = pred_silhouette_sample
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

        sentence_a = ' '.join(example)

        if self.preprocessor == None:
            inputs = self.tokenizer.encode_plus(sentence_a, return_tensors='pt', truncation=True,
                                                add_special_tokens=True)
        else:
            inputs = self.tokenizer.encode_plus(self.preprocessor.preprocess(sentence_a), truncation=True,
                                                return_tensors='pt',
                                                add_special_tokens=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']

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
                head.append(1 - distance.cosine(
                    model1_mat[i][j].flatten(),
                    model2_mat[i][j].flatten()
                ))
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
        self.pretrained_model = AutoModel.from_pretrained(model_path, output_attentions=True,
                                                          output_hidden_states=True).to(self.device)
        self.fine_tuned_model = model.to(self.device)
        self.attention_impact = AttentionSimilarity(self.device,
                                                    self.pretrained_model,
                                                    self.fine_tuned_model,
                                                    tokenizer,
                                                    preprocessor)

    def compute_attention_similarities(self):
        similarities = [self.attention_impact.compute_similarity(example[1]) for example in tqdm(self.data[:500])]
        change_fig = px.imshow(np.array(similarities).mean(0),
                               labels=dict(x="Heads", y="Layers", color="Similarity Score"),
                               )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        return change_fig

    def compute_example_similarities(self, id):
        scores = self.attention_impact.compute_similarity(self.data[id][1])
        change_fig = px.imshow(scores,
                               labels=dict(x="Heads", y="Layers", color="Similarity Score"),
                               )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        change_fig.show()

    def extract_weights(self, layer):
        self_attention_weights = torch.cat([
            layer.attention.self.query.weight,
            layer.attention.self.key.weight,
            layer.attention.self.value.weight
        ], dim=0)

        return self_attention_weights

    def compare_weights(self):
        num_layers = len(self.pretrained_model.encoder.layer)
        num_heads = self.pretrained_model.config.num_attention_heads
        weight_diff_matrix = np.zeros((num_layers, num_heads))

        for layer in range(num_layers):
            for head in range(num_heads):
                pretrained_weight = self.extract_weights(self.pretrained_model.encoder.layer[layer])[:,
                                    head::num_heads].detach().cpu().numpy()
                fine_tuned_weight = self.extract_weights(self.fine_tuned_model.encoder.layer[layer])[:,
                                    head::num_heads].cpu().detach().cpu().numpy()
                weight_diff = 1 - distance.cosine(pretrained_weight.flatten(), fine_tuned_weight.flatten())
                weight_diff_matrix[layer, head] = weight_diff

        return self.weight_difference(weight_diff_matrix)

    def weight_difference(self, weight_diff_matrix):
        change_fig = px.imshow(weight_diff_matrix,
                               labels=dict(x="Heads", y="Layers", color="Similarity Score"),
                               )
        change_fig.layout.height = 700
        change_fig.layout.width = 700
        return change_fig


class ErrorAnalysis:
    def __init__(self, dataset_outputs, batches, tokenization_outputs, model_outputs, results, model):
        self.dataset_outputs = dataset_outputs
        self.batches = batches
        self.tokenization_outputs = tokenization_outputs
        self.model_outputs = model_outputs
        self.results = results
        self.model = model

    def compute_analysis_data(self, mode, model_path):
        if mode == 'train':
            batches = self.batches.train_batches
            toks = self.tokenization_outputs.train_tokenizatin_output
            subwords = self.tokenization_outputs.train_subwords
            md_out = self.model_outputs.train_outputs
            res = self.results.train_metrics
            self.seq_report = res.seq_report
            self.skl_report = res.skl_report
            self.ent = Entity(res.seq_output)
        elif mode == 'val':
            batches = self.batches.val_batches
            toks = self.tokenization_outputs.val_tokenizatin_output
            subwords = self.tokenization_outputs.train_subwords
            md_out = self.model_outputs.val_outputs
            res = self.results.val_metrics
            self.seq_report = res.seq_report
            self.skl_report = res.skl_report
            self.ent = Entity(res.seq_output)
        else:
            batches = self.batches.test_batches
            toks = self.tokenization_outputs.test_tokenizatin_output
            subwords = self.tokenization_outputs.train_subwords
            md_out = self.model_outputs.test_outputs
            res = self.results.test_metrics
            self.seq_report = res.seq_report
            self.skl_report = res.skl_report
            self.ent = Entity(res.seq_output)
        self.dc = DatasetCharacteristics(self.dataset_outputs, batches, toks, subwords, md_out, res)
        self.db = DecisionBoundary(batches, self.dc.analysis_df, self.dataset_outputs)
        self.tr_im = TrainingImpact(mode, self.dataset_outputs, model_path, self.model.bert)


class SaveAnalysis:
    def __init__(self, out_fh, error_analysis, mode, model_path):
        self.ea = error_analysis
        self.out_fh = out_fh
        self.mode = mode
        self.ea.compute_analysis_data(mode, model_path)
        self.generate_split_outputs()

    def generate_confusion(self, ):
        confusion_data = pd.DataFrame()
        confusion_data['truth'] = self.ea.ent.seq_true
        confusion_data['pred'] = self.ea.ent.seq_pred
        entity_prediction = self.ea.ent.entity_prediction
        return confusion_data, entity_prediction

    def generate_clustering(self, ):
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
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_analysis_df.jsonl.gz'),
            lines=True, orient='records'
        )

        self.centroid_df.to_json(
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_centroid_df.jsonl.gz'),
            lines=True, orient='records'
        )

        # this is adding token silhouette score because it is ignoring the IGNORED tokens and only considering entities
        self.token_score_df.to_json(
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_token_score_df.jsonl.gz'),
            lines=True, orient='records'
        )

        self.confusion_data.to_json(
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_confusion_data.jsonl.gz'),
            lines=True, orient='records'
        )

        self.entity_prediction.to_json(
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_entity_prediction.jsonl.gz'),
            lines=True, orient='records'
        )

        self.seq_report.to_csv(
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_seq_report.csv'),
            index=False
        )
        self.skl_report.to_csv(
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_skl_report.csv'),
            index=False
        )

        self.activations.write_json(
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_activations.json')
        )
        self.weights.write_json(
            self.out_fh.cr_fn(f'{self.mode}/{self.mode}_weights.json')
        )


class AnalysisOutputs:
    def __init__(self, fh, out_fh, data_name, model_name, model_path, preprocessor=None):
        self.out_fh = out_fh
        self.model_path = model_path
        self.outputs = fh.load_object(f'evalOutputs/{model_name}_{data_name}_regular_outputs.pkl')
        load_model_path = fh.cr_fn(f'trainOutputs/{model_name}_{data_name}_regular.bin')
        self.model = torch.load(load_model_path)
        self.batch_outputs = BatchOutputs(self.outputs, self.model)
        self.model_outputs = ModelOutputs(self.batch_outputs)
        self.results = ModelResults(self.outputs)
        if preprocessor is not None:
            self.tokenization_outputs = TokenizationOutputs(self.outputs, model_path, preprocessor)
        else:
            self.tokenization_outputs = TokenizationOutputs(self.outputs, model_path)
        self.ea = ErrorAnalysis(self.outputs,
                                self.batch_outputs,
                                self.tokenization_outputs,
                                self.model_outputs,
                                self.results,
                                self.model)
        self.create_folder(out_fh)
        self.out_fh.save_json(self.tokenization_outputs.train_subwords, 'train_subwords.json')
        save_model_path = self.out_fh.cr_fn('initialization')
        torch.save(self.model, f'{save_model_path}/{model_name}_{data_name}_regular.bin')

    def create_folder(self, out_fh):
        os.makedirs(out_fh.cr_fn('train'), exist_ok=True)
        os.makedirs(out_fh.cr_fn('val'), exist_ok=True)
        os.makedirs(out_fh.cr_fn('test'), exist_ok=True)
        os.makedirs(out_fh.cr_fn('initialization'), exist_ok=True)

    def save_analysis(self, mode):
        self.analysis = SaveAnalysis(self.out_fh, self.ea, mode, self.model_path)
        self.analysis.save()




class AuxilariyOutputs:
    def __init__(self, model_name, data_name, model_path, fh, out_fh):
        device = torch.device('cuda')
        outputs = fh.load_object(f'evalOutputs/{model_name}_{data_name}_regular_outputs.pkl')
        pretrained_model = TCModel(len(outputs.data['labels']), model_path)
        self.out_fh = out_fh
        self.batch_outputs = BatchOutputs(outputs, pretrained_model.to(device))
        self.light_train_df = pd.read_json(
            out_fh.cr_fn('train/train_analysis_df.jsonl.gz'),
            lines=True
        )[["token_ids", "words", "agreement", "truth", "pred", "x", "y"]]

        self.light_train_df.to_json(
            out_fh.cr_fn(f'light_train_df.jsonl.gz'),
            lines=True, orient='records'
        )

    def create_df(self, batches):
        flat_states = torch.cat([hidden_state[ids != 0] for batch in batches for ids, hidden_state in
                                 zip(batch['input_ids'], batch['last_hidden_state'])])

        layer_reduced = UMAP(verbose=True, random_state=1).fit_transform(flat_states).transpose()

        analysis_df = pd.DataFrame(
            {'pre_x': layer_reduced[0], 'pre_y': layer_reduced[1]}
        )
        analysis_df = analysis_df.reset_index()
        return analysis_df.rename(columns={'index': 'global_id'})

    def save(self, mode):
        if mode == 'train':
            pre_df = self.create_df(self.batch_outputs.train_batches.batches)
        elif mode == 'val':
            pre_df = self.create_df(self.batch_outputs.val_batches.batches)
        else:
            pre_df = self.create_df(self.batch_outputs.test_batches.batches)

        pre_df.to_json(
            self.out_fh.cr_fn(f'{mode}/{mode}_pre_df.jsonl.gz'),
            lines=True, orient='records'
        )
