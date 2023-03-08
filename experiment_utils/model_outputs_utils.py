import torch
import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from sklearn.metrics import silhouette_samples, silhouette_score


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
                                                                 batch['hidden_states'], batch['input_ids']):
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
                    sentence_score.append(0)
                    silhouette_sample = np.array([0] * len(considered_labels))
                    self.compute_label_score(considered_labels, silhouette_sample)
                    self.errors.append((batch_num, sentence_num, num_of_labels))
                    self.sentence_samples[sentence_num] = [0] * len(considered_labels)
            self.scores.extend(sentence_score)

    def compute_label_score(self, considered_labels, silhouette_sample):
        for lb in self.data_labels:
            # identify the indices of the samples that has silhouette score
            label_indices = considered_labels.detach().cpu().numpy() == self.label_map[lb]
            # for each label assign the samples score that belong to that label
            self.label_score[lb].extend(silhouette_sample[label_indices])

    def generate_split_outputs(self, batches):
        self.compute_silhouette(batches)
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
                                 'logits': outputs['logits'], 'hidden_states': outputs['hidden_states']}))
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
            self.tokens.append(wordpiece_data.tokens)
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
        TOKENIZER = AutoTokenizer.from_pretrained(self.tokenizer_path)
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