import time
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

def current_milli_time():
    return int(round(time.time() * 1000))

class FineTuneConfig:
  def __init__(self) -> None:
      self.MAX_SEQ_LEN = 256
      self.TRAIN_BATCH_SIZE = 16
      self.VALID_BATCH_SIZE = 8
      self.EPOCHS = 4
      self.SPLITS = 4
      self.LEARNING_RATE = 5e-5
      self.WARMUP_RATIO = 0.1
      self.MAX_GRAD_NORM = 1.0
      self.ACCUMULATION_STEPS = 1


class TCDataset:
    def __init__(self, texts, tags, label_list, config, tokenizer, preprocessor=None):
        self.texts = texts
        self.tags = tags
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.config = config
        self.TOKENIZER = tokenizer
        self.PREPROCESSOR = preprocessor

        # Use cross entropy ignore_index as padding label id so that only real label ids contribute to the loss later.
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        textlist = self.texts[item]
        tags = self.tags[item]

        tokens = []
        label_ids = []
        word_ids = []
        for word_id, (word, label) in enumerate(zip(textlist, tags)):
            if self.PREPROCESSOR != None:
                clean_word = self.PREPROCESSOR.preprocess(word)
                word_tokens = self.TOKENIZER.tokenize(clean_word)
            else:
                word_tokens = self.TOKENIZER.tokenize(word)
                # ignore words that are preprocessed because the preprocessor return '' and the tokeniser replace that with empty list which gets ignored here
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([self.label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))
                word_ids.extend([word_id] * (len(word_tokens)))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.TOKENIZER.num_special_tokens_to_add()
        if len(tokens) > self.config.MAX_SEQ_LEN - special_tokens_count:
            tokens = tokens[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            label_ids = label_ids[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            word_ids = word_ids[: (self.config.MAX_SEQ_LEN - special_tokens_count)]

        # Add the [SEP] token
        tokens += [self.TOKENIZER.sep_token]
        label_ids += [self.pad_token_label_id]
        token_type_ids = [0] * len(tokens)
        word_ids += [self.pad_token_label_id]

        # Add the [CLS] TOKEN
        tokens = [self.TOKENIZER.cls_token] + tokens
        label_ids = [self.pad_token_label_id] + label_ids
        token_type_ids = [0] + token_type_ids
        word_ids = [self.pad_token_label_id] + word_ids

        input_ids = self.TOKENIZER.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        sentence_num = [item] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = self.config.MAX_SEQ_LEN - len(input_ids)

        input_ids += [self.TOKENIZER.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [self.pad_token_label_id] * padding_length
        sentence_num += [self.pad_token_label_id] * padding_length
        word_ids += [self.pad_token_label_id] * padding_length

        assert len(input_ids) == self.config.MAX_SEQ_LEN
        assert len(attention_mask) == self.config.MAX_SEQ_LEN
        assert len(token_type_ids) == self.config.MAX_SEQ_LEN
        assert len(label_ids) == self.config.MAX_SEQ_LEN
        assert len(sentence_num) == self.config.MAX_SEQ_LEN
        assert len(word_ids) == self.config.MAX_SEQ_LEN

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'words_ids': torch.tensor(word_ids, dtype=torch.long),
            'sentence_num': torch.tensor(sentence_num, dtype=torch.long),
        }


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
        self.labels = []
        self.tokens = []
        self.sentence_len = 0
        self.wordpieces_len = 0
        for word, label in zip(textlist, tags):
            if self.PREPROCESSOR != None:
                clean_word = self.PREPROCESSOR.preprocess(word)
                word_tokens = self.TOKENIZER.tokenize(clean_word)
            else:
                word_tokens = self.TOKENIZER.tokenize(word)
            if len(word_tokens) > 0:
                self.first_tokens.append(word_tokens[0])
                self.sentence_ind.append(item)
                self.tokens.extend(word_tokens)
                self.wordpieces.append(word_tokens)
                self.words.append(word)
                self.labels.append(label)
                # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.TOKENIZER.num_special_tokens_to_add()
        if len(self.first_tokens) > self.config.MAX_SEQ_LEN - special_tokens_count:
            self.first_tokens = self.first_tokens[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.sentence_ind = self.sentence_ind[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.wordpieces = self.wordpieces[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.words = self.words[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            self.labels = self.labels[: (self.config.MAX_SEQ_LEN - special_tokens_count)]

        # Add the [SEP] token
        self.first_tokens += [self.TOKENIZER.sep_token]
        self.sentence_ind += [self.TOKENIZER.sep_token]
        self.wordpieces += [self.TOKENIZER.sep_token]
        self.labels += [self.TOKENIZER.sep_token]
        self.tokens += [self.TOKENIZER.sep_token]
        # Add the [CLS] TOKEN
        self.first_tokens = [self.TOKENIZER.cls_token] + self.first_tokens
        self.sentence_ind = [self.TOKENIZER.cls_token] + self.sentence_ind
        self.wordpieces = [self.TOKENIZER.cls_token] + self.wordpieces
        self.labels = [self.TOKENIZER.cls_token] + self.labels
        self.tokens = [self.TOKENIZER.cls_token] + self.tokens
        # Length information
        self.sentence_len = len(self.words)
        self.wordpieces_len = len(self.tokens)


class SCModel(nn.Module):
    def __init__(self, num_tag, path):
        super(SCModel, self).__init__()
        self.num_tag = num_tag
        print(f'Loading BERT Model: {path}')
        self.bert = AutoModel.from_pretrained(path, output_attentions=True, output_hidden_states=True)
        self.bert_drop = nn.Dropout(0.3)
        self.output_layer = nn.Linear(self.bert.config.hidden_size, self.num_tag)

    def loss_fn(self, output, target, mask, num_labels):
        # loss function that returns the mean
        lfn = nn.CrossEntropyLoss()
        # loss function that returns the losss fore each sample
        lfns = nn.CrossEntropyLoss(reduction='none')
        # mask to specify the active losses (sentence boundary) based on attention mask
        active_loss = mask.view(-1) == 1
        # this reshape the output dimension from torch.Size([16, 256, 9]) to torch.Size([4096, 9]) now the inner dimensionality match
        active_logits = output.view(-1, num_labels)
        #  the where function takes tensor of condition, tensor of x and tensor of y if the condition is true the value of x will be used in the output tensor if the condition is flase the value of y will be used
        active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(lfn.ignore_index).type_as(target))
        # average_loss
        loss = lfn(active_logits, active_labels)
        # sample loss
        losses = lfns(active_logits, active_labels)
        return loss, losses

    def forward(self, input_ids, attention_mask, token_type_ids, labels, words_ids, sentence_num):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_out = self.bert_drop(output['last_hidden_state'])
        logits = self.output_layer(bert_out)
        average_loss, losses = self.loss_fn(logits, labels, attention_mask, self.num_tag)
        return {'average_loss': average_loss, 'losses': losses, 'logits': logits,
                'hidden_states': output['last_hidden_state']}



