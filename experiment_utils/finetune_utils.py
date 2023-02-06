import time
import torch
from torch import nn

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

