import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import AdamW
from tqdm.notebook import tqdm
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score as seq_f1, precision_score as seq_precision, recall_score as seq_recall, classification_report as seq_classification
from sklearn.metrics import f1_score as skl_f1, precision_score as skl_precision, recall_score as skl_recall, classification_report as skl_classification

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


class TCModel(nn.Module):
    def __init__(self, num_tag, path):
        super(TCModel, self).__init__()
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


class Evaluation:
    def __init__(self, label_ids, predictions, inv_label_map, average_loss) -> None:
        self.truth = label_ids
        self.preds = predictions
        self.inv_labels = inv_label_map
        self.loss = average_loss

    def create_classification_report(self, raw):
        report = raw.strip().split('\n')
        lines = []
        for line in report[1:]:
            tokens = line.split()
            if line != '':
                if len(tokens) > 5:
                    del tokens[1]
                lines.append(tokens)
        return pd.DataFrame(lines, columns=['Tag', 'Precision', 'Recall', 'F1', 'support'])

    def align_predictions(self):

        preds = np.argmax(self.preds, axis=2)
        batch_size, seq_len = preds.shape

        truth_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if self.truth[i, j] != nn.CrossEntropyLoss().ignore_index:
                    truth_list[i].append(self.inv_labels[self.truth[i][j]])
                    preds_list[i].append(self.inv_labels[preds[i][j]])
        return truth_list, preds_list

    def compute_metrics(self):
        truth_list, preds_list = self.align_predictions()
        flat_truth_list = [item for sublist in truth_list for item in sublist]
        flat_preds_list = [item for sublist in preds_list for item in sublist]

        seq_report = seq_classification(y_true=truth_list, y_pred=preds_list, digits=4)
        sk_report = skl_classification(y_true=flat_truth_list, y_pred=flat_preds_list, digits=4)

        return {
            'Seqeval':

                {"Precision": seq_precision(y_true=truth_list, y_pred=preds_list, average='micro'),
                 "Recall": seq_recall(y_true=truth_list, y_pred=preds_list, average='micro'),
                 "F1": seq_f1(y_true=truth_list, y_pred=preds_list, average='micro'),
                 "Loss": self.loss,
                 "classification": self.create_classification_report(seq_report),
                 "output": {'y_true': truth_list, 'y_pred': preds_list}},

            'Sklearn':

                {"Precision": skl_precision(y_true=flat_truth_list, y_pred=flat_preds_list, average='macro'),
                 "Recall": skl_recall(y_true=flat_truth_list, y_pred=flat_preds_list, average='macro'),
                 "F1": skl_f1(y_true=flat_truth_list, y_pred=flat_preds_list, average='macro'),
                 "Loss": self.loss,
                 "classification": self.create_classification_report(sk_report),
                 "output": {'y_true': flat_truth_list, 'y_pred': flat_preds_list}}
        }


class FineTuneUtils:
    def __init__(self) -> None:
        pass

    def train_fn(self, data_loader, model, optimizer, device, scheduler, config):
        model.train()
        final_loss = 0
        for i, data in enumerate(tqdm(data_loader, total=len(data_loader))):
            for k, v in data.items():
                data[k] = v.to(device)
            outputs = model(**data)
            loss = outputs['average_loss']
            # calculate the gradient and we can say loss.grad where the gradient is stored
            loss.backward()
            # item returns the scalar only not the whole tensor
            final_loss += loss.item()
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        return (final_loss / len(data_loader))

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
                loss = outputs['average_loss']
                logits = outputs['logits']
                final_loss += loss.item()
                if logits is not None:
                    preds = logits if preds is None else torch.cat((preds, logits), dim=0)
                if data['labels'] is not None:
                    labels = data['labels'] if labels is None else torch.cat((labels, data['labels']), dim=0)
            preds = preds.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            evaluation = Evaluation(labels, preds, inv_labels, final_loss)
            metrics = evaluation.compute_metrics()
        return metrics, final_loss


class Metrics:
    def __init__(self, metrics) -> None:
        self.seq_metrics = self.get_metrics(metrics, 'Seqeval')
        self.skl_metrics = self.get_metrics(metrics, 'Sklearn')
        self.seq_results, self.seq_report, self.seq_output = self.generate_results(self.seq_metrics)
        self.skl_results, self.skl_report, self.skl_output = self.generate_results(self.skl_metrics)

    def get_metrics(self, metrics, mode):
        return metrics[mode]

    def clean_report(self, report):
        return report[report['Tag'] != 'accuracy']

    def convert_dict(self, dictionary):
        result = {}
        for k, v in dictionary.items():
            result[k] = [round(v, 4)]
        return result

    def slice_dictionary(self, dictionary):
        keys_for_slicing = ["Precision", "Recall", "F1", "Loss"]
        sliced_dict = {key: dictionary[key] for key in keys_for_slicing}
        return sliced_dict

    def generate_results(self, metrics):
        report = self.clean_report(metrics['classification'])
        output = metrics['output']
        results = pd.DataFrame.from_dict(self.convert_dict(self.slice_dictionary(metrics)))
        return results, report, output


class SaveOutputs:
    def __init__(self, data, config, train_dataloader, val_dataloader, test_dataloader, train_metrics, val_metrics,
                 test_metrics) -> None:
        self.data = data
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.compute_metrics()

    def compute_metrics(self):
        print('Compute Train Metrics')
        self.train_metrics = Metrics(self.train_metrics)
        print('Compute Val Metrics')
        self.val_metrics = Metrics(self.val_metrics)
        print('Compute Test Metrics')
        self.test_metrics = Metrics(self.test_metrics)


class Trainer:
    def __init__(self, corpora, data_name, model_path, tokenizer_path, preprocessor_path) -> None:
        self.data_name = data_name
        self.config = FineTuneConfig()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.preprocessor_path = preprocessor_path
        self.device = self.load_device()
        self.data = corpora[self.data_name]
        self.num_tags = len(self.data['labels'])
        self.TOKENIZER, self.PREPROCESSOR = self.load_tokenizer()
        self.train_dataset, self.train_dataloader = self.load_data('train')
        self.val_dataset, self.val_dataloader = self.load_data('val')
        self.test_dataset, self.test_dataloader = self.load_data('test')

        self.model = self.load_model()
        self.fine_tune = FineTuneUtils()

    def load_data(self, mode):
        data_size = 10
        dataset = TCDataset(
            texts=[x[0] for x in self.data[mode][:]],
            tags=[x[1] for x in self.data[mode][:]],
            label_list=self.data['labels'],
            config=self.config,
            tokenizer=self.TOKENIZER,
            preprocessor=self.PREPROCESSOR)

        data_laoder = torch.utils.data.DataLoader(
            dataset=dataset,
            # if the mode is train return the TRAIN_BATCH_SIZE else return VALID_BATCH_SIZE
            batch_size=self.config.TRAIN_BATCH_SIZE if mode == 'train' else self.config.VALID_BATCH_SIZE,
            num_workers=2)

        return dataset, data_laoder

    def load_device(self):
        use_cuda = torch.cuda.is_available()
        return torch.device("cuda:0" if use_cuda else "cpu")

    def load_tokenizer(self):
        if self.preprocessor_path != None:
            print(f'Loading Preprocessor {self.preprocessor_path}')
            PREPROCESSOR = ArabertPreprocessor(self.preprocessor_path)
        else:
            PREPROCESSOR = None
        print(f'Loading Tokenizer {self.tokenizer_path}')
        TOKENIZER = AutoTokenizer.from_pretrained(self.tokenizer_path)

        return TOKENIZER, PREPROCESSOR

    def load_model(self):
        model = SCModel(self.num_tags, self.model_path)
        model.to(self.device)
        print('MODEL LOADED!')
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
        num_train_steps = int(len(self.train_dataset) / self.config.TRAIN_BATCH_SIZE * self.config.EPOCHS)
        print('Number of training steps: ', num_train_steps)
        self.optimizer = AdamW(optimizer_parameters, lr=self.config.LEARNING_RATE)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(self.config.WARMUP_RATIO * num_train_steps),
            num_training_steps=num_train_steps)
        return model

    def train(self):
        training_loss = self.fine_tune.train_fn(self.train_dataloader, self.model, self.optimizer, self.device,
                                                self.scheduler, self.config)
        return training_loss

    def evaluate(self, dataloader, flag=False):
        eval_metrics, eval_loss = self.fine_tune.eval_fn(dataloader, self.model, self.device, self.data['inv_labels'])
        return eval_metrics, eval_loss

    def training_loop(self):
        for epoch in range(self.config.EPOCHS):
            print()
            print(f'Start Training Epoch:{epoch}')
            print()
            start_time = current_milli_time()
            training_loss = self.train()
            print()
            print('Start Train Evaluation')
            train_metrics, train_loss = self.evaluate(self.train_dataloader, epoch)
            print('Start Val Evaluation')
            val_metrics, val_loss = self.evaluate(self.val_dataloader, epoch)
            print('Start Test Evaluation')
            test_metrics, test_loss = self.evaluate(self.test_dataloader, epoch)
            print(test_metrics['Seqeval']['classification'])
            print(test_metrics['Sklearn']['classification'])

            print(
                f"Training Loss = {training_loss}  Train Loss = {train_loss} Val Loss = {val_loss} Test Loss = {test_loss} ")
            end_time = current_milli_time()
            print(f"End Training Time: {round((end_time - start_time) / 60000, 3)} ms")
        return SaveOutputs(self.data, self.config, self.train_dataloader, self.val_dataloader, self.test_dataloader,
                           train_metrics, val_metrics, test_metrics)
