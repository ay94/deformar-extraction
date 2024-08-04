from dataclasses import dataclass, field
import torch
import logging
from tqdm.autonotebook import tqdm

@dataclass
class SentenceData:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_states: torch.Tensor
    losses: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    


@dataclass
class BatchData:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_states: torch.Tensor
    losses: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor

    def detach(self):
        """Detach all tensors to move them to CPU and reduce memory footprint on GPU."""
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.detach().cpu())
            elif isinstance(value, tuple) and all(torch.is_tensor(x) for x in value):
                # If the batch is a tuple of tensors (like hidden states), process each tensor.
                detached_batch = tuple(layer.detach().cpu() for layer in value)
                setattr(self, attr, detached_batch)
            else:
                raise TypeError("Unsupported batch type: {}".format(type(value)))



class ModelOutputProcessor:
    def __init__(self, data_loader, model, device):
        self.data_loader = data_loader
        self.model = model
        self.device = device

    def process_outputs(self):
        self.model.eval()
        sentences = []
        with torch.no_grad():
            for data in tqdm(self.data_loader):
                data = {k: v.to(self.device) for k, v in data.items()}
                data['words_ids'] = data['input_ids']
                data['sentence_num'] = data['input_ids']
                outputs = self.model(**data)
                batch = BatchData(
                    input_ids=data['input_ids'],
                    attention_mask=data['attention_mask'],
                    last_hidden_states=outputs['last_hidden_state'],
                    losses=outputs['losses'],
                    logits=outputs['logits'],
                    labels=data['labels'],
                )
                batch.detach()
                batch_sentences = self.extract_sentences_from_batch(batch)
                sentences.extend(batch_sentences)
        return sentences

    def extract_sentences_from_batch(self, batch: BatchData):
        sentences = []

        loss_start_idx = 0
        unique_values, indices = torch.unique(
            batch.input_ids, return_inverse=True
        )
        active_losses = batch.losses[indices.view(-1) != 0]
        for idx in range(batch.input_ids.size(0)):  # Iterate over each sentence in the batch
            sentence_mask = batch.input_ids[idx] != 0  # Create a mask where the input_ids are not padding

            # Use the mask to filter out padding in attention_mask, last_hidden_state, logits, labels
            if sentence_mask.any():
                input_ids = batch.input_ids[idx][sentence_mask]
                attention_mask = batch.attention_mask[idx][sentence_mask]
                last_hidden_state = batch.last_hidden_states[idx][sentence_mask]
                logits = batch.logits[idx][sentence_mask]
                label = batch.labels[idx][sentence_mask]
                actual_token_count = sentence_mask.sum()  # Number of non-padding tokens
                sentence_loss = active_losses[loss_start_idx:loss_start_idx + actual_token_count]
                loss_start_idx += actual_token_count
                sentence = SentenceData(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                last_hidden_states=last_hidden_state,
                                losses=sentence_loss,
                                logits=logits,
                                labels=label
                            )
                sentences.append(sentence)
        return sentences


class ModelOutputWorkflowManager:
    def __init__(self, model, data_manager, config, split=None):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_manager = data_manager
        self.config = config
        self.splits = list(data_manager.corpus['splits'].keys())
        self.model_outputs = {}
        self.process_model_outputs(split)

    def process_model_outputs(self, split):
        """Process and store model outputs for all configured splits."""
        if split:
            logging.info('Specific Split %s being processed', split)
            self.model_outputs[split] = self.get_split_data(split)
            return
        for split in self.splits:
            if self.get_batch_size(split):  # Ensure the split has a designated batch size
                logging.info('Processing %s Split', split)
                self.model_outputs[split] = self.get_split_data(split)
            else:
                logging.warning('%s split is not configured with a batch size.', split)

    def get_split_data(self, split):
        batch_size = self.get_batch_size(split)
        dataloader = self.data_manager.get_dataloader(split, batch_size)
        if dataloader:
            sentences = ModelOutputProcessor(dataloader, self.model, self.device).process_outputs()
            return sentences
        else:
            logging.warning('No data available for %s split, returning empty list.', split)
            return []
    
    def get_batch_size(self, split):
        """Get the batch size for a given split."""
        return {
            'train': self.config.train_batch_size,
            'test': self.config.test_batch_size,
            'validation': self.config.test_batch_size
        }.get(split, None)

    @property
    def train(self):
        """Return processed outputs for the training split."""
        return self.model_outputs.get('train', [])

    @property
    def test(self):
        """Return processed outputs for the testing split."""
        return self.model_outputs.get('test', [])

    @property
    def validation(self):
        """Return processed outputs for the validation split."""
        return self.model_outputs.get('validation', [])
