import json
import torch
import pickle as pkl
from tqdm.notebook import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split


class GenerateData:
    def __init__(self, fh, dataset, path) -> None:
        self.fh = fh
        self.dataset = dataset
        self.path = path
        print('GENERATE ANERCorp_CamelLab')
        self.ANERCorp_CamelLab = self.read_text()
        print('GENERATE conll2003')
        self.conll2003 = self.generate_conll2003()
        self.corpora = {'ANERCorp_CamelLab': self.ANERCorp_CamelLab, 'conll2003': self.conll2003}

    def read_split(self, split):
        words = []
        tags = []
        sentences = []
        print(f'Generating {split} Split')
        with open(self.fh.cr_fn(f'{self.path}_{split}.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if line != '\n':
                    if '.' == parts[0]:
                        words.append(parts[0])
                        tags.append(parts[1])
                        sentences.append((words, tags))
                        words = []
                        tags = []
                    else:
                        words.append(parts[0])
                        tags.append(parts[1])
        return sentences

    def read_text(self):
        ner_map = {'O': 0, 'B-PERS': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                   'I-MISC': 8}
        ner_inv_map = {v: k for k, v in ner_map.items()}
        tr_dt = self.read_split('train')
        tr, vl = train_test_split(tr_dt, test_size=0.18, random_state=100)
        te = self.read_split('test')
        train = [(id, sen[0], sen[1]) for id, sen in enumerate(tr)]
        val = [(id, sen[0], sen[1]) for id, sen in enumerate(vl)]
        test = [(id, sen[0], sen[1]) for id, sen in enumerate(te)]
        return {'train': train, 'val': val, 'test': test, 'labels': list(ner_map.keys()), 'labels_map': ner_map,
                'inv_labels': ner_inv_map}

    def generate_split_data(self, dataset, split, ner_iv_map):
        sentences = []
        data_split = dataset[split]
        print(f'Generating {split} Split')
        for i in tqdm(range(len(data_split))):
            id = data_split.__getitem__(i)['id']
            tokens = data_split.__getitem__(i)['tokens']
            tags = [ner_iv_map[tid] for tid in data_split.__getitem__(i)['ner_tags']]
            sentences.append((id, tokens, tags))
        return sentences

    def generate_conll2003(self):
        ner_map = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                   'I-MISC': 8}
        ner_inv_map = {v: k for k, v in ner_map.items()}
        tr = self.generate_split_data(self.dataset, 'train', ner_inv_map)
        vl = self.generate_split_data(self.dataset, 'validation', ner_inv_map)
        te = self.generate_split_data(self.dataset, 'test', ner_inv_map)
        return {'train': tr, 'val': vl, 'test': te, 'labels': list(ner_map.keys()), 'labels_map': ner_map,
                'inv_labels': ner_inv_map}


class FileHandler():
    def __init__(self, project_folder: str):
        self.project_folder = project_folder

    def create_filename(self, file_name):
        return f'{self.project_folder}/{file_name}'

    def cr_fn(self, file_name):
        return self.create_filename(file_name)


    def load_corpora(self, dataset, path):
        corpora = GenerateData(self, dataset, path)
        return corpora.corpora

    # def load_data(self, path):
    #     with open(self.cr_fn(path),'r',encoding='utf-8') as f:
    #       data = []
    #       sentence = []
    #       label = []
    #       for line in f:
    #         if len(line.split()) !=0:
    #           if line.split()[0]=='.':
    #             if len(sentence) > 0:
    #               data.append((sentence,label))
    #               sentence = []
    #               label = []
    #             continue
    #           splits = line.split()
    #           if 'TB' not in splits:
    #               sentence.append(splits[0])
    #               label.append(splits[1])
    #     return data
    #
    # def load_corpora(self, data_names, data_modes):
    #     corpora = dict()
    #     for name in data_names:
    #         tr = self.load_data(f'{name}/{name}_{data_modes[0]}.txt')
    #         vl = self.load_data(f'{name}/{name}_{data_modes[1]}.txt')
    #         te = self.load_data(f'{name}/{name}_{data_modes[2]}.txt')
    #         data = tr + vl + te
    #         labels = list(Counter([ label for sentence in data for label in sentence[1]]).keys())
    #         inv_labels = {i: label for i, label in enumerate(labels)}
    #         corpora[f'{name}'] = {'train': tr, 'test': te, 'val': vl, 'labels': labels, 'inv_labels': inv_labels}
    #     return corpora

    def keys_to_int(self, data):
        for key in data.keys():
            wrong_dict = data[key]['inv_labels']
            correct_dict = {int(k): v for k, v in wrong_dict.items()}
            data[key]['inv_labels'] = correct_dict

    def save_json(self, path, data):
        with open(self.cr_fn(path), 'w') as outfile:
            json.dump(data, outfile)

    def load_json(self, path):
        with open(self.cr_fn(path)) as json_file:
          data = json.load(json_file)
          self.keys_to_int(data)
          return data

    def save_object(self, obj, obj_name):
        with open(self.cr_fn(obj_name), 'wb') as output:  # Overwrites any existing file.
            pkl.dump(obj, output, pkl.HIGHEST_PROTOCOL)

    def load_object(self, obj_name):
        with open(self.cr_fn(obj_name), 'rb') as inp:
            obj = pkl.load(inp)
        return obj

    def save_model_state(self, model, model_name):
        torch.save(model.state_dict(), self.cr_fn(model_name))

    def load_model_state(self, model, model_name):
        model.load_state_dict(torch.load(self.cr_fn(model_name)))
        return model

    def save_model(self, model, model_name):
        torch.save(model, self.cr_fn(model_name))

    def load_model(self, model_name):
        model = torch.load(self.cr_fn(model_name))
        model.eval()
        return model



