import json
import torch
import pickle as pkl
from collections import Counter

class FileHandler():
    def __init__(self, project_folder: str):
        self.project_folder = project_folder

    def create_filename(self, file_name):
        return f'{self.project_folder}/{file_name}'

    def cr_fn(self, file_name):
        return self.create_filename(file_name)

    def load_data(self, path):
        with open(self.cr_fn(path),'r',encoding='utf-8') as f:
          data = []
          sentence = []
          label = []
          for line in f:
            if len(line.split()) !=0:
              if line.split()[0]=='.':
                if len(sentence) > 0:
                  data.append((sentence,label))
                  sentence = []
                  label = []
                continue
              splits = line.split()
              if 'TB' not in splits:
                  sentence.append(splits[0])
                  label.append(splits[1])
        return data

    def load_corpora(self, data_names, data_modes):
        corpora = dict()
        for name in data_names:
            tr = self.load_data(f'{name}/{name}_{data_modes[0]}.txt')
            vl = self.load_data(f'{name}/{name}_{data_modes[1]}.txt')
            te = self.load_data(f'{name}/{name}_{data_modes[2]}.txt')
            data = tr + vl + te
            labels = list(Counter([ label for sentence in data for label in sentence[1]]).keys())
            inv_labels = {i: label for i, label in enumerate(labels)}
            corpora[f'{name}'] = {'train': tr, 'test': te, 'val': vl, 'labels': labels, 'inv_labels': inv_labels}
        return corpora

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
          return self.keys_to_int(data)

    def save_object(self, obj, obj_name):
        with open(obj_name, 'wb') as output:  # Overwrites any existing file.
            pkl.dump(obj, output, pkl.HIGHEST_PROTOCOL)

    def load_object(self, obj_name):
        with open(obj_name, 'rb') as inp:
            obj = pkl.load(inp)
        return obj

    def save_model_state(self, model, model_name):
        torch.save(model.state_dict(), model_name)

    def load_model_state(self, model, model_name):
        model.load_state_dict(torch.load(model_name))
        return model

    def save_model(self, model, model_name):
        torch.save(model, model_name)

    def load_model(model_name):
        model = torch.load(model_name)
        model.eval()
        return model



