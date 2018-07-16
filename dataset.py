import os
import io
import json
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from collections import defaultdict

from dictionary import Dictionary

class YelpDataset(Dataset):

    def __init__(self, data_dir, min_occurance=None, size=None, load_from=None):

        self.size = size
        
        data_dir = data_dir
        data_file = os.path.join(data_dir, 'dataset/review.json')

        dictionary_file = os.path.join(data_dir, 'dict.json')
        if not os.path.exists(dictionary_file):
            assert min_occurance is not None
            assert size is not None
            self.dictionary = Dictionary(data_file, min_occurance, size)
            self.dictionary.save(dictionary_file)
        else:
            self.dictionary = Dictionary.load(dictionary_file)

        if load_from is not None:
            self.data = self.load(load_from)
        else:
            dataset_file = os.path.join(data_dir, 'data.json')
            if not os.path.exists(dataset_file):
                self.data = self.create_dataset(data_file)
                self.save(dataset_file)
            self.data = self.load(dataset_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.data[str(idx)]

    def collate_fn(self, data):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        stars = list()
        lengths = list()
        for item in data:
            stars.append(item['stars'])
            lengths.append(item['length'])
        max_length = max(lengths)

        sequences = list()
        for item in data:
            padded = item['words'] + [0] * (max_length - item['length'])
            sequences.append(padded)

        stars = torch.LongTensor(stars).to(device) - 1
        lengths = torch.LongTensor(lengths).to(device)
        sequences = torch.LongTensor(sequences).to(device)
    
        return {
            'stars': stars,
            'lengths': lengths,
            'sequences': sequences
        }

    def create_dataset(self, data_file):

        print("Creating Dataset...")
        data = defaultdict(dict)

        with open(data_file, 'r') as file:
            for line in file:
                review = json.loads(line)
                stars = review['stars']

                words = word_tokenize(review['text'].lower())
                words = self.dictionary.encode(words)
                length = len(words)

                i = len(data)
                data[i]['stars'] = stars
                data[i]['words'] = words
                data[i]['length'] = length

                if i == self.size-1:
                    break
        
        return data

    def save(self, file_name):
        with io.open(file_name, 'wb') as file:
            data = json.dumps(self.data, ensure_ascii=False)
            file.write(data.encode('utf8', 'replace'))

    def load(self, file_name):
        print("Loading Dataset...")
        with open(file_name, 'r', encoding='utf8') as file:
            data = json.load(file)

        return data

                
