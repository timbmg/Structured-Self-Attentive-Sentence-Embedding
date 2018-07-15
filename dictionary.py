import io
import json
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

class Dictionary():

    def __init__(self, data_file=None, min_occurance=None, size=None):

        if data_file is not None:
            self.w2i = dict()
            self.i2w = dict()

            special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
            for st in special_tokens:
                self.i2w[len(self.w2i)] = st
                self.w2i[st] = len(self.w2i)

            print("Creating Dictionary...")
            w2c = OrderedCounter()
            with open(data_file, 'r') as file:
                for i, line in enumerate(file):
                    review = json.loads(line)
                    words = word_tokenize(review['text'].lower())
                    w2c.update(words)

                    if i == size-1:
                        break

            for w, c in w2c.most_common():
                if c >= min_occurance:
                    self.i2w[len(self.w2i)] = w
                    self.w2i[w] = len(self.w2i)

    def __len__(self):
        return len(self.w2i)

    def encode(self, x):
        return [self.w2i.get(w, self.w2i['<unk>']) for w in x]

    def decode(self, x):
        return ' '.join([self.i2w.get(str(i), '<unk>') for i in x])

    def save(self, file_name):
        dictionary = dict(w2i=self.w2i, i2w=self.i2w)
        with io.open(file_name, 'wb') as file:
            data = json.dumps(dictionary, ensure_ascii=False)
            file.write(data.encode('utf8', 'replace'))

    @classmethod
    def load(cls, file_name):
        print("Loading Dictionary...")
        with open(file_name, 'r', encoding='utf8') as file:
            d = json.load(file)

        dictionary = cls()
        dictionary.w2i = d['w2i']
        dictionary.i2w = d['i2w']
        
        return dictionary
