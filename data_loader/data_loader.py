from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from base.base_data_loader import BaseDataLoader
from utils.smiles_tokenizer import SmilesTokenizer

class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        self.X = []
        self.y = []
#        self.config = config
        st = SmilesTokenizer()
        _, self.to_one_hot = st.one_hot_encode('G')

    def get_train_data(self):
        return self.load_smiles().tokenize_smiles().pad_smiles().one_hot_encode()

    def get_test_data(self):
        raise NotImplementedError

    def load_smiles(self, length=0):
        length = self.config.data_length
        print('loading smiles...')
        with open(self.config.data_filename) as f:
            self.smiles = [s.rstrip() for s in f]
        if length != 0:
            self.smiles = self.smiles[:length]
        else:
            self.smiles = self.smiles
        print('done.')
        return self

    def tokenize_smiles(self):
        print('tokenizing {} smiles...'.format(len(self.smiles)))
        st = SmilesTokenizer()
        p = Parallel(n_jobs=-1)
        self.smiles = p([delayed(st.tokenize)(s) for s in tqdm(self.smiles)])
        print('done.')
        return self

    def pad_smiles(self):
        maxlen = max([len(s) for s in self.smiles])
        padded_smiles = []
        print('padding smiles...')
        for s in tqdm(self.smiles):
            padded_s  = ['G'] + s + ['E'] + ['A' for _ in range(maxlen - len(s))]
            padded_smiles.append(padded_s)
        self.smiles = padded_smiles
        print('done.')
        return self

    def one_hot_encode(self):
        print('one hot encoding...')
        for atom in tqdm(self.smiles):
            x = [self.to_one_hot[char] for char in atom[:-1]]
            self.X.append(x)

            y = [self.to_one_hot[char] for char in atom[1:]]
            self.y.append(y)

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        print('done.')
        return self.X, self.y
