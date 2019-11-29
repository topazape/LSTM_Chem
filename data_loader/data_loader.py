from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from base.base_data_loader import BaseDataLoader
from utils.preprocess import Preprocessor
from utils.smiles_tokenizer import SmilesTokenizer

class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        self.X = []
        self.y = []
        self.max_len = 0
        self.pp = Preprocessor()
        self.st = SmilesTokenizer()
        self.one_hot_dict = self.st.one_hot_dict
        self.tokened_smiles = []

    def get_train_data(self):
        return self.load().clean().tokenize().padding().one_hot_encode()

    def load(self, length=0):
        length = self.config.data_length
        print('loading SMILES...')
        with open(self.config.data_filename) as f:
            self.smiles = [s.rstrip() for s in f]
        if length != 0:
            self.smiles = self.smiles[:length]
        print('done.')
        return self

    def clean(self):
        print('cleaning up SMILES...')
        p = Parallel(n_jobs=-1)
        tmp = p([delayed(self.pp.process)(s) for s in self.smiles])
        self.smiles = [s for s in tmp if s]
        print('done.')
        return self

    def tokenize(self):
        print('tokenizing SMILES...')
        p = Parallel(n_jobs=-1)
        self.tokened_smiles = p([delayed(self.st.tokenize)(s) for s in self.smiles])
        for tokened_smi in self.tokened_smiles:
            length = len(tokened_smi)
            if self.max_len < length:
                self.max_len = length
        print('done')
        return self

    def padding(self):
        padded_smiles = []
        print('padding SMILES...')
        for s in tqdm(self.tokened_smiles):
            padded_s  = ['G'] + s + ['E'] + ['A' for _ in range(self.max_len - len(s))]
            padded_smiles.append(padded_s)
        self.tokened_smiles = padded_smiles
        print('done.')
        return self

    def one_hot_encode(self):
        print('one hot encoding...')
        for atom in tqdm(self.smiles):
            x = [self.to_one_hot[char] for char in atom[:-1]]
            self.X.append(x)

            y = [self.to_one_hot[char] for char in atom[1:]]
            self.y.append(y)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        print('done.')
        return self.X, self.y
