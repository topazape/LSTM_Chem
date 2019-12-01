import numpy as np
from joblib import Parallel, delayed
from tensorflow.keras.utils import Sequence
from utils.smiles_tokenizer import SmilesTokenizer


class DataLoader(Sequence):
    def __init__(self, config):
        self.config = config
        self.X = []
        self.y = []
        self.max_len = 0
        self.st = SmilesTokenizer()
        self.one_hot_dict = self.st.one_hot_dict
        self.tokenized_smiles = []
        self.padded_smiles = []

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

    def tokenize(self):
        print('tokenizing SMILES...')
        p = Parallel(n_jobs=-1)
        self.tokenized_smiles = p(
            [delayed(self.st.tokenize)(s) for s in self.smiles])
        for tokenized_smi in self.tokenized_smiles:
            length = len(tokenized_smi)
            if self.max_len < length:
                self.max_len = length
        print('done')
        return self

    def _pad(self, tokenized_smi):
        return ['G'] + tokenized_smi + ['E'] + [
            'A' for _ in range(self.max_len - len(tokenized_smi))
        ]

    def padding(self):
        print('padding SMILES...')
        p = Parallel(n_jobs=-1)
        self.padded_smiles = p(
            [delayed(self._pad)(t_smi) for t_smi in self.tokenized_smiles])
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
