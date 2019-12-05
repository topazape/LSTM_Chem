import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from utils.smiles_tokenizer import SmilesTokenizer


class DataLoader(Sequence):
    def __init__(self, config):
        self.config = config
        self.st = SmilesTokenizer()
        self.max_len = 0

        self.smiles = self.load(length=self.config.data_length)
        self.tokenized_smiles = np.array(self.tokenize(self.smiles))

        self.one_hot_dict = self.st.one_hot_dict
        self.tokenized_smiles = []
        self.padded_smiles = []

    def load(self, length=0):
        length = self.config.data_length
        print('loading SMILES...')
        with open(self.config.data_filename) as f:
            smiles = [s.rstrip() for s in f]
        if length != 0:
            smiles = self.smiles[:length]

        print('done.')
        return smiles

    def tokenize(self, smiles):
        tokenized_smiles = []
        if isinstance(smiles, list):
            print('tokenizing SMILES...')
            tokenized_smiles.append([
                np.array(self.st.tokenize(s)) for s in tqdm(smiles)
            ])
            for tokenized_smi in tokenized_smiles:
                length = len(tokenized_smi)
                if self.max_len < length:
                    self.max_len = length
            print('done.')
        return tokenized_smiles

    def __len__(self):
        ret = int(
            np.ceil(
                len(self.tokenized_smiles) / float(self.config.batch_size)))
        return ret

    def __getitem__(self, idx):
        return

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
