import numpy as np
from joblib import Parallel, delayed
from tensorflow.keras.utils import Sequence
from utils.smiles_tokenizer import SmilesTokenizer


class DataLoader(Sequence):
    def __init__(self, config):
        self.config = config
        self.st = SmilesTokenizer()
        self.max_len = 0

        self.smiles = self._load(length=self.config.data_length)
        self.tokenized_smiles = self._tokenize(self.smiles[:10])

        self.one_hot_dict = self.st.one_hot_dict

    def _load(self, length=0):
        length = self.config.data_length
        print('loading SMILES...')
        with open(self.config.data_filename) as f:
            smiles = [s.rstrip() for s in f]
        if length != 0:
            smiles = smiles[:length]
        print('done.')
        return smiles

    def _tokenize(self, smiles):
        if isinstance(smiles, list):
            print('tokenizing SMILES...')
            if self.config.verbose_training:
                from tqdm import tqdm
                tokenized_smiles = [
                    self.st.tokenize(smi) for smi in tqdm(smiles)
                ]
            else:
                tokenized_smiles = [self.st.tokenize(smi) for smi in smiles]
            tokenized_smiles = tokenized_smiles

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
        data = self.tokenized_smiles[idx * self.config.batch_size:(idx + 1) *
                                     self.config.batch_size]
        data = self._padding(data)
        self.X, self.y = [], []
        for tp_smi in data:
            X = [self.one_hot_dict[symbol] for symbol in tp_smi[:-1]]
            self.X.append(X)
            y = [self.one_hot_dict[symbol] for symbol in tp_smi[1:]]
            self.y.append(y)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        return (self.X, self.y)

    def _pad(self, tokenized_smi):
        return ['G'] + tokenized_smi + ['E'] + [
            'A' for _ in range(self.max_len - len(tokenized_smi))
        ]

    def _padding(self, data):
        print('padding SMILES...')
        padded_smiles = [self._pad(t_smi) for t_smi in data]
        print('done.')
        return padded_smiles
