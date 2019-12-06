import numpy as np
from joblib import Parallel, delayed
from tensorflow.keras.utils import Sequence
from utils.smiles_tokenizer import SmilesTokenizer


class DataLoader(Sequence):
    def __init__(self, config, data_type='train'):
        self.config = config
        self.data_type = data_type
        if self.data_type not in ['train', 'valid']:
            raise NameError(f'data_type: \'{self.data_type}\' is not defined.')

        self.max_len = 0
        self.smiles = self._load(length=self.config.data_length)

        self.st = SmilesTokenizer()
        self.one_hot_dict = self.st.one_hot_dict

        self.tokenized_smiles = self._tokenize(self.smiles[:100000])

    def _set_data(self, tokenized_smiles):
        assert tokenized_smiles
        idx = np.arange(len(tokenized_smiles))
        valid_size = int(
            np.ceil(len(tokenized_smiles) * self.config.validation_split))
        np.random.seed(self.config.seed)
        np.random.shuffle(idx)
        if self.data_type == 'train':
            ret = [tokenized_smiles[idx[i]] for i in idx[valid_size:]]
            return ret
        else:
            ret = [tokenized_smiles[idx[i]] for i in idx[:valid_size]]
            return ret

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
        assert isinstance(smiles, list)
        print('tokenizing SMILES...')
        if self.config.verbose_training:
            from tqdm import tqdm
            tokenized_smiles = [self.st.tokenize(smi) for smi in tqdm(smiles)]
        else:
            tokenized_smiles = [self.st.tokenize(smi) for smi in smiles]

        for tokenized_smi in tokenized_smiles:
            length = len(tokenized_smi)
            if self.max_len < length:
                self.max_len = length
        print('done.')
        return tokenized_smiles

    def __len__(self):
        self.tokenized_smiles = self._set_data(self.tokenized_smiles)
        ret = int(
            np.ceil(
                len(self.tokenized_smiles) / float(self.config.batch_size)))
        return ret

    def __getitem__(self, idx):
        self.tokenized_smiles = self._set_data(self.tokenized_smiles)
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
        padded_smiles = [self._pad(t_smi) for t_smi in data]
        return padded_smiles
