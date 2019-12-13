import numpy as np
from tqdm import tqdm
from lstm_chem.utils.smiles_tokenizer import SmilesTokenizer


class LSTMChemGenerator(object):
    def __init__(self, modeler, config):
        self.config = config
        self.model = modeler.model
        self.st = SmilesTokenizer()

    def sample_with_temp(self, preds):
        streched = np.log(preds) / self.config.sampling_temp
        streched_probs = np.exp(streched) / np.sum(np.exp(streched))
        return np.random.choice(range(len(streched)), p=streched_probs)

    def sample(self, num=10000, start='G'):
        sampled = []
        for _ in tqdm(range(num)):
            start_a = start
            sequence = start_a
            while sequence[-1] != 'E' and len(
                    sequence) < self.config.smiles_max_length:
                x = self.st.one_hot_encode(self.st.tokenize(sequence))
                preds = self.model.predict(x)[0][-1]
                next_a = self.sample_with_temp(preds)
                sequence += self.st.table[next_a]
            sequence = sequence[1:].rstrip('E')
            sampled.append(sequence)
        return sampled
