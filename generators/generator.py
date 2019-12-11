import numpy as np
from tqdm import tqdm
from base.base_generator import BaseGenarator
from utils.smiles_tokenizer import SmilesTokenizer

class LSTMChemGenerator(BaseGenarator):
    def __init__(self, model, config):
        super(LSTMChemGenerator, self).__init__(model, config)
        self.model = model
        self.st = SmilesTokenizer()

    def sample_with_temp(self, preds):
        streched = np.log(preds) / self.config.sampling_temp
        streched_probs = np.exp(streched) / np.sum(np.exp(streched))
        return np.random.choice(len(streched), p=streched_probs)

    def sample(self, num=10, minlen=1, start='G'):
        sampled = []
        for i in tqdm(range(num)):
            start_a = start
            sequence = start_a
            while sequence[-1] == 'E':
                x = self.st.one_hot_encode(st.tokenize(sequence))
                preds = self.model.model.predict(x)[0][-1]
                next_a = self.sample_with_temp(preds)
                sequence += self.st.table[next_a]
            sequence = sequence[1:].rstrip('E')
            if len(sequence) < minlen:
                continue
            else:
                sampled.append(sequence)
        return sampled
