import os
import time
from tensorflow.keras import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.initializers import RandomNormal
from lstm_chem.utils.smiles_tokenizer import SmilesTokenizer


class LSTMChem(object):
    def __init__(self, config, session='train'):
        self.config = config
        self.model = None
        self.weight_init = RandomNormal(mean=0.0,
                                        stddev=0.05,
                                        seed=config.seed)
        st = SmilesTokenizer()

        if session == 'train':
            self.build_model(len(st.table))
        else:
            self.model = self.load(self.config.model_arch_filename,
                                   self.config.model_weight_filename)

    def build_model(self, n_table):
        self.n_table = n_table
        self.model = Sequential()
        self.model.add(
            LSTM(units=self.config.units,
                 input_shape=(None, self.n_table),
                 return_sequences=True,
                 kernel_initializer=self.weight_init,
                 dropout=0.3))
        self.model.add(
            LSTM(units=self.config.units,
                 input_shape=(None, self.n_table),
                 return_sequences=True,
                 kernel_initializer=self.weight_init,
                 dropout=0.5))
        self.model.add(
            Dense(units=self.n_table,
                  activation='softmax',
                  kernel_initializer=self.weight_init))

        arch = self.model.to_json(indent=2)
        self.config.model_arch_filename = os.path.join(self.config.exp_dir,
                                                       'model_arch.json')
        with open(self.config.model_arch_filename, 'w') as f:
            f.write(arch)

        self.model.compile(optimizer=self.config.optimizer,
                           loss='categorical_crossentropy')

    def save(self, checkpoint_path):
        assert self.model, 'You have to build the model first.'

        print('Saving model ...')
        self.model.save_weights(checkpoint_path)
        print('model saved.')

    def load(self, model_arch_file, checkpoint_file):
        print(f'Loading model architecture from {model_arch_file} ...')
        with open(model_arch_file) as f:
            self.model = model_from_json(f.read())
        print(f'Loading model checkpoint from {checkpoint_file} ...')
        self.model.load_weights(checkpoint_file)
        print('Loaded the Model.')
        return self.model
