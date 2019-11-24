from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.initializers import RandomNormal
from base.base_model import BaseModel
from utils.smiles_tokenizer import SmilesTokenizer


class LSTMChem(BaseModel):
    def __init__(self, config):
        super(LSTMChem, self).__init__(config)
        self.weight_init = RandomNormal(mean=0.0,
                                        stddev=0.05,
                                        seed=config.seed)
        st = SmilesTokenizer()
        self.build_model(len(st.table))

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
        self.model.compile(
            optimizer=self.config.optimizer,
            loss='categorical_crossentropy',
        )
