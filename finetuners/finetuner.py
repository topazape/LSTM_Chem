import os
from base.base_trainer import BaseTrain

class LSTMChemFineTuner(BaseTrain):
    def __init__(self, model, data, config):
        super(LSTMChemFineTuner, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.val_loss = []

    def train(self):
        history = self.model.model.fit(
                self.data[0], self.data[1],
                epochs=self.config.finetune_epochs,
                verbose=self.config.verbose_training,
                batch_size=self.config.finetune_batch_size,
                )
#        self.loss.extend(history.history['loss'])
#        self.val_loss.extend(history.history['val_loss'])
