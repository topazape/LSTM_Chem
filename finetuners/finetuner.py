import os
from base.base_trainer import BaseTrain

class LSTMChemFineTuner(object):
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loder = data_loader
        self.config = config
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
