from glob import glob
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class LSTMChemTrainer(object):
    def __init__(self, modeler, train_data_loader, valid_data_loader):
        self.model = modeler.model
        self.config = modeler.config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.checkpoint_dir,
                    '%s-{epoch:02d}-{val_loss:.2f}.hdf5' %
                    self.config.exp_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            ))
        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.tensorboard_log_dir,
                write_graph=self.config.tensorboard_write_graph,
            ))

    def train(self):
#        history = self.model.fit_generator(
        history = self.model.fit(
            self.train_data_loader,
            steps_per_epoch=self.train_data_loader.__len__(),
            epochs=self.config.num_epochs,
            verbose=self.config.verbose_training,
            validation_data=self.valid_data_loader,
            validation_steps=self.valid_data_loader.__len__(),
            use_multiprocessing=True,
            shuffle=True,
            callbacks=self.callbacks)

        last_weight_file = glob(
            os.path.join(
                f'{self.config.checkpoint_dir}',
                f'{self.config.exp_name}-{self.config.num_epochs:02}*.hdf5')
        )[0]

        assert os.path.exists(last_weight_file)
        self.config.model_weight_filename = last_weight_file

        with open(os.path.join(self.config.exp_dir, 'config.json'), 'w') as f:
            f.write(self.config.toJSON(indent=2))
