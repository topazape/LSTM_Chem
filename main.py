from data_loader.data_loader import DataLoader
from models.model import LSTMChem
from trainers.trainer import LSTMChemTrainer
from utils.config import process_config
from utils.dirs import create_dirs

CONFIG_FILE = './configs/LSTMChem_config.json'

def main():
    config = process_config(CONFIG_FILE)

    # create the experiments dirs
    create_dirs([config.tensorboard_log_dir, config.checkpoint_dir])

    print('Create the data generator.')
    data_loader = DataLoader(config)

    print('Create the model.')
    model = LSTMChem(config)

    print('Create the trainer')
    trainer = LSTMChemTrainer(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()

if __name__ == '__main__':
    main()
