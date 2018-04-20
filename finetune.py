from models.model import LSTMChem
from finetuners.finetuner import LSTMChemFineTuner
from finetuners.data_loader import FineTuneDataLoader
from generators.generator import LSTMChemGenerator
from utils.config import process_config

CONFIG_FILE = './configs/LSTMChem_config.json'
WEIGHT_FILE = './path/to/weight_file.hdf5'
def main():
    config = process_config(CONFIG_FILE)
    model = LSTMChem(config)
    model.model.load_weights(WEIGHT_FILE)

    data_loader = FineTuneDataLoader(config)

    finetuner = LSTMChemFineTuner(model, data_loader.get_train_data(), config)

    generator = LSTMChemGenerator(model, config)
    sampled_smiles = generator.sample(num=10)
    return sampled_smiles

if __name__ == '__main__':
    sampled_smiles = main()
    for smiles in sampled_smiles:
        print(smiles)
