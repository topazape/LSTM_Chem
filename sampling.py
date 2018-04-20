from models.model import LSTMChem
from generators.generator import LSTMChemGenerator
from utils.config import process_config

CONFIG_FILE = './configs/LSTMChem_config.json'
WEIGHT_FILE = './path/to/weight_file.hdf5'
def main():
    config = process_config(CONFIG_FILE)
    model = LSTMChem(config)
    model.model.load_weights(WEIGHT_FILE)

    generator = LSTMChemGenerator(model, config)
    print(generator.sample())


if __name__ == '__main__':
    main()
