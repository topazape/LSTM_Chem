from models.model import LSTMChem
from finetuners.finetuner import LSTMChemFineTuner
from finetuners.data_loader import FineTuneDataLoader
from generators.generator import LSTMChemGenerator
from utils.config import process_config

CONFIG_FILE = './configs/LSTMChem_config.json'

def main():
    config = process_config(CONFIG_FILE)
    model = LSTMChem(config)
    model.model.load_weights(config.finetune_weight_filename)

    data_loader = FineTuneDataLoader(config)

    finetuner = LSTMChemFineTuner(model, data_loader.get_train_data(), config)
    finetuner.train()

    generator = LSTMChemGenerator(model, config)
    sampled_smiles = generator.sample(config.finetune_sample_num)
    return sampled_smiles

if __name__ == '__main__':
    sampled_smiles = main()
    with open('TRPM8_agonists.smi', 'w') as f:
        for smi in sampled_smiles:
            f.write(smi + '\n')
