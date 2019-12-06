from utils.config import process_config
from data_loader.data_loader import DataLoader

config = process_config('./configs/LSTMChem_config.json')
dl = DataLoader(config)
