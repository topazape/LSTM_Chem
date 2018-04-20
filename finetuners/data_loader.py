from data_loader.data_loader import DataLoader

class FineTuneDataLoader(DataLoader):
    def __init__(self, config):
        super(FineTuneDataLoader, self).__init__(config)

    def load_smiles(self, length=0):
        length = self.config.data_length
        print('loading smiles...')
        with open(self.config.finetune_data_filename) as f:
            self.smiles = [s.rstrip() for s in f]
        if length != 0:
            self.smiles = self.smiles[:length]
        else:
            self.smiles = self.smiles
        print('done.')
        return self
