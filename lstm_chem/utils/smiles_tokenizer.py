import copy
import numpy as np


class SmilesTokenizer(object):
    def __init__(self):
        atoms = [
            'Li',
            'Na',
            'Al',
            'Si',
            'Cl',
            'Sc',
            'Zn',
            'As',
            'Se',
            'Br',
            'Sn',
            'Te',
            'Cn',
            'H',
            'B',
            'C',
            'N',
            'O',
            'F',
            'P',
            'S',
            'K',
            'V',
            'I',
        ]
        special = [
            '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's'
        ]
        padding = ['G', 'A', 'E']

        self.table = sorted(atoms, key=len, reverse=True) + special + padding
        self.table_len = len(self.table)

        self.one_hot_dict = {}
        for i, symbol in enumerate(self.table):
            vec = np.zeros(self.table_len, dtype=np.float32)
            vec[i] = 1
            self.one_hot_dict[symbol] = vec

    def tokenize(self, smiles):
        N = len(smiles)
        i = 0
        token = []
        while (i < N):
            for j in range(self.table_len):
                symbol = self.table[j]
                if symbol == smiles[i:i + len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
        return token

    def one_hot_encode(self, tokenized_smiles):
        result = np.array(
            [self.one_hot_dict[symbol] for symbol in tokenized_smiles],
            dtype=np.float32)
        result = result.reshape(1, result.shape[0], result.shape[1])
        return result
