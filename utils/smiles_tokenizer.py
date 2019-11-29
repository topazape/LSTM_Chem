import copy
import numpy as np


class SmilesTokenizer(object):
    def __init__(self):
        atoms = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
            'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
            'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
            'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
            'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
            'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
            'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo'
        ]
        special = [
            '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-', 'se', 'te',
            'c', 'n', 'o', 's'
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
