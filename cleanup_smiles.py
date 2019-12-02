#!/usr/bin/env python

import argparse
import os
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import MolStandardize

class Preprocessor(object):
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def process(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normarizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None

def main(input_file, output_file):
    assert os.path.exists(input_file)
    assert not os.path.exists(output_file), f'{output_file} already exists.'

    pp = Preprocessor()

    with open(input_file, 'r') as f:
        smiles = [l.rstrip() for l in f]

    print(f'input SMILES num: {len(smiles)}')
    print('start to clean up')

    p = Parallel(n_jobs=-1)
    pp_smiles = p([delayed(pp.process)(smi) for smi in smiles])
    cl_smiles = [s for s in pp_smiles if s]

    print('done.')
    print(f'output SMILES num: {len(cl_smiles)}')

    with open(output_file, 'w') as f:
        for smi in cl_smiles:
            f.write(smi + '\n')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='remove salts and stereochemical infomation from SMILES')
    parser.add_argument('input', help='input file')
    parser.add_argument('output', help='output file')
    args = parser.parse_args()
    main(args.input, args.output)
