# LSTM_Chem
This is the implementation of the paper - [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111)
## Changelog
### 2020-03-25
* Changed the code to use tensorflow 2.1.0 (tf.keras)
### 2019-12-23
* Reimplimented all code to use tensorflow 2.0.0 (tf.keras)
* Changed data_loader to use generator to reduce memory usage
* Removed some unused atoms and symbols
* Changed directory layout

## Requirements
This model is built using Python 3.7, and utilizes the following packages;

* numpy 1.18.2
* tensorflow 2.1.0
* tqdm 4.43.0
* Bunch 1.0.1
* matplotlib 3.1.2
* RDKit 2019.09.3
* scikit-learn 0.22.2.post1

I strongly recommend using GPU version of tensorflow.
Learning this model with all the data is very slow in CPU mode (about 9 hrs / epoch).
Since tensorflow 2.1.0 depends on CUDA 10.1, be careful that your environment accepts the correct version.  
RDKit and matplotlib are used for SMILES cleanup, validation, and visualization of molecules and their properties.
To install RDKit, I strongly recommend using Anaconda (See [this document](https://www.rdkit.org/docs/Install.html)). Building RDKit from source is hard.  
Scikit-learn is used for PCA.

## Usage
### Training
Just run below. However, all the data is used according to the default setting. So please be careful, it will take a long time.
If you don't have enough time, set `data_length` to a different value in `base_config.json`.
```console
$ python train.py
```
After training, `experiments/{exp_name}/{YYYY-mm-dd}/config.json` is generated.
It's a copy of `base_config.json` with additional settings for internal varibale. Since it is used for generation, be careful when rewriting.
### Generation
See `example_Randomly_generate_SMILES.ipynb`.
### fine-tuning
See `example_Fine-tuning_for_TRPM8.ipynb`.

## Detail
### Configuration
See `base_config.json`. If you want to change, please edit this file before training.

| parameters | meaning |
| ---- | ---- |
| exp_name | experiment name (default: `LSTM_Chem`) |
| data_filename | filepath for training the model (`SMILES file with newline as delimiter`) |
| data_length | number of SMILES for training. If you set 0, all the data is used (default: `0`) |
| units | size of hidden state vector of two LSTM layers (default: `256`, see the paper) |
| num_epochs | number of epochs (default: `22`, see the paper) |
| optimizer | optimizer (default: `adam`) |
| seed | random seed (default: `71`) |
| batch_size | batch size (default: `256`) |
| validation_split | split ratio for validation (default: `0.10`) |
| varbose_training | verbosity mode (default: `True`) |
| checkpoint_monitor | quantity to monitor (default: `val_loss`) |
| checkpoint_mode | one of {`auto`, `min`, `max`} (default: `min`) |
| checkpoint_save_best_only | the latest best model according to the quantity monitored will not be overwritten (default: `False`)|
| checkpoint_save_weights_only | If True, then only the model's weights will be saved (default: `True`)|
| checkpoint_verbose | verbosity mode while `ModelCheckpoint` (default: `1`) |
| tensorboard_write_graph | whether to visualize the graph in TensorBoard (defalut: `True`) |
| sampling_temp | sampling temperature (default: `0.75`, see the paper) |
| smiles_max_length | maximum size of generated SMILES (symbol) length (default: `128`)|
| finetune_epochs | epochs for fine-tuning (default: `12`, see the paper) |
| finetune_batch_size | batch size of finetune (default: `1`) |
| finetune_filename | filepath for fine-tune the model (`SMILES file with newline as delimiter`) |
### Preparing Dataset
#### Get database from ChEMBL
Download SQLite dump for ChEMBL25 (ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_25_sqlite.tar.gz),
which is 3.3 GB compressed, and 16 GB uncompressed.  
Unpack it the usual way, `cd` into the directory, and open the database using sqlite console.
#### Extract SMILES for training
```console
$ sqlite3 chembl_25.db
SQLite version 3.30.1 2019-10-10 20:19:45
Enter ".help" for usage hints.
sqlite> .output dataset.smi
```
You can get SMILES that annotated nM activities according to the following SQL query.
```sql
SELECT
  DISTINCT canonical_smiles
FROM
  compound_structures
WHERE
  molregno IN (
    SELECT
      DISTINCT molregno
    FROM
      activities
    WHERE
      standard_type IN ("Kd", "Ki", "Kb", "IC50", "EC50")
      AND standard_units = "nM"
      AND standard_value < 1000
      AND standard_relation IN ("<", "<<", "<=", "=")
    INTERSECT
    SELECT
      molregno
    FROM
      molecule_dictionary
    WHERE
      molecule_type = "Small molecule"
  );

```
You can get 556134 SMILES in `dataset.smi`. According to the paper,
the dataset was preprocessed and duplicates, salts, and stereochemical information were removed,
SMILES strings with lengths from 34 to 74 (tokens). So I made SMILES clean up script.
Run the following to get cleansed SMILES. It takes about 10 miniutes or more. Please wait.
```console
$ python cleanup_smiles.py datasets/dataset.smi datasets/dataset_cleansed.smi
```
You can get 438552 SMILES. This dataset is used for training.
#### SMILES for fine-tuning
The paper shows 5 TRPM8 antagonists for fine-tuning.
```console
FC(F)(F)c1ccccc1-c1cc(C(F)(F)F)c2[nH]c(C3=NOC4(CCCCC4)C3)nc2c1
O=C(Nc1ccc(OC(F)(F)F)cc1)N1CCC2(CC1)CC(O)c1cccc(Cl)c1O2
O=C(O)c1ccc(S(=O)(=O)N(Cc2ccc(C(F)(F)C3CC3)c(F)c2)c2ncc3ccccc3c2C2CC2)cc1
Cc1cccc(COc2ccccc2C(=O)N(CCCN)Cc2cccs2)c1
CC(c1ccc(F)cc1F)N(Cc1cccc(C(=O)O)c1)C(=O)c1cc2ccccc2cn1
```
You can see this in `datasets/TRPM8_inhibitors_for_fine-tune.smi`.
#### Extract known TRPM8 inhibitors from ChEMBL25
Open the database using sqlite console.
```console
$ sqlite3 chembl_25.db
SQLite version 3.30.1 2019-10-10 20:19:45
Enter ".help" for usage hints.
sqlite> .output known-TRPM8-inhibitors.smi
```
Then issue the following SQL query. I set maximum IC50 activity to 10 uM.
```sql
SELECT
  DISTINCT canonical_smiles
FROM
  activities,
  compound_structures
WHERE
  assay_id IN (
    SELECT
      assay_id
    FROM
      assays
    WHERE
      tid IN (
        SELECT
          tid
        FROM
          target_dictionary
        WHERE
          pref_name = "Transient receptor potential cation channel subfamily M member 8"
      )
  )
  AND standard_type = "IC50"
  AND standard_units = "nM"
  AND standard_value < 10000
  AND standard_relation IN ("<", "<<", "<=", "=")
  AND activities.molregno = compound_structures.molregno;
```
You can get 494 known TRPM8 inhibitors. As described above, clean up the TRPM8 inhibitor SMILES.
Please use the `-ft` option to ignore SMILES strings (tokens) length restriction.
```console
$ python cleanup_smiles.py -ft datasets/known-TRPM8-inhibitors.smi datasets/known_TRPM8-inhibitors_cleansed.smi
```
You can get 477 SMILES. I used this for mere visualization of the results of fine-tuning.
