# LSTM_Chem
This is the implementation of the paper - [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111)
## Usage

## Preparing Dataset
Download SQLite dump for ChEMBL25 (ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_25), which is 3.3 GB compressed, and 16 GB uncompressed.  
Unpack it the usual way, cd into the directory, and open the database using sqlite console.
```console
$ sqlite3 chembl_25.db
SQLite version 3.30.1 2019-10-10 20:19:45
Enter ".help" for usage hints.
sqlite> .output dataset.smi
```
### Extract SMILES for training
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
You can get 556134 SMILES.  
According to the paper, the dataset was preprocessed and duplicates, salts, and stereochemical information were removed. So I made SMILES clean up script. Run the following to get cleansed SMILES. It takes about 10 miniutes or more. Please wait.
```console
$ python cleanup_smiles.py datasets/dataset.smi datasets/dataset_cleansed.smi
```
You can get 524812 SMILES. This dataset is used for training.
### SMILES for fine-tuning
The article shows 5 TRPM8 antagonists for fine-tuning.
```console
FC(F)(F)c1ccccc1-c1cc(C(F)(F)F)c2[nH]c(C3=NOC4(CCCCC4)C3)nc2c1
O=C(Nc1ccc(OC(F)(F)F)cc1)N1CCC2(CC1)CC(O)c1cccc(Cl)c1O2
O=C(O)c1ccc(S(=O)(=O)N(Cc2ccc(C(F)(F)C3CC3)c(F)c2)c2ncc3ccccc3c2C2CC2)cc1
Cc1cccc(COc2ccccc2C(=O)N(CCCN)Cc2cccs2)c1
CC(c1ccc(F)cc1F)N(Cc1cccc(C(=O)O)c1)C(=O)c1cc2ccccc2cn1
```
#### Extract TRPM8 inhibitors
```console
$ sqlite3 chembl_25.db
SQLite version 3.30.1 2019-10-10 20:19:45
Enter ".help" for usage hints.
sqlite> .output known-TRPM8-inhibitors.smi
```
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
```console
$ python cleanup_smiles.py datasets/known-TRPM8-inhibitors.smi datasets/known_TRPM8-inhibitors_cleansed.smi
```
You can get 477 SMILES. I used this for mere visualization of the results of fine-tuning.
