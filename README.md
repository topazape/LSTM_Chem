# LSTM_Chem
This is the implementation of the paper - [Generative Recurrent Networks for De Novo Drug Design.](https://doi.org/10.1002/minf.201700111)
## Preparing Dataset
```sql
SELECT
  DISTINCT canonical_smiles
FROM
  activities, assays, compound_structures
WHERE
  standard_type IN ("Kd", "Ki", "Kb", "IC50", "EC50")
  AND standard_units = "nM"
  AND standard_value < 1000 
  AND standard_relation IN ("<", "<<", "<=", "=")
  AND activities.assay_id = assays.assay_id
  AND activities.molregno = compound_structures.molregno
;
```
You can get 565924 smiles.
