# Data

The raw SCADA exports are **not** stored in this repository (≈1.27 GB).

**Source:** Kelmarsh wind farm data — Zenodo, DOI [10.5281/zenodo.16807551](https://doi.org/10.5281/zenodo.16807551) (ver. 4), CC-BY-4.0.

**Layout expected by `kelmarsh_2022.py`:** place the Greenbyte exports in a folder
named `Kelmarsh_SCADA_2022_4457/` at the repository root:

```
Kelmarsh_SCADA_2022_4457/
    Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv
    Turbine_Data_Kelmarsh_2_2022-01-01_-_2023-01-01_229.csv
    ... up to turbine 6 ...
    Status_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv
    ... up to turbine 6 ...
```

The clustering notebook does **not** need the raw CSV: it reads the frozen preprocessing
artifacts under `artifacts/kelmarsh_2022/` (`windows_2022.csv.gz`, `fitted_preprocessing.pkl`,
`preprocessing_selection.json`, `preprocessed_2022.csv.gz`), which are **included in this
repository**. Raw data is only required to reproduce the preprocessing stage.
