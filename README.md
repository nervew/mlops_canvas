# mlops_canvas
Algotihms for create a full pipeline MLOps

- Pending creation of an environment with the necessary libraries.

- Validate the use and scope of the src and notebooks folders.

- Create a models folder and export the model (DBC)

## Project Structure

```
├── README.md
├── config
│   └── config.json
├── data
│   ├── processed
│   │   ├── backtest_df.parquet
│   │   ├── test_df.parquet
│   │   └── train_df.parquet
│   └── raw
│       └── usuarios_unicos.parquet
├── docs
├── logs
│   └── ts.log
├── notebooks
│   ├── 00 Import_DF.py
│   ├── 01 validation_dataset.py
│   ├── 02 Exploración de datos.py
│   ├── 03 use_split_dataset.py
│   ├── AutoML.py
│   └── Old Notebooks
│       └── (Clone) 02 use_split_dataset.py
├── old
│   └── data
│       ├── processed
│       └── raw
│           └── query.sql
├── reports
│   ├── eda
│   │   └── sweetviz_report.html
│   └── figures
├── requirements.txt
└── src
    ├── Lector_deConsultas.py
    ├── app
    │   ├── EDA
    │   ├── feature_selection
    │   │   ├── factory.py
    │   │   ├── feature_selection.py
    │   │   ├── fs_base.py
    │   │   ├── fs_mutual_info.py
    │   │   ├── fs_rfe.py
    │   │   ├── fs_variance.py
    │   │   └── pipeline.py
    │   └── split_dataset
    │       └── robust_data_splitter.py
    ├── dont_run_me.py
    └── queries
        └── query.sql

```
  