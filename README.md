# RetailMind AI — Bank Customer Churn Project

This project is based on the uploaded `Classification.ipynb`, which uses a bank customer churn dataset with target column `Exited`. It is distinct from the Telco Customer Churn dataset.

## Structure

- `data/`: raw CSV input (ignored by Git by default)
- `notebooks/`: analysis and end-to-end workflow
- `src/`: reusable Python modules as the project grows
- `models/`: fitted pipelines saved with Joblib
- `reports/`: metrics, comparison tables and plots

## Dataset

Place the bank churn CSV in `data/`. The notebook accepts `churn.csv`, `Churn_Modelling.csv`, or a single CSV file in that folder. It must contain `Exited`.

## Run in VS Code

1. Open this folder in VS Code.
2. Install Python and Jupyter extensions.
3. Create/activate a virtual environment if desired.
4. Install: `python -m pip install -r requirements.txt`
5. Open `notebooks/Bank_Churn_End_to_End.ipynb`.
6. Select the matching kernel and run cells from top to bottom.

## Methodology

The notebook splits before fitting preprocessing. Imputation, encoding, scaling and feature selection are inside Scikit-learn pipelines. Grid search and cross-validation use training data only. The hold-out test set is reserved for final evaluation.

The positive class is `Exited = 1`. Recall is used as the primary model-selection metric, with precision, F1, accuracy and ROC-AUC reported alongside it.

PCA and SMOTE are not included by default: the dataset is modest in dimensionality, and resampling is not necessary to demonstrate sound handling of class imbalance. Class weights and feature selection are evaluated instead.


## Model artifacts

The notebook saves separate generations under `models/`:
- `bank_churn_day1_logistic_baseline.joblib`
- `bank_churn_day2_champion.joblib`
- `bank_churn_day3_tuned_champion.joblib`
- `bank_churn_champion.joblib` (final selected pipeline)

Each `.joblib` file contains the fitted preprocessing and estimator pipeline, not just the final classifier.
