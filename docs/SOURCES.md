# Sources

## Official Competition References

| Reference | Role |
|---|---|
| [DACON BARAM 2026](https://dacon.io/competitions/official/236727/overview/description) | Forecasting task and organizer context |
| [Official data page](https://dacon.io/competitions/official/236727/data) | Input dataset descriptions and access conditions |
| [Competition rules](https://dacon.io/competitions/official/236727/overview/rules) | Prediction cutoff and information availability |
| [Competition agreement](https://dacon.io/competitions/official/236727/overview/agreement) | Competition participation and data-use terms |
| [Evaluation](https://dacon.io/competitions/official/236727/overview/evaluation) | Evaluation definition |
| [Evaluation-formula code post](https://dacon.io/competitions/official/236727/codeshare/14035) | Competition metric reference |

## Data and Reference Material

The input data is provided for participation in the competition. Obtain it through the official channel under the applicable terms. This repository excludes original datasets, metadata spreadsheets, sample-submission data, trained models and row-level predictions.

`notebooks/Baseline.ipynb` and `notebooks/Metric.ipynb` are baseline and metric references retained with the project. `src/dacon_wind/metric.py` implements the competition evaluation formula. The links above provide the official task, data and evaluation context.

The rules specify an information cutoff of 14:00 KST on the day before the prediction date. The existing feature aggregation uses forecast timestamps; its implementation is explained in [model design](MODEL_DESIGN.md).

## Dependencies

NumPy, pandas, scikit-learn, LightGBM and joblib support the main model workflow. CatBoost and XGBoost are used in additional experiments. Dependencies are installed separately and retain their upstream licenses.

No project-wide reuse license is declared in this repository.
