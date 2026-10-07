# Sources and Usage Conditions

References checked on 2026-10-06. Development notes reflect the original competition work.

## Official Competition References

| Reference | Purpose |
|---|---|
| [BARAM 2026 competition](https://dacon.io/competitions/official/236727/overview/description) | Task and organizer context |
| [Official data page](https://dacon.io/competitions/official/236727/data) | Dataset provenance, structure and usage restriction |
| [Competition rules](https://dacon.io/competitions/official/236727/overview/rules) | Prediction cutoff, information availability and evaluation obligations |
| [Competition agreement](https://dacon.io/competitions/official/236727/overview/agreement) | Dataset disclosure and code-sharing conditions |
| [Official evaluation page](https://dacon.io/competitions/official/236727/overview/evaluation) | Metric definition |
| [DACON evaluation-formula code post](https://dacon.io/competitions/official/236727/codeshare/14035) | Reference for the competition metric |

The input dataset is supplied for participation in this competition, with other uses restricted by the official data page. Obtain data only through the official channel under the applicable conditions. This repository does not distribute original CSVs, the sample submission, metadata spreadsheets, trained models or row-level predictions. File names are retained to document input and output dependencies.

The rules clarify that information must be available before 14:00 KST on the day preceding the prediction date. This repository's existing aggregation does not enforce that cutoff; see the README limitations before rerunning historical scripts.

## Code Provenance and License Status

`notebooks/Baseline.ipynb` and `notebooks/Metric.ipynb` are baseline/metric references retained with the project. The Python metric implements the competition scoring formula. The exact original source and redistribution license of any copied notebook/code portions have not been established from the repository history; the links above identify the competition references, rather than certifying authorship or a license grant.

No project-wide reuse license is designated. A blanket MIT or similar license has not been applied to material of unverified provenance. If a reuse license is added later, identify which code the author owns and preserve any applicable third-party notices. Installed libraries remain subject to their respective upstream licenses; they are dependencies and are not vendored here.

Public repositories from other participants show that competition code is shared, but do not establish permission for this repository or for dataset redistribution. The specific scope of post-competition GitHub publication has not been confirmed with the organizer.
