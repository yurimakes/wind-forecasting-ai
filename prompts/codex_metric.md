Implement or review the official DACON wind competition metric.

Rules:
- Edit only metric-related source files unless explicitly asked.
- Accept `y_true` and `y_pred` as pandas DataFrames.
- Evaluate only rows where actual generation is at least 10 percent of group capacity.
- Use capacities:
  - `kpx_group_1`: 21600
  - `kpx_group_2`: 21600
  - `kpx_group_3`: 21000
- Calculate group NMAE from normalized absolute error by capacity.
- Calculate `one_minus_nmae = 1 - average_group_nmae`.
- Calculate FICR from settlement scores:
  - hourly normalized error <= 0.06 gives score 4
  - hourly normalized error <= 0.08 gives score 3
  - otherwise gives score 0
- Calculate `ficr = earned_settlement / maximum_possible_settlement`.
- Calculate `total_score = 0.5 * one_minus_nmae + 0.5 * ficr`.
- Add a small sanity check when useful.

Do not modify notebooks, data files, output files, or submission files.
