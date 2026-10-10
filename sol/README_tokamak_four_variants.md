# Tokamak: four feature ablations

## Experimental setup

The notebook `tokamak_four_variant_comparison.ipynb` fits four classifiers on the same grouped train/validation split by `shot_id`, with the same target horizon, classifier hyperparameters and event-level threshold-selection procedure. It reports ROC-AUC, PR-AUC, Caught, Late, Missed and false-alarm rate.

| Variant | Contents | Can be submitted under the no-formula-sheet constraint? |
|---|---|---|
| `Raw` | Measured channels + basic type/NaN/inf cleanup; no engineered features | Yes |
| `Physics` | Raw + diagnostic aggregates/contrasts, D-alpha/Mirnov/ECE/bolometer/SXR summaries, tracking errors, measured `V_loop / I_p` proxy based on the main-text relation `V_loop = I_p R_plasma` | Yes, subject to organizer interpretation |
| `Dynamics` | Physics + causal lags, deltas/slopes and backward-looking rolling statistics; history reset per shot | Yes, subject to exact runtime parity |
| `FormulaSheetProxies` | Physics + approximate q95-like, density/current-like, heating/energy and bolometer/heating ratios inspired by formula-sheet items | Research-only. Do not submit if those formulas/proxies are prohibited |

## Important limits

`FormulaSheetProxies` does not implement every formula in the cheat sheet. Many required physical quantities (for example electron temperature profiles, effective charge, true confinement time and calibrated radiation power) are not directly present in the measured input channels. The added variables are proxies from measured quantities, not exact reconstructions of hidden simulator state.

## Model selection and submission

The notebook marks `FormulaSheetProxies` as not eligible and only selects the best among `Raw`, `Physics`, and `Dynamics` for `best_model.joblib`. It selects using Caught under the selected false-alarm budget, then PR-AUC as tie-breaker. ROC-AUC is retained for the report, but the TokSym Test run emphasizes timely caught disruptions under a false-alarm budget.

The notebook has not been trained in this environment because the original `measured.csv` and `shots.csv` were not present. After running it locally, export `best_model.joblib` to `weights.json`, use the matching `model.py`, run `toksym check`, and then run Test run. Verify on an independent Master seed before making strong claims about generalization.
