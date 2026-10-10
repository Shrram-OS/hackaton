# Tokamak baseline submission

## Contents

- `model.py` — streaming predictor compatible with the `toksym` contract.
- `toksym.json` — manifest for upload/Test run.
- `weights.json` — currently a heuristic smoke-test fallback, **not a trained model**.
- `export_weights.py` — converts `best_model.joblib` saved by the notebook to portable XGBoost JSON weights.

## Build the trained submission (no retraining required if you already ran main_v2.ipynb)

Put this folder next to the files from your notebook run:

- `best_model.joblib`
- `measured.csv`

From that parent folder run:

```bash
python tokamak_baseline_submission/export_weights.py \
  --bundle best_model.joblib \
  --measured measured.csv \
  --out tokamak_baseline_submission/weights.json
```

This exporter requires `joblib` and `xgboost` in the notebook/local environment only. It does not
need `shots.csv` and it does not retrain the model. The resulting `weights.json` embeds the final
XGBoost trees, the exact feature order, the selected feature variant, and the measured-channel list.

Then run the organizer's `toksym check` command on this folder or upload the folder in Test run.
The inference runtime in `model.py` uses the Python standard library only; it does not import
xgboost, scikit-learn, pandas, or numpy.

## Live feature handling

`start_shot(meta)` resets all history. `predict(t, channels)` uses the current measurement and, for
the `Dynamics` variant, only prior rows from the current shot. Lags and rolling windows initialize
with `NaN`/partial-window statistics just as the offline notebook does; no future values or
backfilling are used. The JSON tree evaluator rounds values to float32 to match XGBoost's split
comparisons.

## Important

The bundled fallback weights are there so you can test the interface before export. They are not
trained and should not be used to report model-quality results. Export the trained weights before
running a meaningful Test run. The validation threshold saved in the bundle is diagnostic only;
set Lead time and False-alarm budget in Test run.
