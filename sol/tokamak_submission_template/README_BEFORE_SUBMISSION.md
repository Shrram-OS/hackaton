# TokSym submission scaffold — not ready to upload yet

This folder contains `toksym.json`, `model.py`, and `export_weights.py`. It intentionally does **not** include `weights.json`, because real trained weights must be produced from your own `best_model.joblib`.

## Steps

1. Run `tokamak_four_variant_comparison_roc_auc.ipynb` from top to bottom in the same directory as `measured.csv` and `shots.csv`. This copy chooses by row-level validation ROC-AUC among Raw, Physics, and Dynamics. FormulaSheetProxies is excluded. With the reported metrics, this should choose Dynamics.
2. Check the notebook output: it must say `Выбранный разрешённый вариант для сохранения: Dynamics` (or show the actual best compliant variant if your new dataset gives another result). It must save `best_model.joblib`.
3. Install development dependencies in that Python environment if needed: `pip install joblib pandas numpy xgboost`.
4. From the parent directory containing `best_model.joblib`, `measured.csv`, and this folder, run:

   ```bash
   python tokamak_submission_template/export_weights.py --bundle best_model.joblib --measured measured.csv --out tokamak_submission_template/weights.json
   ```

5. Read exporter output. It must show the expected eligible variant and a positive feature count. For the current metrics, expect `Variant=Dynamics; features=245; input channels=101`.
6. Run the organizer's `toksym check tokamak_submission_template` command (or its Windows equivalent from the supplied README). Only after it passes, upload/drop the **whole folder**.

Do not upload the folder before `weights.json` exists and was just generated from the newly selected `best_model.joblib`. Do not submit `.joblib` directly.

Note: this uses ROC-AUC as a candidate-selection rule because that is your stated priority. TokSym Test run additionally evaluates event-level alarm timing under its Lead time and False-alarm budget controls.
