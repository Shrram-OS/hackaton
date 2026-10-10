"""Convert the notebook's best_model.joblib into a portable standard-library weights.json.

Run this on the machine/environment where best_model.joblib was generated:
    python export_weights.py

Expected in the current directory:
    best_model.joblib
    measured.csv
    tokamak_baseline_submission/  (unzip the provided template next to these files)

Development-time dependencies: joblib, pandas, numpy, xgboost.
None of those are needed by the model runtime once weights.json has been exported.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import joblib


def read_input_columns(measured_path: Path) -> list[str]:
    with measured_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {measured_path}") from exc
    # Match the notebook's removal of accidental pandas index columns.
    columns = [c.strip() for c in columns if not c.strip().lower().startswith("unnamed:")]
    return [c for c in columns if c != "shot_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="best_model.joblib", help="joblib bundle produced by main.ipynb")
    parser.add_argument("--measured", default="measured.csv", help="CSV used to train the model")
    parser.add_argument("--out", default=None, help="output weights.json path; auto-detects submission folder by default")
    args = parser.parse_args()

    bundle_path = Path(args.bundle).resolve()
    measured_path = Path(args.measured).resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    if not measured_path.exists():
        raise FileNotFoundError(f"Measured data not found: {measured_path}")

    saved = joblib.load(bundle_path)
    if not isinstance(saved, dict) or "model" not in saved or "feature_columns" not in saved:
        raise ValueError("Unexpected best_model.joblib format: expected dict with model and feature_columns")

    estimator = saved["model"]
    if not hasattr(estimator, "get_booster"):
        raise TypeError(
            "The saved estimator is not XGBoost. This portable runtime supports XGBoost binary:logistic. "
            "Rerun the notebook in an environment where xgboost is installed."
        )
    feature_columns = list(saved["feature_columns"])
    input_columns = read_input_columns(measured_path)

    with tempfile.TemporaryDirectory(prefix="tokamak_export_") as temp_dir:
        tmp_model = Path(temp_dir) / "xgb_model.json"
        estimator.get_booster().save_model(str(tmp_model))
        booster_json = json.loads(tmp_model.read_text(encoding="utf-8"))

    n_features = int(booster_json["learner"]["learner_model_param"]["num_feature"])
    if n_features != len(feature_columns):
        raise ValueError(
            f"Feature count mismatch: XGBoost has {n_features}, bundle says {len(feature_columns)}."
        )

    portable = {
        "format_version": 1,
        "model_type": "xgboost_json",
        "feature_variant": str(saved.get("feature_variant", "Physics")),
        "input_columns": input_columns,
        "feature_columns": feature_columns,
        "validation_threshold_diagnostic_only": saved.get("validation_threshold_diagnostic_only"),
        "lead_time_s_used_for_label": saved.get("lead_time_s"),
        "false_alarm_budget_used_for_validation": saved.get("false_alarm_budget"),
        "n_training_shots": saved.get("n_training_shots"),
        "backend": saved.get("backend", "XGBoost"),
        "xgb_model": booster_json,
    }
    payload = json.dumps(portable, ensure_ascii=False, separators=(",", ":"))

    if args.out:
        destinations = [Path(args.out).resolve()]
    else:
        package_dir = Path.cwd() / "tokamak_baseline_submission"
        if (package_dir / "toksym.json").exists():
            destinations = [package_dir / "weights.json"]
        else:
            destinations = [Path.cwd() / "weights.json"]

    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(f"Saved: {destination.resolve()} ({destination.stat().st_size:,} bytes)")
    print(
        f"Variant={portable['feature_variant']}; features={len(feature_columns)}; "
        f"input channels={len(input_columns)}. Runtime needs only the Python standard library."
    )


if __name__ == "__main__":
    main()
