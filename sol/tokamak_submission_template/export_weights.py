"""Export best_model.joblib to the portable weights.json used by model.py.

Run from the parent directory that contains best_model.joblib, measured.csv,
and tokamak_submission_template/ (or pass explicit --out).
Development-time dependencies: joblib, pandas, numpy, xgboost.
The runtime model only needs the Python standard library.
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
    columns = [c.strip() for c in columns if not c.strip().lower().startswith("unnamed:")]
    return [c for c in columns if c != "shot_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="best_model.joblib")
    parser.add_argument("--measured", default="measured.csv")
    parser.add_argument("--out", default="tokamak_submission_template/weights.json")
    args = parser.parse_args()

    bundle_path = Path(args.bundle).resolve()
    measured_path = Path(args.measured).resolve()
    out_path = Path(args.out).resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    if not measured_path.is_file():
        raise FileNotFoundError(f"Measured data not found: {measured_path}")

    saved = joblib.load(bundle_path)
    if not isinstance(saved, dict) or "model" not in saved or "feature_columns" not in saved:
        raise ValueError("Unexpected best_model.joblib format: expected dict with model and feature_columns")

    variant = str(saved.get("feature_variant", ""))
    if variant not in {"Raw", "Physics", "Dynamics"}:
        raise ValueError(f"Unsupported submission feature_variant={variant!r}. FormulaSheetProxies is not eligible.")
    estimator = saved["model"]
    if not hasattr(estimator, "get_booster"):
        raise TypeError("The saved estimator is not XGBoost. This portable runtime supports XGBoost binary:logistic.")

    feature_columns = list(saved["feature_columns"])
    input_columns = read_input_columns(measured_path)
    with tempfile.TemporaryDirectory(prefix="tokamak_export_") as temp_dir:
        temp_model = Path(temp_dir) / "xgb_model.json"
        estimator.get_booster().save_model(str(temp_model))
        booster_json = json.loads(temp_model.read_text(encoding="utf-8"))

    n_features = int(booster_json["learner"]["learner_model_param"]["num_feature"])
    if n_features != len(feature_columns):
        raise ValueError(f"Feature count mismatch: XGBoost has {n_features}, bundle says {len(feature_columns)}")

    portable = {
        "format_version": 1,
        "model_type": "xgboost_json",
        "feature_variant": variant,
        "input_columns": input_columns,
        "feature_columns": feature_columns,
        "validation_threshold_diagnostic_only": saved.get("validation_threshold_diagnostic_only"),
        "lead_time_s_used_for_label": saved.get("lead_time_s"),
        "false_alarm_budget_used_for_validation": saved.get("false_alarm_budget"),
        "validation_roc_auc": saved.get("validation_roc_auc"),
        "validation_pr_auc": saved.get("validation_pr_auc"),
        "n_training_shots": saved.get("n_training_shots"),
        "backend": saved.get("backend", "XGBoost"),
        "xgb_model": booster_json,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(portable, ensure_ascii=False, separators=(",", ":"))
    out_path.write_text(payload, encoding="utf-8")
    print(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"Variant={variant}; features={len(feature_columns)}; input channels={len(input_columns)}")
    print("Runtime model uses the Python standard library only.")


if __name__ == "__main__":
    main()
