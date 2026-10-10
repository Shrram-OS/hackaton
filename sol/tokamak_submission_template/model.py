"""TokSym streaming predictor for the compliant Raw / Physics / Dynamics variants.

This runtime intentionally does not implement the FormulaSheetProxies research-only variant.
It loads a trained weights.json exported from the notebook and uses only Python's standard library.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

NAN = float("nan")
EPS = 1e-12


def _num(value: Any) -> float:
    if value is None:
        return NAN
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return NAN
    return result if math.isfinite(result) else NAN


def _finite(values: List[float]) -> List[float]:
    return [x for x in values if math.isfinite(x)]


def _mean(values: List[float]) -> float:
    values = _finite(values)
    return sum(values) / len(values) if values else NAN


def _sum_min1(values: List[float]) -> float:
    values = _finite(values)
    return sum(values) if values else NAN


def _max(values: List[float]) -> float:
    values = _finite(values)
    return max(values) if values else NAN


def _min(values: List[float]) -> float:
    values = _finite(values)
    return min(values) if values else NAN


def _std_sample(values: List[float]) -> float:
    values = _finite(values)
    if len(values) < 2:
        return NAN
    mu = sum(values) / len(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _safe_ratio(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or abs(b) <= EPS:
        return NAN
    result = a / b
    return result if math.isfinite(result) else NAN


def _log1p_nonnegative(x: float) -> float:
    return math.log1p(max(0.0, x)) if math.isfinite(x) else NAN


def _logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -700.0))
    return z / (1.0 + z)


def _row_aggregate(raw: Mapping[str, float], names: List[str], how: str) -> float:
    values = [_num(raw.get(name)) for name in names if name in raw]
    values = _finite(values)
    if not values:
        return NAN
    if how == "mean":
        return sum(values) / len(values)
    if how == "max":
        return max(values)
    if how == "min":
        return min(values)
    if how == "rms":
        return math.sqrt(sum(x * x for x in values) / len(values))
    if how == "abs_sum":
        return sum(abs(x) for x in values)
    raise ValueError(f"Unknown aggregation: {how}")


class Model:
    """Stateful online predictor; the shot history is reset at every shot_start."""

    def __init__(self) -> None:
        weights_path = Path(__file__).with_name("weights.json")
        with weights_path.open("r", encoding="utf-8") as handle:
            self.bundle = json.load(handle)

        self.variant = str(self.bundle.get("feature_variant", ""))
        self.input_columns = list(self.bundle.get("input_columns", []))
        self.feature_columns = list(self.bundle.get("feature_columns", []))
        self.model_type = self.bundle.get("model_type")
        self.xgb_model = self.bundle.get("xgb_model")
        self.history: List[Dict[str, float]] = []
        self.shot_meta: Dict[str, Any] = {}

        if self.model_type != "xgboost_json":
            raise ValueError(
                "weights.json is not an exported trained XGBoost model. "
                "Run export_weights.py on the freshly trained best_model.joblib before submission."
            )
        if self.variant not in {"Raw", "Physics", "Dynamics"}:
            raise ValueError(
                f"Unsupported submission feature variant {self.variant!r}; "
                "only Raw, Physics, and Dynamics are allowed in this runtime."
            )
        if not self.feature_columns or not isinstance(self.xgb_model, dict):
            raise ValueError("weights.json is missing feature columns or XGBoost model data")
        self._parse_xgb()

    def start_shot(self, meta: Optional[Mapping[str, Any]] = None) -> None:
        self.history = []
        self.shot_meta = dict(meta or {})

    def predict(self, t: float, channels: Mapping[str, Any]) -> float:
        if not isinstance(channels, Mapping):
            return 0.0
        raw: Dict[str, float] = {}
        names = self.input_columns or list(channels.keys())
        for name in names:
            if name == "shot_id":
                continue
            raw[name] = _num(channels.get(name))
        raw["t"] = _num(t)

        if self.variant == "Raw":
            features = dict(raw)
        else:
            features = self._physics_features(raw)
            if self.variant == "Dynamics":
                features.update(self._dynamic_features(features))

        row = [_num(features.get(name)) for name in self.feature_columns]
        probability = self._xgb_predict(row)
        self.history.append(dict(features))
        if len(self.history) > 100:
            self.history = self.history[-100:]
        if not math.isfinite(probability):
            return 0.0
        return max(0.0, min(1.0, float(probability)))

    def _physics_features(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Feature engineering matched to make_physics_features() in the notebook.

        These are sensor aggregations and diagnostic contrasts, not reconstructed hidden states
        or computations of equations from the end-of-methodical formula sheet.
        """
        out = dict(raw)

        heating_cols = [c for c in ("nbi_power_MW", "ich_power_MW", "ech_power_MW") if c in out]
        if heating_cols:
            out["feat_input_power_MW"] = _sum_min1([
                max(0.0, out[c]) if math.isfinite(out[c]) else NAN for c in heating_cols
            ])

        chord_cols = [c for c in out if c.startswith("ifr_chord_")]
        if chord_cols:
            out["feat_chord_mean"] = _row_aggregate(out, chord_cols, "mean")
            hi, lo = _row_aggregate(out, chord_cols, "max"), _row_aggregate(out, chord_cols, "min")
            out["feat_chord_spread"] = hi - lo if math.isfinite(hi) and math.isfinite(lo) else NAN
            if all(c in out for c in ("ifr_chord_core", "ifr_chord_edge", "ifr_chord_far_edge")):
                edge_proxy = _mean([abs(out["ifr_chord_edge"]), abs(out["ifr_chord_far_edge"])])
                out["feat_ifr_edge_core_ratio_proxy"] = _safe_ratio(
                    edge_proxy, abs(out["ifr_chord_core"]) + 0.01
                )

        mirnov_cols = [c for c in out if c.startswith("mag_mirnov_")]
        if mirnov_cols:
            out["feat_mirnov_rms"] = _row_aggregate(out, mirnov_cols, "rms")

        dalpha_cols = [c for c in out if c.startswith("dalpha_")]
        if dalpha_cols:
            abs_values = [abs(out[c]) if math.isfinite(out[c]) else NAN for c in dalpha_cols]
            out["feat_dalpha_mean_abs"] = _mean(abs_values)
            out["feat_dalpha_max_abs"] = _max(abs_values)
            out["feat_dalpha_peak_ratio_proxy"] = _safe_ratio(
                out["feat_dalpha_max_abs"], out["feat_dalpha_mean_abs"] + 1e-4
            )

        ece_cols = sorted(c for c in out if c.startswith("ece_"))
        if ece_cols:
            ece_values = [out[c] for c in ece_cols]
            out["feat_ece_mean"] = _mean(ece_values)
            ece_max, ece_min = _max(ece_values), _min(ece_values)
            out["feat_ece_spread"] = ece_max - ece_min if math.isfinite(ece_max) and math.isfinite(ece_min) else NAN
            if len(ece_cols) >= 4:
                mid = len(ece_cols) // 2
                out["feat_ece_half_contrast"] = (
                    _mean([out[c] for c in ece_cols[mid:]])
                    - _mean([out[c] for c in ece_cols[:mid]])
                )

        bolo_cols = [c for c in out if c.startswith("bolo_")]
        if bolo_cols:
            bolo_abs = [abs(out[c]) if math.isfinite(out[c]) else NAN for c in bolo_cols]
            out["feat_bolo_abs_sum"] = _sum_min1(bolo_abs)
            out["feat_bolo_max_abs"] = _max(bolo_abs)
            out["feat_bolo_concentration_proxy"] = _safe_ratio(
                out["feat_bolo_max_abs"], out["feat_bolo_abs_sum"] + 1e-6
            )

        sxr_cols = [c for c in out if c.startswith("sxr_")]
        if sxr_cols:
            sxr_abs = [abs(out[c]) if math.isfinite(out[c]) else NAN for c in sxr_cols]
            out["feat_sxr_abs_sum"] = _sum_min1(sxr_abs)
            out["feat_sxr_max_abs"] = _max(sxr_abs)

        # Ohmic diagnostic proxy V_loop / I_p from the main text's V_loop = I_p R_plasma relation.
        # This does not implement the Spitzer temperature law from the formula sheet.
        if all(c in out for c in ("mag_vloop_V", "mag_ip_MA")):
            out["feat_vloop_per_current_proxy"] = _safe_ratio(
                abs(out["mag_vloop_V"]), abs(out["mag_ip_MA"]) + 0.03
            )

        if "mag_saddle_lm" in out:
            out["feat_locked_mode_abs"] = abs(out["mag_saddle_lm"]) if math.isfinite(out["mag_saddle_lm"]) else NAN
        if "neutron_total" in out:
            out["feat_neutron_log1p"] = _log1p_nonnegative(out["neutron_total"])

        if all(c in out for c in ("mag_ip_MA", "prog_ip_MA")):
            a, b = out["mag_ip_MA"], out["prog_ip_MA"]
            out["feat_ip_tracking_error"] = a - b if math.isfinite(a) and math.isfinite(b) else NAN
        if all(c in out for c in ("mag_bt_T", "prog_bt_T")):
            a, b = out["mag_bt_T"], out["prog_bt_T"]
            out["feat_bt_tracking_error"] = a - b if math.isfinite(a) and math.isfinite(b) else NAN

        prog_heating_cols = [c for c in ("prog_p_nbi_MW", "prog_p_ech_MW", "prog_p_ich_MW") if c in out]
        if prog_heating_cols and "feat_input_power_MW" in out:
            prog_sum = _sum_min1([
                max(0.0, out[c]) if math.isfinite(out[c]) else NAN for c in prog_heating_cols
            ])
            out["feat_heating_tracking_error"] = (
                out["feat_input_power_MW"] - prog_sum
                if math.isfinite(out["feat_input_power_MW"]) and math.isfinite(prog_sum)
                else NAN
            )

        if all(c in out for c in ("gas_flow_ne_meas", "gas_flow_d2_meas")):
            out["feat_neon_to_d2_flow_proxy"] = _safe_ratio(
                abs(out["gas_flow_ne_meas"]), abs(out["gas_flow_d2_meas"]) + 1e-8
            )

        pf_cols = [c for c in ("coil_pf_shape_kA", "coil_pf_vert_kA", "coil_pf_radial_kA") if c in out]
        if pf_cols:
            out["feat_pf_coil_abs_sum"] = _sum_min1([
                abs(out[c]) if math.isfinite(out[c]) else NAN for c in pf_cols
            ])
        if all(c in out for c in ("coil_pf_vert_kA", "coil_pf_radial_kA")):
            a, b = out["coil_pf_radial_kA"], out["coil_pf_vert_kA"]
            out["feat_pf_radial_minus_vert"] = a - b if math.isfinite(a) and math.isfinite(b) else NAN

        return {k: (v if math.isfinite(v) else NAN) for k, v in out.items()}

    def _dynamic_features(self, current: Dict[str, float]) -> Dict[str, float]:
        timeline = self.history[-100:] + [current]
        now_idx = len(timeline) - 1
        dyn: Dict[str, float] = {}
        lag_sources = [
            "mag_ip_MA", "prog_ip_MA", "mag_saddle_lm", "mag_diamag_MJ",
            "mag_vloop_V", "ifr_chord_core", "neutron_total", "feat_dalpha_max_abs",
            "feat_mirnov_rms", "feat_input_power_MW", "feat_bolo_abs_sum", "feat_sxr_abs_sum",
        ]
        for col in lag_sources:
            if col not in current:
                continue
            for lag in (1, 3, 10, 20):
                past_idx = now_idx - lag
                dyn[f"dyn__{col}__lag{lag}"] = timeline[past_idx].get(col, NAN) if past_idx >= 0 else NAN

        delta_sources = [
            "mag_ip_MA", "prog_ip_MA", "mag_saddle_lm", "mag_diamag_MJ", "neutron_total",
            "feat_dalpha_max_abs", "feat_mirnov_rms", "feat_bolo_abs_sum", "feat_input_power_MW",
        ]
        for col in delta_sources:
            if col not in current:
                continue
            for lag in (10, 50):
                past_idx = now_idx - lag
                past = timeline[past_idx] if past_idx >= 0 else None
                previous = past.get(col, NAN) if past is not None else NAN
                now = current.get(col, NAN)
                delta = now - previous if math.isfinite(now) and math.isfinite(previous) else NAN
                dyn[f"dyn__{col}__delta{lag}"] = delta
                if col in {"mag_ip_MA", "prog_ip_MA", "mag_saddle_lm", "feat_bolo_abs_sum", "feat_input_power_MW"}:
                    previous_t = past.get("t", NAN) if past is not None else NAN
                    current_t = current.get("t", NAN)
                    elapsed = current_t - previous_t if math.isfinite(current_t) and math.isfinite(previous_t) else NAN
                    dyn[f"dyn__{col}__slope{lag}"] = (
                        delta / elapsed if math.isfinite(delta) and math.isfinite(elapsed) and elapsed > 0 else NAN
                    )

        rolling_sources = [
            "mag_ip_MA", "mag_saddle_lm", "mag_diamag_MJ", "neutron_total", "feat_dalpha_max_abs",
            "feat_dalpha_peak_ratio_proxy", "feat_mirnov_rms", "feat_bolo_abs_sum", "feat_input_power_MW",
        ]
        for col in rolling_sources:
            if col not in current:
                continue
            for window in (10, 50):
                rows = timeline[max(0, len(timeline) - window):]
                values = [row.get(col, NAN) for row in rows]
                dyn[f"dyn__{col}__mean{window}"] = _mean(values)
                dyn[f"dyn__{col}__std{window}"] = _std_sample(values)
                if col in {"mag_saddle_lm", "feat_dalpha_max_abs", "feat_bolo_abs_sum", "feat_sxr_abs_sum"}:
                    dyn[f"dyn__{col}__max{window}"] = _max(values)

        if "feat_dalpha_peak_ratio_proxy" in current:
            rows = timeline[max(0, len(timeline) - 100):]
            dyn["dyn__dalpha_high_steps_100"] = float(sum(
                1 for row in rows
                if math.isfinite(_num(row.get("feat_dalpha_peak_ratio_proxy")))
                and _num(row.get("feat_dalpha_peak_ratio_proxy")) > 2.0
            ))
        return dyn

    def _parse_xgb(self) -> None:
        learner = self.xgb_model["learner"]
        booster = learner["gradient_booster"]
        if booster.get("name", "gbtree") == "dart":
            raise ValueError("DART boosters are not supported by this runtime")
        self.trees = booster.get("model", {}).get("trees", [])
        if not self.trees:
            raise ValueError("weights.json contains no XGBoost trees")
        raw_base = str(learner.get("learner_model_param", {}).get("base_score", "[0.5]")).strip()
        if raw_base.startswith("[") and raw_base.endswith("]"):
            raw_base = raw_base[1:-1].split(",")[0].strip()
        base_score = float(raw_base)
        self.base_margin = math.log(base_score / (1.0 - base_score)) if 0.0 < base_score < 1.0 else 0.0
        if learner.get("objective", {}).get("name") != "binary:logistic":
            raise ValueError("Only XGBoost objective=binary:logistic is supported")

    def _xgb_predict(self, row: List[float]) -> float:
        margin = self.base_margin
        for tree in self.trees:
            left = tree["left_children"]
            right = tree["right_children"]
            split_idx = tree["split_indices"]
            split_cond = tree["split_conditions"]
            default_left = tree["default_left"]
            node = 0
            hops = 0
            while left[node] != -1:
                feature_idx = int(split_idx[node])
                value = row[feature_idx] if 0 <= feature_idx < len(row) else NAN
                if not math.isfinite(value):
                    go_left = bool(default_left[node])
                else:
                    value32 = struct.unpack("f", struct.pack("f", value))[0]
                    threshold32 = struct.unpack("f", struct.pack("f", float(split_cond[node])))[0]
                    go_left = value32 < threshold32
                node = left[node] if go_left else right[node]
                hops += 1
                if hops > len(left):
                    return 0.0
            margin += float(split_cond[node])
        return _logistic(margin)


if __name__ == "__main__":
    from toksym_model import serve
    serve(Model())
