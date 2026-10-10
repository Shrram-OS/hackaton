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
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return NAN
    return out if math.isfinite(out) else NAN


def _finite(values: List[float]) -> List[float]:
    return [x for x in values if math.isfinite(x)]


def _mean(values: List[float]) -> float:
    xs = _finite(values)
    return sum(xs) / len(xs) if xs else NAN


def _sum_min1(values: List[float]) -> float:
    xs = _finite(values)
    return sum(xs) if xs else NAN


def _max(values: List[float]) -> float:
    xs = _finite(values)
    return max(xs) if xs else NAN


def _min(values: List[float]) -> float:
    xs = _finite(values)
    return min(xs) if xs else NAN


def _std_sample(values: List[float]) -> float:
    xs = _finite(values)
    if len(xs) < 2:
        return NAN
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _safe_ratio(a: float, b: float, eps: float = EPS) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or abs(b) <= eps:
        return NAN
    out = a / b
    return out if math.isfinite(out) else NAN


def _log1p_nonnegative(x: float) -> float:
    if not math.isfinite(x):
        return NAN
    return math.log1p(max(0.0, x))


def _logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -700.0))
    return z / (1.0 + z)


class Model:

    def __init__(self) -> None:
        path = Path(__file__).with_name("weights.json")
        with path.open("r", encoding="utf-8") as f:
            self.bundle = json.load(f)
        self.variant = str(self.bundle.get("feature_variant", "Physics"))
        self.input_columns = list(self.bundle.get("input_columns", []))
        self.feature_columns = list(self.bundle.get("feature_columns", []))
        self.model_type = self.bundle.get("model_type", "heuristic_fallback")
        self.xgb_model = self.bundle.get("xgb_model")
        self.history: List[Dict[str, float]] = []
        self.shot_meta: Dict[str, Any] = {}

        if self.model_type == "xgboost_json":
            if not isinstance(self.xgb_model, dict) or not self.feature_columns:
                raise ValueError("weights.json: incomplete XGBoost bundle")
            if self.variant not in {"Raw", "Physics", "Dynamics"}:
                raise ValueError("weights.json: unsupported feature_variant")
            self._parse_xgb()

    def start_shot(self, meta: Optional[Mapping[str, Any]] = None) -> None:
        self.history = []
        self.shot_meta = dict(meta or {})

    def predict(self, t: float, channels: Mapping[str, Any]) -> float:
        if not isinstance(channels, Mapping):
            return 0.0
        current_t = _num(t)
        raw: Dict[str, float] = {}
        names = self.input_columns or list(channels.keys())
        for name in names:
            if name == "shot_id":
                continue
            raw[name] = _num(channels.get(name))
        
        raw["t"] = current_t

        if self.variant == "Raw" and self.model_type == "xgboost_json":
            features = dict(raw)
        else:
            features = self._physics_features(raw)
            if self.variant == "Dynamics" and self.model_type == "xgboost_json":
                features.update(self._dynamic_features(features))

        if self.model_type == "xgboost_json":
            p = self._xgb_predict([_num(features.get(name)) for name in self.feature_columns])
        else:
            p = self._heuristic_probability(raw, features)

        
        
        self.history.append(dict(features))
        if len(self.history) > 100:
            self.history = self.history[-100:]
        if not math.isfinite(p):
            return 0.0
        return max(0.0, min(1.0, float(p)))

    
    def _physics_features(self, raw: Dict[str, float]) -> Dict[str, float]:
        out = dict(raw)
        eps_ip = 0.03
        eps = 1e-6

        heating_cols = [c for c in ("nbi_power_MW", "ich_power_MW", "ech_power_MW") if c in out]
        if heating_cols:
            out["feat_input_power_MW"] = _sum_min1([max(0.0, out[c]) if math.isfinite(out[c]) else NAN for c in heating_cols])

        if "mag_bt_T" in out and "mag_ip_MA" in out:
            out["feat_q95_base_proxy"] = 5.0 * (0.95 ** 2) * abs(out["mag_bt_T"]) / (2.89 * (abs(out["mag_ip_MA"]) + eps_ip)) if math.isfinite(out["mag_bt_T"]) and math.isfinite(out["mag_ip_MA"]) else NAN
            out["feat_ip_over_bt"] = _safe_ratio(abs(out["mag_ip_MA"]), abs(out["mag_bt_T"]) + 0.05)

        chord_cols = sorted(c for c in out if c.startswith("ifr_chord_"))
        if chord_cols:
            vals = [out[c] for c in chord_cols]
            out["feat_chord_mean"] = _mean(vals)
            out["feat_chord_spread"] = _max(vals) - _min(vals) if math.isfinite(_max(vals)) and math.isfinite(_min(vals)) else NAN
            if "mag_ip_MA" in out:
                out["feat_chord_per_current_proxy"] = _safe_ratio(abs(out["feat_chord_mean"]), abs(out["mag_ip_MA"]) + eps_ip)
            if all(c in out for c in ("ifr_chord_core", "ifr_chord_edge", "ifr_chord_far_edge")):
                edge_vals = [abs(out["ifr_chord_edge"]), abs(out["ifr_chord_far_edge"])]
                edge_proxy = _mean(edge_vals)
                out["feat_ifr_edge_core_ratio_proxy"] = _safe_ratio(edge_proxy, abs(out["ifr_chord_core"]) + 0.01)

        mirnov_cols = sorted(c for c in out if c.startswith("mag_mirnov_"))
        if mirnov_cols:
            vals = _finite([out[c] for c in mirnov_cols])
            out["feat_mirnov_rms"] = math.sqrt(sum(x * x for x in vals) / len(vals)) if vals else NAN

        dalpha_cols = sorted(c for c in out if c.startswith("dalpha_"))
        if dalpha_cols:
            abs_vals = [abs(out[c]) if math.isfinite(out[c]) else NAN for c in dalpha_cols]
            out["feat_dalpha_mean_abs"] = _mean(abs_vals)
            out["feat_dalpha_max_abs"] = _max(abs_vals)
            out["feat_dalpha_peak_ratio_proxy"] = _safe_ratio(out["feat_dalpha_max_abs"], out["feat_dalpha_mean_abs"] + 1e-4)

        ece_cols = sorted(c for c in out if c.startswith("ece_"))
        if ece_cols:
            ece_vals = [out[c] for c in ece_cols]
            out["feat_ece_mean"] = _mean(ece_vals)
            ece_max, ece_min = _max(ece_vals), _min(ece_vals)
            out["feat_ece_spread"] = ece_max - ece_min if math.isfinite(ece_max) and math.isfinite(ece_min) else NAN
            if len(ece_cols) >= 4:
                mid = len(ece_cols) // 2
                out["feat_ece_half_contrast"] = _mean([out[c] for c in ece_cols[mid:]]) - _mean([out[c] for c in ece_cols[:mid]])

        bolo_cols = sorted(c for c in out if c.startswith("bolo_"))
        if bolo_cols:
            bolo_abs = [abs(out[c]) if math.isfinite(out[c]) else NAN for c in bolo_cols]
            out["feat_bolo_abs_sum"] = _sum_min1(bolo_abs)
            out["feat_bolo_max_abs"] = _max(bolo_abs)
            out["feat_bolo_concentration_proxy"] = _safe_ratio(out["feat_bolo_max_abs"], out["feat_bolo_abs_sum"] + 1e-6)
            if "feat_input_power_MW" in out:
                den = abs(out["feat_input_power_MW"]) if math.isfinite(out["feat_input_power_MW"]) else NAN
                out["feat_bolo_to_heating_proxy"] = _safe_ratio(out["feat_bolo_abs_sum"], max(den, 0.5) if math.isfinite(den) else NAN)

        sxr_cols = sorted(c for c in out if c.startswith("sxr_"))
        if sxr_cols:
            sxr_abs = [abs(out[c]) if math.isfinite(out[c]) else NAN for c in sxr_cols]
            out["feat_sxr_abs_sum"] = _sum_min1(sxr_abs)
            out["feat_sxr_max_abs"] = _max(sxr_abs)

        if "feat_input_power_MW" in out and "mag_diamag_MJ" in out:
            den = abs(out["mag_diamag_MJ"]) + 0.02 if math.isfinite(out["mag_diamag_MJ"]) else NAN
            out["feat_heating_to_energy_s_inv_proxy"] = _safe_ratio(out["feat_input_power_MW"], den)
        if "mag_vloop_V" in out and "mag_ip_MA" in out:
            out["feat_vloop_per_current_proxy"] = _safe_ratio(abs(out["mag_vloop_V"]), abs(out["mag_ip_MA"]) + eps_ip)
        if "mag_saddle_lm" in out:
            out["feat_locked_mode_abs"] = abs(out["mag_saddle_lm"]) if math.isfinite(out["mag_saddle_lm"]) else NAN
        if "neutron_total" in out:
            out["feat_neutron_log1p"] = _log1p_nonnegative(out["neutron_total"])
        if "mag_ip_MA" in out and "prog_ip_MA" in out:
            out["feat_ip_tracking_error"] = out["mag_ip_MA"] - out["prog_ip_MA"] if math.isfinite(out["mag_ip_MA"]) and math.isfinite(out["prog_ip_MA"]) else NAN
        if "mag_bt_T" in out and "prog_bt_T" in out:
            out["feat_bt_tracking_error"] = out["mag_bt_T"] - out["prog_bt_T"] if math.isfinite(out["mag_bt_T"]) and math.isfinite(out["prog_bt_T"]) else NAN
        prog_heating_cols = [c for c in ("prog_p_nbi_MW", "prog_p_ech_MW", "prog_p_ich_MW") if c in out]
        if prog_heating_cols and "feat_input_power_MW" in out:
            prog_sum = _sum_min1([max(0.0, out[c]) if math.isfinite(out[c]) else NAN for c in prog_heating_cols])
            out["feat_heating_tracking_error"] = out["feat_input_power_MW"] - prog_sum if math.isfinite(out["feat_input_power_MW"]) and math.isfinite(prog_sum) else NAN
        if "gas_flow_ne_meas" in out and "gas_flow_d2_meas" in out:
            out["feat_neon_to_d2_flow_proxy"] = _safe_ratio(abs(out["gas_flow_ne_meas"]), abs(out["gas_flow_d2_meas"]) + 1e-8)

        pf_cols = [c for c in ("coil_pf_shape_kA", "coil_pf_vert_kA", "coil_pf_radial_kA") if c in out]
        if pf_cols:
            out["feat_pf_coil_abs_sum"] = _sum_min1([abs(out[c]) if math.isfinite(out[c]) else NAN for c in pf_cols])
        if "coil_pf_vert_kA" in out and "coil_pf_radial_kA" in out:
            out["feat_pf_radial_minus_vert"] = out["coil_pf_radial_kA"] - out["coil_pf_vert_kA"] if math.isfinite(out["coil_pf_radial_kA"]) and math.isfinite(out["coil_pf_vert_kA"]) else NAN

        return out

    
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
            "mag_ip_MA", "prog_ip_MA", "mag_saddle_lm", "mag_diamag_MJ",
            "neutron_total", "feat_dalpha_max_abs", "feat_mirnov_rms",
            "feat_bolo_abs_sum", "feat_input_power_MW",
        ]
        for col in delta_sources:
            if col not in current:
                continue
            for lag in (10, 50):
                past_idx = now_idx - lag
                past = timeline[past_idx] if past_idx >= 0 else None
                pv = past.get(col, NAN) if past is not None else NAN
                cv = current.get(col, NAN)
                delta = cv - pv if math.isfinite(cv) and math.isfinite(pv) else NAN
                dyn[f"dyn__{col}__delta{lag}"] = delta
                if col in {"mag_ip_MA", "prog_ip_MA", "mag_saddle_lm", "feat_bolo_abs_sum", "feat_input_power_MW"}:
                    pt = past.get("t", NAN) if past is not None else NAN
                    ct = current.get("t", NAN)
                    elapsed = ct - pt if math.isfinite(ct) and math.isfinite(pt) else NAN
                    dyn[f"dyn__{col}__slope{lag}"] = delta / elapsed if math.isfinite(delta) and math.isfinite(elapsed) and elapsed > 0 else NAN

        rolling_sources = [
            "mag_ip_MA", "mag_saddle_lm", "mag_diamag_MJ", "neutron_total",
            "feat_dalpha_max_abs", "feat_dalpha_peak_ratio_proxy", "feat_mirnov_rms",
            "feat_bolo_abs_sum", "feat_input_power_MW",
        ]
        for col in rolling_sources:
            if col not in current:
                continue
            for window in (10, 50):
                rows = timeline[max(0, len(timeline) - window):]
                vals = [r.get(col, NAN) for r in rows]
                dyn[f"dyn__{col}__mean{window}"] = _mean(vals)
                dyn[f"dyn__{col}__std{window}"] = _std_sample(vals)
                if col in {"mag_saddle_lm", "feat_dalpha_max_abs", "feat_bolo_abs_sum", "feat_sxr_abs_sum"}:
                    dyn[f"dyn__{col}__max{window}"] = _max(vals)

        if "feat_dalpha_peak_ratio_proxy" in current:
            rows = timeline[max(0, len(timeline) - 100):]
            dyn["dyn__dalpha_high_steps_100"] = float(sum(
                1 for r in rows
                if math.isfinite(r.get("feat_dalpha_peak_ratio_proxy", NAN))
                and r.get("feat_dalpha_peak_ratio_proxy", NAN) > 2.0
            ))
        return dyn

    
    def _parse_xgb(self) -> None:
        learner = self.xgb_model["learner"]
        gb = learner["gradient_booster"]
        
        if gb.get("name", "gbtree") == "dart":
            raise ValueError("DART booster is not supported by this minimal runtime")
        model = gb.get("model", {})
        self.trees = model.get("trees", [])
        if not self.trees:
            raise ValueError("weights.json contains no XGBoost trees")
        param = learner.get("learner_model_param", {})
        raw_base = str(param.get("base_score", "[0.5]")).strip()
        if raw_base.startswith("[") and raw_base.endswith("]"):
            raw_base = raw_base[1:-1].split(",")[0].strip()
        self.base_score = float(raw_base)
        
        self.base_margin = math.log(self.base_score / (1.0 - self.base_score)) if 0.0 < self.base_score < 1.0 else 0.0
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
                fidx = int(split_idx[node])
                value = row[fidx] if 0 <= fidx < len(row) else NAN
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

    
    def _heuristic_probability(self, raw: Dict[str, float], features: Dict[str, float]) -> float:
        saddle = abs(_num(raw.get("mag_saddle_lm")))
        saddle_hist = [abs(_num(r.get("mag_saddle_lm"))) for r in self.history[:100]]
        base_vals = _finite(saddle_hist)
        baseline = _mean(base_vals) if base_vals else 0.003
        
        ratio = saddle / max(baseline if math.isfinite(baseline) else 0.003, 1e-5) if math.isfinite(saddle) else 1.0
        z_score = math.log(max(1.0, ratio))

        
        bolo = _num(features.get("feat_bolo_abs_sum"))
        bolo_hist = _finite([_num(r.get("feat_bolo_abs_sum")) for r in self.history[-50:]])
        bolo_base = _mean(bolo_hist) if bolo_hist else (bolo if math.isfinite(bolo) else 0.0)
        bolo_ratio = bolo / max(bolo_base if math.isfinite(bolo_base) else 0.0, 1e-4) if math.isfinite(bolo) else 1.0
        mirnov = _num(features.get("feat_mirnov_rms"))
        mirnov_hist = _finite([_num(r.get("feat_mirnov_rms")) for r in self.history[-50:]])
        mirnov_base = _mean(mirnov_hist) if mirnov_hist else (mirnov if math.isfinite(mirnov) else 0.0)
        mirnov_ratio = mirnov / max(mirnov_base if math.isfinite(mirnov_base) else 0.0, 1e-4) if math.isfinite(mirnov) else 1.0

        score = -6.0 + 1.15 * z_score + 0.35 * math.log(max(1.0, bolo_ratio)) + 0.35 * math.log(max(1.0, mirnov_ratio))
        
        measured_ip = _num(raw.get("mag_ip_MA"))
        prog_ip = _num(raw.get("prog_ip_MA"))
        if math.isfinite(measured_ip) and math.isfinite(prog_ip) and prog_ip > 0.15 and measured_ip < 0.85 * prog_ip:
            score -= 3.0
        return _logistic(score)


if __name__ == "__main__":
    
    m = Model()
    m.start_shot({"shot_id": 0, "duration": 1.0})
    p = m.predict(0.005, {"mag_saddle_lm": 0.002, "mag_ip_MA": 0.01, "mag_bt_T": 3.3})
    print(f"smoke_test_p={p:.6f}")
