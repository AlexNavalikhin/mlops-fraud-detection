import pandas as pd
import numpy as np
import json
import os
import logging
import pickle

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, config):
        self.threshold  = config["drift"]["threshold"]
        self.report_dir = config["drift"]["report_dir"]
        os.makedirs(self.report_dir, exist_ok=True)
        self.reference  = None

    def set_reference(self, df):
        self.reference = self._compute_stats(df)
        logger.info(f"Эталон установлен: {len(self.reference)} признаков")

    def detect(self, df, batch_index):
        if self.reference is None:
            logger.warning("Эталон не установлен, вызови set_reference() на первом батче")
            return {}

        current = self._compute_stats(df)
        report  = {
            "batch_index": batch_index,
            "drifted_features": [],
            "feature_deltas":   {},
            "drift_detected":   False,
        }

        for feature, ref_val in self.reference.items():
            if feature not in current:
                continue
            cur_val = current[feature]
            if ref_val == 0:
                delta = abs(cur_val)
            else:
                delta = abs(cur_val - ref_val) / abs(ref_val)

            report["feature_deltas"][feature] = {
                "reference": round(ref_val, 4),
                "current":   round(cur_val, 4),
                "delta_pct": round(delta * 100, 2),
                "drifted":   delta > self.threshold,
            }

            if delta > self.threshold:
                report["drifted_features"].append(feature)

        report["drift_detected"] = len(report["drifted_features"]) > 0
        self._save_report(report, batch_index)
        self._print_summary(report)
        return report

    def _compute_stats(self, df):
        stats = {}

        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            stats[f"{col}_mean"] = float(df[col].mean())
            stats[f"{col}_std"]  = float(df[col].std())

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            shares = df[col].value_counts(normalize=True)
            for val, share in shares.items():
                stats[f"{col}={val}_share"] = float(share)

        if "Is_Fraud" in df.columns:
            stats["fraud_rate"] = float(df["Is_Fraud"].mean())

        return stats

    def _save_report(self, report, batch_index):
        path = os.path.join(self.report_dir, f"drift_batch_{batch_index:04d}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _print_summary(self, report):
        if report["drift_detected"]:
            logger.info(f"[Drift] Батч {report['batch_index']}: "
                  f"дрейф обнаружен в {len(report['drifted_features'])} признаках")
            for feature in report["drifted_features"]:
                delta = report["feature_deltas"][feature]
                logger.info(f"  {feature}: "
                      f"{delta['reference']} => {delta['current']} "
                      f"(+{delta['delta_pct']}%)")
        else:
            logger.warning(f"[Drift] Батч {report['batch_index']}: дрейф не обнаружен")

    def load_history(self):
        history = []
        for f in sorted(os.listdir(self.report_dir)):
            if f.startswith("drift_batch_") and f.endswith(".json"):
                with open(os.path.join(self.report_dir, f)) as fp:
                    history.append(json.load(fp))
        return history

    def save(self):
        path = os.path.join(self.report_dir, "drift_detector.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(report_dir):
        path = os.path.join(report_dir, "drift_detector.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"DriftDetector не найден: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
