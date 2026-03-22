import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


class DataQuality:
    def __init__(self, config):
        self.thresholds = config["quality"]
        self.report_dir = config["quality"]["report_dir"]
        os.makedirs(self.report_dir, exist_ok=True)

    def evaluate(self, df, batch_index):
        report = {
            "batch_index": batch_index,
            "evaluated_at": datetime.now().isoformat(),
            "n_rows": len(df),

            "missing_per_column": df.isnull().sum().to_dict(),
            "missing_pct_total": round(df.isnull().mean().mean() * 100, 2),

            "duplicates_count": int(df.duplicated().sum()),
            "duplicates_pct": round(df.duplicated().mean() * 100, 2),

            "fraud_rate": round(df["Is_Fraud"].mean() * 100, 2),
            "fraud_count": int(df["Is_Fraud"].sum()),
            "legit_count": int((df["Is_Fraud"] == 0).sum()),

            "amount_mean": round(float(df["Transaction_Amount"].mean()), 2),
            "amount_std": round(float(df["Transaction_Amount"].std()), 2),
            "balance_mean": round(float(df["Account_Balance"].mean()), 2),
        }

        report["passed"] = self._check_thresholds(report)
        self._save_report(report, batch_index)
        return report

    def _check_thresholds(self, report):
        checks = {
            "missing_ok": report["missing_pct_total"] <= self.thresholds["max_missing_pct"],
            "duplicates_ok": report["duplicates_pct"] <= self.thresholds["max_duplicates_pct"],
            "fraud_rate_ok": report["fraud_rate"] <= self.thresholds["max_fraud_rate"],
        }
        report["checks"] = checks
        return all(checks.values())

    def _save_report(self, report, batch_index):
        path = os.path.join(
            self.report_dir, f"quality_batch_{batch_index:04d}.json"
        )

        def convert(obj):
            if isinstance(obj, (np.integer)): return int(obj)
            if isinstance(obj, (np.floating)): return float(obj)
            if isinstance(obj, (np.bool_)): return bool(obj)
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(v) for v in obj]
            return obj

        with open(path, "w") as f:
            json.dump(convert(report), f, indent=2, ensure_ascii=False)

    def load_history(self):
        history = []
        for f in sorted(os.listdir(self.report_dir)):
            if f.startswith("quality_batch_") and f.endswith(".json"):
                with open(os.path.join(self.report_dir, f)) as fp:
                    history.append(json.load(fp))
        return history
