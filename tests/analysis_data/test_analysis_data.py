import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile

from analysis_data.quality import DataQuality
from analysis_data.cleaner import DataCleaner
from analysis_data.apriori import AssociationRulesMiner
from analysis_data.eda import AutoEDA
from analysis_data.drift import DriftDetector


def make_fake_df(n=500):
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "Customer_ID": [f"C{i:05d}" for i in range(n)],
        "Transaction_ID": [f"T{i:06d}" for i in range(n)],
        "Transaction_Date": dates,
        "Transaction_Amount": np.round(np.random.exponential(500, n), 2),
        "Account_Balance": np.round(np.random.uniform(100, 50000, n), 2),
        "Transaction_Type": np.random.choice(["Withdrawal", "Deposit", "Transfer"], n),
        "Merchant_Category": np.random.choice(["Retail", "Online", "Travel"], n),
        "Device_Type": np.random.choice(["Smartphone", "Laptop"], n),
        "Account_Type": np.random.choice(["Savings", "Checking"], n),
        "State": np.random.choice(["NY", "CA", "TX"], n),
        "Age": np.random.randint(18, 80, n),
        "Is_Fraud": np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })
    df.loc[:25, "Transaction_Amount"] = np.nan
    return df


def make_config(tmp_dir):
    return {
        "quality": {
            "report_dir": os.path.join(tmp_dir, "reports/quality"),
            "max_missing_pct": 30.0,
            "max_duplicates_pct": 5.0,
            "max_fraud_rate": 50.0,
        },
        "apriori": {
            "report_dir": os.path.join(tmp_dir, "reports/apriori"),
            "min_support": 0.05,
            "min_confidence": 0.3,
            "min_lift": 1.0,
            "n_rules": 10,
        },
        "eda": {
            "report_dir": os.path.join(tmp_dir, "reports/eda"),
        },
        "drift": {
            "report_dir": os.path.join(tmp_dir, "reports/drift"),
            "threshold":  0.2,
        },
    }


class TestDataQuality(unittest.TestCase):

    def test_evaluate_returns_required_keys(self):
        tmp = tempfile.mkdtemp()
        report = DataQuality(make_config(tmp)).evaluate(make_fake_df(), 0)
        for key in ["n_rows", "missing_pct_total", "fraud_rate", "passed"]:
            self.assertIn(key, report)
        shutil.rmtree(tmp)

    def test_evaluate_saves_json(self):
        tmp    = tempfile.mkdtemp()
        config = make_config(tmp)
        DataQuality(config).evaluate(make_fake_df(), 0)
        path = os.path.join(config["quality"]["report_dir"], "quality_batch_0000.json")
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(tmp)

    def test_passed_false_when_missing_exceeds_threshold(self):
        tmp = tempfile.mkdtemp()
        df  = make_fake_df()
        df.loc[:, "Transaction_Amount"] = np.nan
        config = make_config(tmp)
        config["quality"]["max_missing_pct"] = 5.0
        report = DataQuality(config).evaluate(df, 0)
        self.assertFalse(report["passed"])
        shutil.rmtree(tmp)


class TestDataCleaner(unittest.TestCase):

    def test_fill_missing_no_nans_after_clean(self):
        tmp     = tempfile.mkdtemp()
        df      = make_fake_df()
        config  = make_config(tmp)
        report  = DataQuality(config).evaluate(df, 0)
        cleaned = DataCleaner(config).clean(df, report)
        self.assertEqual(cleaned["Transaction_Amount"].isnull().sum(), 0)
        shutil.rmtree(tmp)

    def test_remove_duplicates(self):
        tmp    = tempfile.mkdtemp()
        df     = make_fake_df(200)
        df     = pd.concat([df, df.iloc[:20]], ignore_index=True)
        config = make_config(tmp)
        report = DataQuality(config).evaluate(df, 0)
        cleaned = DataCleaner(config).clean(df, report)
        self.assertEqual(cleaned.duplicated().sum(), 0)
        shutil.rmtree(tmp)

    def test_remove_invalid_rows(self):
        tmp    = tempfile.mkdtemp()
        df     = make_fake_df()
        df.loc[:5, "Transaction_Amount"] = -999
        config  = make_config(tmp)
        report  = DataQuality(config).evaluate(df, 0)
        cleaned = DataCleaner(config).clean(df, report)
        self.assertTrue((cleaned["Transaction_Amount"] >= 0).all())
        shutil.rmtree(tmp)


class TestAssociationRulesMiner(unittest.TestCase):

    def test_fit_returns_dataframe(self):
        tmp   = tempfile.mkdtemp()
        rules = AssociationRulesMiner(make_config(tmp)).fit(make_fake_df(500), 0)
        self.assertIsInstance(rules, pd.DataFrame)
        shutil.rmtree(tmp)

    def test_fit_saves_json(self):
        tmp    = tempfile.mkdtemp()
        config = make_config(tmp)
        AssociationRulesMiner(config).fit(make_fake_df(500), 0)
        path = os.path.join(config["apriori"]["report_dir"], "rules_batch_0000.json")
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(tmp)

    def test_rules_have_required_columns(self):
        tmp   = tempfile.mkdtemp()
        rules = AssociationRulesMiner(make_config(tmp)).fit(make_fake_df(500), 0)
        if not rules.empty:
            for col in ["antecedents", "consequents", "support", "confidence", "lift"]:
                self.assertIn(col, rules.columns)
        shutil.rmtree(tmp)


class TestAutoEDA(unittest.TestCase):

    def test_run_returns_stats(self):
        tmp   = tempfile.mkdtemp()
        stats = AutoEDA(make_config(tmp)).run(make_fake_df(), 0)
        self.assertIn("n_rows", stats)
        self.assertIn("fraud_rate", stats)
        shutil.rmtree(tmp)

    def test_run_saves_stats_json(self):
        tmp    = tempfile.mkdtemp()
        config = make_config(tmp)
        AutoEDA(config).run(make_fake_df(), 0)
        path = os.path.join(config["eda"]["report_dir"], "batch_0000", "stats.json")
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(tmp)

    def test_run_saves_plots(self):
        tmp    = tempfile.mkdtemp()
        config = make_config(tmp)
        AutoEDA(config).run(make_fake_df(), 0)
        batch_dir = os.path.join(config["eda"]["report_dir"], "batch_0000")
        self.assertTrue(os.path.exists(
            os.path.join(batch_dir, "numeric_distributions.png")))
        shutil.rmtree(tmp)


class TestDriftDetector(unittest.TestCase):

    def test_no_drift_on_same_data(self):
        tmp      = tempfile.mkdtemp()
        df       = make_fake_df()
        detector = DriftDetector(make_config(tmp))
        detector.set_reference(df)
        report = detector.detect(df, 1)
        self.assertFalse(report["drift_detected"])
        shutil.rmtree(tmp)

    def test_drift_detected_on_different_data(self):
        tmp      = tempfile.mkdtemp()
        detector = DriftDetector(make_config(tmp))
        df_ref   = make_fake_df()
        df_ref["Transaction_Amount"] = 100
        detector.set_reference(df_ref)
        df_new   = make_fake_df()
        df_new["Transaction_Amount"] = 100000
        report = detector.detect(df_new, 1)
        self.assertTrue(report["drift_detected"])
        shutil.rmtree(tmp)

    def test_detect_saves_report(self):
        tmp      = tempfile.mkdtemp()
        config   = make_config(tmp)
        detector = DriftDetector(config)
        detector.set_reference(make_fake_df())
        detector.detect(make_fake_df(), 1)
        path = os.path.join(config["drift"]["report_dir"], "drift_batch_0001.json")
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
