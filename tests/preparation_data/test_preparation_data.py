import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile

from preparation_data.preprocessor import DataPreprocessor


def make_fake_df(n=500):
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "Customer_ID": [f"C{i:05d}" for i in range(n)],
        "Customer_Name": [f"Name_{i}" for i in range(n)],
        "Transaction_ID": [f"T{i:06d}" for i in range(n)],
        "Transaction_Date":     dates,
        "Transaction_Amount":   np.round(np.random.exponential(500, n), 2),
        "Account_Balance": np.round(np.random.uniform(100, 50000, n), 2),
        "Age": np.random.randint(18, 80, n),
        "Transaction_Type": np.random.choice(["Withdrawal", "Deposit", "Transfer"], n),
        "Merchant_Category": np.random.choice(["Retail", "Online", "Travel"], n),
        "Device_Type": np.random.choice(["Smartphone", "Laptop"], n),
        "Account_Type": np.random.choice(["Savings", "Checking"], n),
        "State": np.random.choice(["NY", "CA", "TX"], n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Is_Fraud": np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })
    df.loc[:25, "Transaction_Amount"] = np.nan
    df.loc[:10, "Device_Type"] = np.nan
    return df


def make_config(tmp_dir, num_strategy="standard", cat_strategy="ordinal"):
    return {
        "preprocessor": {
            "save_dir":     os.path.join(tmp_dir, "models/preprocessor"),
            "num_strategy": num_strategy,
            "cat_strategy": cat_strategy,
        }
    }


class TestDataPreprocessor(unittest.TestCase):

    def test_fit_transform_no_nans(self):
        tmp = tempfile.mkdtemp()
        X, y = DataPreprocessor(make_config(tmp)).fit_transform(make_fake_df())
        self.assertEqual(X.isnull().sum().sum(), 0)
        shutil.rmtree(tmp)

    def test_fit_transform_returns_correct_shape(self):
        tmp = tempfile.mkdtemp()
        df = make_fake_df(500)
        X, y = DataPreprocessor(make_config(tmp)).fit_transform(df)
        self.assertEqual(len(X), len(df))
        self.assertEqual(len(y), len(df))
        shutil.rmtree(tmp)

    def test_transform_after_fit_no_nans(self):
        tmp = tempfile.mkdtemp()
        prep = DataPreprocessor(make_config(tmp))
        prep.fit_transform(make_fake_df())
        X, _ = prep.transform(make_fake_df(200))
        self.assertEqual(X.isnull().sum().sum(), 0)
        shutil.rmtree(tmp)

    def test_save_and_load(self):
        tmp = tempfile.mkdtemp()
        prep = DataPreprocessor(make_config(tmp))
        prep.fit_transform(make_fake_df())
        loaded = DataPreprocessor.load(make_config(tmp)["preprocessor"]["save_dir"])
        X, _ = loaded.transform(make_fake_df(100))
        self.assertEqual(X.isnull().sum().sum(), 0)
        shutil.rmtree(tmp)

    def test_onehot_encoding_adds_columns(self):
        tmp = tempfile.mkdtemp()
        cfg = make_config(tmp, cat_strategy="onehot")
        X_oh, _ = DataPreprocessor(cfg).fit_transform(make_fake_df())
        cfg2 = make_config(tmp, cat_strategy="ordinal")
        X_or, _ = DataPreprocessor(cfg2).fit_transform(make_fake_df())
        self.assertGreater(X_oh.shape[1], X_or.shape[1])
        shutil.rmtree(tmp)

    def test_minmax_values_in_range(self):
        tmp  = tempfile.mkdtemp()
        cfg  = make_config(tmp, num_strategy="minmax")
        X, _ = DataPreprocessor(cfg).fit_transform(make_fake_df())
        for col in ["Transaction_Amount", "Account_Balance", "Age"]:
            if col in X.columns:
                self.assertGreaterEqual(X[col].min(), -0.01)
                self.assertLessEqual(X[col].max(), 1.01)
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
