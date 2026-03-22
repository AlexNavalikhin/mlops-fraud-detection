import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile

from collection_data.collector import DataCollector
from collection_data.storage import RawStorage
from collection_data.meta import MetaCalculator


def make_fake_df(n=3000):
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
        "Is_Fraud": np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })
    df.loc[:50, "Transaction_Amount"] = np.nan
    return df


def make_storage():
    tmp = tempfile.mkdtemp()
    storage = RawStorage(os.path.join(tmp, "raw"))
    return storage, tmp


def make_collector():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "data.csv")
    make_fake_df(3000).to_csv(src, index=False)
    config = {
        "data": {
            "source_path": src,
            "raw_dir":     os.path.join(tmp, "raw"),
            "date_column": "Transaction_Date",
            "batch_size":  1000,
        }
    }
    return DataCollector(config), tmp

class TestRawStorage(unittest.TestCase):
    def test_save_creates_csv(self):
        storage, tmp = make_storage()
        storage.save_batch(make_fake_df(100), {}, batch_index=0)
        path = os.path.join(storage.raw_dir, "batch_0000", "data.csv")
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(tmp)

    def test_save_creates_meta_json(self):
        storage, tmp = make_storage()
        storage.save_batch(make_fake_df(100), {"x": 1}, batch_index=0)
        path = os.path.join(storage.raw_dir, "batch_0000", "meta.json")
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(tmp)

    def test_load_all_batches_correct_size(self):
        storage, tmp = make_storage()
        storage.save_batch(make_fake_df(100), {}, batch_index=0)
        storage.save_batch(make_fake_df(100), {}, batch_index=1)
        self.assertEqual(len(storage.load_all_batches()), 200)
        shutil.rmtree(tmp)


class TestMetaCalculator(unittest.TestCase):
    def test_n_rows_correct(self):
        meta = MetaCalculator().calculate(make_fake_df(500), batch_index=0)
        self.assertEqual(meta["n_rows"], 500)

    def test_fraud_rate_in_range(self):
        meta = MetaCalculator().calculate(make_fake_df(500), batch_index=0)
        self.assertGreaterEqual(meta["fraud_rate"], 0.0)
        self.assertLessEqual(meta["fraud_rate"], 100.0)


class TestDataCollector(unittest.TestCase):
    def test_load_source_returns_dataframe(self):
        collector, tmp = make_collector()
        self.assertIsInstance(collector.load_source(), pd.DataFrame)
        shutil.rmtree(tmp)

    def test_load_source_not_empty(self):
        collector, tmp = make_collector()
        self.assertGreater(len(collector.load_source()), 0)
        shutil.rmtree(tmp)

    def test_split_no_data_loss(self):
        collector, tmp = make_collector()
        df = collector.load_source()
        total = sum(len(b) for b in collector.split_into_batches(df))
        self.assertEqual(total, len(df))
        shutil.rmtree(tmp)

    def test_stream_returns_dataframe(self):
        collector, tmp = make_collector()
        self.assertIsInstance(collector.stream_next_batch(), pd.DataFrame)
        shutil.rmtree(tmp)

    def test_stream_returns_none_when_exhausted(self):
        collector, tmp = make_collector()
        while collector.stream_next_batch() is not None:
            pass
        self.assertIsNone(collector.stream_next_batch())
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
