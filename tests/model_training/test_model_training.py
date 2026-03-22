import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile

from model_training.trainer import ModelTrainer


def make_xy(n=300):
    np.random.seed(42)
    X = pd.DataFrame({
        "Transaction_Amount": np.random.exponential(500, n),
        "Account_Balance":    np.random.uniform(100, 50000, n),
        "Age":                np.random.randint(18, 80, n),
        "Transaction_Type":   np.random.choice([0, 1, 2], n),
        "Device_Type":        np.random.choice([0, 1], n),
        "Account_Type":       np.random.choice([0, 1], n),
    })
    y = np.random.choice([0, 1], n, p=[0.97, 0.03])
    return X, y


def make_config(tmp_dir):
    return {
        "trainer": {
            "save_dir":        os.path.join(tmp_dir, "models/trainer"),
            "batch_size":      10,
            "rf_n_estimators": 20,
            "rf_max_depth":    5,
            "mlp_layers":      (16, 8),
        }
    }


class TestModelTrainer(unittest.TestCase):

    def test_fit_initial_sets_fitted_flags(self):
        tmp = tempfile.mkdtemp()
        trainer = ModelTrainer(make_config(tmp))
        X, y = make_xy()
        trainer.fit(X, y)
        self.assertTrue(trainer.rf_fitted)
        self.assertTrue(trainer.mlp_fitted)
        shutil.rmtree(tmp)

    def test_predict_returns_correct_shape(self):
        tmp = tempfile.mkdtemp()
        trainer = ModelTrainer(make_config(tmp))
        X, y = make_xy()
        trainer.fit(X, y)
        for model in ["rf", "mlp"]:
            preds = trainer.predict(X, model=model)
            self.assertEqual(len(preds), len(X))
        shutil.rmtree(tmp)

    def test_predict_before_fit_raises(self):
        tmp = tempfile.mkdtemp()
        trainer = ModelTrainer(make_config(tmp))
        X, _ = make_xy()
        with self.assertRaises(RuntimeError):
            trainer.predict(X, model="rf")
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)