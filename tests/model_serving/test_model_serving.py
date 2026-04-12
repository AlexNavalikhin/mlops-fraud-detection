import unittest
import numpy as np
import pandas as pd
import os
import shutil
import tempfile

from model_training.trainer import ModelTrainer
from model_validation.validator import ModelValidator
from model_serving.serving import ModelServing


def make_xy(n=300):
    np.random.seed(42)
    X = pd.DataFrame({
        "Transaction_Amount": np.random.exponential(500, n),
        "Account_Balance": np.random.uniform(100, 50000, n),
        "Age": np.random.randint(18, 80, n),
        "Transaction_Type": np.random.choice([0, 1, 2], n),
        "Device_Type": np.random.choice([0, 1], n),
        "Account_Type": np.random.choice([0, 1], n),
    })
    y = np.random.choice([0, 1], n, p=[0.97, 0.03])
    return X, y


def make_configs(tmp_dir):
    return {
        "trainer": {
            "save_dir": os.path.join(tmp_dir, "models/trainer"),
            "rf_new_trees": 10,
            "rf_n_estimators": 20,
            "rf_max_depth": 5,
            "mlp_layers": (16, 8),
        },
        "validator": {
            "save_dir": os.path.join(tmp_dir, "models/validator"),
            "n_splits": 3,
            "best_metric": "f1",
            "min_batches": 0,
            "val_size": 0.2,
        },
        "serving": {
            "save_dir": os.path.join(tmp_dir, "models/serving"),
        },
    }


def make_serving(tmp_dir):
    config = make_configs(tmp_dir)
    X, y = make_xy()

    trainer = ModelTrainer(config)
    trainer.fit(X, y)

    validator = ModelValidator(config)
    validator.evaluate(trainer, X, y, batch_index=0)

    serving = ModelServing(config)
    serving.load_model(validator)
    return serving, X


class TestModelServing(unittest.TestCase):

    def test_predict_returns_correct_shape(self):
        tmp = tempfile.mkdtemp()
        serving, X = make_serving(tmp)
        preds = serving.predict(X)
        self.assertEqual(len(preds), len(X))
        shutil.rmtree(tmp)

    def test_predict_without_model_raises(self):
        tmp = tempfile.mkdtemp()
        serving = ModelServing(make_configs(tmp))
        X, _ = make_xy()
        with self.assertRaises(RuntimeError):
            serving.predict(X)
        shutil.rmtree(tmp)

    def test_save_and_load_production_model(self):
        tmp = tempfile.mkdtemp()
        config = make_configs(tmp)
        serving, X = make_serving(tmp)
        serving.save_production_model()
        model, meta = ModelServing.load_production_model(config["serving"]["save_dir"])
        self.assertIsNotNone(model)
        self.assertIn("model_name", meta)
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
