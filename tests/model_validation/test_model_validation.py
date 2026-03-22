import unittest
import numpy as np
import pandas as pd
import os
import shutil
import tempfile

from model_training.trainer import ModelTrainer
from model_validation.validator import ModelValidator


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


def make_trainer_config(tmp_dir):
    return {
        "trainer": {
            "save_dir": os.path.join(tmp_dir, "models/trainer"),
            "batch_size": 10,
            "rf_n_estimators": 20,
            "rf_max_depth": 5,
            "mlp_layers": (16, 8),
        }
    }


def make_validator_config(tmp_dir):
    return {
        "validator": {
            "save_dir": os.path.join(tmp_dir, "models/validator"),
            "n_splits": 3,
            "best_metric": "f1",
        }
    }


def make_fitted_trainer(tmp_dir):
    trainer = ModelTrainer(make_trainer_config(tmp_dir))
    X, y = make_xy()
    trainer.fit(X, y)
    return trainer


class TestModelValidator(unittest.TestCase):

    def test_best_model_saved_after_evaluate(self):
        tmp = tempfile.mkdtemp()
        config = make_validator_config(tmp)
        trainer = make_fitted_trainer(tmp)
        validator = ModelValidator(config)
        X, y = make_xy()
        validator.evaluate(trainer, X, y, batch_index=0)
        path = os.path.join(config["validator"]["save_dir"], "best_model.pkl")
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(tmp)

    def test_load_best_model_returns_model_and_meta(self):
        tmp = tempfile.mkdtemp()
        trainer = make_fitted_trainer(tmp)
        validator = ModelValidator(make_validator_config(tmp))
        X, y = make_xy()
        validator.evaluate(trainer, X, y, batch_index=0)
        model, meta = validator.load_best_model()
        self.assertIsNotNone(model)
        self.assertIn("model_name", meta)
        self.assertIn("score", meta)
        shutil.rmtree(tmp)

    def test_cross_validate_returns_mean_f1(self):
        tmp = tempfile.mkdtemp()
        trainer = make_fitted_trainer(tmp)
        validator = ModelValidator(make_validator_config(tmp))
        X, y = make_xy()
        result = validator.cross_validate(trainer, X, y, model_name="rf")
        self.assertIn("mean_f1", result)
        self.assertIn("std_f1", result)
        self.assertGreaterEqual(result["mean_f1"], 0.0)
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
