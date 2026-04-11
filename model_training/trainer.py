# 4_model_training/trainer.py

import numpy as np
import pickle
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config):
        self.save_dir = config["trainer"]["save_dir"]
        self.rf_new_trees = config["trainer"]["rf_new_trees"]
        os.makedirs(self.save_dir, exist_ok=True)

        self.rf = RandomForestClassifier(
            n_estimators = config["trainer"]["rf_n_estimators"],
            max_depth = config["trainer"]["rf_max_depth"],
            class_weight = "balanced",
            random_state = 42,
            warm_start = True,
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes = config["trainer"]["mlp_layers"],
            max_iter = 1,
            warm_start = True,
            random_state = 42,
        )

        self.rf_fitted = False
        self.mlp_fitted = False
        self.batch_count = 0

    def fit(self, X, y):
        if self.batch_count == 0:
            self._initial_fit(X, y)
        else:
            self._incremental_fit(X, y)
        self.batch_count += 1
        self.save()

    def _initial_fit(self, X, y):
        logger.info(f"Первичное обучение на {len(X)} строках...")
        weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weight = {0: weights[0], 1: weights[1]}
        self.rf.set_params(class_weight=class_weight)
        self.rf.fit(X, y)
        self.rf_fitted = True
        logger.info(f"RandomForest обучен: {self.rf.n_estimators} деревьев")
        self.mlp.partial_fit(X, y, classes=np.unique(y))
        self.mlp_fitted = True
        logger.info("MLP обучен")

    def _incremental_fit(self, X, y):
        logger.info(f"Дообучение на батче {self.batch_count}, строк: {len(X)}")
        self.rf.n_estimators += self.rf_new_trees
        self.rf.fit(X, y)
        logger.info(f"RandomForest: теперь {self.rf.n_estimators} деревьев")
        self.mlp.partial_fit(X, y)
        logger.info("MLP: веса обновлены")

    def predict(self, X, model="rf"):
        m = self._get_model(model)
        return m.predict(X)

    def predict_proba(self, X, model="rf"):
        m = self._get_model(model)
        return m.predict_proba(X)

    def _get_model(self, name):
        if name == "rf":
            if not self.rf_fitted:
                raise RuntimeError("RandomForest не обучен")
            return self.rf
        if name == "mlp":
            if not self.mlp_fitted:
                raise RuntimeError("MLP не обучен")
            return self.mlp
        raise ValueError(f"Неизвестная модель: {name}. Доступны: rf, mlp")

    def save(self):
        path = os.path.join(self.save_dir, f"trainer_batch_{self.batch_count:04d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Модели сохранены: {path}")

    @staticmethod
    def load(save_dir):
        files = sorted([
            f for f in os.listdir(save_dir)
            if f.startswith("trainer_batch_") and f.endswith(".pkl")
        ])
        if not files:
            raise FileNotFoundError(f"Нет сохранённых моделей в {save_dir}")
        path = os.path.join(save_dir, files[-1])
        with open(path, "rb") as f:
            trainer = pickle.load(f)
        logger.info(f"Модели загружены: {path}")
        return trainer