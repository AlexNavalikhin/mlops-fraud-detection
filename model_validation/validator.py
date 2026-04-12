import numpy as np
import pandas as pd
import json
import os
import pickle
import logging
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class ModelValidator:
    def __init__(self, config):
        self.save_dir = config["validator"]["save_dir"]
        self.n_splits = config["validator"]["n_splits"]
        self.metric = config["validator"]["best_metric"]
        self.min_batches = config["validator"].get("min_batches", 5)
        self.val_size = config["validator"].get("val_size", 0.2)
        os.makedirs(self.save_dir, exist_ok=True)
        self.history = []
        self.best = {"score": 0.0, "batch_index": None, "model": None}

    def evaluate(self, trainer, X, y, batch_index):
        report = {"batch_index": batch_index, "models": {}, "can_use_model": False}

        split = int(len(X) * (1 - self.val_size))
        X_val = X.iloc[split:]
        y_val = y[split:]

        for model_name in ["rf", "mlp"]:
            try:
                metrics = self._compute_metrics(trainer, X_val, y_val, model_name)
                report["models"][model_name] = metrics
                logger.info(
                    f"Батч {batch_index} [{model_name}] "
                    f"F1={metrics['f1']:.3f} "
                    f"AUC={metrics['roc_auc']:.3f}"
                )
            except Exception as e:
                logger.warning(f"Ошибка оценки {model_name}: {e}")

        if batch_index >= self.min_batches:
            self._update_best(trainer, report, batch_index)
            report["can_use_model"] = True
        else:
            logger.info(f"Батч {batch_index}: пропускаем выбор лучшей модели "
                        f"(min_batches={self.min_batches})")

        self._save_report(report, batch_index)
        self.history.append(report)
        return report

    def _compute_metrics(self, trainer, X, y, model_name):
        preds = trainer.predict(X, model=model_name)
        proba = trainer.predict_proba(X, model=model_name)[:, 1]
        return {
            "f1":        round(float(f1_score(y, preds, zero_division=0)), 4),
            "roc_auc":   round(float(roc_auc_score(y, proba)), 4),
            "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
            "recall":    round(float(recall_score(y, preds, zero_division=0)), 4),
        }

    def cross_validate(self, trainer, X, y, model_name="rf"):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            trainer._get_model(model_name).fit(X_tr, y_tr)
            preds = trainer.predict(X_val, model=model_name)
            score = f1_score(y_val, preds, zero_division=0)
            fold_scores.append(score)
            logger.info(f"  Фолд {fold+1}/{self.n_splits}: F1={score:.3f}")

        mean_score = round(float(np.mean(fold_scores)), 4)
        std_score = round(float(np.std(fold_scores)), 4)
        logger.info(f"TimeSeriesCV [{model_name}]: F1={mean_score} +/- {std_score}")
        return {"mean_f1": mean_score, "std_f1": std_score, "folds": fold_scores}  # ← добавили

    def _update_best(self, trainer, report, batch_index):
        for model_name, metrics in report["models"].items():
            score = metrics.get(self.metric, 0.0)
            if score >= self.best["score"]:
                self.best = {
                    "score":       score,
                    "batch_index": batch_index,
                    "model":       model_name,
                }
                self._save_best_model(trainer, model_name, score, batch_index)
                logger.info(
                    f"Новая лучшая модель: {model_name} "
                    f"{self.metric}={score:.3f} "
                    f"(батч {batch_index})"
                )

    def _save_best_model(self, trainer, model_name, score, batch_index):
        model = trainer._get_model(model_name)
        path  = os.path.join(self.save_dir, "best_model.pkl")
        meta  = {
            "model_name":  model_name,
            "score":       score,
            "metric":      self.metric,
            "batch_index": batch_index,
        }
        with open(path, "wb") as f:
            pickle.dump({"model": model, "meta": meta}, f)
        with open(os.path.join(self.save_dir, "best_model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load_best_model(self):
        path = os.path.join(self.save_dir, "best_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Лучшая модель не найдена {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Загружена лучшая модель: {data['meta']}")
        return data["model"], data["meta"]

    def _save_report(self, report, batch_index):
        path = os.path.join(
            self.save_dir, f"validation_batch_{batch_index:04d}.json"
        )
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def load_history(self):
        history = []
        for f in sorted(os.listdir(self.save_dir)):
            if f.startswith("validation_batch_") and f.endswith(".json"):
                with open(os.path.join(self.save_dir, f)) as fp:
                    history.append(json.load(fp))
        return history
