import time
import tracemalloc
import pickle
import json
import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelServing:
    def __init__(self, config):
        self.save_dir = config["serving"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = None
        self.meta  = None
        self.perf_log = []

    def load_model(self, validator):
        self.model, self.meta = validator.load_best_model()
        logger.info(f"Модель загружена: {self.meta}")

    def save_production_model(self):
        if self.model is None:
            raise RuntimeError("Сначала вызови load_model()")
        path = os.path.join(self.save_dir, "production_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "meta": self.meta}, f)
        logger.info(f"Продуктовая модель сохранена: {path}")

    @staticmethod
    def load_production_model(save_dir):
        path = os.path.join(save_dir, "production_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Продуктовая модель не найдена: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Продуктовая модель загружена: {data['meta']}")
        return data["model"], data["meta"]

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        tracemalloc.start()
        start = time.perf_counter()

        preds = self.model.predict(X)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_kb = round(peak_kb / 1024, 2)

        self._log_performance(len(X), elapsed_ms, peak_kb)
        return preds

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Модель не загружена")
        return self.model.predict_proba(X)

    def _log_performance(self, n_rows, elapsed_ms, peak_kb):
        entry = {
            "n_rows": n_rows,
            "elapsed_ms": elapsed_ms,
            "peak_kb": peak_kb,
            "ms_per_row": round(elapsed_ms / max(n_rows, 1), 4),
        }
        self.perf_log.append(entry)
        logger.info(
            f"Inference: {n_rows} строк | "
            f"{elapsed_ms}ms | "
            f"{peak_kb}KB peak"
        )
        self._save_perf_log()

    def _save_perf_log(self):
        path = os.path.join(self.save_dir, "performance_log.json")
        with open(path, "w") as f:
            json.dump(self.perf_log, f, indent=2)

    def get_performance_summary(self):
        if not self.perf_log:
            return {}
        elapsed = [e["elapsed_ms"] for e in self.perf_log]
        memory  = [e["peak_kb"] for e in self.perf_log]
        return {
            "n_calls": len(self.perf_log),
            "avg_elapsed_ms": round(float(np.mean(elapsed)), 2),
            "max_elapsed_ms": round(float(np.max(elapsed)), 2),
            "avg_peak_kb": round(float(np.mean(memory)), 2),
            "max_peak_kb": round(float(np.max(memory)), 2),
        }