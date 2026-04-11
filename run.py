import argparse
import sys
import yaml
import pandas as pd
import logging
import os

from collection_data.collector import DataCollector
from analysis_data.quality import DataQuality
from analysis_data.cleaner import DataCleaner
from analysis_data.apriori import AssociationRulesMiner
from analysis_data.eda import AutoEDA
from analysis_data.drift import DriftDetector
from preparation_data.preprocessor import DataPreprocessor
from model_training.trainer import ModelTrainer
from model_validation.validator import ModelValidator
from model_serving.serving import ModelServing

import warnings
warnings.filterwarnings("ignore")

def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml не найден по пути: {os.path.abspath(path)}")
    with open(path) as f:
        return yaml.safe_load(f)

logger = logging.getLogger(__name__)

def build_pipeline(config):
    return {
        "collector": DataCollector(config),
        "quality": DataQuality(config),
        "cleaner": DataCleaner(config),
        "apriori": AssociationRulesMiner(config),
        "eda": AutoEDA(config),
        "drift": DriftDetector(config),
        "preprocessor": DataPreprocessor(config),
        "trainer": ModelTrainer(config),
        "validator": ModelValidator(config),
        "serving": ModelServing(config),
    }

def _process_batch(p, batch, batch_index):
    quality_report = p["quality"].evaluate(batch, batch_index)
    if not quality_report["passed"]:
        logger.warning(f"Батч {batch_index} не прошёл контроль качества")

    clean_batch = p["cleaner"].clean(batch, quality_report)
    p["apriori"].fit(clean_batch, batch_index)
    p["eda"].run(clean_batch, batch_index)

    if batch_index == 0:
        p["drift"].set_reference(clean_batch)
        p["drift"].save()
        X, y = p["preprocessor"].fit_transform(clean_batch)
    else:
        p["drift"] = DriftDetector.load(p["drift"].report_dir)
        p["drift"].detect(clean_batch, batch_index)
        p["preprocessor"] = DataPreprocessor.load(p["preprocessor"].save_dir)
        X, y = p["preprocessor"].transform(clean_batch)

    p["trainer"].fit(X, y)
    p["validator"].evaluate(p["trainer"], X, y, batch_index)
    p["serving"].load_model(p["validator"])
    p["serving"].save_production_model()

    logger.info(f"Батч {batch_index} успешно обработан")
    print(f"Батч {batch_index} обработан")

def run_update(config, process_all=False, n_batches=None):
    try:
        p = build_pipeline(config)
        count = 0

        while True:
            if n_batches is not None and count >= n_batches:
                break

            batch = p["collector"].stream_next_batch()
            if batch is None:
                break

            batch_index = p["collector"].storage.get_next_batch_index() - 1
            _process_batch(p, batch, batch_index)
            count += 1

            if not process_all and n_batches is None:
                break

        return True

    except Exception as e:
        logger.error(f"Ошибка в update: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference(config, file_path):
    try:
        df = pd.read_csv(file_path)

        preprocessor = DataPreprocessor.load(config["preprocessor"]["save_dir"])
        X, _ = preprocessor.transform(df)

        model, meta = ModelServing.load_production_model(config["serving"]["save_dir"])
        preds = model.predict(X)

        df["predict"] = preds
        out_path = os.path.join(config["serving"]["save_dir"], "inference_result.csv")
        df.to_csv(out_path, index=False)

        logger.info(f"Inference выполнен: {len(df)} строк -> {out_path}")
        print(out_path)
        return out_path

    except Exception as e:
        logger.error(f"Ошибка в inference: {e}")
        raise


def run_summary(config):
    try:
        report = {}

        quality = DataQuality(config)
        report["data_quality"] = quality.load_history()

        validator = ModelValidator(config)
        report["model_metrics"] = validator.load_history()

        try:
            _, meta = ModelServing.load_production_model(config["serving"]["save_dir"])
            report["best_model"] = meta
        except FileNotFoundError:
            report["best_model"] = None

        serving = ModelServing(config)
        perf_path = os.path.join(config["serving"]["save_dir"], "performance_log.json")
        if os.path.exists(perf_path):
            import json
            with open(perf_path) as f:
                serving.perf_log = json.load(f)
            report["performance"] = serving.get_performance_summary()

        import json
        out_path = "reports/summary.json"
        os.makedirs("reports", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary сохранён: {out_path}")
        print(out_path)
        return out_path

    except Exception as e:
        logger.error(f"Ошибка в summary: {e}")
        raise

def run_reset(config):
    import shutil

    dirs_to_clean = [
        config["data"]["raw_dir"],
        config["preprocessor"]["save_dir"],
        config["trainer"]["save_dir"],
        config["validator"]["save_dir"],
        config["serving"]["save_dir"],
        config["quality"]["report_dir"],
        config["drift"]["report_dir"],
        config["eda"]["report_dir"],
        config["apriori"]["report_dir"],
        "reports",
    ]

    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            logger.info(f"Очищено: {d}")

    print("Хранилища очищены. Запусти update для обучения с нуля.")
    return True

def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline")
    parser.add_argument("-mode", required=True, choices=["update", "inference", "summary", "reset"])
    parser.add_argument("-file", default=None, help="Путь к файлу для inference")
    parser.add_argument("-config", default="config.yaml", help="Путь к конфигу")
    parser.add_argument("-all", action="store_true", help="Обработать все батчи")
    parser.add_argument("-batches", type=int, default=None, help="Количество батчей для обработки")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "update":
        result = run_update(
            config,
            process_all=args.all,
            n_batches=args.batches
        )
        print(result)

    elif args.mode == "inference":
        if not args.file:
            print("Укажи файл через -file")
            sys.exit(1)
        run_inference(config, args.file)

    elif args.mode == "summary":
        run_summary(config)

    elif args.mode == "reset":
        run_reset(config)


if __name__ == "__main__":
    main()
