import pandas as pd
import os
import logging
from datetime import datetime
from collection_data.storage import RawStorage
from collection_data.meta import MetaCalculator

import warnings
warnings.filterwarnings("ignore")

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename='logs/collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataCollector:
    def __init__(self, config):
        self.source_path = config['data']['source_path']
        self.batch_size = config['data']['batch_size']
        self.date_col = config['data']['date_column']
        self.storage = RawStorage(config['data']['raw_dir'])
        self.meta = MetaCalculator()

    def load_source(self):
        try:
            df = pd.read_csv(self.source_path)
            df[self.date_col] = pd.to_datetime(df[self.date_col], dayfirst=True)
            df = df.sort_values(self.date_col).reset_index(drop=True)
            logging.info(f"Датасет загружен: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Ошибка загрузки: {e}")
            raise

    def split_into_batches(self, df):
        batches = []
        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i : i + self.batch_size].copy()
            batches.append(batch)
        logging.info(f"Создано батчей: {len(batches)}")
        return batches

    def stream_next_batch(self):
        next_idx = self.storage.get_next_batch_index()
        all_batches = self.split_into_batches(self.load_source())

        if next_idx >= len(all_batches):
            logging.info("Новых батчей нет")
            return None

        batch = all_batches[next_idx]
        meta  = self.meta.calculate(batch, batch_index=next_idx)
        self.storage.save_batch(batch, meta, batch_index=next_idx)
        logging.info(f"Батч {next_idx} сохранён, строк: {len(batch)}")
        return batch
