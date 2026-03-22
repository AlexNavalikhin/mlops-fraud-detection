import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, config):
        self.max_missing_pct = config["quality"]["max_missing_pct"]

    def clean(self, df, quality_report):
        df = df.copy()
        original_len = len(df)

        df = self._remove_duplicates(df)
        df = self._drop_high_missing_columns(df, quality_report)
        df = self._fill_missing(df)
        df = self._remove_invalid_rows(df)

        logger.info(
            f"Очистка: {original_len} в {len(df)} строк "
            f"(удалено {original_len - len(df)})"
        )
        return df

    def _remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Дубликаты удалены: {before - len(df)} строк")
        return df

    def _drop_high_missing_columns(self, df, quality_report):
        to_drop = []
        missing = quality_report.get("missing_per_column", {})
        for col, count in missing.items():
            col_pct = count / len(df) * 100
            if col_pct > self.max_missing_pct:
                to_drop.append(col)

        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"Удалены колонки с высоким % пропусков: {to_drop}")
        return df

    def _fill_missing(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def _remove_invalid_rows(self, df):
        before = len(df)

        if "Transaction_Amount" in df.columns:
            df = df[df["Transaction_Amount"] >= 0]

        if "Account_Balance" in df.columns:
            df = df[df["Account_Balance"] >= 0]

        if "Age" in df.columns:
            df = df[df["Age"].between(0, 120)]

        logger.info(f"Некорректные строки удалены: {before - len(df)}")
        return df
