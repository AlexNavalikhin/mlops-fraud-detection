import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging

logger = logging.getLogger(__name__)


class AutoEDA:
    def __init__(self, config):
        self.report_dir = config["eda"]["report_dir"]
        os.makedirs(self.report_dir, exist_ok=True)

    def run(self, df, batch_index):
        batch_dir = os.path.join(self.report_dir, f"batch_{batch_index:04d}")
        os.makedirs(batch_dir, exist_ok=True)

        stats = self._compute_stats(df)
        self._save_stats(stats, batch_dir)
        self._plot_numeric(df, batch_dir)
        self._plot_categorical(df, batch_dir)
        self._plot_correlation(df, batch_dir)
        self._plot_fraud_breakdown(df, batch_dir)

        logger.info(f"EDA батч {batch_index} сохранён в {batch_dir}")
        return stats

    def _compute_stats(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        stats = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "numeric_cols": num_cols,
            "cat_cols": cat_cols,
            "fraud_rate": round(df["Is_Fraud"].mean() * 100, 2),
        }

        desc = df[num_cols].describe().round(2)
        stats["numeric_stats"] = desc.to_dict()

        stats["cat_top3"] = {}
        for col in cat_cols:
            top3 = df[col].value_counts().head(3).to_dict()
            stats["cat_top3"][col] = {str(k): int(v) for k, v in top3.items()}

        return stats

    def _save_stats(self, stats, batch_dir):
        path = os.path.join(batch_dir, "stats.json")
        with open(path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    def _plot_numeric(self, df, batch_dir):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != "Is_Fraud"]
        if not num_cols:
            return

        n = len(num_cols)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(num_cols):
            axes[i].hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel("Значение")
            axes[i].set_ylabel("Частота")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Распределение числовых признаков", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, "numeric_distributions.png"), dpi=100)
        plt.close()

    def _plot_categorical(self, df, batch_dir):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if df[c].nunique() <= 20]
        if not cat_cols:
            return

        n    = len(cat_cols)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(cat_cols):
            counts = df[col].value_counts()
            axes[i].bar(counts.index, counts.values, color="#55A868", edgecolor="white")
            axes[i].set_title(col, fontsize=11)
            axes[i].tick_params(axis="x", rotation=30)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Распределение категориальных признаков", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, "categorical_distributions.png"), dpi=100)
        plt.close()

    def _plot_correlation(self, df, batch_dir):
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            return

        corr = df[num_cols].corr().round(2)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=45, ha="right")
        ax.set_yticklabels(num_cols)

        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                ax.text(j, i, corr.iloc[i, j],
                        ha="center", va="center", fontsize=8)

        ax.set_title("Корреляционная матрица", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, "correlation.png"), dpi=100)
        plt.close()

    def _plot_fraud_breakdown(self, df, batch_dir):
        cat_cols = ["Transaction_Type", "Merchant_Category", "Device_Type", "Account_Type"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        if not cat_cols:
            return

        fig, axes = plt.subplots(1, len(cat_cols),
                                 figsize=(5 * len(cat_cols), 4))
        if len(cat_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, cat_cols):
            fraud_rate = df.groupby(col)["Is_Fraud"].mean() * 100
            fraud_rate = fraud_rate.sort_values(ascending=False)
            bars = ax.bar(fraud_rate.index, fraud_rate.values, color="#C44E52", edgecolor="white")
            ax.set_title(f"Fraud rate по {col}", fontsize=11)
            ax.set_ylabel("Fraud rate (%)")
            ax.tick_params(axis="x", rotation=30)

            for bar, val in zip(bars, fraud_rate.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", fontsize=9)

        plt.suptitle("Fraud rate по категориям", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, "fraud_breakdown.png"), dpi=100)
        plt.close()
