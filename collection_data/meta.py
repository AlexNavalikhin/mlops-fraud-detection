import pandas as pd
import numpy as np

class MetaCalculator:
    def calculate(self, df, batch_index):
        return {
            "batch_index": batch_index,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "date_min": str(df['Transaction_Date'].min()),
            "date_max": str(df['Transaction_Date'].max()),
            "missing_total": int(df.isnull().sum().sum()),
            "missing_pct": round(df.isnull().mean().mean() * 100, 2),
            "duplicates": int(df.duplicated().sum()),
            "fraud_count": int(df['Is_Fraud'].sum()),
            "fraud_rate": round(df['Is_Fraud'].mean() * 100, 2),
            "amount_mean": round(float(df['Transaction_Amount'].mean()), 2),
            "amount_std":    round(float(df['Transaction_Amount'].std()), 2),
            "balance_mean":  round(float(df['Account_Balance'].mean()), 2),
            "top_transaction_type": str(df['Transaction_Type'].mode()[0]),
            "top_device_type": str(df['Device_Type'].mode()[0]),
            "unique_states": int(df['State'].nunique()),
        }
