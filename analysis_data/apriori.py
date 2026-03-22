import pandas as pd
import numpy as np
import json
import os
import logging
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

logger = logging.getLogger(__name__)


class AssociationRulesMiner:
    def __init__(self, config):
        self.min_support = config["apriori"]["min_support"]
        self.min_confidence = config["apriori"]["min_confidence"]
        self.min_lift = config["apriori"]["min_lift"]
        self.n_rules = config["apriori"]["n_rules"]
        self.report_dir = config["apriori"]["report_dir"]
        os.makedirs(self.report_dir, exist_ok=True)

    def fit(self, df, batch_index):
        binary_df = self._binarize(df)
        frequent  = apriori(binary_df, min_support=self.min_support, use_colnames=True)

        if frequent.empty:
            logger.warning(f"Батч {batch_index}: частые наборы не найдены, попробуй снизить min_support")
            return pd.DataFrame()

        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=self.min_confidence
        )
        rules = rules[rules["lift"] >= self.min_lift]
        rules = rules.sort_values("lift", ascending=False)

        self._save_rules(rules, batch_index)
        self._log_top_rules(rules)
        return rules

    def _binarize(self, df):
        result = pd.DataFrame()

        cat_cols = ["Transaction_Type", "Merchant_Category", "Device_Type", "Account_Type", "State", "Gender"]
        for col in cat_cols:
            if col not in df.columns:
                continue
            for val in df[col].dropna().unique():
                col_name = f"{col}={val}"
                result[col_name] = (df[col] == val)

        if "Transaction_Amount" in df.columns:
            median = df["Transaction_Amount"].median()
            result["Amount_high"] = (df["Transaction_Amount"] > median)
            result["Amount_low"] = (df["Transaction_Amount"] <= median)

        if "Account_Balance" in df.columns:
            median = df["Account_Balance"].median()
            result["Balance_high"] = (df["Account_Balance"] > median)
            result["Balance_low"] = (df["Account_Balance"] <= median)

        if "Is_Fraud" in df.columns:
            result["Is_Fraud=1"] = (df["Is_Fraud"] == 1)
            result["Is_Fraud=0"] = (df["Is_Fraud"] == 0)

        return result.astype(bool)

    def get_fraud_rules(self, rules, top_n=5):
        if rules.empty:
            return rules
        mask = rules["consequents"].apply(lambda x: "Is_Fraud=1" in x)
        return rules[mask].head(top_n)

    def _save_rules(self, rules, batch_index):
        path = os.path.join(
            self.report_dir, f"rules_batch_{batch_index:04d}.json"
        )
        records = []
        for _, row in rules.head(self.n_rules).iterrows():
            records.append({
                "antecedents": list(row["antecedents"]),
                "consequents": list(row["consequents"]),
                "support": round(float(row["support"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4),
            })
        with open(path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"Сохранено {len(records)} правил в {path}")

    def _log_top_rules(self, rules):
        fraud_rules = self.get_fraud_rules(rules, top_n=5)
        if fraud_rules.empty:
            logger.info("Правил с Is_Fraud=1 не найдено")
            return
        logger.info("Топ правил (Is_Fraud=1):")
        for _, row in fraud_rules.iterrows():
            ant = ", ".join(row["antecedents"])
            con = ", ".join(row["consequents"])
            logger.info(f"  [{ant}] => [{con}] | conf={row['confidence']:.2f} lift={row['lift']:.2f}")
