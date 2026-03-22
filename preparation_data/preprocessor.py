import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    def __init__(self, config):
        self.num_strategy = config["preprocessor"]["num_strategy"]   # standard / minmax
        self.cat_strategy = config["preprocessor"]["cat_strategy"]   # ordinal / onehot
        self.save_dir = config["preprocessor"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        self.num_cols = [
            "Transaction_Amount", "Account_Balance", "Age"
        ]
        self.cat_cols = [
            "Transaction_Type", "Merchant_Category",
            "Device_Type", "Account_Type", "State", "Gender"
        ]
        self.drop_cols = [
            "Customer_ID", "Customer_Name", "Transaction_ID",
            "Customer_Email", "Customer_Contact", "Merchant_ID",
            "Transaction_Location", "Transaction_Time",
            "Transaction_Currency", "Transaction_Description",
            "Bank_Branch", "City"
        ]
        self.target_col = "Is_Fraud"

        self._num_imputer = SimpleImputer(strategy="median")
        self._cat_imputer = SimpleImputer(strategy="most_frequent")
        self._num_scaler  = self._make_scaler()
        self._cat_encoder = self._make_encoder()

    def fit_transform(self, df):
        df = self._prepare(df)
        X, y = self._split_xy(df)

        X[self.num_cols] = self._num_imputer.fit_transform(X[self.num_cols])
        X[self.cat_cols] = self._cat_imputer.fit_transform(X[self.cat_cols])
        X = self._encode_cat(X, fit=True)
        X = self._scale_num(X, fit=True)

        self.save()
        return X, y

    def transform(self, df):
        df = self._prepare(df)
        X, y = self._split_xy(df)

        X[self.num_cols] = self._num_imputer.transform(X[self.num_cols])
        X[self.cat_cols] = self._cat_imputer.transform(X[self.cat_cols])
        X = self._encode_cat(X, fit=False)
        X = self._scale_num(X, fit=False)
        return X, y

    def _prepare(self, df):
        df = df.copy()
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        cols_to_drop += ["Transaction_Date"]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        return df

    def _split_xy(self, df):
        y = df[self.target_col].values if self.target_col in df.columns else None
        X = df.drop(columns=[self.target_col], errors="ignore")

        keep = [c for c in self.num_cols + self.cat_cols if c in X.columns]
        return X[keep], y

    def _make_scaler(self):
        if self.num_strategy == "minmax":
            return MinMaxScaler()
        return StandardScaler()

    def _make_encoder(self):
        if self.cat_strategy == "onehot":
            return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def _scale_num(self, X, fit):
        num_cols = [c for c in self.num_cols if c in X.columns]
        if not num_cols:
            return X
        if fit:
            X[num_cols] = self._num_scaler.fit_transform(X[num_cols])
        else:
            X[num_cols] = self._num_scaler.transform(X[num_cols])
        return X

    def _encode_cat(self, X, fit):
        cat_cols = [c for c in self.cat_cols if c in X.columns]
        if not cat_cols:
            return X

        if self.cat_strategy == "onehot":
            if fit:
                encoded = self._cat_encoder.fit_transform(X[cat_cols])
            else:
                encoded = self._cat_encoder.transform(X[cat_cols])
            new_cols = self._cat_encoder.get_feature_names_out(cat_cols)
            encoded_df = pd.DataFrame(encoded, columns=new_cols, index=X.index)
            X = X.drop(columns=cat_cols).join(encoded_df)
        else:
            if fit:
                X[cat_cols] = self._cat_encoder.fit_transform(X[cat_cols])
            else:
                X[cat_cols] = self._cat_encoder.transform(X[cat_cols])
        return X

    def save(self):
        path = os.path.join(self.save_dir, "preprocessor.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(save_dir):
        path = os.path.join(save_dir, "preprocessor.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
