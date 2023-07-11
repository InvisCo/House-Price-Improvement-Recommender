from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class FeatureInfo:
    def __init__(self, yaml_path: Path):
        self.feature_data = yaml.load(yaml_path)

    def get_type(self, feature):
        return self.feature_data[feature]["type"]

    def get_min(self, feature):
        return self.feature_data[feature].get("min")

    def get_max(self, feature):
        return self.feature_data[feature].get("max")

    def get_values(self, feature):
        return self.feature_data[feature].get("values")

    def get_labels(self, feature):
        return self.feature_data[feature].get("labels")

    def get_na(self, feature):
        return self.feature_data[feature].get("NA")

    def get_features(self):
        return list(self.feature_data.keys())

    def get_numerical_features(self):
        return [
            feature
            for feature in self.feature_data
            if self.get_type(feature) == "numerical"
        ]

    def get_categorical_features(self):
        return [
            feature
            for feature in self.feature_data
            if self.get_type(feature) == "categorical"
        ]

    def get_value_label_map(self):
        output = {}
        for feature in self.get_categorical_features():
            output.update(
                {
                    self.get_values(feature): self.get_labels(feature),
                    self.get_labels(feature): self.get_values(feature),
                }
            )
        return output


def process_data(
    filepath: Path, features: list[str]
) -> Tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    # Read data from csv file
    df = pd.read_csv(filepath)

    # Sum values in columns "BsmtFinSF1" and "BsmtFinSF2" to "BsmtFinSF"
    df["BsmtFinSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]

    # Basement Condition blank values are NA
    df["BsmtCond"] = df["BsmtCond"].fillna("NA")

    # Features and target
    x = df[features].fillna(0)
    y = df["SalePrice"]

    # Split data into training and testing sets
    train_test_split = int(0.8 * len(df.values))
    return (
        x[:train_test_split],  # x_train
        x[train_test_split:],  # x_test
        y[:train_test_split],  # y_train
        y[train_test_split:],  # y_test
    )


def generate_models(filename: str, feature_info: FeatureInfo):
    # Set filepath to the data
    filepath = Path(filename)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), feature_info.get_numerical_features()),
            ("cat", OneHotEncoder(), feature_info.get_categorical_features()),
        ]
    )
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LinearRegression())]
    )

    # Process data and train model
    X_train, X_test, y_train, y_test = process_data(filepath)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    test_results = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return model, test_results

