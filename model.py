from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

FEATURE_INFO_PATH = Path("data/features.yaml")
DATA_PATH = Path("data/train.csv")


class FeatureInfo:
    def __init__(self, yaml_path: Path) -> None:
        with yaml_path.open("r") as f:
            self.__feature_data = yaml.load(f.read(), Loader=yaml.SafeLoader)

    def get_type(self, feature: str) -> str:
        return self.__feature_data[feature].get("type")

    def get_input(self, feature: str) -> str:
        return self.__feature_data[feature].get("input")

    def get_label(self, feature: str) -> str:
        return self.__feature_data[feature].get("label")

    def get_min(self, feature: str) -> Optional[int]:
        return int(self.__feature_data[feature].get("min"))

    def get_max(self, feature: str) -> Optional[int]:
        return int(self.__feature_data[feature].get("max"))

    def get_values(self, feature: str) -> Optional[List[str]]:
        return self.__feature_data[feature].get("values")

    def get_labels(self, feature: str) -> Optional[List[str]]:
        return self.__feature_data[feature].get("labels")

    def get_na(self, feature: str) -> Optional[str]:
        return self.__feature_data[feature].get("NA")

    def get_features(self) -> List[str]:
        return list(self.__feature_data.keys())

    def get_numerical_features(self) -> List[str]:
        return [
            feature
            for feature in self.__feature_data
            if self.get_type(feature) == "numerical"
        ]

    def get_categorical_features(self) -> List[str]:
        return [
            feature
            for feature in self.__feature_data
            if self.get_type(feature) == "categorical"
        ]

    def map_label_to_value(self, feature: str, label: str) -> str:
        return self.get_values(feature)[self.get_labels(feature).index(label)]

    def map_value_to_label(self, feature: str, value: str) -> str:
        return self.get_label(feature)[self.get_value(feature).index(value)]


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


def generate_models(file: Path, feature_info: FeatureInfo):
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#column-transformer-with-mixed-types
    # Assign numerical features to MinMaxScaler and categorical features to OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), feature_info.get_numerical_features()),
            ("cat", OneHotEncoder(), feature_info.get_categorical_features()),
        ]
    )
    # Create model pipeline where data is preprocessed and then fed into a LinearRegression model
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LinearRegression())]
    )

    # Process data and train model
    X_train, X_test, y_train, y_test = process_data(file, feature_info.get_features())
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    test_results = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return model, test_results

