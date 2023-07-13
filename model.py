from math import floor, log10
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

FEATURE_INFO_PATH = Path("data/features.yaml")
DATA_PATH = Path("data/train.csv")


class FeatureInfo:
    def __init__(self, yaml_path: Path) -> None:
        with yaml_path.open("r") as f:
            # Load feature data from yaml file
            self.__feature_data: Dict[str, Dict[str, Any]] = yaml.load(
                f.read(), Loader=yaml.SafeLoader
            )

    # Getters
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

    def is_renovatable(self, feature: str) -> bool:
        return self.__feature_data[feature].get("renovation")

    # Getters for lists of features
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

    def get_renovatable_features(self) -> List[str]:
        return [
            feature for feature in self.__feature_data if self.is_renovatable(feature)
        ]

    # Mapping functions
    def convert_labels_to_values(self, data: Dict[str, str]) -> Dict[str, str]:
        data = data.copy()
        # Change categorical feature values to labels
        for feature in self.get_categorical_features():
            data[feature] = self.get_values(feature)[
                self.get_labels(feature).index(data[feature])
            ]
        return data


def process_data(
    filepath: Path, features: list[str]
) -> Tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    # Read data from csv file
    df = pd.read_csv(filepath)

    # Shuffle data and reset incices
    df = df.sample(frac=1, random_state=3654).reset_index(drop=True)

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
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                feature_info.get_categorical_features(),
            ),
        ]
    )
    # Create model pipeline where data is preprocessed and then fed into a LinearRegression model
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LinearRegression())]
    )

    # Process data
    X_train, X_test, y_train, y_test = process_data(file, feature_info.get_features())
    # Train model
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    # Get mean absolute error, mean squared error and r2 score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # round to 3 significant figures
    test_results = {
        "mae": round(mae, -int(floor(log10(abs(mae)))) + 2),
        "mse": round(mse, -int(floor(log10(abs(mse)))) + 2),
        "r2": round(r2, -int(floor(log10(abs(r2)))) + 2),
    }

    return model, test_results


def find_price_increases(
    model: Pipeline,
    input_data: dict,
    feature_info: FeatureInfo,
) -> pd.DataFrame:
    # Convert input data into a dataframe with the correct columns
    input_df = pd.DataFrame(index=[0], columns=feature_info.get_features())
    for feature, value in input_data.items():
        # Assign value to feature in dataframe
        input_df[feature] = value

    # Get list of renovatable features
    renovatable_features = feature_info.get_renovatable_features()

    results = []
    # For each renovatable feature...
    for feature in renovatable_features:
        # Make a copy of the data
        X_up = input_df.copy()
        X_down = input_df.copy()

        # Determine the step
        if feature_info.get_type(feature) == "numerical":
            # If the feature is small (likely number of rooms), step is 1
            if feature_info.get_max(feature) <= 10:
                step = 1
            # If feature value is below 100, step is entire value
            elif X_up[feature][0] < 100:
                step = X_up[feature][0]
            # Else, step is 100
            else:
                step = 100

            # Increment and decrement the feature value by the step
            X_up[feature] += step
            X_down[feature] -= step
        else:  # feature type = categorical
            values = feature_info.get_values(feature)
            current_index = values.index(input_df[feature][0])

            # Get the next and previous values in the list of values
            next_index = min(current_index + 1, len(values) - 1)
            prev_index = max(current_index - 1, 0)

            # Set the new values
            X_up[feature] = values[next_index]
            X_down[feature] = values[prev_index]

        # Predict the new price and calculate the change
        for X in (X_up, X_down):
            try:
                # Predict the price
                new_price = model.predict(X)[0]
                old_price = model.predict(input_df)[0]
                # Calculate the change
                change = new_price - old_price

                # Only store the result if the price increased by at least 1000
                if change >= 1000:
                    # Round the change to the nearest integer
                    change = int(round(change, 0))
                    # Get the feature name and values
                    feature_label = feature_info.get_label(feature)
                    from_val = input_df[feature][0]
                    to_val = X[feature][0]
                    # Create a string describing the renovation
                    renovation = f"Change {feature_label} from {from_val} to {to_val}"
                    results.append(
                        (feature_label, from_val, to_val, renovation, change)
                    )
            except ValueError:
                # Exception raised if X contains a categorical value that is not in the training data
                continue

    return pd.DataFrame(
        results, columns=["Feature", "From", "To", "Renovation", "Price Increase"]
    )
